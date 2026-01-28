from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf

import discord
from discord import app_commands
from discord.ext import voice_recv  # pip install discord-ext-voice-recv

from app.config import (
    CLONE_MIN_SECONDS,
    CLONE_RECORD_SECONDS,
    CLONE_SAMPLE_TEXT_ES,
)
from app.tts.audio import prepare_tts_pcm, trim_silence_energy
from app.tts.engine import TTSEngine
from app.tts.text import preprocess_discord_text

LOG = logging.getLogger("qwen-discord-tts")


# -----------------------------
# Voice receiving sink: record ONE user only
# -----------------------------
class SingleUserPCMCollector(voice_recv.AudioSink):
    """Collect decoded PCM for one user only; everyone else ignored.

    We keep the PCM in memory and the command reads it directly.
    """

    def __init__(self, target_user_id: int):
        super().__init__()
        self.target_user_id = int(target_user_id)
        self._buf = bytearray()

        # discord voice decode is typically 48kHz, stereo, 16-bit PCM
        self.sample_rate = 48000
        self.channels = 2

    def wants_opus(self) -> bool:
        return False  # we want decoded PCM

    def write(self, user: discord.abc.User | discord.Member | None, data: voice_recv.VoiceData):
        if user is None or int(getattr(user, "id", -1)) != self.target_user_id:
            return
        pcm = getattr(data, "pcm", None)
        if pcm:
            self._buf.extend(pcm)

    def cleanup(self) -> None:
        # Required by discord.ext.voice_recv.AudioSink (abstract). We don't need to do anything here.
        return

    def mono_float32(self) -> np.ndarray:
        """Return the captured audio as mono float32 in [-1, 1]. Empty array if none."""
        if not self._buf:
            return np.zeros((0,), dtype=np.float32)

        x = np.frombuffer(bytes(self._buf), dtype=np.int16)
        if self.channels == 2:
            n = (x.size // 2) * 2
            x = x[:n].reshape(-1, 2).astype(np.float32).mean(axis=1)
        else:
            x = x.astype(np.float32)

        # int16 -> float32 [-1,1]
        x = x / np.float32(32768.0)
        return np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)


# -----------------------------
# Voice client with safety guard for voice-recv cleanup races
# -----------------------------
class SafeVoiceRecvClient(voice_recv.VoiceRecvClient):
    """Guard against voice-recv cleanup races when stopping listening."""

    def _remove_ssrc(self, user_id: int) -> None:  # pragma: no cover - lib-level guard
        reader = getattr(self, "_reader", None)
        if reader is None or reader is discord.utils.MISSING:
            return
        if not hasattr(reader, "speaking_timer"):
            return
        super()._remove_ssrc(user_id=user_id)


# -----------------------------
# Guild voice player with queue
# -----------------------------
@dataclass
class TTSJob:
    tts_future: "asyncio.Future[Optional[bytes]]"


@dataclass
class GuildState:
    voice_client: discord.VoiceClient
    voice_channel_id: int
    text_channel_id: int
    queue: "asyncio.Queue[TTSJob]"
    worker_task: asyncio.Task


class Bot(discord.Client):
    def __init__(self, tts: TTSEngine):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.voice_states = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.tts = tts
        self.guild_state: Dict[int, GuildState] = {}

        # Prevent concurrent enrollments per guild + suppress TTS for enrolling user
        self._clone_lock: Dict[int, asyncio.Lock] = {}
        self._cloning_users: set[int] = set()
        self._admins: set[int] = {441597233150951425}
        self._disguises: Dict[int, int] = {}

    def _is_admin(self, user_id: int) -> bool:
        return int(user_id) in self._admins

    async def setup_hook(self):
        LOG.info("Syncing application commands")
        guild_id = os.environ.get("DISCORD_SYNC_GUILD_ID")
        if guild_id:
            guild = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            LOG.info("Synced commands to guild_id=%s", guild_id)
        else:
            await self.tree.sync()

    async def on_ready(self):
        LOG.info("Logged in as %s (%s)", self.user, self.user.id)

    async def _player_worker(self, guild_id: int) -> None:
        """Single ordered pipeline per guild: dequeue chat messages in order, synthesize, then play.

        This guarantees voice playback follows chat order even if later messages would synth faster.
        """
        st = self.guild_state[guild_id]
        LOG.info("Player worker started guild_id=%s", guild_id)

        while True:
            job = await st.queue.get()
            queue_after_get = st.queue.qsize()
            wav_bytes: Optional[bytes] = None
            try:
                # Wait for voice client connectivity before consuming the job.
                while True:
                    # Refresh voice client reference in case it was replaced/reconnected.
                    st = self.guild_state.get(guild_id, st)
                    vc = st.voice_client
                    if vc.is_connected():
                        break
                    LOG.info("Voice client disconnected guild_id=%s; attempting reconnect", guild_id)
                    await self._ensure_voice_connected(guild_id)
                    await asyncio.sleep(1.0)

                # Await the engine batch queue in-order for this guild
                try:
                    wait_start = time.monotonic()
                    wav_bytes = await job.tts_future
                    LOG.debug(
                        "TTS future resolved guild_id=%s wait_ms=%.2f",
                        guild_id,
                        (time.monotonic() - wait_start) * 1000.0,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOG.warning("TTS error: %s", exc)
                    continue
                if wav_bytes is None:
                    # Prompt missing (e.g., user forgot voice mid-queue). Skip.
                    LOG.debug("TTS returned empty guild_id=%s", guild_id)
                    continue

                prep_start = time.monotonic()
                pcm_bytes, _sr, _ch = await asyncio.to_thread(prepare_tts_pcm, wav_bytes, 48000, True)
                LOG.debug(
                    "PCM prep guild_id=%s in_bytes=%d out_bytes=%d prep_ms=%.2f queue_after_get=%d",
                    guild_id,
                    len(wav_bytes),
                    len(pcm_bytes),
                    (time.monotonic() - prep_start) * 1000.0,
                    queue_after_get,
                )
                if not pcm_bytes:
                    LOG.debug("PCM conversion returned empty guild_id=%s", guild_id)
                    continue

                src_buf = io.BytesIO(pcm_bytes)
                src = discord.PCMAudio(src_buf)

                done = asyncio.Event()

                def after_play(err: Optional[Exception]) -> None:
                    if err:
                        LOG.warning("Playback error: %s", err)
                    done.set()

                LOG.debug("Playback start guild_id=%s bytes=%d", guild_id, len(pcm_bytes))
                if vc.is_playing():
                    LOG.warning("Voice client already playing guild_id=%s; stopping stale audio", guild_id)
                    vc.stop()
                try:
                    vc.play(src, after=after_play)
                except discord.ClientException as exc:
                    LOG.warning("Playback start failed guild_id=%s err=%s", guild_id, exc)
                    continue

                playback_timeout = 0.0
                if _sr > 0 and _ch > 0:
                    playback_timeout = len(pcm_bytes) / float(_sr * _ch * 2) + 5.0
                else:
                    playback_timeout = 15.0
                try:
                    await asyncio.wait_for(done.wait(), timeout=playback_timeout)
                except asyncio.TimeoutError:
                    LOG.warning("Playback timeout guild_id=%s timeout=%.2f", guild_id, playback_timeout)
                    vc.stop()
                    continue
                duration_sec = 0.0
                if _sr > 0 and _ch > 0:
                    duration_sec = len(pcm_bytes) / float(_sr * _ch * 2)
                LOG.info(
                    "Playback done guild_id=%s seconds=%.2f queue_after=%d",
                    guild_id,
                    duration_sec,
                    st.queue.qsize(),
                )

            finally:
                st.queue.task_done()
        LOG.info("Player worker stopped guild_id=%s", guild_id)

    async def _join(self, interaction: discord.Interaction, channel: discord.VoiceChannel | None = None) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return

        if channel is None:
            member = interaction.user
            if not isinstance(member, discord.Member):
                member = interaction.guild.get_member(interaction.user.id)

            if not member or not member.voice or not member.voice.channel:
                await interaction.response.send_message("You are not in a voice channel.", ephemeral=True)
                return

            channel = member.voice.channel

        vc = interaction.guild.voice_client
        if vc and vc.is_connected():
            await vc.move_to(channel)
        else:
            # IMPORTANT: VoiceRecvClient enables receiving audio for /clone
            vc = await channel.connect(cls=SafeVoiceRecvClient)

        q: asyncio.Queue[TTSJob] = asyncio.Queue()
        guild_id = int(interaction.guild_id)

        # Replace existing state cleanly
        old = self.guild_state.get(guild_id)
        if old:
            try:
                old.worker_task.cancel()
            except Exception:
                pass
            while True:
                try:
                    job = old.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                job.tts_future.cancel()
                old.queue.task_done()

        # Create state first, then start worker (avoid KeyError race)
        placeholder_task = asyncio.create_task(asyncio.sleep(0))
        self.guild_state[guild_id] = GuildState(
            voice_client=vc,
            voice_channel_id=int(channel.id),
            text_channel_id=int(interaction.channel_id),
            queue=q,
            worker_task=placeholder_task,
        )
        self.guild_state[guild_id].worker_task.cancel()  # cancel placeholder
        self.guild_state[guild_id].worker_task = asyncio.create_task(self._player_worker(guild_id))

        await interaction.response.send_message(
            f"Joined **{channel.name}**. I will speak messages from <#{interaction.channel_id}>.",
            ephemeral=True,
        )
        LOG.info("Joined voice guild_id=%s channel_id=%s", interaction.guild_id, channel.id)

    async def _leave(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        st = self.guild_state.get(int(interaction.guild_id))
        if not st:
            await interaction.response.send_message("I'm not connected.", ephemeral=True)
            return

        try:
            st.worker_task.cancel()
        except Exception:
            pass
        while True:
            try:
                job = st.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            job.tts_future.cancel()
            st.queue.task_done()
        await st.voice_client.disconnect(force=True)
        self.guild_state.pop(int(interaction.guild_id), None)
        await interaction.response.send_message("Left voice chat.", ephemeral=True)
        LOG.info("Left voice guild_id=%s", interaction.guild_id)

    async def _ensure_voice_connected(self, guild_id: int) -> None:
        st = self.guild_state.get(guild_id)
        if not st:
            return
        vc = st.voice_client
        if vc.is_connected():
            return
        guild = self.get_guild(guild_id)
        if not guild:
            return
        channel = guild.get_channel(int(st.voice_channel_id))
        if not isinstance(channel, discord.VoiceChannel):
            return
        try:
            new_vc = await channel.connect(cls=SafeVoiceRecvClient, reconnect=True)
        except Exception as exc:
            LOG.warning("Reconnect failed guild_id=%s channel_id=%s err=%s", guild_id, st.voice_channel_id, exc)
            return
        st.voice_client = new_vc

    async def _set_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        st = self.guild_state.get(int(interaction.guild_id))
        if not st:
            await interaction.response.send_message("I'm not in voice. Use /join first.", ephemeral=True)
            return
        st.text_channel_id = int(channel.id)
        await interaction.response.send_message(f"Now reading messages from {channel.mention}.", ephemeral=True)
        LOG.info("Set text channel guild_id=%s channel_id=%s", interaction.guild_id, channel.id)

    async def _clone(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return

        member = interaction.user
        if not isinstance(member, discord.Member):
            member = interaction.guild.get_member(interaction.user.id)

        if not member or not member.voice or not member.voice.channel:
            await interaction.response.send_message("You are not in a voice channel.", ephemeral=True)
            return

        guild_id = int(interaction.guild_id)
        lock = self._clone_lock.setdefault(guild_id, asyncio.Lock())
        if lock.locked():
            await interaction.response.send_message("A clone session is already running in this guild.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        LOG.info("Clone started guild_id=%s user_id=%s", interaction.guild_id, interaction.user.id)

        async with lock:
            # Ensure bot is connected with VoiceRecvClient
            st = self.guild_state.get(guild_id)
            if not st or not st.voice_client.is_connected():
                # connect and create state
                vc = await member.voice.channel.connect(cls=SafeVoiceRecvClient)
                q: asyncio.Queue[TTSJob] = asyncio.Queue()

                placeholder_task = asyncio.create_task(asyncio.sleep(0))
                self.guild_state[guild_id] = GuildState(
                    voice_client=vc,
                    voice_channel_id=int(member.voice.channel.id),
                    text_channel_id=int(interaction.channel_id),
                    queue=q,
                    worker_task=placeholder_task,
                )
                self.guild_state[guild_id].worker_task.cancel()
                self.guild_state[guild_id].worker_task = asyncio.create_task(self._player_worker(guild_id))
                st = self.guild_state[guild_id]
            else:
                # move to caller channel (so we can hear them)
                try:
                    await st.voice_client.move_to(member.voice.channel)
                    st.voice_channel_id = int(member.voice.channel.id)
                except Exception:
                    pass

            vc = st.voice_client

            if not hasattr(vc, "listen") or not hasattr(vc, "stop_listening"):
                await interaction.followup.send(
                    "Voice receive is not enabled. Ensure you're connected with VoiceRecvClient "
                    "(discord-ext-voice-recv) and not plain discord.py VoiceClient.",
                    ephemeral=True,
                )
                return

            self._cloning_users.add(int(member.id))

            try:
                await interaction.followup.send(
                    "🎙️ **Voice clone enrollment**\n\n"
                    f"Read the following text out loud **now** (natural pace, clear mic).\n"
                    f"I will record you for **{CLONE_RECORD_SECONDS}s** and ignore everyone else:\n\n"
                    f"```{CLONE_SAMPLE_TEXT_ES}```",
                    ephemeral=True,
                )

                sink = SingleUserPCMCollector(int(member.id))

                loop = asyncio.get_running_loop()
                done_fut: asyncio.Future[Optional[Exception]] = loop.create_future()

                def after_listen(err: Optional[Exception]) -> None:
                    if not done_fut.done():
                        loop.call_soon_threadsafe(done_fut.set_result, err)

                vc.listen(sink, after=after_listen)
                await asyncio.sleep(CLONE_RECORD_SECONDS)
                vc.stop_listening()

                # Give the recv thread a moment to flush the last frames
                await asyncio.sleep(0.25)

                err = await done_fut
                if err:
                    raise RuntimeError(f"recording error: {err}")

                audio = sink.mono_float32()
                sr = sink.sample_rate

                # Trim leading/trailing silence (energy-based VAD)
                audio = trim_silence_energy(audio, int(sr))

                if audio.size < int(sr * CLONE_MIN_SECONDS):
                    await interaction.followup.send(
                        "I captured too little audio (or none).\n"
                        "Fixes: make sure Discord input is working, speak continuously, disable push-to-talk, "
                        "and check that the bot has permission to connect/speak in the channel.",
                        ephemeral=True,
                    )
                    return

                # Optional: save the capture for debugging
                if os.environ.get("QWEN_TTS_CLONE_DEBUG_WAV", "0") == "1":
                    debug_dir = Path(os.environ.get("QWEN_TTS_CLONE_DEBUG_DIR", "clone_debug")).resolve()
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        str(debug_dir / f"{member.id}.wav"),
                        (audio * 32767).astype(np.int16),
                        int(sr),
                        subtype="PCM_16",
                    )

                prompt_items = self.tts.build_clone_prompt_items(
                    ref_audio=(audio, int(sr)),
                    ref_text=CLONE_SAMPLE_TEXT_ES,
                )

                out_pt = self.tts.prompt_path(int(member.id))
                self.tts.save_prompt_items_pt(prompt_items, out_pt)

                # Clear cache so next synthesis reloads the fresh prompt
                self.tts.prompt_cache.pop(out_pt, None)
                self.tts.set_prompt_exists(int(member.id), True)
                LOG.info("Clone saved guild_id=%s user_id=%s prompt=%s", interaction.guild_id, member.id, out_pt)

                await interaction.followup.send(
                    "✅ Enrolled your voice.",
                    ephemeral=True,
                )

            finally:
                self._cloning_users.discard(int(member.id))
                LOG.info("Clone finished guild_id=%s user_id=%s", interaction.guild_id, member.id)

    async def _forget(self, interaction: discord.Interaction) -> None:
        """Forget the invoking user's enrolled voice (delete VOICES_DIR/<user_id>.pt)."""
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return

        user_id = int(interaction.user.id)

        if user_id in self._cloning_users:
            await interaction.response.send_message("You're currently enrolling. Finish /clone first.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        changed = await asyncio.to_thread(self.tts.forget_user, user_id, True)

        if changed:
            await interaction.followup.send("🧹 Deleted your enrolled prompt file.", ephemeral=True)
        else:
            await interaction.followup.send("No enrolled prompt file was found for you.", ephemeral=True)
        LOG.info("Forget user_id=%s deleted=%s", user_id, changed)

    async def _disguise(self, interaction: discord.Interaction, target: discord.Member) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        if not self._is_admin(int(interaction.user.id)):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return

        target_id = int(target.id)
        caller_id = int(interaction.user.id)
        if target_id == caller_id:
            self._disguises.pop(caller_id, None)
            await interaction.response.send_message("Disguise cleared. Using your own voice.", ephemeral=True)
            return

        if not self.tts.prompt_exists(target_id):
            await interaction.response.send_message(
                f"{target.mention} does not have an enrolled voice.",
                ephemeral=True,
            )
            return

        self._disguises[caller_id] = target_id
        await interaction.response.send_message(
            f"Disguise set. You will speak using {target.mention}'s voice.",
            ephemeral=True,
        )
        LOG.info("Disguise set admin_id=%s target_id=%s", caller_id, target_id)

    async def _add_admin(self, interaction: discord.Interaction, target: discord.Member) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        if int(interaction.user.id) != 441597233150951425:
            await interaction.response.send_message("Only txoka can add admins.", ephemeral=True)
            return

        target_id = int(target.id)
        self._admins.add(target_id)
        await interaction.response.send_message(f"Added admin: {target.mention}.", ephemeral=True)
        LOG.info("Admin added by txoka target_id=%s", target_id)

    async def _sync_commands(self, interaction: discord.Interaction) -> None:
        if int(interaction.user.id) != 441597233150951425:
            await interaction.response.send_message("Only txoka can sync commands.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        guild_id_env = os.environ.get("DISCORD_SYNC_GUILD_ID")
        if guild_id_env:
            guild = discord.Object(id=int(guild_id_env))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            await interaction.followup.send(f"Synced commands to guild_id={guild_id_env}.", ephemeral=True)
            LOG.info("Manual sync to guild_id=%s by user_id=%s", guild_id_env, interaction.user.id)
            return

        if interaction.guild:
            guild = discord.Object(id=int(interaction.guild.id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            await interaction.followup.send("Synced commands to this guild.", ephemeral=True)
            LOG.info("Manual sync to guild_id=%s by user_id=%s", interaction.guild.id, interaction.user.id)
            return

        await self.tree.sync()
        await interaction.followup.send("Synced commands globally.", ephemeral=True)
        LOG.info("Manual global sync by user_id=%s", interaction.user.id)

    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        st = self.guild_state.get(int(message.guild.id))
        if not st:
            return

        # Only speak from selected text channel
        if int(message.channel.id) != int(st.text_channel_id):
            LOG.debug(
                "Skip message wrong channel guild_id=%s channel_id=%s expected=%s",
                message.guild.id,
                message.channel.id,
                st.text_channel_id,
            )
            return

        # Resolve voice id (admin disguise)
        author_id = int(message.author.id)
        voice_id = self._disguises.get(author_id, author_id)

        # Only speak if target voice has an enrolled prompt file
        if not self.tts.prompt_exists(voice_id):
            LOG.debug(
                "Skip message no prompt guild_id=%s user_id=%s voice_id=%s",
                message.guild.id,
                author_id,
                voice_id,
            )
            return

        # If user is currently enrolling, don't queue TTS for them
        if author_id in self._cloning_users:
            LOG.debug("Skip message user cloning guild_id=%s user_id=%s", message.guild.id, message.author.id)
            return

        # Avoid massive / empty messages
        text = preprocess_discord_text(message).strip()
        if not text:
            return

        # Enqueue TTS in the engine, then queue playback in-order for this guild
        tts_future = await self.tts.enqueue(voice_id, text)
        await st.queue.put(TTSJob(tts_future=tts_future))
        qsize = st.queue.qsize()
        if qsize and qsize % 10 == 0:
            LOG.info("Guild queue size=%d guild_id=%s", qsize, message.guild.id)
        LOG.debug(
            "Enqueued message guild_id=%s user_id=%s voice_id=%s chars=%d",
            message.guild.id,
            message.author.id,
            voice_id,
            len(text),
        )
