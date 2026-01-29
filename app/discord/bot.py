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
    GUILD_QUEUE_LIMIT,
    GLOBAL_QUEUE_LIMIT,
)
from app.discord.admin_store import AdminStore
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
    def __init__(self, tts: TTSEngine, admin_store: AdminStore):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        intents.voice_states = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.tts = tts
        self.admin_store = admin_store
        self._debug_guilds: set[int] = set()
        self.guild_state: Dict[int, GuildState] = {}

        # Prevent concurrent enrollments per guild + suppress TTS for enrolling user
        self._clone_lock: Dict[int, asyncio.Lock] = {}
        self._cloning_users: set[int] = set()
        self._cloning_channels: Dict[int, int] = {}
        self._clone_joined: Dict[int, bool] = {}
        self._disguises: Dict[int, Dict[int, int]] = {}
        self._admin_commands: list[app_commands.Command] = []
        self._register_admin_command_objects()

    def _register_admin_command_objects(self) -> None:
        async def disguise(interaction: discord.Interaction, target: discord.Member):
            await self._disguise(interaction, target)

        disguise = app_commands.describe(target="User whose voice you want to use")(disguise)
        disguise_cmd = app_commands.Command(
            name="disguise",
            description="Admins only: speak using someone else's voice.",
            callback=disguise,
        )

        async def addadmin(
            interaction: discord.Interaction,
            target: discord.Member,
            role: app_commands.Choice[str] | None = None,
        ):
            await self._add_admin(interaction, target, role.value if role else "admin")

        addadmin = app_commands.describe(target="User to grant admin access")(addadmin)
        addadmin = app_commands.describe(role="admin or superadmin")(addadmin)
        addadmin = app_commands.choices(
            role=[
                app_commands.Choice(name="admin", value="admin"),
                app_commands.Choice(name="superadmin", value="superadmin"),
            ]
        )(addadmin)
        addadmin_cmd = app_commands.Command(
            name="addadmin",
            description="Admins only: add an admin or superadmin.",
            callback=addadmin,
        )

        async def removeadmin(interaction: discord.Interaction, target: discord.Member):
            await self._remove_admin(interaction, target)

        removeadmin = app_commands.describe(target="User to remove from admins")(removeadmin)
        removeadmin_cmd = app_commands.Command(
            name="removeadmin",
            description="Admins only: remove an admin or superadmin.",
            callback=removeadmin,
        )

        async def adminlist(interaction: discord.Interaction):
            await self._admin_list(interaction)

        adminlist_cmd = app_commands.Command(
            name="adminlist",
            description="Admins only: list current admins and roles.",
            callback=adminlist,
        )

        async def sync_cmd(interaction: discord.Interaction):
            await self._sync_commands(interaction)

        sync_cmd_obj = app_commands.Command(
            name="sync",
            description="Admins only: sync slash commands to this guild.",
            callback=sync_cmd,
        )

        async def adddebugguild(interaction: discord.Interaction, guild_id: int):
            await self._add_debug_guild(interaction, guild_id)

        adddebugguild = app_commands.describe(guild_id="Guild ID to enable admin commands for")(adddebugguild)
        adddebugguild_cmd = app_commands.Command(
            name="adddebugguild",
            description="Admins only: enable admin commands for a guild.",
            callback=adddebugguild,
        )

        async def removedebugguild(interaction: discord.Interaction, guild_id: int):
            await self._remove_debug_guild(interaction, guild_id)

        removedebugguild = app_commands.describe(guild_id="Guild ID to disable admin commands for")(removedebugguild)
        removedebugguild_cmd = app_commands.Command(
            name="removedebugguild",
            description="Admins only: disable admin commands for a guild.",
            callback=removedebugguild,
        )

        async def debugguildlist(interaction: discord.Interaction):
            await self._debug_guild_list(interaction)

        debugguildlist_cmd = app_commands.Command(
            name="debugguildlist",
            description="Admins only: list guilds with admin commands enabled.",
            callback=debugguildlist,
        )

        self._admin_commands = [
            disguise_cmd,
            addadmin_cmd,
            removeadmin_cmd,
            adminlist_cmd,
            sync_cmd_obj,
            adddebugguild_cmd,
            removedebugguild_cmd,
            debugguildlist_cmd,
        ]

    def _register_admin_commands_for_guild(self, guild_id: int) -> None:
        guild = discord.Object(id=int(guild_id))
        for cmd in self._admin_commands:
            try:
                self.tree.add_command(cmd, guild=guild)
            except Exception:
                continue

    def _unregister_admin_commands_for_guild(self, guild_id: int) -> None:
        guild = discord.Object(id=int(guild_id))
        for cmd in self._admin_commands:
            self.tree.remove_command(cmd.name, guild=guild)

    def _is_debug_guild(self, guild_id: int | None) -> bool:
        return guild_id is not None and int(guild_id) in self._debug_guilds

    def _is_admin(self, user_id: int, guild_id: int | None) -> bool:
        if not self._is_debug_guild(guild_id):
            return False
        return self.admin_store.is_admin(int(user_id))

    def _is_superadmin(self, user_id: int, guild_id: int | None) -> bool:
        if not self._is_debug_guild(guild_id):
            return False
        return self.admin_store.is_superadmin(int(user_id))

    async def setup_hook(self):
        LOG.info("Syncing application commands")
        await self.tree.sync()
        self._debug_guilds = set(self.admin_store.list_debug_guilds())
        for gid in self._debug_guilds:
            self._register_admin_commands_for_guild(gid)
            await self.tree.sync(guild=discord.Object(id=int(gid)))

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
        guild_id = int(interaction.guild_id)
        lock = self._clone_lock.get(guild_id)
        if lock and lock.locked():
            target_channel_id = self._cloning_channels.get(guild_id)
            if channel is not None and target_channel_id is not None and int(channel.id) != target_channel_id:
                await interaction.response.send_message(
                    "I'm currently cloning in another channel. Try again when cloning finishes.",
                    ephemeral=True,
                )
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
        if self._cloning_channels.get(guild_id) == int(channel.id):
            self._clone_joined[guild_id] = True

        await interaction.response.send_message(
            f"Joined **{channel.name}**. I will speak messages from <#{interaction.channel_id}>.",
            ephemeral=True,
        )
        LOG.info("Joined voice guild_id=%s channel_id=%s", interaction.guild_id, channel.id)

    async def _leave(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        guild_id = int(interaction.guild_id)
        lock = self._clone_lock.get(guild_id)
        if lock and lock.locked():
            await interaction.response.send_message("Can't leave while cloning is in progress.", ephemeral=True)
            return
        st = self.guild_state.get(guild_id)
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
            channel_id = int(member.voice.channel.id)
            self._cloning_channels[guild_id] = channel_id
            self._clone_joined.pop(guild_id, None)

            st = self.guild_state.get(guild_id)
            preexisting_connected = False
            if st and st.voice_client.is_connected() and int(st.voice_channel_id) == channel_id:
                preexisting_connected = True
            # Ensure bot is connected with VoiceRecvClient
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
                self._cloning_channels.pop(guild_id, None)
                if not preexisting_connected and not self._clone_joined.get(guild_id, False):
                    st = self.guild_state.get(guild_id)
                    if st:
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
                        self.guild_state.pop(guild_id, None)
                        LOG.info("Left voice after clone guild_id=%s", interaction.guild_id)
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
        guild_id = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), guild_id):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return

        target_id = int(target.id)
        caller_id = int(interaction.user.id)
        if target_id == caller_id:
            self._disguises.get(guild_id, {}).pop(caller_id, None)
            await interaction.response.send_message("Disguise cleared. Using your own voice.", ephemeral=True)
            return

        if not self.tts.prompt_exists(target_id):
            await interaction.response.send_message(
                f"{target.mention} does not have an enrolled voice.",
                ephemeral=True,
            )
            return

        self._disguises.setdefault(guild_id, {})[caller_id] = target_id
        await interaction.response.send_message(
            f"Disguise set. You will speak using {target.mention}'s voice.",
            ephemeral=True,
        )
        LOG.info("Disguise set admin_id=%s target_id=%s", caller_id, target_id)

    async def _add_admin(self, interaction: discord.Interaction, target: discord.Member, role: str = "admin") -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        guild_id = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), guild_id):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return
        if not self._is_superadmin(int(interaction.user.id), guild_id):
            await interaction.response.send_message("Superadmins only.", ephemeral=True)
            return

        target_id = int(target.id)
        role = role.strip().lower()
        if role not in {"admin", "superadmin"}:
            await interaction.response.send_message("Role must be 'admin' or 'superadmin'.", ephemeral=True)
            return

        self.admin_store.upsert_admin(target_id, role)
        await interaction.response.send_message(f"Added {role}: {target.mention}.", ephemeral=True)
        LOG.info("Admin added role=%s by user_id=%s target_id=%s", role, interaction.user.id, target_id)

    async def _remove_admin(self, interaction: discord.Interaction, target: discord.Member) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        guild_id = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), guild_id):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return
        if not self._is_superadmin(int(interaction.user.id), guild_id):
            await interaction.response.send_message("Superadmins only.", ephemeral=True)
            return

        target_id = int(target.id)
        removed = self.admin_store.remove_admin(target_id)
        if removed:
            await interaction.response.send_message(f"Removed admin: {target.mention}.", ephemeral=True)
            LOG.info("Admin removed by user_id=%s target_id=%s", interaction.user.id, target_id)
            return

        await interaction.response.send_message(
            "That user is not removable (or is a master superadmin).",
            ephemeral=True,
        )

    async def _admin_list(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        guild_id = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), guild_id):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return

        records = self.admin_store.list_admins()
        if not records:
            await interaction.response.send_message("No admins found.", ephemeral=True)
            return

        lines = [f"<@{r.user_id}> — {r.role}" for r in records]
        out = "Admins:\n" + "\n".join(lines)
        await interaction.response.send_message(out[:1900], ephemeral=True)

    async def _sync_commands(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        guild_id = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), guild_id):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        guild = discord.Object(id=int(interaction.guild.id))
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        await interaction.followup.send("Synced commands to this guild.", ephemeral=True)
        LOG.info("Manual sync to guild_id=%s by user_id=%s", interaction.guild.id, interaction.user.id)

    async def _add_debug_guild(self, interaction: discord.Interaction, guild_id: int) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        current_gid = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), current_gid):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return
        target_gid = int(guild_id)
        self.admin_store.add_debug_guild(target_gid)
        self._debug_guilds.add(target_gid)
        self._register_admin_commands_for_guild(target_gid)
        await self.tree.sync(guild=discord.Object(id=target_gid))
        await interaction.response.send_message(f"Enabled admin commands for guild_id={target_gid}.", ephemeral=True)
        LOG.info("Added debug guild_id=%s by user_id=%s", target_gid, interaction.user.id)

    async def _remove_debug_guild(self, interaction: discord.Interaction, guild_id: int) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        current_gid = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), current_gid):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return
        target_gid = int(guild_id)
        removed = self.admin_store.remove_debug_guild(target_gid)
        self._debug_guilds.discard(target_gid)
        self._disguises.pop(target_gid, None)
        self._unregister_admin_commands_for_guild(target_gid)
        await self.tree.sync(guild=discord.Object(id=target_gid))
        if removed:
            await interaction.response.send_message(f"Disabled admin commands for guild_id={target_gid}.", ephemeral=True)
            LOG.info("Removed debug guild_id=%s by user_id=%s", target_gid, interaction.user.id)
        else:
            await interaction.response.send_message("Guild was not enabled.", ephemeral=True)

    async def _debug_guild_list(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return
        current_gid = int(interaction.guild.id)
        if not self._is_admin(int(interaction.user.id), current_gid):
            await interaction.response.send_message("Admins only.", ephemeral=True)
            return
        guilds = self.admin_store.list_debug_guilds()
        if not guilds:
            await interaction.response.send_message("No debug guilds configured.", ephemeral=True)
            return
        lines = [str(gid) for gid in guilds]
        out = "Debug guilds:\n" + "\n".join(lines)
        await interaction.response.send_message(out[:1900], ephemeral=True)

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
        voice_id = author_id
        if self._is_debug_guild(int(message.guild.id)):
            voice_id = self._disguises.get(int(message.guild.id), {}).get(author_id, author_id)

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

        # Enforce queue limits (global + per-guild)
        guild_qsize = st.queue.qsize()
        global_qsize = sum(s.queue.qsize() for s in self.guild_state.values())
        if (GUILD_QUEUE_LIMIT > 0 and guild_qsize >= GUILD_QUEUE_LIMIT) or (
            GLOBAL_QUEUE_LIMIT > 0 and global_qsize >= GLOBAL_QUEUE_LIMIT
        ):
            try:
                await message.add_reaction("❌")
            except (discord.Forbidden, discord.HTTPException) as exc:
                LOG.warning("Failed to add reject reaction message_id=%s err=%s", message.id, exc)
            return

        # Enqueue TTS in the engine, then queue playback in-order for this guild
        tts_future = await self.tts.enqueue(int(message.guild.id), voice_id, text)
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
