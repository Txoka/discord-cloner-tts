from __future__ import annotations

import asyncio
import logging
import os
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
from app.tts.audio import trim_silence_energy
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
# Guild voice player with queue
# -----------------------------
@dataclass
class TTSJob:
    user_id: int
    text: str


@dataclass
class GuildState:
    voice_client: discord.VoiceClient
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

    async def setup_hook(self):
        await self.tree.sync()

    async def on_ready(self):
        LOG.info("Logged in as %s (%s)", self.user, self.user.id)

    async def _player_worker(self, guild_id: int) -> None:
        """Single ordered pipeline per guild: dequeue chat messages in order, synthesize, then play.

        This guarantees voice playback follows chat order even if later messages would synth faster.
        """
        st = self.guild_state[guild_id]
        vc = st.voice_client

        while True:
            job = await st.queue.get()
            wav_path: Optional[Path] = None
            try:
                if not vc.is_connected():
                    break

                # Synthesize in-order (in a thread so we don't block the loop)
                wav_path = await asyncio.to_thread(self.tts.synth_to_wavfile, int(job.user_id), job.text)
                if wav_path is None:
                    # Prompt missing (e.g., user forgot voice mid-queue). Skip.
                    continue

                src = discord.FFmpegPCMAudio(
                    executable="ffmpeg",
                    source=str(wav_path),
                    options="-loglevel warning -vn",
                )

                done = asyncio.Event()

                def after_play(err: Optional[Exception]) -> None:
                    if err:
                        LOG.warning("Playback error: %s", err)
                    done.set()

                vc.play(src, after=after_play)
                await done.wait()

            finally:
                if wav_path is not None:
                    try:
                        wav_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                st.queue.task_done()

    async def _join(self, interaction: discord.Interaction) -> None:
        if not interaction.guild:
            await interaction.response.send_message("Guild-only command.", ephemeral=True)
            return

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
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient)

        q: asyncio.Queue[TTSJob] = asyncio.Queue()
        guild_id = int(interaction.guild_id)

        # Replace existing state cleanly
        old = self.guild_state.get(guild_id)
        if old:
            try:
                old.worker_task.cancel()
            except Exception:
                pass

        # Create state first, then start worker (avoid KeyError race)
        placeholder_task = asyncio.create_task(asyncio.sleep(0))
        self.guild_state[guild_id] = GuildState(
            voice_client=vc,
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
        await st.voice_client.disconnect(force=True)
        self.guild_state.pop(int(interaction.guild_id), None)
        await interaction.response.send_message("Left voice chat.", ephemeral=True)

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

        async with lock:
            # Ensure bot is connected with VoiceRecvClient
            st = self.guild_state.get(guild_id)
            if not st or not st.voice_client.is_connected():
                # connect and create state
                vc = await member.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
                q: asyncio.Queue[TTSJob] = asyncio.Queue()

                placeholder_task = asyncio.create_task(asyncio.sleep(0))
                self.guild_state[guild_id] = GuildState(
                    voice_client=vc,
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

                await interaction.followup.send(
                    "✅ Enrolled your voice.",
                    ephemeral=True,
                )

            finally:
                self._cloning_users.discard(int(member.id))

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

    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        st = self.guild_state.get(int(message.guild.id))
        if not st:
            return

        # Only speak from selected text channel
        if int(message.channel.id) != int(st.text_channel_id):
            return

        # Only speak if author has an enrolled prompt file
        if not self.tts.prompt_exists(int(message.author.id)):
            return

        # If user is currently enrolling, don't queue TTS for them
        if int(message.author.id) in self._cloning_users:
            return

        # Avoid massive / empty messages
        text = preprocess_discord_text(message).strip()
        if not text:
            return

        # Enqueue the text job; the guild worker will synth+play strictly in chat order
        await st.queue.put(TTSJob(user_id=int(message.author.id), text=text))
