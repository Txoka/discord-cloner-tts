from __future__ import annotations

import asyncio

import numpy as np
import pytest

import app.discord.bot as bot_mod
from app.discord.bot import Bot, GuildState
from tests.helpers.fakes import (
    FakeGuild,
    FakeInteraction,
    FakeUser,
    FakeVoiceChannel,
    FakeVoiceClient,
)


class FakeMember:
    def __init__(self, user_id: int, voice_channel: FakeVoiceChannel):
        self.id = int(user_id)
        self.voice = type("Voice", (), {"channel": voice_channel})


class FakeCollector:
    def __init__(self, target_user_id: int):
        self.target_user_id = int(target_user_id)
        self.sample_rate = 48000

    def mono_float32(self) -> np.ndarray:
        return np.ones((self.sample_rate,), dtype=np.float32) * 0.1


class FakeTTS:
    def __init__(self, voices_dir):
        self._voices_dir = voices_dir
        self.prompt_cache = {}
        self.exists_cache = {}

    def prompt_path(self, user_id: int):
        return self._voices_dir / f"{user_id}.pt"

    def build_clone_prompt_items(self, ref_audio, ref_text):
        return [object()]

    def save_prompt_items_pt(self, prompt_items, out_pt):
        out_pt.write_bytes(b"ok")

    def set_prompt_exists(self, user_id: int, exists: bool) -> None:
        self.exists_cache[int(user_id)] = bool(exists)


class FakeAdminStore:
    def is_admin(self, user_id: int) -> bool:
        return True

    def is_superadmin(self, user_id: int) -> bool:
        return True

    def list_debug_guilds(self):
        return [1]

    def list_admins(self):
        return []

    def add_debug_guild(self, guild_id: int) -> None:
        return None

    def remove_debug_guild(self, guild_id: int) -> bool:
        return True


@pytest.mark.asyncio
async def test_clone_creates_prompt_file(monkeypatch, tmp_path):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]

    vc = FakeVoiceClient()
    channel = FakeVoiceChannel(vc)
    member = FakeMember(123, channel)
    guild = FakeGuild(1, member=member)

    # Existing connected state so we don't call connect
    q = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=vc,
        voice_channel_id=1,
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    bot.guild_state[1].worker_task.cancel()

    interaction = FakeInteraction(guild_id=1, channel_id=10, user=member, guild=guild)

    monkeypatch.setattr(bot_mod, "SingleUserPCMCollector", FakeCollector)
    monkeypatch.setattr(bot_mod, "CLONE_RECORD_SECONDS", 0)
    monkeypatch.setattr(bot_mod, "CLONE_MIN_SECONDS", 0)
    monkeypatch.setattr(bot_mod.discord, "Member", FakeMember)

    await bot._clone(interaction)

    prompt_path = tts.prompt_path(member.id)
    assert prompt_path.exists()
    assert interaction.followup.messages
