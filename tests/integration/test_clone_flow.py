from __future__ import annotations

import asyncio

import numpy as np
import pytest

import app.discord.bot as bot_mod
from app.discord.bot import Bot, GuildState
from tests.helpers.asyncio_utils import cancel_task
from tests.helpers.fakes import (
    FakeGuild,
    FakeInteraction,
    FakeUser,
    FakeVoiceChannel,
    FakeVoiceClient,
    FakeVoiceRecvClient,
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
    def __init__(self):
        self._disguises: dict[int, int] = {}

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

    def list_disguises(self) -> dict[int, int]:
        return dict(self._disguises)

    def set_disguise(self, user_id: int, target_id: int) -> None:
        self._disguises[int(user_id)] = int(target_id)

    def clear_disguise(self, user_id: int) -> bool:
        return self._disguises.pop(int(user_id), None) is not None


@pytest.mark.asyncio
async def test_clone_creates_prompt_file(monkeypatch, tmp_path):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]

    vc = FakeVoiceClient()
    recv_vc = FakeVoiceRecvClient()
    channel = FakeVoiceChannel(vc, channel_id=5, recv_voice_client=recv_vc)
    member = FakeMember(123, channel)
    guild = FakeGuild(1, member=member)

    # Existing connected receive-capable state so we don't call connect
    q = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=recv_vc,
        voice_channel_id=1,
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
        voice_receive_enabled=True,
    )
    await cancel_task(bot.guild_state[1].worker_task)

    interaction = FakeInteraction(guild_id=1, channel_id=10, user=member, guild=guild)

    monkeypatch.setattr(bot_mod, "SingleUserPCMCollector", FakeCollector)
    monkeypatch.setattr(bot_mod, "CLONE_RECORD_SECONDS", 0)
    monkeypatch.setattr(bot_mod, "CLONE_MIN_SECONDS", 0)
    monkeypatch.setattr(bot_mod.discord, "Member", FakeMember)

    await bot._clone(interaction)

    prompt_path = tts.prompt_path(member.id)
    assert prompt_path.exists()
    assert interaction.followup.messages


@pytest.mark.asyncio
async def test_clone_disconnects_when_not_prejoined(monkeypatch, tmp_path):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]

    vc = FakeVoiceClient()
    recv_vc = FakeVoiceRecvClient()
    channel = FakeVoiceChannel(vc, recv_voice_client=recv_vc)
    member = FakeMember(123, channel)
    guild = FakeGuild(1, member=member)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=member, guild=guild)

    monkeypatch.setattr(bot_mod, "SingleUserPCMCollector", FakeCollector)
    monkeypatch.setattr(bot_mod, "CLONE_RECORD_SECONDS", 0)
    monkeypatch.setattr(bot_mod, "CLONE_MIN_SECONDS", 0)
    monkeypatch.setattr(bot_mod.discord, "Member", FakeMember)

    await bot._clone(interaction)

    assert 1 not in bot.guild_state


@pytest.mark.asyncio
async def test_clone_keeps_prejoined_channel(monkeypatch, tmp_path):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]

    vc = FakeVoiceClient()
    recv_vc = FakeVoiceRecvClient()
    channel = FakeVoiceChannel(vc, recv_voice_client=recv_vc)
    member = FakeMember(123, channel)
    guild = FakeGuild(1, member=member)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=member, guild=guild)

    q = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=recv_vc,
        voice_channel_id=5,
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
        voice_receive_enabled=True,
    )
    await cancel_task(bot.guild_state[1].worker_task)
    bot.guild_state[1].voice_channel_id = int(member.voice.channel.id)

    monkeypatch.setattr(bot_mod, "SingleUserPCMCollector", FakeCollector)
    monkeypatch.setattr(bot_mod, "CLONE_RECORD_SECONDS", 0)
    monkeypatch.setattr(bot_mod, "CLONE_MIN_SECONDS", 0)
    monkeypatch.setattr(bot_mod.discord, "Member", FakeMember)

    await bot._clone(interaction)

    assert 1 in bot.guild_state
    assert bot.guild_state[1].voice_client.is_connected()


@pytest.mark.asyncio
async def test_join_blocked_when_cloning_other_channel(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    guild = FakeGuild(1)
    user = FakeUser(5)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=user, guild=guild)

    lock = bot._clone_lock.setdefault(1, asyncio.Lock())
    await lock.acquire()
    try:
        bot._cloning_channels[1] = 111
        other_channel = FakeVoiceChannel(FakeVoiceClient(), channel_id=222, members=[user])
        await bot._join(interaction, other_channel)
        assert interaction.response.messages
        assert "cloning" in interaction.response.messages[-1].lower()
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_join_blocked_when_cloning_other_channel_without_arg(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    vc = FakeVoiceClient()
    channel = FakeVoiceChannel(vc, channel_id=222)
    member = FakeMember(5, channel)
    channel.members.append(member)
    guild = FakeGuild(1, member=member)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=member, guild=guild)

    lock = bot._clone_lock.setdefault(1, asyncio.Lock())
    await lock.acquire()
    try:
        bot._cloning_channels[1] = 111
        await bot._join(interaction, None)
        assert interaction.response.messages
        assert "cloning" in interaction.response.messages[-1].lower()
        assert 1 not in bot.guild_state
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_leave_blocked_when_cloning(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    guild = FakeGuild(1)
    user = FakeUser(5)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=user, guild=guild)

    lock = bot._clone_lock.setdefault(1, asyncio.Lock())
    await lock.acquire()
    try:
        await bot._leave(interaction)
        assert interaction.response.messages
        assert "can't leave" in interaction.response.messages[-1].lower()
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_join_during_clone_same_channel_sets_keep(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    guild = FakeGuild(1)
    user = FakeUser(5)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=user, guild=guild)
    channel = FakeVoiceChannel(FakeVoiceClient(), channel_id=123, members=[user])

    lock = bot._clone_lock.setdefault(1, asyncio.Lock())
    await lock.acquire()
    worker = None
    try:
        bot._cloning_channels[1] = 123
        await bot._join(interaction, channel)
        worker = bot.guild_state.get(1).worker_task if bot.guild_state.get(1) else None
        assert bot._clone_joined.get(1) is True
    finally:
        if worker is not None:
            await cancel_task(worker)
        lock.release()
