from __future__ import annotations

import asyncio

import numpy as np
import pytest

import app.discord.bot as bot_mod
from app.discord.bot import Bot, GuildState, TTSJob
from tests.helpers.fakes import (
    FakeChannel,
    FakeGuild,
    FakeInteraction,
    FakeUser,
    FakeVoiceChannel,
    FakeVoiceClient,
    FakeMessage,
)


class FakeAdminStore:
    def is_admin(self, user_id: int) -> bool:
        return True

    def is_superadmin(self, user_id: int) -> bool:
        return True

    def list_debug_guilds(self):
        return [1, 2]

    def list_admins(self):
        return []

    def add_debug_guild(self, guild_id: int) -> None:
        return None

    def remove_debug_guild(self, guild_id: int) -> bool:
        return True


class FakeTTS:
    def __init__(self, voices_dir):
        self._voices_dir = voices_dir
        self.prompt_cache = {}
        self.exists_cache = {}
        self.enqueued: list[tuple[int, int, str]] = []

    def prompt_path(self, user_id: int):
        return self._voices_dir / f"{user_id}.pt"

    def prompt_exists(self, user_id: int) -> bool:
        return True

    def build_clone_prompt_items(self, ref_audio, ref_text):
        return [object()]

    def save_prompt_items_pt(self, prompt_items, out_pt):
        out_pt.write_bytes(b"ok")

    def set_prompt_exists(self, user_id: int, exists: bool) -> None:
        self.exists_cache[int(user_id)] = bool(exists)

    async def enqueue(self, guild_id: int, user_id: int, text: str):
        self.enqueued.append((guild_id, user_id, text))
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        fut.set_result(b"wav")
        return fut


class FakeCollector:
    def __init__(self, target_user_id: int):
        self.target_user_id = int(target_user_id)
        self.sample_rate = 48000

    def mono_float32(self) -> np.ndarray:
        return np.ones((self.sample_rate,), dtype=np.float32) * 0.1


class FakeMember:
    def __init__(self, user_id: int, voice_channel: FakeVoiceChannel):
        self.id = int(user_id)
        self.voice = type("Voice", (), {"channel": voice_channel})


@pytest.mark.asyncio
async def test_clone_rejects_second_attempt_same_guild(tmp_path, monkeypatch):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())

    vc = FakeVoiceClient()
    channel = FakeVoiceChannel(vc)
    member = FakeMember(123, channel)
    guild = FakeGuild(1, member=member)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=member, guild=guild)

    monkeypatch.setattr(bot_mod, "SingleUserPCMCollector", FakeCollector)
    monkeypatch.setattr(bot_mod, "CLONE_RECORD_SECONDS", 0)
    monkeypatch.setattr(bot_mod, "CLONE_MIN_SECONDS", 0)
    monkeypatch.setattr(bot_mod.discord, "Member", FakeMember)

    lock = bot._clone_lock.setdefault(1, asyncio.Lock())
    await lock.acquire()
    try:
        await bot._clone(interaction)
        assert interaction.response.messages
        assert "already running" in interaction.response.messages[-1].lower()
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_clone_allows_parallel_guilds(tmp_path, monkeypatch):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())

    vc1 = FakeVoiceClient()
    vc2 = FakeVoiceClient()
    channel1 = FakeVoiceChannel(vc1)
    channel2 = FakeVoiceChannel(vc2)
    member1 = FakeMember(111, channel1)
    member2 = FakeMember(222, channel2)
    guild1 = FakeGuild(1, member=member1)
    guild2 = FakeGuild(2, member=member2)
    interaction1 = FakeInteraction(guild_id=1, channel_id=10, user=member1, guild=guild1)
    interaction2 = FakeInteraction(guild_id=2, channel_id=20, user=member2, guild=guild2)

    monkeypatch.setattr(bot_mod, "SingleUserPCMCollector", FakeCollector)
    monkeypatch.setattr(bot_mod, "CLONE_RECORD_SECONDS", 0)
    monkeypatch.setattr(bot_mod, "CLONE_MIN_SECONDS", 0)
    monkeypatch.setattr(bot_mod.discord, "Member", FakeMember)

    await asyncio.gather(bot._clone(interaction1), bot._clone(interaction2))

    assert tts.prompt_path(member1.id).exists()
    assert tts.prompt_path(member2.id).exists()


@pytest.mark.asyncio
async def test_on_message_ignored_for_cloning_user(tmp_path):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())

    guild = FakeGuild(1)
    channel = FakeChannel(10)
    user = FakeUser(5)
    msg = FakeMessage(guild=guild, channel=channel, author=user, clean_content="hi")

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=FakeVoiceClient(),
        voice_channel_id=1,
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    bot.guild_state[1].worker_task.cancel()

    bot._cloning_users.add(user.id)
    await bot.on_message(msg)
    assert not tts.enqueued


@pytest.mark.asyncio
async def test_player_worker_handles_none_and_exception(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    vc = FakeVoiceClient()
    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=vc,
        voice_channel_id=1,
        text_channel_id=1,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    bot.guild_state[1].worker_task.cancel()

    worker = asyncio.create_task(bot._player_worker(1))

    loop = asyncio.get_running_loop()
    fut1 = loop.create_future()
    fut2 = loop.create_future()
    await q.put(TTSJob(tts_future=fut1))
    await q.put(TTSJob(tts_future=fut2))

    fut1.set_result(None)
    fut2.set_exception(RuntimeError("boom"))

    await q.join()
    worker.cancel()

    assert vc.play_calls == []
