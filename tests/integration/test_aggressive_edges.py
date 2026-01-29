from __future__ import annotations

import asyncio

import pytest

import app.discord.bot as bot_mod
from app.discord.admin_store import AdminStore
from app.discord.bot import Bot, GuildState, TTSJob
from app.tts.engine import TTSEngine
from tests.helpers.asyncio_utils import cancel_task
from tests.helpers.fakes import (
    FakeChannel,
    FakeGuild,
    FakeInteraction,
    FakeMessage,
    FakeUser,
    FakeVoiceChannel,
    FakeVoiceClient,
)


class FakeAdminStore:
    def is_admin(self, user_id: int) -> bool:
        return True

    def is_superadmin(self, user_id: int) -> bool:
        return True

    def list_debug_guilds(self):
        return [1, 2, 3]

    def list_admins(self):
        return []

    def add_debug_guild(self, guild_id: int) -> None:
        return None

    def remove_debug_guild(self, guild_id: int) -> bool:
        return True


class FakeTTS:
    def __init__(self, voices_dir):
        self._voices_dir = voices_dir
        self.enqueued: list[tuple[int, int, str]] = []

    def prompt_exists(self, user_id: int) -> bool:
        return True

    async def enqueue(self, guild_id: int, user_id: int, text: str):
        self.enqueued.append((guild_id, user_id, text))
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        fut.set_result(b"wav")
        return fut


@pytest.mark.asyncio
async def test_round_robin_fairness_under_skew(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)

    def fake_get_prompt(_user_id: int):
        return [object()]

    monkeypatch.setattr(engine, "get_prompt", fake_get_prompt)
    monkeypatch.setattr(engine, "prompt_exists", lambda _uid: True)
    monkeypatch.setattr("app.tts.engine.MAX_BATCH_SIZE", 3)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(engine, "_process_batch", lambda batch: [(r, b"wav", None) for r in batch])

    futs = []
    for i in range(5):
        futs.append(await engine.enqueue(1, 10, f"a{i}"))
    futs.append(await engine.enqueue(2, 20, "b0"))
    futs.append(await engine.enqueue(3, 30, "c0"))

    await asyncio.gather(*futs)

    await cancel_task(engine._worker_task)


@pytest.mark.asyncio
async def test_global_queue_limit_concurrent(monkeypatch, tmp_path):
    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=FakeAdminStore())

    for gid in (1, 2, 3):
        q: asyncio.Queue[TTSJob] = asyncio.Queue()
        bot.guild_state[gid] = GuildState(
            voice_client=FakeVoiceClient(),
            voice_channel_id=gid,
            text_channel_id=10 + gid,
            queue=q,
            worker_task=asyncio.create_task(asyncio.sleep(0)),
        )
        await cancel_task(bot.guild_state[gid].worker_task)

    monkeypatch.setattr(bot_mod, "GLOBAL_QUEUE_LIMIT", 2)
    monkeypatch.setattr(bot_mod, "GUILD_QUEUE_LIMIT", 10)

    msgs = [
        FakeMessage(FakeGuild(1), FakeChannel(11), FakeUser(1), "m1"),
        FakeMessage(FakeGuild(2), FakeChannel(12), FakeUser(2), "m2"),
        FakeMessage(FakeGuild(3), FakeChannel(13), FakeUser(3), "m3"),
    ]

    await asyncio.gather(*(bot.on_message(m) for m in msgs))
    assert len(tts.enqueued) <= 2


@pytest.mark.asyncio
async def test_remove_debug_guild_race_with_messages(tmp_path, monkeypatch):
    admin_store = AdminStore(tmp_path / "admins.sqlite3", master_superadmins=[1], debug_guild_ids=[1])
    admin_store.init_schema()
    admin_store.bootstrap_superadmins()
    admin_store.bootstrap_debug_guilds()

    tts = FakeTTS(tmp_path)
    bot = Bot(tts=tts, admin_store=admin_store)
    bot._debug_guilds.add(1)
    bot._disguises[1] = {5: 99}

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=FakeVoiceClient(),
        voice_channel_id=1,
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    await cancel_task(bot.guild_state[1].worker_task)

    async def fake_sync(*_args, **_kwargs):
        return None

    bot.tree.sync = fake_sync  # type: ignore[assignment]

    msg = FakeMessage(FakeGuild(1), FakeChannel(10), FakeUser(5), "hello")

    await asyncio.gather(bot.on_message(msg), bot._remove_debug_guild(FakeInteraction(1, 10, FakeUser(1), FakeGuild(1)), 1))
    assert 1 not in bot._disguises


@pytest.mark.asyncio
async def test_join_leave_sequence_during_clone(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    guild = FakeGuild(1)
    user = FakeUser(5)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=user, guild=guild)
    channel = FakeVoiceChannel(FakeVoiceClient(), channel_id=123)
    other = FakeVoiceChannel(FakeVoiceClient(), channel_id=999)

    lock = bot._clone_lock.setdefault(1, asyncio.Lock())
    await lock.acquire()
    try:
        bot._cloning_channels[1] = 123
        await bot._join(interaction, channel)
        await bot._join(interaction, other)
        await bot._leave(interaction)
        assert bot._clone_joined.get(1) is True
        assert interaction.response.messages
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_player_handles_playback_error(monkeypatch):
    class ErrVoiceClient(FakeVoiceClient):
        def play(self, src: object, after=None) -> None:
            raise bot_mod.discord.ClientException("boom")

    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    vc = ErrVoiceClient()
    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=vc,
        voice_channel_id=1,
        text_channel_id=1,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    await cancel_task(bot.guild_state[1].worker_task)

    monkeypatch.setattr(bot_mod, "prepare_tts_pcm", lambda b, *_args: (b, 48000, 2))

    worker = asyncio.create_task(bot._player_worker(1))
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await q.put(TTSJob(tts_future=fut))
    fut.set_result(b"wav")
    await q.join()
    await cancel_task(worker)
