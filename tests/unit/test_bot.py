from __future__ import annotations

import asyncio
import pytest

import app.discord.bot as bot_mod
from app.discord.admin_store import AdminRecord
from app.discord.bot import Bot, GuildState, TTSJob
from tests.helpers.asyncio_utils import cancel_task
from tests.helpers.fakes import (
    FakeChannel,
    FakeGuild,
    FakeInteraction,
    FakeMessage,
    FakeUser,
    FakeVoiceChannel,
    FakeVoiceClient,
    SpyVoiceClient,
)


class FakePCMAudio:
    def __init__(self, source):
        self.source = source


class FakeAdminStore:
    def __init__(self):
        self._disguises: dict[int, int] = {}

    def is_admin(self, user_id: int) -> bool:
        return True

    def is_superadmin(self, user_id: int) -> bool:
        return True

    def has_role(self, user_id: int, role: str) -> bool:
        return True

    def list_admins(self):
        return []

    def list_debug_guilds(self):
        return []

    def add_debug_guild(self, guild_id: int) -> None:
        return None

    def remove_debug_guild(self, guild_id: int) -> bool:
        return True

    def list_debug_guilds(self):
        return [1, 2]

    def list_disguises(self) -> dict[int, int]:
        return dict(self._disguises)

    def set_disguise(self, user_id: int, target_id: int) -> None:
        self._disguises[int(user_id)] = int(target_id)

    def clear_disguise(self, user_id: int) -> bool:
        return self._disguises.pop(int(user_id), None) is not None

    def get_disguise(self, user_id: int):
        return self._disguises.get(int(user_id))


def test_single_user_pcm_collector_ignores_other_user():
    collector = bot_mod.SingleUserPCMCollector(target_user_id=1)
    class Data:
        pcm = b"\x00\x01" * 4

    collector.write(FakeUser(id=2), Data())
    assert collector.mono_float32().size == 0


def test_log_latency_breakdown(caplog):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    loop = asyncio.new_event_loop()
    try:
        fut = loop.create_future()
    finally:
        loop.close()
    job = TTSJob(
        tts_future=fut,
        trace={
            "msg_recv_ts": 100.00,
            "preprocess_done_ts": 100.05,
            "enqueue_call_ts": 100.06,
            "enqueue_return_ts": 100.07,
            "tts_queue_enqueued_ts": 100.07,
            "tts_queue_dequeued_ts": 100.27,
            "tts_infer_start_ts": 100.27,
            "tts_infer_end_ts": 100.77,
            "tts_future_wait_start_ts": 100.20,
            "tts_future_wait_end_ts": 100.80,
            "guild_queue_put_ts": 100.08,
            "guild_queue_get_ts": 101.08,
            "vc_wait_start_ts": 101.08,
            "vc_wait_end_ts": 101.28,
            "pcm_prep_start_ts": 101.28,
            "pcm_prep_end_ts": 101.38,
            "playback_start_ts": 101.38,
            "playback_end_ts": 102.38,
        },
        message_id=123,
        author_id=456,
        voice_id=789,
    )

    caplog.set_level("INFO")
    bot._log_latency(1, job, "ok")

    joined = "\n".join(rec.message for rec in caplog.records if "Latency stage=ok" in rec.message)
    assert "Latency stage=ok guild_id=1 message_id=123 author_id=456 voice_id=789" in joined
    assert "total_ms=2380.00" in joined
    assert "preproc_ms=50.00" in joined
    assert "enqueue_ms=10.00" in joined
    assert "tts_queue_ms=200.00" in joined
    assert "tts_wait_ms=600.00" in joined
    assert "tts_infer_ms=500.00" in joined
    assert "guild_queue_ms=1000.00" in joined
    assert "vc_wait_ms=200.00" in joined
    assert "pcm_ms=100.00" in joined
    assert "playback_ms=1000.00" in joined


@pytest.mark.asyncio
async def test_join_refuses_empty_channel():
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    guild = FakeGuild(1)
    user = FakeUser(5)
    interaction = FakeInteraction(guild_id=1, channel_id=10, user=user, guild=guild)
    empty_channel = FakeVoiceChannel(FakeVoiceClient(), channel_id=123, members=[])

    await bot._join(interaction, empty_channel)
    assert interaction.response.messages
    assert "empty voice channel" in interaction.response.messages[-1].lower()


@pytest.mark.asyncio
async def test_auto_leave_triggers_when_alone(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    monkeypatch.setattr(bot_mod, "AUTO_LEAVE_SECONDS", 0.01)

    vc = FakeVoiceClient()
    class DummyVoiceChannel:
        def __init__(self) -> None:
            self.id = 1
            self.name = "voice"
            self.members = []

    monkeypatch.setattr(bot_mod.discord, "VoiceChannel", DummyVoiceChannel)
    channel = DummyVoiceChannel()
    guild = FakeGuild(1, channel=channel)

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=vc,
        voice_channel_id=1,
        text_channel_id=1,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    await cancel_task(bot.guild_state[1].worker_task)

    bot.get_guild = lambda _gid: guild  # type: ignore[assignment]

    left = asyncio.Event()

    async def fake_leave(guild_id: int, reason: str) -> None:
        left.set()

    bot._leave_guild = fake_leave  # type: ignore[assignment]

    bot._schedule_auto_leave(1)
    await asyncio.wait_for(left.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_player_starts_stream_after_enqueue(monkeypatch):
    vc = SpyVoiceClient()
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(bot_mod, "prepare_tts_pcm", lambda b, *_args: (b"pcm:" + b, 48000, 2))

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    guild_id = 1
    bot.guild_state[guild_id] = GuildState(
        voice_client=vc,
        voice_channel_id=1,
        text_channel_id=1,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    await cancel_task(bot.guild_state[guild_id].worker_task)
    worker = asyncio.create_task(bot._player_worker(guild_id))

    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await q.put(TTSJob(tts_future=fut))
    fut.set_result(b"one")

    await q.join()
    await cancel_task(worker)

    assert vc.play_sources
    stream = vc.play_sources[0]
    assert getattr(stream, "_closed", True) is False


@pytest.mark.asyncio
async def test_on_message_filters_and_enqueues(monkeypatch):
    class FakeTTS:
        def __init__(self) -> None:
            self.enqueued = []

        def prompt_exists(self, user_id: int) -> bool:
            return True

        async def enqueue(self, guild_id: int, user_id: int, text: str):
            self.enqueued.append((guild_id, user_id, text))
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(b"wav")
            return fut

    tts = FakeTTS()
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]
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
    await cancel_task(bot.guild_state[1].worker_task)

    await bot.on_message(msg)
    assert tts.enqueued
    assert q.qsize() == 1


@pytest.mark.asyncio
async def test_on_message_rejects_when_queue_full(monkeypatch):
    class FakeTTS:
        def __init__(self) -> None:
            self.enqueued = []

        def prompt_exists(self, user_id: int) -> bool:
            return True

        async def enqueue(self, guild_id: int, user_id: int, text: str):
            self.enqueued.append((guild_id, user_id, text))
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(b"wav")
            return fut

    monkeypatch.setattr(bot_mod, "GUILD_QUEUE_LIMIT", 1)
    monkeypatch.setattr(bot_mod, "GLOBAL_QUEUE_LIMIT", 1)

    tts = FakeTTS()
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]
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
    await cancel_task(bot.guild_state[1].worker_task)

    # Fill queue to limit
    loop = asyncio.get_running_loop()
    await q.put(TTSJob(tts_future=loop.create_future()))

    await bot.on_message(msg)

    assert not tts.enqueued
    assert "❌" in msg.reactions


@pytest.mark.asyncio
async def test_admin_disabled_ignores_disguise(monkeypatch):
    class FakeTTS:
        def prompt_exists(self, user_id: int) -> bool:
            return True

        async def enqueue(self, guild_id: int, user_id: int, text: str):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(b"wav")
            self.last = (guild_id, user_id, text)
            return fut

    tts = FakeTTS()
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    bot._disguises[5] = 99

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
    await cancel_task(bot.guild_state[1].worker_task)

    await bot.on_message(msg)
    assert tts.last[1] == 5


@pytest.mark.asyncio
async def test_admin_enabled_applies_disguise_across_debug_guilds(monkeypatch):
    class FakeTTS:
        def prompt_exists(self, user_id: int) -> bool:
            return True

        async def enqueue(self, guild_id: int, user_id: int, text: str):
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(b"wav")
            self.last = (guild_id, user_id, text)
            return fut

    tts = FakeTTS()
    bot = Bot(tts=tts, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    bot._debug_guilds.update({1, 2})
    bot._disguises[5] = 99

    guild = FakeGuild(2)
    channel = FakeChannel(10)
    user = FakeUser(5)
    msg = FakeMessage(guild=guild, channel=channel, author=user, clean_content="hi")

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[2] = GuildState(
        voice_client=FakeVoiceClient(),
        voice_channel_id=2,
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    await cancel_task(bot.guild_state[2].worker_task)

    await bot.on_message(msg)
    assert tts.last[1] == 99


@pytest.mark.asyncio
async def test_admin_list(monkeypatch):
    class FakeStore(FakeAdminStore):
        def list_admins(self):
            return [AdminRecord(user_id=1, role="superadmin")]

    class FakeInteraction:
        def __init__(self):
            self.guild = type("Guild", (), {"id": 1})()
            self.user = FakeUser(1)
            self.response = type("Resp", (), {"send_message": self._send})()
            self.messages = []

        async def _send(self, content: str, ephemeral: bool = True):
            self.messages.append(content)

    bot = Bot(tts=None, admin_store=FakeStore())  # type: ignore[arg-type]
    bot._debug_guilds.add(1)
    interaction = FakeInteraction()
    await bot._admin_list(interaction)
    assert interaction.messages
    assert "superadmin" in interaction.messages[0]


@pytest.mark.asyncio
async def test_debug_guild_list():
    class FakeStore(FakeAdminStore):
        def list_debug_guilds(self):
            return [2, 5]

    class FakeInteraction:
        def __init__(self):
            self.guild = type("Guild", (), {"id": 2})()
            self.user = FakeUser(1)
            self.response = type("Resp", (), {"send_message": self._send})()
            self.messages = []

        async def _send(self, content: str, ephemeral: bool = True):
            self.messages.append(content)

    bot = Bot(tts=None, admin_store=FakeStore())  # type: ignore[arg-type]
    bot._debug_guilds.add(2)
    interaction = FakeInteraction()
    await bot._debug_guild_list(interaction)
    assert "2" in interaction.messages[0]


@pytest.mark.asyncio
async def test_debug_command_enables_current_guild(monkeypatch):
    class FakeStore(FakeAdminStore):
        def __init__(self):
            super().__init__()
            self._guilds: set[int] = set()

        def list_debug_guilds(self):
            return list(self._guilds)

        def add_debug_guild(self, guild_id: int) -> None:
            self._guilds.add(int(guild_id))

        def remove_debug_guild(self, guild_id: int) -> bool:
            existed = int(guild_id) in self._guilds
            self._guilds.discard(int(guild_id))
            return existed

    class FakeInteraction:
        def __init__(self):
            self.guild = type("Guild", (), {"id": 9})()
            self.user = FakeUser(1)
            self.response = type("Resp", (), {"send_message": self._send})()
            self.messages = []

        async def _send(self, content: str, ephemeral: bool = True):
            self.messages.append(content)

    bot = Bot(tts=None, admin_store=FakeStore())  # type: ignore[arg-type]

    async def fake_sync(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot.tree, "sync", fake_sync)
    interaction = FakeInteraction()
    await bot._debug_current_guild(interaction)
    assert 9 in bot._debug_guilds
    assert interaction.messages
    interaction.messages = []
    await bot._debug_current_guild(interaction)
    assert 9 not in bot._debug_guilds
    assert interaction.messages


def test_debug_command_can_be_disabled():
    bot = Bot(tts=None, admin_store=FakeAdminStore(), enable_debug_command=False)  # type: ignore[arg-type]
    names = [cmd.name for cmd in bot.tree.get_commands()]
    assert "debug" not in names


@pytest.mark.asyncio
async def test_debug_command_disabled_not_synced(monkeypatch):
    bot = Bot(tts=None, admin_store=FakeAdminStore(), enable_debug_command=False)  # type: ignore[arg-type]

    async def fake_sync(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot.tree, "sync", fake_sync)
    await bot.setup_hook()
    names = [cmd.name for cmd in bot.tree.get_commands()]
    assert "debug" not in names
