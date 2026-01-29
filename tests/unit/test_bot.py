from __future__ import annotations

import asyncio
import pytest

import app.discord.bot as bot_mod
from app.discord.bot import Bot, GuildState, TTSJob
from tests.helpers.fakes import FakeChannel, FakeGuild, FakeMessage, FakeUser, FakeVoiceClient


class FakePCMAudio:
    def __init__(self, source):
        self.source = source


class FakeAdminStore:
    def is_admin(self, user_id: int) -> bool:
        return True

    def is_superadmin(self, user_id: int) -> bool:
        return True

    def has_role(self, user_id: int, role: str) -> bool:
        return True

    def list_admins(self):
        return []


def test_single_user_pcm_collector_ignores_other_user():
    collector = bot_mod.SingleUserPCMCollector(target_user_id=1)
    class Data:
        pcm = b"\x00\x01" * 4

    collector.write(FakeUser(id=2), Data())
    assert collector.mono_float32().size == 0


@pytest.mark.asyncio
async def test_player_worker_orders_playback(monkeypatch):
    vc = FakeVoiceClient()
    bot = Bot(tts=None, admin_store=FakeAdminStore())  # type: ignore[arg-type]
    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(bot_mod, "prepare_tts_pcm", lambda b, *_args: (b"pcm:" + b, 48000, 2))
    monkeypatch.setattr(bot_mod.discord, "PCMAudio", FakePCMAudio)

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    guild_id = 1
    bot.guild_state[guild_id] = GuildState(
        voice_client=vc,
        voice_channel_id=1,
        text_channel_id=1,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    bot.guild_state[guild_id].worker_task.cancel()
    worker = asyncio.create_task(bot._player_worker(guild_id))

    loop = asyncio.get_running_loop()
    fut1 = loop.create_future()
    fut2 = loop.create_future()
    await q.put(TTSJob(tts_future=fut1))
    await q.put(TTSJob(tts_future=fut2))

    fut1.set_result(b"one")
    fut2.set_result(b"two")

    await q.join()
    worker.cancel()

    assert vc.play_calls == [b"pcm:one", b"pcm:two"]


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
    monkeypatch.setattr(bot_mod, "DISCORD_ADMIN_ENABLED", True)
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
    bot.guild_state[1].worker_task.cancel()

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
    monkeypatch.setattr(bot_mod, "DISCORD_ADMIN_ENABLED", True)
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
    bot.guild_state[1].worker_task.cancel()

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

    monkeypatch.setattr(bot_mod, "DISCORD_ADMIN_ENABLED", False)
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
    bot.guild_state[1].worker_task.cancel()

    await bot.on_message(msg)
    assert tts.last[1] == 5


@pytest.mark.asyncio
async def test_admin_list(monkeypatch):
    class FakeStore(FakeAdminStore):
        def list_admins(self):
            return [bot_mod.AdminRecord(user_id=1, role="superadmin")]

    class FakeInteraction:
        def __init__(self):
            self.guild = object()
            self.user = FakeUser(1)
            self.response = type("Resp", (), {"send_message": self._send})()
            self.messages = []

        async def _send(self, content: str, ephemeral: bool = True):
            self.messages.append(content)

    monkeypatch.setattr(bot_mod, "DISCORD_ADMIN_ENABLED", True)
    bot = Bot(tts=None, admin_store=FakeStore())  # type: ignore[arg-type]
    interaction = FakeInteraction()
    await bot._admin_list(interaction)
    assert interaction.messages
    assert "superadmin" in interaction.messages[0]
