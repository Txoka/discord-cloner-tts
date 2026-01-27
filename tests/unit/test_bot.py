from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest

import app.discord.bot as bot_mod
from app.discord.bot import Bot, GuildState, TTSJob
from tests.helpers.fakes import (
    FakeChannel,
    FakeGuild,
    FakeMessage,
    FakeUser,
    FakeVoiceClient,
)


class FakeFFmpegPCMAudio:
    def __init__(self, executable: str, source: io.BytesIO, pipe: bool, options: str):
        self.source = source


def test_single_user_pcm_collector_ignores_other_user():
    collector = bot_mod.SingleUserPCMCollector(target_user_id=1)
    class Data:
        pcm = b"\x00\x01" * 4

    collector.write(FakeUser(id=2), Data())
    assert collector.mono_float32().size == 0


def test_trim_tts_wav_invalid_bytes_returns_original():
    bot = Bot(tts=None)  # type: ignore[arg-type]
    bad = b"not a wav"
    assert bot._trim_tts_wav(bad) == bad


@pytest.mark.asyncio
async def test_player_worker_orders_playback(monkeypatch):
    vc = FakeVoiceClient()
    bot = Bot(tts=None)  # type: ignore[arg-type]
    monkeypatch.setattr(bot, "_trim_tts_wav", lambda b: b)
    monkeypatch.setattr(bot_mod.discord, "FFmpegPCMAudio", FakeFFmpegPCMAudio)

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    guild_id = 1
    bot.guild_state[guild_id] = GuildState(
        voice_client=vc,
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

    assert vc.play_calls == [b"one", b"two"]


@pytest.mark.asyncio
async def test_on_message_filters_and_enqueues(monkeypatch):
    class FakeTTS:
        def __init__(self) -> None:
            self.enqueued = []

        def prompt_exists(self, user_id: int) -> bool:
            return True

        async def enqueue(self, user_id: int, text: str):
            self.enqueued.append((user_id, text))
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            fut.set_result(b"wav")
            return fut

    tts = FakeTTS()
    bot = Bot(tts=tts)  # type: ignore[arg-type]
    guild = FakeGuild(1)
    channel = FakeChannel(10)
    user = FakeUser(5)
    msg = FakeMessage(guild=guild, channel=channel, author=user, clean_content="hi")

    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=FakeVoiceClient(),
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    bot.guild_state[1].worker_task.cancel()

    await bot.on_message(msg)
    assert tts.enqueued
    assert q.qsize() == 1
