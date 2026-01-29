from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

import app.discord.bot as bot_mod
from app.discord.admin_store import AdminStore
from app.discord.bot import Bot, GuildState, TTSJob
from app.tts.engine import TTSEngine
from tests.helpers.asyncio_utils import cancel_task
from tests.helpers.fakes import FakeChannel, FakeGuild, FakeMessage, FakeUser, FakeVoiceClient


class FakePCMAudio:
    def __init__(self, source):
        self.source = source


@dataclass
class FakePromptItem:
    ref_code: object | None = None
    ref_spk_embedding: object | None = None
    x_vector_only_mode: bool = False
    icl_mode: bool = True
    ref_text: str | None = None


class FakeModel:
    def __init__(self, wav_codes: dict[str, int]) -> None:
        self.wav_codes = wav_codes
        self.calls: list[tuple[list[str], list[object]]] = []

    def generate_voice_clone(self, text, language, voice_clone_prompt, max_new_tokens):
        texts = list(text)
        self.calls.append((texts, list(voice_clone_prompt)))
        wavs = []
        for t in texts:
            code = float(self.wav_codes.get(t, 0))
            wavs.append(np.array([code], dtype=np.float32))
        return wavs, 16000


class FakeAdminStore:
    def is_admin(self, user_id: int) -> bool:
        return True

    def is_superadmin(self, user_id: int) -> bool:
        return True

    def list_debug_guilds(self):
        return [1, 2]

    def list_admins(self):
        return []


class FakeTTS:
    def __init__(self) -> None:
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
async def test_full_pipeline_round_robin_and_model_batch(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)
    wav_codes = {"g1-a": 11, "g2-a": 22, "g1-b": 33}
    engine._model = FakeModel(wav_codes)

    def fake_get_prompt(user_id: int):
        return [FakePromptItem(ref_spk_embedding=object())]

    monkeypatch.setattr(engine, "get_prompt", fake_get_prompt)
    monkeypatch.setattr(engine, "prompt_exists", lambda user_id: True)
    monkeypatch.setattr("app.tts.engine.MAX_BATCH_SIZE", 3)

    def fake_write_wav(wav, sr: int):
        code = int(float(wav[0]))
        return f"{code}".encode()

    monkeypatch.setattr(engine, "_write_wav", fake_write_wav)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    bot = Bot(tts=engine, admin_store=FakeAdminStore())
    monkeypatch.setattr(bot_mod, "prepare_tts_pcm", lambda b, *_args: (b, 48000, 2))
    monkeypatch.setattr(bot_mod.discord, "PCMAudio", FakePCMAudio)

    for gid in (1, 2):
        vc = FakeVoiceClient()
        q: asyncio.Queue = asyncio.Queue()
        bot.guild_state[gid] = GuildState(
            voice_client=vc,
            voice_channel_id=gid,
            text_channel_id=100 + gid,
            queue=q,
            worker_task=asyncio.create_task(bot._player_worker(gid)),
        )

    msg1 = FakeMessage(
        guild=FakeGuild(1),
        channel=FakeChannel(101),
        author=FakeUser(10),
        clean_content="g1-a",
    )
    msg2 = FakeMessage(
        guild=FakeGuild(2),
        channel=FakeChannel(102),
        author=FakeUser(20),
        clean_content="g2-a",
    )
    msg3 = FakeMessage(
        guild=FakeGuild(1),
        channel=FakeChannel(101),
        author=FakeUser(10),
        clean_content="g1-b",
    )

    try:
        await bot.on_message(msg1)
        await bot.on_message(msg2)
        await bot.on_message(msg3)

        await bot.guild_state[1].queue.join()
        await bot.guild_state[2].queue.join()

        assert bot.guild_state[1].voice_client.play_calls == [b"11", b"33"]
        assert bot.guild_state[2].voice_client.play_calls == [b"22"]

        calls = engine._model.calls
        assert calls
        texts, prompts = calls[0]
        assert texts == ["g1-a", "g2-a", "g1-b"]
        assert len(prompts) == len(texts)
    finally:
        tasks = []
        for st in bot.guild_state.values():
            tasks.append(cancel_task(st.worker_task))
        tasks.append(cancel_task(engine._worker_task))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_debug_guild_command_gating(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)
    admin_store = AdminStore(tmp_path / "admins.sqlite3", master_superadmins=[1], debug_guild_ids=[1])
    admin_store.init_schema()
    admin_store.bootstrap_superadmins()
    admin_store.bootstrap_debug_guilds()

    bot = Bot(tts=engine, admin_store=admin_store)

    async def fake_sync(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot.tree, "sync", fake_sync)
    await bot.setup_hook()

    assert bot._is_admin(1, 1) is True
    assert bot._is_admin(1, 2) is False

    guild_cmds = [c.name for c in bot.tree.get_commands(guild=bot_mod.discord.Object(id=1))]
    assert "disguise" in guild_cmds
    assert "adddebugguild" in guild_cmds
    other_cmds = [c.name for c in bot.tree.get_commands(guild=bot_mod.discord.Object(id=2))]
    assert "disguise" not in other_cmds


@pytest.mark.asyncio
async def test_add_remove_debug_guild(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)
    admin_store = AdminStore(tmp_path / "admins.sqlite3", master_superadmins=[1], debug_guild_ids=[1])
    admin_store.init_schema()
    admin_store.bootstrap_superadmins()
    admin_store.bootstrap_debug_guilds()

    bot = Bot(tts=engine, admin_store=admin_store)
    bot._debug_guilds.add(1)
    bot._disguises[3] = {10: 99}

    async def fake_sync(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot.tree, "sync", fake_sync)

    class FakeInteraction:
        def __init__(self, guild_id: int):
            self.guild = type("Guild", (), {"id": guild_id})()
            self.user = FakeUser(1)
            self.response = type("Resp", (), {"send_message": self._send, "defer": self._defer})()
            self.followup = type("Follow", (), {"send": self._send})()
            self.messages: list[str] = []

        async def _send(self, content: str, ephemeral: bool = True):
            self.messages.append(content)

        async def _defer(self, ephemeral: bool = True):
            return None

    await bot._add_debug_guild(FakeInteraction(1), 3)
    assert 3 in admin_store.list_debug_guilds()
    assert 3 in bot._debug_guilds

    await bot._remove_debug_guild(FakeInteraction(1), 3)
    assert 3 not in admin_store.list_debug_guilds()
    assert 3 not in bot._debug_guilds
    assert 3 not in bot._disguises


@pytest.mark.asyncio
async def test_full_pipeline_rejects_when_queues_full(monkeypatch, tmp_path):
    tts = FakeTTS()
    bot = Bot(tts=tts, admin_store=FakeAdminStore())

    vc = FakeVoiceClient()
    q: asyncio.Queue[TTSJob] = asyncio.Queue()
    bot.guild_state[1] = GuildState(
        voice_client=vc,
        voice_channel_id=1,
        text_channel_id=10,
        queue=q,
        worker_task=asyncio.create_task(asyncio.sleep(0)),
    )
    await cancel_task(bot.guild_state[1].worker_task)

    loop = asyncio.get_running_loop()
    await q.put(TTSJob(tts_future=loop.create_future()))

    monkeypatch.setattr(bot_mod, "GUILD_QUEUE_LIMIT", 1)
    monkeypatch.setattr(bot_mod, "GLOBAL_QUEUE_LIMIT", 1)

    msg = FakeMessage(
        guild=FakeGuild(1),
        channel=FakeChannel(10),
        author=FakeUser(5),
        clean_content="hello",
    )

    await bot.on_message(msg)

    assert not tts.enqueued
    assert "❌" in msg.reactions
