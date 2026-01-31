from __future__ import annotations

import asyncio
import pytest

import app.discord.bot as bot_mod
from app.discord.bot import Bot, GuildState
from app.tts.engine import TTSEngine
from tests.helpers.asyncio_utils import cancel_task
from tests.helpers.fakes import FakeChannel, FakeGuild, FakeMessage, FakeUser, FakeVoiceClient


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

    def list_debug_guilds(self):
        return [1, 2]

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
async def test_pipeline_global_engine_per_guild_order(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(engine, "prompt_exists", lambda user_id: True)

    def fake_process(batch):
        results = []
        for req in batch:
            results.append((req, f"{req.user_id}:{req.text}".encode(), None))
        return results

    engine._process_batch = fake_process  # type: ignore[assignment]

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
        clean_content="hello",
    )
    msg2 = FakeMessage(
        guild=FakeGuild(2),
        channel=FakeChannel(102),
        author=FakeUser(20),
        clean_content="world",
    )
    msg3 = FakeMessage(
        guild=FakeGuild(1),
        channel=FakeChannel(101),
        author=FakeUser(10),
        clean_content="again",
    )

    await bot.on_message(msg1)
    await bot.on_message(msg2)
    await bot.on_message(msg3)

    await bot.guild_state[1].queue.join()
    await bot.guild_state[2].queue.join()

    assert bot.guild_state[1].stream.items == [b"10:hello", b"10:again"]
    assert bot.guild_state[2].stream.items == [b"20:world"]

    tasks = []
    for st in bot.guild_state.values():
        tasks.append(cancel_task(st.worker_task))
    tasks.append(cancel_task(engine._worker_task))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
