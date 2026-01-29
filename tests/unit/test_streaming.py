from __future__ import annotations

import asyncio

import app.discord.bot as bot_mod


REAL_STREAM = bot_mod.GuildPCMStream


def test_stream_ends_when_empty(monkeypatch):
    monkeypatch.setattr(bot_mod, "GuildPCMStream", REAL_STREAM)
    stream = REAL_STREAM(guild_id=1)

    out = stream.read()
    assert out == b""
    assert stream._closed is True


def test_stream_consumes_and_closes(monkeypatch):
    monkeypatch.setattr(bot_mod, "GuildPCMStream", REAL_STREAM)
    stream = REAL_STREAM(guild_id=1)

    class Job:
        trace = {}

    job = Job()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stream.enqueue(job, b"\x01" * 100, loop=loop)

    frame = stream.read()
    assert len(frame) == bot_mod.PCM_FRAME_BYTES
    assert job.trace.get("playback_start_ts") is not None
    assert job.trace.get("playback_end_ts") is not None

    out = stream.read()
    assert out == b""
    assert stream._closed is True

    asyncio.set_event_loop(None)
    loop.close()
