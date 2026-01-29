from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from app.tts.engine import TTSEngine


@dataclass
class FakePromptItem:
    ref_code: torch.Tensor | None
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool = False
    icl_mode: bool = True
    ref_text: str | None = None


class FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def generate_voice_clone(self, text, language, voice_clone_prompt, max_new_tokens):
        if isinstance(text, list):
            wavs = [np.ones((10,), dtype=np.float32) * 0.1 for _ in text]
        else:
            wavs = [np.ones((10,), dtype=np.float32) * 0.1]
        return wavs, 16000

    def create_voice_clone_prompt(self, *args, **kwargs):
        return []


def test_prompt_path_and_exists(tmp_path):
    engine = TTSEngine(tmp_path)
    path = engine.prompt_path(123)
    assert path.name == "123.pt"
    assert not engine.prompt_exists(123)
    path.write_bytes(b"x")
    assert engine.prompt_exists(123)


def test_save_and_load_prompt_items(tmp_path):
    engine = TTSEngine(tmp_path)
    engine._model = FakeModel()
    item = FakePromptItem(ref_code=torch.tensor([1]), ref_spk_embedding=torch.tensor([2]))
    out_pt = engine.prompt_path(1)
    engine.save_prompt_items_pt([item], out_pt)
    loaded = engine._load_prompt_items_from_pt(out_pt)
    assert len(loaded) == 1


def test_get_prompt_uses_cache(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)
    engine._model = FakeModel()
    out_pt = engine.prompt_path(1)
    item = FakePromptItem(ref_code=None, ref_spk_embedding=torch.tensor([2]))
    engine.save_prompt_items_pt([item], out_pt)

    calls = {"count": 0}

    def fake_load(pt):
        calls["count"] += 1
        return [item]

    monkeypatch.setattr(engine, "_load_prompt_items_from_pt", fake_load)
    first = engine.get_prompt(1)
    second = engine.get_prompt(1)
    assert first == second
    assert calls["count"] == 1


def test_forget_user(tmp_path):
    engine = TTSEngine(tmp_path)
    out_pt = engine.prompt_path(1)
    out_pt.write_bytes(b"x")
    assert engine.forget_user(1, delete_pt=True) is True
    assert not out_pt.exists()


def test_process_batch_builds_results(tmp_path, monkeypatch):
    engine = TTSEngine(tmp_path)
    engine._model = FakeModel()

    def fake_get_prompt(user_id: int):
        return [FakePromptItem(ref_code=None, ref_spk_embedding=torch.tensor([1]))]

    monkeypatch.setattr(engine, "get_prompt", fake_get_prompt)

    req1 = type("Req", (), {"user_id": 1, "text": "hi", "future": None})
    req2 = type("Req", (), {"user_id": 2, "text": "yo", "future": None})
    results = engine._process_batch([req1, req2])
    assert len(results) == 2
    assert results[0][1] is not None


@pytest.mark.asyncio
async def test_queue_worker_resolves_future(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    def fake_process(batch):
        return [(r, b"wav", None) for r in batch]

    engine._process_batch = fake_process  # type: ignore[assignment]
    fut = await engine.enqueue(1, 1, "hi")
    out = await asyncio.wait_for(fut, 1)
    assert out == b"wav"
    if engine._worker_task:
        engine._worker_task.cancel()


@pytest.mark.asyncio
async def test_queue_worker_skips_cancelled(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    def fake_process(batch):
        return [(r, b"wav", None) for r in batch]

    engine._process_batch = fake_process  # type: ignore[assignment]
    fut = await engine.enqueue(1, 1, "hi")
    fut.cancel()
    await asyncio.sleep(0)
    if engine._worker_task:
        engine._worker_task.cancel()
    assert fut.cancelled()


@pytest.mark.asyncio
async def test_queue_worker_round_robin(monkeypatch, tmp_path):
    engine = TTSEngine(tmp_path)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr("app.tts.engine.MAX_BATCH_SIZE", 3)

    captured: list[list[tuple[int, str]]] = []

    def fake_process(batch):
        captured.append([(r.guild_id, r.text) for r in batch])
        return [(r, b"wav", None) for r in batch]

    engine._process_batch = fake_process  # type: ignore[assignment]

    futs = [
        await engine.enqueue(1, 10, "g1-a"),
        await engine.enqueue(1, 10, "g1-b"),
        await engine.enqueue(2, 20, "g2-a"),
    ]

    for fut in futs:
        await asyncio.wait_for(fut, 1)

    assert captured
    assert captured[0] == [(1, "g1-a"), (2, "g2-a"), (1, "g1-b")]
    if engine._worker_task:
        engine._worker_task.cancel()
