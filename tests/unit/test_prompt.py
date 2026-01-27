from __future__ import annotations

from dataclasses import dataclass

import torch

import app.tts.prompt as prompt


@dataclass
class DummyPromptItem:
    ref_code: torch.Tensor | None
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None


def test_torch_load_fallback(monkeypatch, tmp_path):
    path = tmp_path / "x.pt"
    torch.save({"items": []}, path)

    calls = {"count": 0}

    def fake_load(*args, **kwargs):
        calls["count"] += 1
        if kwargs.get("weights_only") is True:
            raise TypeError("no weights_only")
        return {"items": []}

    monkeypatch.setattr(prompt.torch, "load", fake_load)
    out = prompt.torch_load(path)
    assert out == {"items": []}
    assert calls["count"] == 2


def test_load_prompt_items_from_pt_dict(monkeypatch, tmp_path):
    payload = {
        "items": [
            {
                "ref_code": [1, 2],
                "ref_spk_embedding": [3, 4],
                "x_vector_only_mode": False,
                "icl_mode": True,
                "ref_text": "hi",
            }
        ]
    }
    pt = tmp_path / "p.pt"
    torch.save(payload, pt)

    monkeypatch.setattr(prompt, "VoiceClonePromptItem", DummyPromptItem)
    items = prompt.load_prompt_items_from_pt(pt)
    assert isinstance(items[0], DummyPromptItem)
    assert torch.is_tensor(items[0].ref_spk_embedding)


def test_load_prompt_items_from_pt_missing_spk(tmp_path):
    payload = {"items": [{"ref_code": [1, 2]}]}
    pt = tmp_path / "p.pt"
    torch.save(payload, pt)

    try:
        prompt.load_prompt_items_from_pt(pt)
    except ValueError as exc:
        assert "ref_spk_embedding" in str(exc)
    else:
        raise AssertionError("expected ValueError")
