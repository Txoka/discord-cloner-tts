from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qwen_tts import VoiceClonePromptItem


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_prompt_items_from_pt(pt_path: Path) -> Any:
    payload = torch_load(pt_path)
    items_raw = payload["items"] if isinstance(payload, dict) and "items" in payload else payload

    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError(f"Bad prompt file (no items list): {pt_path}")

    # If already prompt objects, pass-through
    if not isinstance(items_raw[0], dict):
        return items_raw

    out = []
    for d in items_raw:
        ref_code = d.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)

        ref_spk = d.get("ref_spk_embedding")
        if ref_spk is None:
            raise ValueError(f"Prompt item missing ref_spk_embedding in {pt_path}")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)

        xvec = bool(d.get("x_vector_only_mode", False))
        icl = bool(d.get("icl_mode", (not xvec)))

        out.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=xvec,
                icl_mode=icl,
                ref_text=d.get("ref_text"),
            )
        )
    return out
