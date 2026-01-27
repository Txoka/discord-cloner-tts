from __future__ import annotations

import sys
from pathlib import Path
import types
from dataclasses import dataclass

import torch


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


# Stub vllm_omni qwen3_tts import chain if not installed.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

root = _ensure_module("vllm_omni")
_ensure_module("vllm_omni.model_executor")
_ensure_module("vllm_omni.model_executor.models")
_ensure_module("vllm_omni.model_executor.models.qwen3_tts")
qwen_mod = _ensure_module("vllm_omni.model_executor.models.qwen3_tts.qwen3_tts")


@dataclass
class VoiceClonePromptItem:
    ref_code: torch.Tensor | None
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None


class Qwen3TTSModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "Qwen3TTSModel":
        return cls()

    def generate_voice_clone(self, *args, **kwargs):
        raise NotImplementedError

    def create_voice_clone_prompt(self, *args, **kwargs):
        return []


qwen_mod.Qwen3TTSModel = Qwen3TTSModel
qwen_mod.VoiceClonePromptItem = VoiceClonePromptItem


# Stub qwen_tts VoiceClonePromptItem if not installed.
qwen_tts_mod = _ensure_module("qwen_tts")
qwen_tts_mod.VoiceClonePromptItem = VoiceClonePromptItem
