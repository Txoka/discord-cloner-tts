from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from app.config import (
    DEFAULT_LANGUAGE,
    DEVICE,
    DTYPE,
    MAX_CHARS_PER_MESSAGE,
    MAX_NEW_TOKENS,
    MODEL_ID,
)
from app.tts.audio import normalize_audio, to_mono_float32
from app.tts.prompt import load_prompt_items_from_pt

LOG = logging.getLogger("qwen-discord-tts")


class TTSEngine:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        self.prompt_cache: Dict[Path, Any] = {}
        self._model: Any = None

    def _pick_attn(self) -> str:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception:
            return "sdpa"

    def load_model(self) -> None:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        attn = self._pick_attn()

        LOG.info("Loading Qwen3-TTS model=%s device=%s dtype=%s attn=%s", MODEL_ID, DEVICE, DTYPE, attn)
        self._model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=dtype_map[DTYPE],
            attn_implementation=attn,
        )

    def prompt_path(self, user_id: int) -> Path:
        return (self.voices_dir / f"{int(user_id)}.pt").resolve()

    def prompt_exists(self, user_id: int) -> bool:
        return self.prompt_path(user_id).exists()

    def get_prompt(self, user_id: int) -> Optional[Any]:
        pt = self.prompt_path(user_id)
        if not pt.exists():
            return None
        if pt not in self.prompt_cache:
            self.prompt_cache[pt] = load_prompt_items_from_pt(pt)
        return self.prompt_cache[pt]

    def build_clone_prompt_items(self, ref_audio: Any, ref_text: str) -> Any:
        """Create reusable prompt items from (audio, transcript). NOT x-vector-only."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

    def save_prompt_items_pt(self, prompt_items: Any, out_pt: Path) -> None:
        """Save in a stable, class-independent format: list[dict] with tensors."""
        serial = []
        for it in prompt_items:
            ref_code = getattr(it, "ref_code", None)
            ref_spk = getattr(it, "ref_spk_embedding", None)
            if ref_spk is None:
                raise ValueError("Prompt item missing ref_spk_embedding")

            if torch.is_tensor(ref_code):
                ref_code = ref_code.detach().cpu()
            if torch.is_tensor(ref_spk):
                ref_spk = ref_spk.detach().cpu()

            serial.append(
                {
                    "ref_code": ref_code,
                    "ref_spk_embedding": ref_spk,
                    "x_vector_only_mode": bool(getattr(it, "x_vector_only_mode", False)),
                    "icl_mode": bool(getattr(it, "icl_mode", True)),
                    "ref_text": getattr(it, "ref_text", None),
                }
            )
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"items": serial}, out_pt)

    def forget_user(self, user_id: int, delete_pt: bool = True) -> bool:
        """Delete VOICES_DIR/<user_id>.pt and clear cache. Returns True if deleted."""
        pt = self.prompt_path(user_id)
        self.prompt_cache.pop(pt, None)

        if delete_pt and pt.exists():
            try:
                pt.unlink()
                return True
            except Exception:
                return False
        return False

    def synth_to_wavfile(self, user_id: int, text: str) -> Optional[Path]:
        """Blocking call. Returns a temp WAV path or None if user has no prompt file."""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        prompt = self.get_prompt(user_id)
        if prompt is None:
            return None

        # Keep it sane
        text = text.strip().replace("\n", " ")
        if len(text) > MAX_CHARS_PER_MESSAGE:
            text = text[:MAX_CHARS_PER_MESSAGE].rstrip() + "…"

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=DEFAULT_LANGUAGE,
            voice_clone_prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        wav = normalize_audio(to_mono_float32(wavs[0]))

        td = tempfile.NamedTemporaryFile(prefix="qwen_discord_", suffix=".wav", delete=False)
        td.close()
        out = Path(td.name)
        sf.write(str(out), wav, int(sr), subtype="PCM_16")
        return out
