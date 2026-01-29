from __future__ import annotations

import asyncio
import io
import logging
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Deque, Set

import soundfile as sf
import torch

# vLLM-Omni Qwen3-TTS wrapper + prompt item type
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts import (  # type: ignore
    Qwen3TTSModel,
    VoiceClonePromptItem,
)

from app.config import (
    DEFAULT_LANGUAGE,
    DEVICE,
    DTYPE,
    MAX_BATCH_SIZE,
    MAX_CHARS_PER_MESSAGE,
    MAX_NEW_TOKENS,
    MODEL_ID,
)
from app.tts.audio import normalize_audio, to_mono_float32

LOG = logging.getLogger("qwen-discord-tts")


@dataclass
class _TTSRequest:
    guild_id: int
    user_id: int
    text: str
    future: "asyncio.Future[Optional[bytes]]"


class TTSEngine:
    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        # Cache: <user_id>.pt -> list[VoiceClonePromptItem]
        self.prompt_cache: Dict[Path, List[VoiceClonePromptItem]] = {}
        self._prompt_exists_cache: Dict[int, bool] = {}

        self._model: Optional[Qwen3TTSModel] = None
        self._guild_queues: Dict[int, Deque[_TTSRequest]] = {}
        self._active_guilds: Deque[int] = deque()
        self._active_set: Set[int] = set()
        self._queue_event = asyncio.Event()
        self._queue_lock = asyncio.Lock()
        self._pending_total = 0
        self._worker_task: Optional[asyncio.Task] = None
        self._worker_lock = asyncio.Lock()

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

        LOG.info("Loading vLLM-Omni Qwen3-TTS model=%s device=%s dtype=%s attn=%s", MODEL_ID, DEVICE, DTYPE, attn)

        # vLLM-Omni wrapper uses HF-style from_pretrained and forwards kwargs to AutoModel.from_pretrained
        self._model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=dtype_map[DTYPE],
            attn_implementation=attn,
        )

    async def _ensure_worker(self) -> None:
        async with self._worker_lock:
            if self._worker_task is None or self._worker_task.done():
                LOG.info("Starting TTS queue worker")
                self._worker_task = asyncio.create_task(self._queue_worker())

    async def enqueue(self, guild_id: int, user_id: int, text: str) -> "asyncio.Future[Optional[bytes]]":
        """Queue a TTS request and return a future that resolves to in-memory WAV bytes (or None)."""
        await self._ensure_worker()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Optional[bytes]] = loop.create_future()
        req = _TTSRequest(guild_id=int(guild_id), user_id=int(user_id), text=text, future=fut)
        async with self._queue_lock:
            q = self._guild_queues.setdefault(int(guild_id), deque())
            q.append(req)
            self._pending_total += 1
            if int(guild_id) not in self._active_set:
                self._active_guilds.append(int(guild_id))
                self._active_set.add(int(guild_id))
            self._queue_event.set()
        LOG.debug("Enqueued TTS guild_id=%s user_id=%s pending=%d", guild_id, user_id, self._pending_total)
        return fut

    async def _queue_worker(self) -> None:
        LOG.info("TTS worker loop running")
        while True:
            await self._queue_event.wait()

            batch: list[_TTSRequest] = []
            while len(batch) < MAX_BATCH_SIZE:
                async with self._queue_lock:
                    if not self._active_guilds:
                        self._queue_event.clear()
                        break
                    guild_id = self._active_guilds.popleft()
                    self._active_set.discard(guild_id)
                    q = self._guild_queues.get(guild_id)
                    if not q:
                        continue
                    req = q.popleft()
                    self._pending_total -= 1
                    if q:
                        self._active_guilds.append(guild_id)
                        self._active_set.add(guild_id)
                    else:
                        self._guild_queues.pop(guild_id, None)
                batch.append(req)

            if not batch:
                continue

            LOG.info(
                "TTS queue batch size=%d pending_after=%d users=%s",
                len(batch),
                self._pending_total,
                [r.user_id for r in batch],
            )

            active: list[_TTSRequest] = []
            for r in batch:
                if r.future.cancelled():
                    continue
                active.append(r)

            if not active:
                continue

            try:
                # Keep your “don’t block loop” design.
                results = await asyncio.to_thread(self._process_batch, active)
            except Exception as exc:
                for r in active:
                    if not r.future.cancelled():
                        r.future.set_exception(exc)
                continue

            for r, wav_bytes, err in results:
                if r.future.cancelled():
                    continue
                if err is not None:
                    r.future.set_exception(err)
                else:
                    r.future.set_result(wav_bytes)

    # -----------------------------
    # Prompt storage / cache
    # -----------------------------
    def prompt_path(self, user_id: int) -> Path:
        return (self.voices_dir / f"{int(user_id)}.pt").resolve()

    def prompt_exists(self, user_id: int) -> bool:
        user_id = int(user_id)
        cached = self._prompt_exists_cache.get(user_id)
        if cached is not None:
            if cached:
                return True
            exists = self.prompt_path(user_id).exists()
            if exists:
                self._prompt_exists_cache[user_id] = True
            return exists
        exists = self.prompt_path(user_id).exists()
        self._prompt_exists_cache[user_id] = exists
        LOG.debug("Prompt exists user_id=%s exists=%s", user_id, exists)
        return exists

    def set_prompt_exists(self, user_id: int, exists: bool) -> None:
        self._prompt_exists_cache[int(user_id)] = bool(exists)

    def _load_prompt_items_from_pt(self, pt: Path) -> List[VoiceClonePromptItem]:
        """
        Load list[VoiceClonePromptItem] from our stable torch.save() format:
          {"items": [{"ref_code": Tensor|None, "ref_spk_embedding": Tensor, "x_vector_only_mode": bool,
                      "icl_mode": bool, "ref_text": str|None}, ...]}
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        payload = torch.load(pt, map_location="cpu")
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError(f"Invalid prompt file format (items not a list): {pt}")

        # Put tensors onto the model device to avoid device mismatch surprises later.
        model_device = getattr(self._model, "device", None)
        if model_device is None:
            # fallback: try to read underlying HF model device
            model_device = getattr(getattr(self._model, "model", None), "device", torch.device("cpu"))

        out: List[VoiceClonePromptItem] = []
        for d in raw_items:
            ref_code = d.get("ref_code", None)
            ref_spk = d.get("ref_spk_embedding", None)
            if ref_spk is None:
                raise ValueError(f"Prompt item missing ref_spk_embedding in {pt}")

            if torch.is_tensor(ref_code):
                ref_code = ref_code.to(model_device)
            elif ref_code is not None:
                raise ValueError(f"ref_code must be Tensor|None in {pt}")

            if not torch.is_tensor(ref_spk):
                raise ValueError(f"ref_spk_embedding must be Tensor in {pt}")
            ref_spk = ref_spk.to(model_device)

            out.append(
                VoiceClonePromptItem(
                    ref_code=ref_code,
                    ref_spk_embedding=ref_spk,
                    x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                    icl_mode=bool(d.get("icl_mode", True)),
                    ref_text=d.get("ref_text", None),
                )
            )
        LOG.info("Loaded prompt items count=%d path=%s", len(out), pt)
        return out

    def get_prompt(self, user_id: int) -> Optional[List[VoiceClonePromptItem]]:
        pt = self.prompt_path(user_id)
        if not pt.exists():
            return None
        if pt not in self.prompt_cache:
            self.prompt_cache[pt] = self._load_prompt_items_from_pt(pt)
        return self.prompt_cache[pt]

    def build_clone_prompt_items(self, ref_audio: Any, ref_text: str) -> Any:
        """Create reusable prompt items from (audio, transcript)."""
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
        self._prompt_exists_cache[int(out_pt.stem)] = True
        LOG.info("Saved prompt items count=%d path=%s", len(serial), out_pt)

    def forget_user(self, user_id: int, delete_pt: bool = True) -> bool:
        """Delete VOICES_DIR/<user_id>.pt and clear cache. Returns True if deleted."""
        user_id = int(user_id)
        pt = self.prompt_path(user_id)
        self.prompt_cache.pop(pt, None)
        self._prompt_exists_cache[user_id] = False

        if delete_pt and pt.exists():
            try:
                pt.unlink()
                LOG.info("Deleted prompt file user_id=%s path=%s", user_id, pt)
                return True
            except Exception:
                LOG.warning("Failed deleting prompt file user_id=%s path=%s", user_id, pt)
                return False
        return False

    # -----------------------------
    # Synthesis helpers
    # -----------------------------
    def _sanitize_text(self, text: str) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) > MAX_CHARS_PER_MESSAGE:
            text = text[:MAX_CHARS_PER_MESSAGE].rstrip() + "…"
        return text

    def _write_wav(self, wav: Any, sr: int) -> bytes:
        wav = normalize_audio(to_mono_float32(wav))
        buf = io.BytesIO()
        sf.write(buf, wav, int(sr), subtype="PCM_16", format="WAV")
        return buf.getvalue()

    # -----------------------------
    # Correct batching (multi-speaker)
    # -----------------------------
    def _process_batch(self, batch: list[_TTSRequest]) -> list[tuple[_TTSRequest, Optional[bytes], Optional[Exception]]]:
        """
        Correct batch behavior for vLLM-Omni:

        - Build texts[] and one VoiceClonePromptItem per request (speaker).
        - Call generate_voice_clone(text=list[str], voice_clone_prompt=list[VoiceClonePromptItem]).
          Lengths must match. This is the “batch speakers” path. :contentReference[oaicite:3]{index=3}
        - If anything fails, fall back to per-item generation.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        results: list[tuple[_TTSRequest, Optional[bytes], Optional[Exception]]] = []
        valid_reqs: list[_TTSRequest] = []
        texts: list[str] = []
        batch_prompt_items: list[VoiceClonePromptItem] = []

        # Gather
        for req in batch:
            prompt_items = self.get_prompt(req.user_id)
            if prompt_items is None or len(prompt_items) == 0:
                LOG.debug("Prompt missing for user_id=%s", req.user_id)
                results.append((req, None, None))
                continue

            # If you stored multiple prompt items for a user, batching becomes ambiguous.
            # Pick the first for batching; the single-item path still supports multi-item prompts.
            one_item = prompt_items[0]

            valid_reqs.append(req)
            texts.append(self._sanitize_text(req.text))
            batch_prompt_items.append(one_item)

        if not valid_reqs:
            return results

        # Fast path: real batch (multiple speakers)
        wavs, sr = self._model.generate_voice_clone(
            text=texts,
            language=[DEFAULT_LANGUAGE] * len(texts),
            voice_clone_prompt=batch_prompt_items,  # <-- FIXED: list[VoiceClonePromptItem], len == len(texts)
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for req, wav in zip(valid_reqs, wavs, strict=False):
            results.append((req, self._write_wav(wav, int(sr)), None))

        LOG.info("TTS batch finished")

        return results
