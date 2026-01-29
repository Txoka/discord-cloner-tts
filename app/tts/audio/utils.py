from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
import torch

from app.config import MAX_GAIN_DB, NORM_MODE, TARGET_PEAK_DBFS, TARGET_RMS_DBFS

LOG = logging.getLogger("qwen-discord-tts")


def to_mono_float32(wav: Any) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        x = wav.detach().cpu().float().numpy()
    else:
        x = np.asarray(wav)

    if x.ndim == 2:
        x = x.mean(axis=-1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        denom = float(max(abs(info.min), info.max))
        x = x.astype(np.float32) / denom
    else:
        x = x.astype(np.float32, copy=False)
    return np.clip(x, -1.0, 1.0)


def resample_linear(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    if sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")

    x = x.astype(np.float32, copy=False)
    n_in = x.size
    n_out = int(round(n_in * float(target_sr) / float(sr)))
    if n_out <= 1:
        return x[:1].astype(np.float32, copy=False)

    t_in = np.linspace(0.0, 1.0, num=n_in, endpoint=True, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=True, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


def mono_to_stereo_int16_bytes(x: np.ndarray) -> bytes:
    if x.size == 0:
        return b""
    x = np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)
    mono_i16 = (x * np.float32(32767.0)).astype(np.int16, copy=False)
    stereo = np.repeat(mono_i16[:, None], 2, axis=1)
    return stereo.tobytes()


def prepare_tts_pcm(
    wav_bytes: bytes,
    target_sr: int = 48000,
    trim: bool = True,
) -> tuple[bytes, int, int]:
    """Decode WAV bytes, optionally trim, resample, and return (pcm_bytes, sr, channels)."""
    start = time.monotonic()
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile is required to decode WAV bytes") from exc

    try:
        with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
            data = f.read(dtype="float32", always_2d=False)
            sr = f.samplerate
    except Exception:
        LOG.debug("Failed decoding WAV bytes")
        return b"", 0, 0

    audio = to_mono_float32(data)
    if trim:
        audio = trim_silence_energy(audio, int(sr))
    resampled = resample_linear(audio, int(sr), int(target_sr))
    out = mono_to_stereo_int16_bytes(resampled)
    LOG.debug(
        "Prepared PCM in_bytes=%d out_bytes=%d sr_in=%d sr_out=%d ms=%.2f",
        len(wav_bytes),
        len(out),
        int(sr),
        int(target_sr),
        (time.monotonic() - start) * 1000.0,
    )
    return out, int(target_sr), 2


def rms_dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + eps))
    return 20.0 * float(np.log10(max(rms, eps)))


def peak_dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    peak = float(np.max(np.abs(x)) + eps)
    return 20.0 * float(np.log10(max(peak, eps)))


def normalize_audio(x: np.ndarray) -> np.ndarray:
    mode = NORM_MODE
    if mode == "none":
        return x

    x = x.astype(np.float32, copy=False)
    if x.size == 0 or float(np.max(np.abs(x))) < 1e-5:
        return x

    def apply_gain_db(sig: np.ndarray, gain_db: float) -> np.ndarray:
        g = 10.0 ** (gain_db / 20.0)
        return sig * np.float32(g)

    if mode == "peak":
        gain_db = min(TARGET_PEAK_DBFS - peak_dbfs(x), MAX_GAIN_DB)
        return np.clip(apply_gain_db(x, gain_db), -1.0, 1.0)

    if mode == "rms":
        gain_db = min(TARGET_RMS_DBFS - rms_dbfs(x), MAX_GAIN_DB)
        y = apply_gain_db(x, gain_db)
        # peak safety
        pk = peak_dbfs(y)
        if pk > TARGET_PEAK_DBFS:
            y = apply_gain_db(y, TARGET_PEAK_DBFS - pk)
        return np.clip(y, -1.0, 1.0)

    return x


def trim_silence_energy(
    x: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    hop_ms: int = 10,
    noise_percentile: float = 10.0,
    snr_db: float = 10.0,
    floor_db: float = -45.0,
    pad_ms: int = 150,
    min_keep_ms: int = 300,
    min_voiced_ms: int = 60,
) -> np.ndarray:
    """Energy-based VAD trim.

    Removes leading/trailing non-voice regions by thresholding short-time RMS.
    Threshold is estimated from the quietest frames (noise_percentile) + snr_db.

    Returns possibly-trimmed audio. If no voiced region is detected, returns original x.
    """
    if x.size == 0:
        return x

    x = x.astype(np.float32, copy=False)
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    pad = int(sr * pad_ms / 1000)
    min_keep = int(sr * min_keep_ms / 1000)

    if x.size < frame:
        return x

    # Compute frame RMS dB
    starts = np.arange(0, x.size - frame + 1, hop, dtype=np.int64)
    if starts.size == 0:
        return x

    # Vectorized framing via striding is possible but keep it simple and robust.
    rms = np.empty((starts.size,), dtype=np.float32)
    eps = np.float32(1e-12)
    for i, s in enumerate(starts):
        seg = x[s : s + frame]
        rms[i] = np.sqrt(np.mean(seg * seg) + eps)

    db = 20.0 * np.log10(np.maximum(rms, eps))

    def find_voiced(thr_db: float, floor_db_value: float) -> np.ndarray:
        return db >= max(thr_db, floor_db_value)

    def find_voiced_bounds(mask: np.ndarray) -> tuple[int, int] | None:
        min_frames = max(1, int(round(min_voiced_ms / max(hop_ms, 1))))
        if mask.size < min_frames:
            return None
        if min_frames == 1:
            first_idx = int(np.argmax(mask))
            if not mask[first_idx]:
                return None
            last_idx = int(len(mask) - 1 - np.argmax(mask[::-1]))
            return first_idx, last_idx

        window = np.ones((min_frames,), dtype=np.int32)
        hits = np.convolve(mask.astype(np.int32), window, mode="valid")
        idx = np.flatnonzero(hits >= min_frames)
        if idx.size == 0:
            return None
        first_idx = int(idx[0])
        last_idx = int(idx[-1] + min_frames - 1)
        return first_idx, last_idx

    db_median = float(np.median(db))
    mad = float(np.median(np.abs(db - db_median)))
    if mad <= 1e-6:
        mad = 1e-6
    thr_db = db_median + 3.0 * mad

    voiced = find_voiced(thr_db, floor_db)
    if not np.any(voiced):
        return x

    bounds = find_voiced_bounds(voiced)
    if bounds is None:
        return x
    first, last = bounds

    start_samp = int(starts[first])
    end_samp = int(starts[last] + frame)

    # Pad and clamp
    start_samp = max(0, start_samp - pad)
    end_samp = min(x.size, end_samp + pad)

    if end_samp - start_samp < min_keep:
        return x

    return x[start_samp:end_samp]
