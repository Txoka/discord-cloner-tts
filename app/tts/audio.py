from __future__ import annotations

from typing import Any

import numpy as np
import torch

from app.config import (
    MAX_GAIN_DB,
    NORM_MODE,
    TARGET_PEAK_DBFS,
    TARGET_RMS_DBFS,
)


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
    pad_ms: int = 150,
    min_keep_ms: int = 300,
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

    noise_db = float(np.percentile(db, noise_percentile))
    thr_db = noise_db + float(snr_db)

    voiced = db >= thr_db
    if not np.any(voiced):
        return x

    first = int(np.argmax(voiced))
    last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))

    start_samp = int(starts[first])
    end_samp = int(starts[last] + frame)

    # Pad and clamp
    start_samp = max(0, start_samp - pad)
    end_samp = min(x.size, end_samp + pad)

    if end_samp - start_samp < min_keep:
        return x

    return x[start_samp:end_samp]
