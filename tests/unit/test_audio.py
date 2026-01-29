from __future__ import annotations

import numpy as np
import torch

import app.tts.audio as audio
import app.tts.audio.utils as audio_utils


def test_to_mono_float32_from_tensor():
    x = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
    out = audio.to_mono_float32(x)
    assert out.dtype == np.float32
    assert out.shape == (2,)


def test_to_mono_float32_int_scaling():
    x = np.array([0, 32767, -32768], dtype=np.int16)
    out = audio.to_mono_float32(x)
    assert np.max(out) <= 1.0
    assert np.min(out) >= -1.0


def test_rms_peak_dbfs_ranges():
    x = np.ones((1000,), dtype=np.float32) * 0.5
    assert audio.peak_dbfs(x) < 0.0
    assert audio.rms_dbfs(x) < 0.0


def test_normalize_audio_none(monkeypatch):
    monkeypatch.setattr(audio_utils, "NORM_MODE", "none")
    x = np.array([0.1, -0.1], dtype=np.float32)
    out = audio.normalize_audio(x)
    assert np.allclose(out, x)


def test_normalize_audio_peak(monkeypatch):
    monkeypatch.setattr(audio_utils, "NORM_MODE", "peak")
    monkeypatch.setattr(audio_utils, "TARGET_PEAK_DBFS", -1.0)
    monkeypatch.setattr(audio_utils, "MAX_GAIN_DB", 12.0)
    x = np.array([0.1, -0.1], dtype=np.float32)
    out = audio.normalize_audio(x)
    assert np.max(np.abs(out)) <= 1.0


def test_normalize_audio_rms(monkeypatch):
    monkeypatch.setattr(audio_utils, "NORM_MODE", "rms")
    monkeypatch.setattr(audio_utils, "TARGET_RMS_DBFS", -20.0)
    monkeypatch.setattr(audio_utils, "TARGET_PEAK_DBFS", -1.0)
    monkeypatch.setattr(audio_utils, "MAX_GAIN_DB", 12.0)
    x = np.ones((1000,), dtype=np.float32) * 0.01
    out = audio.normalize_audio(x)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 1.0


def test_trim_silence_energy_basic():
    sr = 16000
    silence = np.zeros((sr // 2,), dtype=np.float32)
    tone = np.ones((sr // 2,), dtype=np.float32) * 0.2
    x = np.concatenate([silence, tone, silence])
    out = audio.trim_silence_energy(x, sr)
    assert out.size <= x.size
    assert out.size > 0


def test_trim_silence_energy_all_silence():
    sr = 16000
    x = np.zeros((sr,), dtype=np.float32)
    out = audio.trim_silence_energy(x, sr)
    assert out.size == x.size


def test_prepare_tts_pcm_invalid_bytes_returns_empty():
    out, sr, ch = audio.prepare_tts_pcm(b"not a wav")
    assert out == b""
    assert sr == 0
    assert ch == 0


def test_trim_silence_energy_from_file():
    from pathlib import Path
    import pytest
    import subprocess
    import wave

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    ogg_path = assets_dir / "vad_crop_audio.ogg"
    if not ogg_path.exists():
        pytest.skip("vad_crop_audio.ogg not provided yet")

    wav_path = assets_dir / "vad_raw.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(ogg_path),
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n)

    wav_path.unlink(missing_ok=True)

    assert sampwidth == 2
    x = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=-1)
    info = np.iinfo(np.int16)
    denom = float(max(abs(info.min), info.max))
    mono = np.clip(x.astype(np.float32) / denom, -1.0, 1.0)
    trimmed = audio.trim_silence_energy(mono, int(sr))

    def compute_bounds(x: np.ndarray, sr: int) -> tuple[int, int]:
        frame_ms = 30
        hop_ms = 10
        floor_db = -45.0
        pad_ms = 150
        min_keep_ms = 300
        min_voiced_ms = 60

        if x.size == 0:
            return 0, x.size
        frame = max(1, int(sr * frame_ms / 1000))
        hop = max(1, int(sr * hop_ms / 1000))
        pad = int(sr * pad_ms / 1000)
        min_keep = int(sr * min_keep_ms / 1000)
        if x.size < frame:
            return 0, x.size
        starts = np.arange(0, x.size - frame + 1, hop, dtype=np.int64)
        if starts.size == 0:
            return 0, x.size

        rms = np.empty((starts.size,), dtype=np.float32)
        eps = np.float32(1e-12)
        for i, s in enumerate(starts):
            seg = x[s : s + frame]
            rms[i] = np.sqrt(np.mean(seg * seg) + eps)

        db = 20.0 * np.log10(np.maximum(rms, eps))
        db_median = float(np.median(db))
        mad = float(np.median(np.abs(db - db_median)))
        if mad <= 1e-6:
            mad = 1e-6
        thr_db = db_median + 3.0 * mad
        mask = db >= max(thr_db, floor_db)

        min_frames = max(1, int(round(min_voiced_ms / max(hop_ms, 1))))
        if mask.size < min_frames:
            return 0, x.size

        window = np.ones((min_frames,), dtype=np.int32)
        hits = np.convolve(mask.astype(np.int32), window, mode="valid")
        idx = np.flatnonzero(hits >= min_frames)
        if idx.size == 0:
            return 0, x.size

        first = int(idx[0])
        last = int(idx[-1] + min_frames - 1)
        start_samp = int(starts[first])
        end_samp = int(starts[last] + frame)
        start_samp = max(0, start_samp - pad)
        end_samp = min(x.size, end_samp + pad)
        if end_samp - start_samp < min_keep:
            return 0, x.size
        return start_samp, end_samp

    start_samp, end_samp = compute_bounds(mono, int(sr))
    start_ms = start_samp * 1000.0 / float(sr)
    end_ms = end_samp * 1000.0 / float(sr)

    expected_start_ms = 7070.0
    expected_end_ms = 14100.0
    tolerance_ms = 60.0

    print(f"vad_start_ms={start_ms:.1f} vad_end_ms={end_ms:.1f}")
    assert abs(start_ms - expected_start_ms) <= tolerance_ms
    assert abs(end_ms - expected_end_ms) <= tolerance_ms
    assert trimmed.size == end_samp - start_samp
