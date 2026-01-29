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
    import soundfile as sf
    import pytest

    wav_path = Path(__file__).resolve().parents[1] / "assets" / "vad_sample.wav"
    if not wav_path.exists():
        pytest.skip("vad_sample.wav not provided yet")

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    mono = audio.to_mono_float32(data)
    trimmed = audio.trim_silence_energy(mono, int(sr))

    # Expected crop window for the provided sample (see tests/assets/vad_sample.wav).
    expected_start_ms = 0
    expected_end_ms = 410
    tolerance_ms = 50

    expected_len = int(round((expected_end_ms - expected_start_ms) * int(sr) / 1000.0))
    tol = int(round(tolerance_ms * int(sr) / 1000.0))
    assert abs(trimmed.size - expected_len) <= tol
