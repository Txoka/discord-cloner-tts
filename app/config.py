from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

# -----------------------------
# Config
# -----------------------------
MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DTYPE = os.environ.get("QWEN_TTS_DTYPE", "bfloat16")  # bfloat16|float16|float32
DEVICE = os.environ.get("QWEN_TTS_DEVICE", "cuda")  # cuda|cpu
DEFAULT_LANGUAGE = os.environ.get("QWEN_TTS_LANG", "Auto")
MAX_NEW_TOKENS = int(os.environ.get("QWEN_TTS_MAX_NEW_TOKENS", "2048"))
LOG_LEVEL = os.environ.get("QWEN_TTS_LOG_LEVEL", "INFO")

# Admins
DISCORD_SUPERADMIN_IDS = os.environ.get("DISCORD_SUPERADMIN_IDS", "")
if not DISCORD_SUPERADMIN_IDS:
    DISCORD_SUPERADMIN_IDS = os.environ.get("DISCORD_SUPERADMIN_ID", "")
SUPERADMIN_IDS = [int(x) for x in (s.strip() for s in DISCORD_SUPERADMIN_IDS.split(",")) if x]

DISCORD_DEBUG_GUILD_IDS = os.environ.get("DISCORD_DEBUG_GUILD_IDS", "")
DEBUG_GUILD_IDS = [int(x) for x in (s.strip() for s in DISCORD_DEBUG_GUILD_IDS.split(",")) if x]

# Volume normalization (simple RMS normalization per utterance)
NORM_MODE = os.environ.get("QWEN_TTS_NORM", "none")  # none|rms|peak
TARGET_RMS_DBFS = float(os.environ.get("QWEN_TTS_TARGET_RMS_DBFS", "-20"))
TARGET_PEAK_DBFS = float(os.environ.get("QWEN_TTS_TARGET_PEAK_DBFS", "-1"))
MAX_GAIN_DB = float(os.environ.get("QWEN_TTS_MAX_GAIN_DB", "12"))

# Text handling
MAX_CHARS_PER_MESSAGE = int(os.environ.get("QWEN_TTS_MAX_CHARS", "1024"))
MAX_BATCH_SIZE = max(1, int(os.environ.get("QWEN_TTS_MAX_BATCH_SIZE", "4")))
GLOBAL_QUEUE_LIMIT = int(os.environ.get("QWEN_TTS_GLOBAL_QUEUE_LIMIT", "200"))
GUILD_QUEUE_LIMIT = int(os.environ.get("QWEN_TTS_GUILD_QUEUE_LIMIT", "10"))

# Voice-clone capture
CLONE_RECORD_SECONDS = int(os.environ.get("QWEN_TTS_CLONE_SECONDS", "20"))
CLONE_MIN_SECONDS = float(os.environ.get("QWEN_TTS_CLONE_MIN_SECONDS", "3.0"))

# Auto-leave when the bot is alone in voice
AUTO_LEAVE_SECONDS = int(os.environ.get("QWEN_TTS_AUTO_LEAVE_SECONDS", "300"))

# Spanish-heavy sample (covers rr, ll, ñ, j, z/ce/ci, numbers, punctuation, etc.)
CLONE_SAMPLE_TEXT_ES = os.environ.get(
    "QWEN_TTS_CLONE_TEXT",
    (
        "Hoy, al salir de casa en Bilbao con lluvia fina, pasaron un carro y un perro curiosos. "
        "Una niña con gafas me dijo: \"¿Quieres café y churros?\" Respondí \"¡Claro!\"; el reloj marcaba las once. "
        "Caminamos sin prisa y charlamos de ciencia, música y ferrocarril, hasta llegar a la plaza."
    ),
)

# Storage
VOICES_DIR = Path(os.environ.get("VOICES_DIR", str(BASE_DIR / "voices"))).resolve()
ADMIN_DB_PATH = Path(os.environ.get("DISCORD_ADMIN_DB_PATH", str(BASE_DIR / "data" / "admins.sqlite3"))).resolve()
