# Discord Cloner TTS

A Discord bot that joins a voice channel, reads a chosen text channel aloud, and uses per-user voice cloning powered by Qwen3-TTS via vLLM-Omni.

## What it does
- Joins a voice channel and speaks messages from a selected text channel.
- Lets each user enroll a voice by reading a short Spanish sample in voice chat.
- Stores voice prompt files per user in `voices/<user_id>.pt` and reuses them for TTS.
- Plays messages in order per guild, even if synthesis times vary.

## How it works (high level)
1. You run the bot and invite it to your server.
2. Use `/join` to connect it to your voice channel.
3. Use `/setchannel` to select which text channel it should read.
4. Each user runs `/clone` and reads the sample text for ~20s.
5. Messages in the selected text channel are synthesized and played with that user's cloned voice.

## Quick start (Docker)
Prereqs: Docker, Docker Compose, and an NVIDIA GPU/runtime (the default compose file uses `runtime: nvidia`).

1) Create `.env` and set your bot token:
```env
DISCORD_TOKEN=your_token_here
```

2) Build and run:
```bash
make build
make up
```

3) Tail logs:
```bash
make logs
```

## Discord commands
- `/join [channel]` - Join the caller's voice channel, or an explicitly provided one (must have at least one human).
- `/leave` - Leave voice chat.
- `/setchannel #channel` - Choose which text channel to read aloud.
- `/clone` - Record a 20s Spanish sample to enroll your voice.
- `/forget` - Delete your enrolled voice prompt file.
- `/disguise @user` - Admins only: speak using someone else's voice.
- `/addadmin @user [role]` - Superadmins only: add an admin or superadmin.
- `/removeadmin @user` - Superadmins only: remove an admin or superadmin.
- `/adminlist` - Admins only: list all admins and their roles.
- `/debug` - Admins only: toggle admin commands for the current guild.
- `/adddebugguild <guild_id>` - Admins only: enable admin commands for a guild.
- `/removedebugguild <guild_id>` - Admins only: disable admin commands for a guild.
- `/debugguildlist` - Admins only: list guilds with admin commands enabled.
- `/sync` - Admins only: sync slash commands to the current guild.

## Configuration
Environment variables (set in `.env`):
```env
# Required
DISCORD_TOKEN=...
DISCORD_SUPERADMIN_IDS=441597233150951425   # Comma-separated IDs
DISCORD_DEBUG_GUILD_IDS=                   # Comma-separated guild IDs
DISCORD_DEBUG_COMMAND_ENABLED=true         # Enable global /debug (still available in debug guilds)

# Model and runtime
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
QWEN_TTS_DTYPE=bfloat16        # bfloat16|float16|float32
QWEN_TTS_DEVICE=cuda           # cuda|cpu
QWEN_TTS_LANG=Auto
QWEN_TTS_MAX_NEW_TOKENS=2048
QWEN_TTS_MAX_BATCH_SIZE=16   # Max TTS batch size (queue worker)

# Volume normalization
QWEN_TTS_NORM=none             # none|rms|peak
QWEN_TTS_TARGET_RMS_DBFS=-20
QWEN_TTS_TARGET_PEAK_DBFS=-1
QWEN_TTS_MAX_GAIN_DB=12

# Text handling
QWEN_TTS_MAX_CHARS=1024
QWEN_TTS_GLOBAL_QUEUE_LIMIT=200
QWEN_TTS_GUILD_QUEUE_LIMIT=50

# Voice-clone capture
QWEN_TTS_CLONE_SECONDS=20
QWEN_TTS_CLONE_MIN_SECONDS=3.0
QWEN_TTS_CLONE_TEXT=...         # Optional custom sample text

# Auto-leave
QWEN_TTS_AUTO_LEAVE_SECONDS=300  # Leave when alone in voice for this many seconds

# Storage
VOICES_DIR=/app/voices
DISCORD_ADMIN_DB_PATH=/app/data/admins.sqlite3
```

## Notes
- The bot only speaks messages from users who have enrolled a voice.
- Voice enrollment uses `discord-ext-voice-recv` to capture decoded PCM from Discord.
- Audio is synthesized in-memory (WAV bytes) and decoded to PCM for streaming playback.
- Model weights and cache are mounted to `./models` by docker-compose.
- Admin roles are stored in a SQLite database under `./data`.
- TTS scheduling is round-robin across guilds to keep fairness while batching.
- Messages are rejected with a ❌ reaction when queue limits are exceeded.
- If the bot is alone in voice for `QWEN_TTS_AUTO_LEAVE_SECONDS`, it disconnects.
- Admin commands only exist in guilds listed in `DISCORD_DEBUG_GUILD_IDS` (and stored in the DB).
- During `/clone`, `/leave` is blocked and `/join` can only target the cloning channel; if the bot wasn’t already in that channel and no `/join` happens during cloning, it disconnects afterward.

## Troubleshooting
- `ClientException: Already connected to a voice channel` on `/join` or `/clone`:
  - Usually means Discord has an active voice connection while the bot is trying to `connect()` again.
  - Workaround: run `/leave` then `/join` again, or wait a few seconds and retry.
- `NotFound: Unknown interaction` on `/join`:
  - The interaction token likely expired before the response was sent.
  - Retry the command; if it keeps happening, the command handler may need to defer responses sooner.
- `Mooncake not available` / `Datasystem not available` warnings:
  - These are optional connectors; safe to ignore unless you use those integrations.
- `WS payload has extra keys: {'seq': ...}` from `discord.ext.voice_recv`:
  - Informational from the voice receive library; generally safe to ignore.

## Repo layout
- `app/main.py` - Entry point, command wiring.
- `app/discord/` - Voice channel logic, recording, playback queue, and stream.
- `app/tts/` - Text cleanup, audio processing, prompt handling, TTS engine.
- `voices/` - Stored voice prompt files (mounted into the container).
