# Architecture

## Overview
This project runs a Discord bot that reads messages from a selected text channel, synthesizes speech with Qwen3-TTS via vLLM-Omni, and plays audio in a guild voice channel. Each user must enroll a voice sample which is stored on disk and reused for future synthesis.

Dependency compatibility note: `vllm` and `vllm-omni` must stay version-aligned to avoid runtime import errors during startup (for example, missing `vllm.multimodal.processing.context` when APIs drift).

## Core Components
- `app/main.py`
  - Process entry point. Loads config, initializes `TTSEngine`, wires Discord slash commands.
- `app/discord/bot.py`
  - Discord client implementation.
  - Manages per-guild voice connections and playback queues.
  - Handles enrollment (`/clone`), per-message routing, and admin actions.
- `app/tts/engine.py`
  - TTS queue manager and model wrapper.
  - Loads and caches prompt items for each user.
  - Performs batching of synthesis requests.
- `app/tts/text.py`
  - Sanitizes and normalizes Discord message text for TTS.
- `app/tts/audio/utils.py`
  - Audio normalization, trimming, resampling, and PCM conversion.
  - Energy-based VAD uses a median+MAD threshold with an absolute dB floor and a minimum voiced window.

## Data & State
- Voice prompt files: `voices/<user_id>.pt`.
- Admin database (SQLite): persistent storage for admin roles and admin disguises.
- In-memory state:
  - Per-guild playback queues with ordered delivery.
  - Per-guild PCM stream feeding a single Discord audio source.
  - Fairness scheduler with per-guild request queues (round-robin).

## Message → Audio Pipeline (Current)
1. Discord message received (`on_message`).
2. Validate guild + channel + prompt presence + cloning status.
3. Enqueue TTS request into per-guild scheduler (round-robin) → get a future for WAV bytes.
4. Enqueue TTS future into per-guild playback queue.
5. Playback worker consumes futures in order and enqueues PCM into a per-guild stream (single Discord audio source).

## Behavior Notes
- Scheduler batches across guilds while ensuring round-robin fairness (one per guild per cycle; fills remaining slots if fewer guilds).
- Queue limits are enforced at ingest (global + per-guild); rejected messages get a ❌ reaction.
- Admin roles are stored in SQLite with two levels: admin and superadmin (master superadmins from env).
- Admin commands are only registered in debug guilds; other guilds have no admin behavior.
- `/debug` is a global command that toggles admin commands for the current guild.
- Admin disguises are stored per admin user and apply in any debug guild.
- `/join` uses a standard Discord voice connection for playback; `/clone` upgrades or reconnects with `discord-ext-voice-recv` only when capture is needed.
- While cloning, `/leave` is blocked and `/join` is restricted to the cloning channel; the bot disconnects after clone unless already joined (or joined during cloning).
- `/join` refuses empty voice channels, and the bot auto-leaves after being alone for the configured timeout.

## Config (Environment)
- `DISCORD_TOKEN`: bot token.
- `DISCORD_SUPERADMIN_IDS`: comma-separated list of superadmin IDs.
- `DISCORD_DEBUG_GUILD_IDS`: comma-separated list of guild IDs that get admin commands.
- `DISCORD_ADMIN_DB_PATH`: SQLite DB file path.
- `DISCORD_DEBUG_COMMAND_ENABLED`: enable or disable the global `/debug` toggle (still available in debug guilds).
- `QWEN_TTS_*`: model/runtime settings.
- `QWEN_TTS_GLOBAL_QUEUE_LIMIT` / `QWEN_TTS_GUILD_QUEUE_LIMIT`: queue limits.
- `QWEN_TTS_AUTO_LEAVE_SECONDS`: seconds before auto-leaving when alone in voice.
