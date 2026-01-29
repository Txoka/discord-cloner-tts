# Architecture

## Overview
This project runs a Discord bot that reads messages from a selected text channel, synthesizes speech with Qwen3-TTS via vLLM-Omni, and plays audio in a guild voice channel. Each user must enroll a voice sample which is stored on disk and reused for future synthesis.

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

## Data & State
- Voice prompt files: `voices/<user_id>.pt`.
- Planned admin database (SQLite): persistent storage for admin roles.
- In-memory state:
  - Per-guild playback queues with ordered delivery.
  - Global TTS request queue and worker(s).

## Message → Audio Pipeline (Current)
1. Discord message received (`on_message`).
2. Validate guild + channel + prompt presence + cloning status.
3. Enqueue TTS request (global) → get a future for WAV bytes.
4. Enqueue TTS future into per-guild playback queue.
5. Playback worker consumes futures in order and plays audio.

## Planned Updates
- Replace global TTS queue with a fairness-aware scheduler that pulls requests in round-robin across guilds, while still batching across guilds.
- Add queue limits (global + per-guild). If exceeded, react with :x:.
- Introduce SQLite admin database with two roles: admin and superadmin.
- Update bot commands to add/remove admins/superadmins.
- Expand tests to cover new scheduling, limits, and DB behavior.

## Config (Environment)
- `DISCORD_TOKEN`: bot token.
- `DISCORD_SUPERADMIN_ID`: comma-separated list of superadmin IDs (must include txoka).
- `QWEN_TTS_*`: model/runtime settings.
- `QWEN_TTS_QUEUE_LIMIT_*`: queue limits (planned).

