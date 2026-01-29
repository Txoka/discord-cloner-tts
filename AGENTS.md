# Repository Guidelines

## Project Structure & Module Organization
- `app/` holds the Python application code.
  - `app/main.py` is the entry point; it wires the Discord bot to the TTS engine.
  - `app/discord/` contains bot and voice-channel behavior.
  - `app/tts/` contains text handling, audio processing, prompts, and the TTS engine.
  - `app/config.py` centralizes environment-driven settings.
- `tests/` holds unit and integration tests with helper fakes under `tests/helpers/`.
- `voices/` stores voice prompt data and generated assets (mounted into the container).
- `Dockerfile`, `docker-compose.yml`, and `Makefile` define the containerized workflow.
- `requirements.txt` pins Python dependencies (CUDA-enabled PyTorch wheels).

## Build, Test, and Development Commands
- `make build` — build the Docker image with your user/group IDs.
- `make up` — run the service via Docker Compose (GPU-enabled; rebuilds if needed).
- `make down` — stop and remove the Compose stack.
- `make logs` — tail container logs.
- `make test` — run the test suite via Docker Compose.

## Coding Style & Naming Conventions
- Python 3.12; use 4-space indentation and match existing import/order patterns.
- Keep type hints consistent with current modules (`from __future__ import annotations`).
- Prefer descriptive module/function names (e.g., `engine.py`, `prompt.py`).
- No formatter or linter is configured—keep changes minimal and consistent.

## Testing Guidelines
- Automated tests are present under `tests/`.
- Use `make test` to run the suite via Docker Compose.

## Commit & Pull Request Guidelines
- This repository uses Git with a `main` branch.
- Use clear, imperative commit messages (e.g., “Add voice normalization options”).
- PRs should describe behavior changes, configuration updates, and any new dependencies.
- Include usage notes if new environment variables are introduced.

## Security & Configuration Tips
- Store secrets in `.env` (e.g., `DISCORD_TOKEN`) and avoid committing real tokens.
- Model and runtime settings are controlled via `QWEN_TTS_*` env vars in `app/config.py`.
- Voice data under `voices/` can contain user content; treat it as sensitive.

## TTS Pipeline (Global Engine + Per-Guild Playback)
```
Discord message (guild text channel)
  └─ on_message() in app/discord/bot.py
      ├─ checks: guild connected, correct channel, user has prompt, not cloning
      ├─ preprocess text
      ├─ TTSEngine.enqueue(user_id, text)   <-- GLOBAL TTS QUEUE (all guilds)
      │    └─ _queue_worker batches mixed users/guilds
      │        └─ _process_batch -> model.generate_voice_clone(...)
      │             └─ Future resolves to WAV bytes
      └─ GuildState.queue.put(TTSJob(future))  <-- PER-GUILD QUEUE
           └─ _player_worker(guild_id)
               ├─ await future (synth result)
               ├─ trim silence
               └─ play in voice channel (ordered per guild)
```
