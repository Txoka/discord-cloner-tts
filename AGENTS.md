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
- Update or add tests for any functional change, and always run the test suite before merging into `dev`.

## Commit & Pull Request Guidelines
- This repository uses Git with a `main` branch.
- Make incremental commits while working; avoid large monolithic commits.
- Commit frequently within each feature branch using clear, meaningful messages.
- Create a feature branch before significant changes and merge it into `dev`.
- Only the repository owner merges `dev` into `main`.
- If merge conflicts arise, resolve them unless they involve user-owned changes; when uncertain, ask first.
- Use clear, imperative commit messages (e.g., “Add voice normalization options”).
- PRs should describe behavior changes, configuration updates, and any new dependencies.
- Include usage notes if new environment variables are introduced.

## Required Workflow for New Features/Fixes
For any new feature or fix, always follow this sequence:
1. Create and switch to a new feature branch before making changes.
2. Implement the change.
3. Update or add tests as needed.
4. Run the test suite and iterate until all tests pass.
5. Update documentation after tests pass.
6. Merge the feature branch into `dev`.

## Documentation & Planning
- Read and keep `ARCHITECTURE.md` current when behavior or structure changes.
- Use `TODO.md` to track active work and update it as tasks are completed.
- Update user-facing docs (e.g., `README.md`) whenever behavior or configuration changes.

## Clarifications
- If any requirement is ambiguous or uncertain, ask for clarification before coding or making assumptions.

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
