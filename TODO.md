# TODO

## Planning / Design
- Define fairness scheduler (round-robin per guild, allow extra from guilds when batch slots remain).
- Define queue limit defaults + env variable names (global + per-guild).
- Confirm admin command names and parameters (add/remove + role).

## Data Layer
- Add SQLite dependency + DB file path.
- Create admin table schema and migration/initialization logic.
- Bootstrap superadmins from `DISCORD_SUPERADMIN_ID` (comma-separated).

## Bot Behavior
- Replace global FIFO queue with fairness-aware scheduling.
- Ensure per-guild playback order remains strict.
- Enforce queue limits; on rejection react with :x: (log warning if reaction fails).
- Add admin commands: add/remove admin or superadmin.

## Config / Runtime
- Add new env vars to `.env` + `app/config.py`.
- Add DB volume to `docker-compose.yml`.
- Add DB install step to `Dockerfile` if needed.
- Add DB path to `.gitignore`.

## Tests
- Expand unit tests for:
  - Scheduler fairness and batching behavior.
  - Queue limit rejections + reaction handling.
  - Admin DB operations and bootstrap.
- Update fakes to match runtime API behavior.
- Run test suite and adjust as needed.

## Docs
- Update `README.md` for new commands, config vars, and DB storage.
- Ensure `AGENTS.md` remains accurate.
