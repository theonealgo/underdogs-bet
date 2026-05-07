# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a multi-sport AI prediction and betting platform (**underdogs.bet**). The main product is a Flask web app; there are also optional microservices (odds engine, social media app, player props).

### Services

| Service | Command | Port | Notes |
|---------|---------|------|-------|
| Main Flask app | `uv run python NHL77FINAL.py` | 5000 | Core product. Uses SQLite (auto-created). |
| Player Props backend | `uvicorn app.main:app --host 0.0.0.0 --port 8101 --reload` (from `standalone-player-props/backend/`) | 8101 | FastAPI. Module path is `app.main:app` (not `app:app`). |
| Player Props frontend | `npm run dev` (from `standalone-player-props/frontend/`) | 5179 | React/Vite |
| Social Media app | `npm run dev` (from `social-media-app/`) | 3000 | Next.js 14. Requires Node 22.5+ for `--experimental-sqlite`. |
| Odds Engine | `docker compose up --build` (from `odds_engine/`) | 7000 | Requires Docker. Needs Postgres+Redis. |

### Running the main app

```bash
cd /workspace
uv run python NHL77FINAL.py
```

The app serves on `http://0.0.0.0:5000`. Routes use `/<slug>` pattern (e.g. `/nba-picks`, `/nhl-picks`). The sport pages redirect from `/nba` -> `/nba-picks`.

### Lint & Typecheck

- **social-media-app**: `cd social-media-app && npx next lint` and `npx tsc --noEmit`
- No Python linters configured at repository level.

### Tests

- `scripts/verify_public_routes.py` — smoke-tests Flask public routes (requires the app to be running on port 5000)
- `test_threshold_picks.py` — tests the ATS threshold pick system locally
- `test_apis.py` — requires external API keys; tests external sports data APIs
- No pytest/jest framework configured.

### Key gotchas

- The `uv.lock` file is present; always use `uv sync` (not `pip install`) for the root Python project.
- The `social-media-app` lint requires ESLint 8 + `eslint-config-next@14.2.3` (not ESLint 9, which is incompatible with Next.js 14).
- The player-props backend entry point is `app.main:app`, not `app:app`.
- Shell scripts in the repo root (e.g. `start_app.sh`, `daily_predictions.sh`) hardcode macOS paths and are NOT usable in Cloud Agent VMs.
- The `DISABLE_PREMIUM_PAYWALL=1` env var bypasses premium checks for local dev.
