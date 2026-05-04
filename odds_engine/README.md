# Odds Engine Stack

This folder contains the full sportsbook-style odds engine stack (Fastify + Python + Postgres + Redis).

## Quick start

```bash
docker compose up --build
```

Services:
- Fastify API: http://localhost:7000
- Python model service: http://localhost:7001
- Postgres: localhost:5433 (db: `odds_engine`)
- Redis: localhost:6380

## API endpoints

- `GET /health`
- `GET /games`
- `GET /odds?gameId=...` or `GET /odds?sport=...&home=...&away=...`
- `POST /player-props/batch` (body: `{ sport, items: [{ player_id, player_name, team, prop_type }] }`)
- `GET /model?gameId=...`
- `POST /bet` (body: `{ gameId, market, side, stake }`)
- `GET /risk?gameId=...`

## Sample data

Sample games and team stats are seeded via `db/init.sql`.

## Spreads (how the line is read)

Point spreads have **nothing to do with home or away** as labels: the board is **favorite vs underdog** and **by how many** (expected margin / handicap). The favorite always gets the **minus** number; the underdog gets the **plus** number of the same size. Venue is irrelevant to that relationship—only relative team strength and the handicap matter.

## Local integration (Flask app)

Set the Flask app to read odds from this service:

```bash
export ODDS_ENGINE_URL=http://localhost:7000
```
