#!/usr/bin/env python3
"""
Smoke-check public HTML routes before deploy.

Fails if any listed path returns HTTP 500 (or higher). Accepts redirects,
auth redirects (302), 404 on optional paths, etc.

Usage (from repo root):
    python3 scripts/verify_public_routes.py
"""
from __future__ import annotations

import os
import sys

# Repo root (parent of scripts/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Curated paths: high-traffic pages + areas often touched by template/nav work.
DEFAULT_PATHS = (
    "/",
    "/robots.txt",
    "/sitemap.xml",
    "/nba",
    "/nhl",
    "/privacy",
    "/terms",
    "/login",
    "/plans",
    "/results",
    "/sport/NBA/predictions",
    "/performance",
    "/player-props",
    "/tutorial",
    "/what-are-ai-sports-betting-picks",
    "/contact",
    "/donate",
    "/responsible-gaming",
    "/ai-sports-betting-picks-today",
    "/our-model-vs-sportsbooks",
    "/sport/NBA/spreads",
    "/sport/NBA/ats",
)


def main() -> int:
    import NHL77FINAL as m  # import after sys.path (repo root)

    client = m.app.test_client()
    bad: list[tuple[str, int]] = []
    for path in DEFAULT_PATHS:
        resp = client.get(path, follow_redirects=False)
        if resp.status_code >= 500:
            bad.append((path, resp.status_code))
        else:
            print(f"{path}\t{resp.status_code}")
    if bad:
        print("FAIL: server errors", file=sys.stderr)
        for path, code in bad:
            print(f"  {path}\t{code}", file=sys.stderr)
        return 1
    print("verify_public_routes: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
