#!/bin/bash
# Double-click in Finder: starts Flask (default port 5059). Override: LOCAL_PORT=8899 ./run_local.command
cd "$(dirname "$0")" || exit 1
export OPEN_BROWSER=1
export LOCAL_PORT="${LOCAL_PORT:-5059}"
exec python3 NHL77FINAL.py
