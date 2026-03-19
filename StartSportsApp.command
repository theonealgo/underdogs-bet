#!/bin/bash
cd "/Users/nimamesghali/Documents/2025sports/SportStatsAPI (v2)"
if ! pgrep -f NHL77FINAL.py > /dev/null; then
    nohup python3 NHL77FINAL.py > app.log 2>&1 &
    echo "Sports app started on http://localhost:5001"
    sleep 2
fi
