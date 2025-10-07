#!/usr/bin/env python3
"""Test all sports APIs"""
from datetime import datetime, date
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("TESTING ALL SPORTS APIs")
print("=" * 60)

# Test NBA
print("\n1. NBA API Test:")
try:
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    live_board = live_scoreboard.ScoreBoard()
    live_games = live_board.get_dict()
    if live_games and 'scoreboard' in live_games and 'games' in live_games['scoreboard']:
        games = live_games['scoreboard']['games']
        print(f"   Found {len(games)} NBA games:")
        for game in games:
            away = game.get('awayTeam', {}).get('teamTricode', 'N/A')
            home = game.get('homeTeam', {}).get('teamTricode', 'N/A')
            print(f"   - {away} @ {home}")
    else:
        print("   No NBA games found")
except Exception as e:
    print(f"   Error: {e}")

# Test NHL
print("\n2. NHL API Test:")
try:
    from src.data_collectors.nhl_collector import NHLDataCollector
    nhl = NHLDataCollector()
    games = nhl.get_todays_games()
    if not games.empty:
        print(f"   Found {len(games)} NHL games:")
        for _, game in games.iterrows():
            print(f"   - {game['away_team_id']} @ {game['home_team_id']}")
    else:
        print("   No NHL games found")
except Exception as e:
    print(f"   Error: {e}")

# Test WNBA
print("\n3. WNBA API Test:")
try:
    from src.data_collectors.wnba_collector import WNBADataCollector
    wnba = WNBADataCollector()
    # Test for tomorrow (Oct 8)
    tomorrow = date(2025, 10, 8)
    games = wnba.get_todays_games(game_date=tomorrow)
    if not games.empty:
        print(f"   Found {len(games)} WNBA games for Oct 8:")
        for _, game in games.iterrows():
            print(f"   - {game['away_team_id']} @ {game['home_team_id']}")
    else:
        print("   No WNBA games found for Oct 8")
except Exception as e:
    print(f"   Error: {e}")

# Test NCAAF
print("\n4. NCAAF API Test:")
try:
    from src.data_collectors.ncaaf_collector import NCAAFDataCollector
    ncaaf = NCAAFDataCollector()
    # Test for tomorrow (Oct 8)
    tomorrow = date(2025, 10, 8)
    games = ncaaf.get_todays_games(game_date=tomorrow)
    if not games.empty:
        print(f"   Found {len(games)} NCAAF games for Oct 8:")
        for _, game in games.iterrows():
            print(f"   - {game['away_team_id']} @ {game['home_team_id']}")
    else:
        print("   No NCAAF games found for Oct 8")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
