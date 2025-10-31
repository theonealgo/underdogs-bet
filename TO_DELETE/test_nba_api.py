#!/usr/bin/env python3
"""Test NBA API to see what games are available"""
from datetime import datetime, date
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

# Test live scoreboard
try:
    print("Testing NBA Live Scoreboard...")
    live_board = live_scoreboard.ScoreBoard()
    live_games = live_board.get_dict()
    
    if live_games and 'scoreboard' in live_games and 'games' in live_games['scoreboard']:
        games = live_games['scoreboard']['games']
        print(f"\nFound {len(games)} games on live scoreboard:")
        for game in games:
            away_team = game.get('awayTeam', {}).get('teamTricode', 'N/A')
            home_team = game.get('homeTeam', {}).get('teamTricode', 'N/A')
            game_time = game.get('gameTimeUTC', 'N/A')
            status = game.get('gameStatus', 0)
            print(f"  {away_team} @ {home_team} - Time: {game_time}, Status: {status}")
    else:
        print("No games found in live scoreboard")
        print(f"Response keys: {live_games.keys() if live_games else 'None'}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
