#!/usr/bin/env python3
"""
Schedule Loader - Import schedules into database
=================================================
Loads NFL and NHL schedules from schedules/*.py files into the database.
Run this once per season to update game schedules.

Usage: python load_schedules.py [--sport NFL|NHL|ALL]
"""

import sqlite3
import sys
from datetime import datetime

DATABASE = 'sports_predictions_original.db'

def load_nfl_schedule():
    """Load NFL schedule from schedules/nfl_schedule.py"""
    print("📥 Loading NFL schedule...")
    
    # Import the schedule
    sys.path.insert(0, 'schedules')
    from nfl_schedule import get_nfl_schedule
    
    games = get_nfl_schedule()
    print(f"   Found {len(games)} NFL games")
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing NFL games
    cursor.execute("DELETE FROM games WHERE sport = 'NFL'")
    print(f"   Cleared old NFL data")
    
    # Insert new games
    inserted = 0
    for game in games:
        cursor.execute('''
            INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, home_score, away_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'NFL',
            'NFL',
            f"NFL_{game['match_id']}",
            2025,
            game['date'],
            game['home_team'],
            game['away_team'],
            parse_score(game.get('result'), 'home') if game.get('result') else None,
            parse_score(game.get('result'), 'away') if game.get('result') else None
        ))
        inserted += 1
    
    conn.commit()
    conn.close()
    
    print(f"   ✓ Inserted {inserted} NFL games")
    return inserted

def load_nhl_schedule():
    """Load NHL schedule from nhlschedules.py"""
    print("📥 Loading NHL schedule...")
    
    # Import the schedule (2025-2026 season)
    from nhlschedules import get_nhl_2025_schedule
    
    games = get_nhl_2025_schedule()
    print(f"   Found {len(games)} NHL games")
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing NHL games
    cursor.execute("DELETE FROM games WHERE sport = 'NHL'")
    print(f"   Cleared old NHL data")
    
    # Insert new games
    inserted = 0
    for game in games:
        # Convert date format: '2025-10-07' or 'DD/MM/YYYY' to 'DD/MM/YYYY'
        game_date = convert_date_format(game['date'])
        
        cursor.execute('''
            INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, home_score, away_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'NHL',
            'NHL',
            f"NHL_{game['match_id']}",
            2025,
            game_date,
            game['home_team'],
            game['away_team'],
            game.get('home_score'),
            game.get('away_score')
        ))
        inserted += 1
    
    conn.commit()
    conn.close()
    
    print(f"   ✓ Inserted {inserted} NHL games")
    return inserted

def convert_date_format(date_str):
    """Convert various date formats to DD/MM/YYYY"""
    # Try YYYY-MM-DD format first
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%d/%m/%Y')
    except:
        pass
    
    # Try DD/MM/YYYY HH:MM format
    try:
        dt = datetime.strptime(date_str.split()[0], '%d/%m/%Y')
        return dt.strftime('%d/%m/%Y')
    except:
        pass
    
    # Already in DD/MM/YYYY format
    return date_str.split()[0] if ' ' in date_str else date_str

def parse_score(result_str, team):
    """Parse score from result string like '24 - 20'"""
    if not result_str or result_str == 'None':
        return None
    
    try:
        parts = result_str.split('-')
        if len(parts) == 2:
            home_score = int(parts[0].strip())
            away_score = int(parts[1].strip())
            return home_score if team == 'home' else away_score
    except:
        return None
    
    return None

def main():
    """Main loader function"""
    sport = sys.argv[1].upper() if len(sys.argv) > 1 else 'ALL'
    
    print("\n🏈🏒 jackpotpicks.bet - Schedule Loader")
    print("=" * 50)
    
    total = 0
    
    if sport in ['NFL', 'ALL']:
        total += load_nfl_schedule()
    
    if sport in ['NHL', 'ALL']:
        total += load_nhl_schedule()
    
    print("=" * 50)
    print(f"✓ Successfully loaded {total} games into database\n")

if __name__ == '__main__':
    main()
