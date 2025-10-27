#!/usr/bin/env python3
"""
Load Complete NFL 2025 Schedule
================================
Loads ALL NFL 2025 season games from schedules/nfl_schedule.py into database.
"""

import sqlite3
import sys
sys.path.insert(0, '.')
from schedules.nfl_schedule import get_nfl_schedule

DATABASE = 'sports_predictions_original.db'

def load_complete_schedule():
    """Load complete NFL 2025 schedule into database"""
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Get all games from schedule file
    all_games = get_nfl_schedule()
    
    print(f"Loading {len(all_games)} NFL 2025 games...")
    
    # Clear existing NFL 2025 data
    print("Clearing existing NFL 2025 data...")
    cursor.execute("DELETE FROM games WHERE sport = 'NFL' AND season = 2025")
    cursor.execute("DELETE FROM predictions WHERE sport = 'NFL' AND game_date LIKE '%/2025%'")
    
    # Insert all games
    inserted = 0
    for game in all_games:
        try:
            # Parse result if available
            home_score = None
            away_score = None
            if game.get('result'):
                scores = game['result'].split(' - ')
                if len(scores) == 2:
                    away_score = int(scores[0].strip())
                    home_score = int(scores[1].strip())
            
            # Create unique game_id
            game_id = f"NFL_2025_{game['date'].replace('/', '').replace(':', '').replace(' ', '_')}_{game['match_id']}"
            
            cursor.execute('''
                INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'NFL',
                'NFL',
                game_id,
                2025,
                game['date'],
                game['home_team'],
                game['away_team'],
                home_score,
                away_score
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting game {game['match_id']}: {e}")
    
    conn.commit()
    conn.close()
    
    # Count completed vs upcoming
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    completed = cursor.execute("SELECT COUNT(*) FROM games WHERE sport = 'NFL' AND season = 2025 AND home_score IS NOT NULL").fetchone()[0]
    upcoming = cursor.execute("SELECT COUNT(*) FROM games WHERE sport = 'NFL' AND season = 2025 AND home_score IS NULL").fetchone()[0]
    conn.close()
    
    print(f"\n✅ Successfully loaded {inserted} NFL 2025 games!")
    print(f"📊 {completed} completed games")
    print(f"📅 {upcoming} upcoming games")
    print(f"🏈 Full season ready!")
    
    return inserted

if __name__ == '__main__':
    load_complete_schedule()
