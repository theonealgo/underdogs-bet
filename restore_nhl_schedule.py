#!/usr/bin/env python3
"""
EMERGENCY SCHEDULE RESTORE
Restore missing NHL games from nhlschedules.py to database
WITHOUT touching predictions or any other data
"""

import sqlite3
from nhlschedules import get_nhl_2025_schedule
from datetime import datetime

def restore_schedule():
    # Get correct schedule from file
    schedule = get_nhl_2025_schedule()
    print(f"✓ Loaded {len(schedule)} games from nhlschedules.py")
    
    # Connect to database
    conn = sqlite3.connect('sports_predictions_original.db')
    cursor = conn.cursor()
    
    # Check current state
    cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NHL' AND season=2025")
    current_count = cursor.fetchone()[0]
    print(f"✓ Database currently has {current_count} NHL games for 2025 season")
    
    # Clear ONLY the NHL 2025 games (not predictions!)
    print(f"⚠ Deleting existing {current_count} NHL 2025 games...")
    cursor.execute("DELETE FROM games WHERE sport='NHL' AND season=2025")
    print(f"✓ Cleared NHL 2025 games")
    
    # Insert all games from schedule file
    print(f"✓ Inserting {len(schedule)} games from schedule file...")
    inserted = 0
    
    for game in schedule:
        # Parse date from DD/MM/YYYY format
        date_parts = game['date'].split('/')
        game_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"  # Convert to YYYY-MM-DD
        
        try:
            cursor.execute('''
                INSERT INTO games (sport, league, game_id, season, game_date, 
                                 home_team_id, away_team_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'NHL',
                'NHL',
                f"NHL_2025_{game['match_id']}",
                2025,
                game_date,
                game['home_team'],
                game['away_team'],
                'scheduled'
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting game {game['match_id']}: {e}")
    
    conn.commit()
    print(f"✓ Inserted {inserted} games")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NHL' AND season=2025")
    final_count = cursor.fetchone()[0]
    print(f"✓ Database now has {final_count} NHL games")
    
    # Check date range
    cursor.execute("SELECT MIN(game_date), MAX(game_date) FROM games WHERE sport='NHL' AND season=2025")
    min_date, max_date = cursor.fetchone()
    print(f"✓ Date range: {min_date} to {max_date}")
    
    conn.close()
    
    if final_count == len(schedule):
        print(f"\n✅ SUCCESS: All {final_count} games restored")
    else:
        print(f"\n⚠ WARNING: Expected {len(schedule)} but got {final_count}")

if __name__ == "__main__":
    restore_schedule()
