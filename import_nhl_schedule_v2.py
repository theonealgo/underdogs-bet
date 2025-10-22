"""
Import NHL 2025-2026 schedule into V2 database
Uses data from nhlschedules.py
"""

import sqlite3
import sys

# Import the schedule
sys.path.insert(0, 'backups/v2')
from nhlschedules_nhl_v2 import get_nhl_2025_schedule, import_nhl_schedules_to_database

DATABASE = 'backups/v2/sports_predictions_nhl_v2.db'

def import_schedule():
    """Import NHL schedule into V2 database"""
    
    print("\n" + "="*70)
    print("IMPORTING NHL 2025-2026 SCHEDULE INTO V2 DATABASE")
    print("="*70)
    
    # Get schedule data
    schedule = get_nhl_2025_schedule()
    print(f"\n✓ Loaded {len(schedule)} games from schedule")
    
    # Connect to V2 database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing NHL games
    cursor.execute("DELETE FROM games WHERE sport = 'NHL'")
    print(f"✓ Cleared existing NHL games")
    
    # Import games
    inserted = 0
    for game in schedule:
        try:
            cursor.execute('''
                INSERT INTO games 
                (sport, league, game_id, season, game_date, home_team_id, away_team_id, 
                 status, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'NHL',
                'NHL',
                str(game['match_id']),
                2026,
                game['date'],
                game['home_team'],
                game['away_team'],
                'scheduled',
                game.get('home_score'),
                game.get('away_score')
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting game {game['match_id']}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✓ Inserted {inserted} games into V2 database")
    
    # Verify
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    total = cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NHL'").fetchone()[0]
    upcoming = cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NHL' AND home_score IS NULL").fetchone()[0]
    completed = cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NHL' AND home_score IS NOT NULL").fetchone()[0]
    
    # Get date range
    dates = cursor.execute('''
        SELECT MIN(game_date), MAX(game_date) 
        FROM games 
        WHERE sport='NHL' AND home_score IS NULL
    ''').fetchone()
    
    conn.close()
    
    print(f"\n{'='*70}")
    print("VERIFICATION:")
    print(f"{'='*70}")
    print(f"Total games: {total}")
    print(f"Upcoming games: {upcoming}")
    print(f"Completed games: {completed}")
    print(f"Date range: {dates[0]} to {dates[1]}")
    print(f"\n✅ Schedule import complete!\n")


if __name__ == "__main__":
    import_schedule()
