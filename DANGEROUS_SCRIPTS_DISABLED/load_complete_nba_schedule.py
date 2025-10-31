#!/usr/bin/env python3
"""
Load COMPLETE NBA 2024-25 schedule from CSV
Loads ALL 1232 games exactly as provided - no modifications
"""
import sqlite3
import csv
from datetime import datetime

DATABASE = 'sports_predictions_original.db'
CSV_FILE = 'attached_assets/nba-2024-UTC (1)_1761665748391.csv'

def parse_score(result_str):
    """Parse '132 - 109' format to (home_score, away_score)"""
    if not result_str or result_str.strip() == '':
        return None, None
    try:
        parts = result_str.split('-')
        home_score = int(parts[0].strip())
        away_score = int(parts[1].strip())
        return home_score, away_score
    except:
        return None, None

def convert_date_to_sortable(date_str):
    """Convert 'DD/MM/YYYY HH:MM' to 'YYYY-MM-DD' for proper sorting"""
    try:
        # Parse: "22/10/2024 23:30"
        dt = datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"ERROR parsing date '{date_str}': {e}")
        return None

def load_complete_schedule():
    """Load ALL 1232 games from CSV exactly as provided"""
    print("=" * 60)
    print("LOADING COMPLETE NBA 2024-25 SCHEDULE")
    print("=" * 60)
    
    # Read CSV
    games = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            home_score, away_score = parse_score(row.get('Result', ''))
            game_date = convert_date_to_sortable(row['Date'])
            
            if game_date is None:
                print(f"SKIPPING game {row['Match Number']} - invalid date")
                continue
            
            games.append({
                'match_number': int(row['Match Number']),
                'round': int(row['Round Number']),
                'date': game_date,
                'home_team': row['Home Team'],
                'away_team': row['Away Team'],
                'home_score': home_score,
                'away_score': away_score
            })
    
    print(f"✓ Parsed {len(games)} games from CSV")
    
    # Connect to database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing NBA games
    cursor.execute("DELETE FROM games WHERE sport = 'NBA'")
    print(f"✓ Cleared old NBA data")
    
    # Insert ALL games
    inserted = 0
    for game in games:
        cursor.execute('''
            INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, home_score, away_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'NBA',
            'NBA',
            f"NBA_{game['match_number']}",
            2024,  # 2024-25 season
            game['date'],
            game['home_team'],
            game['away_team'],
            game['home_score'],
            game['away_score']
        ))
        inserted += 1
        
        if inserted % 100 == 0:
            print(f"   Inserted {inserted}/{len(games)} games...")
    
    conn.commit()
    conn.close()
    
    print("=" * 60)
    print(f"✓ SUCCESS: Loaded {inserted} NBA games into database")
    print("=" * 60)
    
    return inserted

if __name__ == '__main__':
    load_complete_schedule()
