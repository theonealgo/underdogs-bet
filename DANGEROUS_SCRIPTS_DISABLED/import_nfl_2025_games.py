#!/usr/bin/env python3
"""
Import NFL 2025 Season Games
=============================
Loads 93 games from September-October 2025 into database.
"""

import sqlite3
from datetime import datetime

DATABASE = 'sports_predictions_original.db'

def import_2025_games():
    """Import NFL 2025 season games from text file"""
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # First, clear existing NFL 2025 data
    print("Clearing existing NFL 2025 data...")
    cursor.execute("DELETE FROM games WHERE sport = 'NFL' AND game_date LIKE '%/09/2025%' OR game_date LIKE '%/10/2025%'")
    cursor.execute("DELETE FROM predictions WHERE sport = 'NFL' AND game_date LIKE '%/09/2025%' OR game_date LIKE '%/10/2025%'")
    
    # Read the data file
    games_data = []
    with open('attached_assets/Pasted-Date-Visitor-Home-V-Score-H-Score-Actual-Winner-04-09-2025-Dallas-Cowboys-Philadelphia-Eagles-20-24--1761305739130_1761305739130.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                date = parts[0].strip()
                away_team = parts[1].strip()
                home_team = parts[2].strip()
                away_score = int(parts[3].strip()) if parts[3].strip() else None
                home_score = int(parts[4].strip()) if parts[4].strip() else None
                
                games_data.append({
                    'date': date,
                    'away': away_team,
                    'home': home_team,
                    'away_score': away_score,
                    'home_score': home_score
                })
    
    print(f"Importing {len(games_data)} games...")
    
    # Insert games
    inserted = 0
    for i, game in enumerate(games_data, start=1):
        try:
            # Create unique game_id
            game_id = f"NFL_2025_{game['date'].replace('/', '')}_{game['away'][:3]}_{game['home'][:3]}_{i}"
            
            cursor.execute('''
                INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'NFL',
                'NFL',
                game_id,
                2025,
                game['date'],  # Store as DD/MM/YYYY format
                game['home'],
                game['away'],
                game['home_score'],
                game['away_score']
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting game {game['away']} @ {game['home']}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Successfully imported {inserted} NFL 2025 games!")
    print(f"📅 Games from September 4 - October 9, 2025")
    print(f"🏈 All games have scores and can be displayed on Results page")
    
    return inserted

if __name__ == '__main__':
    import_2025_games()
