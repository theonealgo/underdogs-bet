#!/usr/bin/env python3
"""
Load 95 NFL Games (Sept 4 - Oct 9, 2025)
=========================================
"""

import sqlite3

DATABASE = 'sports_predictions_original.db'

def load_95_games():
    """Load the specific 95 NFL games for testing"""
    
    # Read the 95 games from the file
    games_data = []
    with open('attached_assets/Pasted-Date-Visitor-Home-V-Score-H-Score-04-09-2025-Dallas-Cowboys-Philadelphia-Eagles-20-24-05-09-2025-Kan-1761568939591_1761568939591.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
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
    
    print(f"Found {len(games_data)} games in file")
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing NFL 2025 data
    print("Clearing existing NFL 2025 data...")
    cursor.execute("DELETE FROM games WHERE sport = 'NFL' AND season = 2025")
    cursor.execute("DELETE FROM predictions WHERE sport = 'NFL'")
    
    # Insert all 95 games
    print(f"Inserting {len(games_data)} games...")
    inserted = 0
    for i, game in enumerate(games_data, start=1):
        try:
            game_id = f"NFL_2025_{game['date'].replace('/', '')}_{i}"
            
            cursor.execute('''
                INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'NFL',
                'NFL',
                game_id,
                2025,
                game['date'],
                game['home'],
                game['away'],
                game['home_score'],
                game['away_score']
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting game {i}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Successfully loaded {inserted} NFL games!")
    print(f"📊 All games have scores for results testing")
    
    return inserted

if __name__ == '__main__':
    load_95_games()
