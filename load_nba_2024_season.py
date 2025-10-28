import sqlite3
import csv
from datetime import datetime

def parse_date(date_str):
    """Parse DD/MM/YYYY HH:MM format to YYYY-MM-DD"""
    dt = datetime.strptime(date_str, "%d/%m/%Y %H:%M")
    return dt.strftime("%Y-%m-%d")

def parse_result(result_str):
    """Parse 'score1 - score2' format to (score1, score2)"""
    parts = result_str.split(' - ')
    return int(parts[0].strip()), int(parts[1].strip())

def load_2024_nba_season():
    conn = sqlite3.connect('sports_predictions_original.db')
    cursor = conn.cursor()
    
    # First, check if 2024 data already exists
    cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NBA' AND season=2024")
    existing_count = cursor.fetchone()[0]
    
    if existing_count > 0:
        print(f"⚠️ Found {existing_count} existing 2024 NBA games. Removing them first...")
        cursor.execute("DELETE FROM games WHERE sport='NBA' AND season=2024")
        conn.commit()
        print(f"✓ Removed {existing_count} old 2024 NBA games")
    
    # Load the CSV file
    csv_path = 'attached_assets/nba-2024-UTC (1)_1761658402903.csv'
    
    games_loaded = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            match_number = row['Match Number']
            game_date = parse_date(row['Date'])
            home_team = row['Home Team']
            away_team = row['Away Team']
            result = row['Result']
            
            # Parse result
            home_score, away_score = parse_result(result)
            
            # Create game_id
            game_id = f"NBA_2024_{match_number}"
            
            # Insert into database
            cursor.execute("""
                INSERT INTO games (sport, league, game_id, season, game_date, 
                                 home_team_id, away_team_id, status, home_score, away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ('NBA', 'NBA', game_id, 2024, game_date, 
                  home_team, away_team, 'final', home_score, away_score))
            
            games_loaded += 1
            
            if games_loaded % 100 == 0:
                print(f"Loaded {games_loaded} games...")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Successfully loaded {games_loaded} 2024 NBA games")
    print(f"   Status: final (completed)")
    print(f"   Season: 2024")
    print(f"   These games will be used for model training only")
    
    # Verify the data
    conn = sqlite3.connect('sports_predictions_original.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*), MIN(game_date), MAX(game_date) FROM games WHERE sport='NBA' AND season=2024")
    count, min_date, max_date = cursor.fetchone()
    print(f"\n📊 Verification:")
    print(f"   Total 2024 games: {count}")
    print(f"   Date range: {min_date} to {max_date}")
    
    cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NBA' AND season=2025")
    count_2025 = cursor.fetchone()[0]
    print(f"   Total 2025 games: {count_2025}")
    
    conn.close()

if __name__ == "__main__":
    load_2024_nba_season()
