import sqlite3
import pandas as pd
from datetime import datetime

# Parse the NHL schedule file
schedule_file = "attached_assets/Pasted-Date-Time-Visitor-G-Home-G-Att-LOG-Notes-2025-10-07-5-00-PM-Chicago-Blackhawks-2-Florida-Panthers--1761842548476_1761842548477.txt"

# Read the file
with open(schedule_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse games (skip header rows that repeat)
games = []
match_id = 1

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    parts = line.split('\t')
    
    # Skip header rows
    if len(parts) > 0 and (parts[0] == 'Date' or parts[0].strip() == 'Date'):
        continue
    
    # Valid game row has at least 5 columns (Date, Time, Visitor, G, Home)
    if len(parts) >= 5:
        date = parts[0].strip()
        time = parts[1].strip() if len(parts) > 1 else ''
        visitor = parts[2].strip() if len(parts) > 2 else ''
        visitor_goals = parts[3].strip() if len(parts) > 3 else ''
        home = parts[4].strip() if len(parts) > 4 else ''
        home_goals = parts[5].strip() if len(parts) > 5 else ''
        
        # Skip if missing critical data
        if not date or not visitor or not home:
            continue
        
        # Skip if this looks like a header
        if date == 'Date' or visitor == 'Visitor':
            continue
        
        # Convert date to YYYY-MM-DD format
        try:
            dt = datetime.strptime(date, '%Y-%m-%d')
            formatted_date = dt.strftime('%Y-%m-%d')
        except:
            continue
        
        # Parse scores (empty if not played yet)
        home_score = None
        away_score = None
        if visitor_goals and visitor_goals.isdigit():
            away_score = int(visitor_goals)
        if home_goals and home_goals.isdigit():
            home_score = int(home_goals)
        
        games.append({
            'match_id': match_id,
            'date': formatted_date,
            'visitor': visitor,
            'home': home,
            'home_score': home_score,
            'away_score': away_score
        })
        match_id += 1

print(f"Parsed {len(games)} games from file")
print(f"Date range: {games[0]['date']} to {games[-1]['date']}")
print(f"Match IDs: 1 to {len(games)}")

# Connect to database
conn = sqlite3.connect('sports_predictions_original.db')
cursor = conn.cursor()

# Delete all existing NHL 2025 schedule
print("\nDeleting existing NHL 2025 schedule...")
cursor.execute("DELETE FROM games WHERE sport='NHL' AND season=2025")
deleted_count = cursor.rowcount
print(f"Deleted {deleted_count} existing games")

# Insert all 1,312 games
print(f"\nInserting {len(games)} games...")
inserted = 0

for game in games:
    game_id = f"NHL_2025_{game['match_id']}"
    
    cursor.execute("""
        INSERT INTO games (game_id, sport, league, season, game_date, home_team_id, away_team_id, home_score, away_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_id,
        'NHL',
        'NHL',
        2025,
        game['date'],
        game['home'],
        game['visitor'],
        game['home_score'],
        game['away_score']
    ))
    inserted += 1

conn.commit()
print(f"Successfully inserted {inserted} games")

# Verify
cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NHL' AND season=2025")
total = cursor.fetchone()[0]
print(f"\nVerification: {total} NHL 2025 games in database")

# Show first 10 and last 10
print("\nFirst 10 games in database:")
cursor.execute("""
    SELECT game_id, game_date, away_team_id, home_team_id 
    FROM games 
    WHERE sport='NHL' AND season=2025 
    ORDER BY game_id 
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"  {row[0]:20s} | {row[1]} | {row[2]:25s} @ {row[3]:25s}")

print("\nLast 10 games in database:")
cursor.execute("""
    SELECT game_id, game_date, away_team_id, home_team_id 
    FROM games 
    WHERE sport='NHL' AND season=2025 
    ORDER BY game_id DESC
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"  {row[0]:20s} | {row[1]} | {row[2]:25s} @ {row[3]:25s}")

conn.close()
print("\n✓ Schedule rebuild complete!")
