import sqlite3
from datetime import datetime

# Read the schedule file
file_path = "attached_assets/Pasted-Date-Time-Visitor-G-Home-G-Att-LOG-Notes-2025-10-07-5-00-PM-Chicago-Blackhawks-2-F-1761844646116_1761844646117.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

games = []
match_id = 1

for line in lines:
    parts = line.strip().split('\t')
    
    # Skip empty lines or header lines
    if len(parts) < 5:
        continue
    if parts[0] in ['Date', '']:
        continue
    
    date = parts[0].strip()
    visitor = parts[2].strip() if len(parts) > 2 else ''
    home = parts[4].strip() if len(parts) > 4 else ''
    
    # Skip if missing data
    if not date or not visitor or not home or date == 'Date':
        continue
    
    # Parse scores
    visitor_goals = parts[3].strip() if len(parts) > 3 else ''
    home_goals = parts[5].strip() if len(parts) > 5 else ''
    
    away_score = int(visitor_goals) if visitor_goals.isdigit() else None
    home_score = int(home_goals) if home_goals.isdigit() else None
    
    games.append({
        'match_id': match_id,
        'date': date,
        'visitor': visitor,
        'home': home,
        'home_score': home_score,
        'away_score': away_score
    })
    match_id += 1

print(f"Total games parsed: {len(games)}")
print(f"\nFirst game: {games[0]}")
print(f"Last game: {games[-1]}")

# Connect and rebuild database
conn = sqlite3.connect('sports_predictions_original.db')
cursor = conn.cursor()

# Delete existing NHL 2025
cursor.execute("DELETE FROM games WHERE sport='NHL' AND season=2025")
print(f"\nDeleted {cursor.rowcount} existing games")

# Insert all games
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

conn.commit()

# Verify
cursor.execute("SELECT COUNT(*) FROM games WHERE sport='NHL' AND season=2025")
total = cursor.fetchone()[0]
print(f"\n✓ Inserted {total} games")

# Show samples
print("\nFirst 5 games:")
cursor.execute("""
    SELECT game_id, game_date, away_team_id, home_team_id 
    FROM games WHERE sport='NHL' AND season=2025 
    ORDER BY CAST(SUBSTR(game_id, 10) AS INTEGER) LIMIT 5
""")
for row in cursor.fetchall():
    print(f"  {row[0]:15s} | {row[1]} | {row[2]:20s} @ {row[3]:20s}")

print(f"\nGames 146-150:")
cursor.execute("""
    SELECT game_id, game_date, away_team_id, home_team_id 
    FROM games WHERE sport='NHL' AND season=2025 
    AND CAST(SUBSTR(game_id, 10) AS INTEGER) BETWEEN 146 AND 150
    ORDER BY CAST(SUBSTR(game_id, 10) AS INTEGER)
""")
for row in cursor.fetchall():
    print(f"  {row[0]:15s} | {row[1]} | {row[2]:20s} @ {row[3]:20s}")

conn.close()
