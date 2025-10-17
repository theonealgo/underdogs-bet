#!/usr/bin/env python3
import sqlite3

# Get predictions from database
conn = sqlite3.connect('sports_predictions.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT game_date, away_team_id, home_team_id, predicted_winner
    FROM predictions 
    WHERE sport = 'NFL'
    ORDER BY game_date
""")

print("Date\tAway\tHome\tPredicted Winner")
print("="*100)

for row in cursor.fetchall():
    date, away, home, winner = row
    print(f"{date}\t{away}\t{home}\t{winner}")

conn.close()
