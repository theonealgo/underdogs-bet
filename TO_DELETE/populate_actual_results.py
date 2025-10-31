#!/usr/bin/env python3
import sqlite3

# Read user results
with open('attached_assets/Pasted-Week-1-Date-Visitor-Home-Box-Score-09-04-2025-Dallas-Cowboys-20-Philadelphia-Eagles-24-Box-Score-1760725176797_1760725176798.txt', 'r') as f:
    result_lines = f.readlines()

# Parse results
results = []
for line in result_lines:
    if '\t' in line and ('09/' in line or '10/' in line):
        parts = line.split('\t')
        if len(parts) >= 5:
            date = parts[0].strip()
            away = parts[1].strip()
            try:
                away_score = int(parts[2].strip())
                home = parts[3].strip()
                home_score = int(parts[4].strip())
                winner = home if home_score > away_score else away
                # Convert MM/DD/YYYY to DD/MM/YYYY for database matching
                m, d, y = date.split('/')
                date_db = f"{d}/{m}/{y}"
                results.append({
                    'date': date_db,
                    'away': away,
                    'home': home,
                    'home_score': home_score,
                    'away_score': away_score,
                    'winner': winner
                })
            except:
                pass

# Update database
conn = sqlite3.connect('sports_predictions.db')
cursor = conn.cursor()

updated = 0
for res in results:
    cursor.execute("""
        UPDATE predictions
        SET actual_winner = ?,
            actual_home_score = ?,
            actual_away_score = ?,
            win_prediction_correct = CASE WHEN predicted_winner = ? THEN 1 ELSE 0 END,
            result_updated_at = CURRENT_TIMESTAMP
        WHERE sport = 'NFL'
        AND game_date = ?
        AND home_team_id = ?
        AND away_team_id = ?
    """, (res['winner'], res['home_score'], res['away_score'], res['winner'], 
          res['date'], res['home'], res['away']))
    
    if cursor.rowcount > 0:
        updated += 1

conn.commit()
conn.close()

print(f"✅ Updated {updated} predictions with actual results")
print(f"Total results processed: {len(results)}")
