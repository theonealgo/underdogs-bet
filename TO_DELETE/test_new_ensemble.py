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
                    'winner': winner
                })
            except:
                pass

# Get predictions from database
conn = sqlite3.connect('sports_predictions.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT game_date, away_team_id, home_team_id, predicted_winner
    FROM predictions 
    WHERE sport = 'NFL'
""")

predictions = []
for row in cursor.fetchall():
    predictions.append({
        'date': row[0],
        'away': row[1],
        'home': row[2],
        'predicted_winner': row[3]
    })

conn.close()

# Match and calculate
matches = []
for pred in predictions:
    for res in results:
        if pred['date'] == res['date'] and pred['away'] == res['away'] and pred['home'] == res['home']:
            matches.append({**pred, **res})
            break

print(f"📊 NEW ENSEMBLE ACCURACY (70% XGB / 30% Elo)")
print("="*80)
print(f"Matched games: {len(matches)}")

if len(matches) > 0:
    correct = sum(1 for m in matches if m['predicted_winner'] == m['winner'])
    total = len(matches)
    
    print(f"\n✅ Correct: {correct}/{total} = {correct/total*100:.1f}%")
    
    # Show a few examples
    print(f"\nSample predictions:")
    for m in matches[:5]:
        result = "✅" if m['predicted_winner'] == m['winner'] else "❌"
        print(f"  {result} {m['date']}: {m['away']} @ {m['home']} - Predicted: {m['predicted_winner']}, Actual: {m['winner']}")
