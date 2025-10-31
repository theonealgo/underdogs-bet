import sqlite3
import re

# Parse the user's file
predictions = []
with open('attached_assets/Pasted-10-07-2025-Chicago-Blackhawks-2-Florida-Panthers-3-48-90-55-60-52-30-52-60-Incorrect-Correct-Cor-1761272915258_1761272915259.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        
        date = parts[0].strip()
        # Convert from MM/DD/YYYY or YYYY-MM-DD to DD/MM/YYYY format used in database
        if '-' in date:
            # Format: 2025-10-11
            y, m, d = date.split('-')
            date = f"{d}/{m}/{y}"
        elif '/' in date:
            # Format: 10/07/2025 (MM/DD/YYYY)
            m, d, y = date.split('/')
            date = f"{d.zfill(2)}/{m.zfill(2)}/{y}"
        
        away_team = parts[1].strip()
        away_score = int(parts[2].strip())
        home_team = parts[3].strip()
        home_score = int(parts[4].strip())
        
        # Probabilities are for HOME team
        elo_pct = float(parts[5].strip().replace('%', '')) / 100
        xgb_pct = float(parts[6].strip().replace('%', '')) / 100
        log_pct = float(parts[7].strip().replace('%', '')) / 100
        ensemble_pct = float(parts[8].strip().replace('%', '')) / 100
        
        predictions.append({
            'date': date,
            'away_team': away_team,
            'home_team': home_team,
            'away_score': away_score,
            'home_score': home_score,
            'elo_home_prob': elo_pct,
            'xgboost_home_prob': xgb_pct,
            'logistic_home_prob': log_pct,
            'ensemble_home_prob': ensemble_pct
        })

print(f"Parsed {len(predictions)} predictions from file")
print(f"First date converted: {predictions[0]['date']}")

# Check what dates exist in database
conn = sqlite3.connect('sports_predictions_original.db')
db_sample = conn.execute('''
    SELECT game_date FROM predictions WHERE sport = 'NHL' 
    ORDER BY game_date LIMIT 5
''').fetchall()
print(f"Sample DB dates: {[r[0] for r in db_sample]}")

# Update database
cursor = conn.cursor()
updated = 0
not_found = []

for pred in predictions:
    cursor.execute('''
        UPDATE predictions
        SET elo_home_prob = ?,
            xgboost_home_prob = ?,
            logistic_home_prob = ?,
            win_probability = ?
        WHERE sport = 'NHL'
          AND game_date = ?
          AND home_team_id = ?
          AND away_team_id = ?
    ''', (
        pred['elo_home_prob'],
        pred['xgboost_home_prob'],
        pred['logistic_home_prob'],
        pred['ensemble_home_prob'],
        pred['date'],
        pred['home_team'],
        pred['away_team']
    ))
    
    if cursor.rowcount > 0:
        updated += 1
    else:
        not_found.append(f"{pred['date']}: {pred['away_team']} @ {pred['home_team']}")

conn.commit()
conn.close()

print(f"\nUpdated {updated}/{len(predictions)} predictions in database")
if not_found and len(not_found) <= 10:
    print(f"\nFirst {len(not_found)} not found:")
    for nf in not_found[:10]:
        print(f"  {nf}")

print("\nVerifying first 3 updated predictions:")
conn = sqlite3.connect('sports_predictions_original.db')
for row in conn.execute('''
    SELECT game_date, away_team_id, home_team_id,
           ROUND(elo_home_prob * 100, 1) as elo,
           ROUND(xgboost_home_prob * 100, 1) as xgb,
           ROUND(win_probability * 100, 1) as ens
    FROM predictions
    WHERE sport = 'NHL' AND elo_home_prob < 0.80
    ORDER BY game_date
    LIMIT 3
'''):
    print(f"  {row[0]}: {row[1]} @ {row[2]} - Elo:{row[3]}% XGB:{row[4]}% Ens:{row[5]}%")
conn.close()
