#!/usr/bin/env python3

# Read predictions
with open('attached_assets/Pasted-Date-Away-Home-Pick-XGB-Home-Win-Elo-Home-Win-Confidence-04-09-2025-Dallas-Cowboys-Philade-1760725147746_1760725147747.txt', 'r') as f:
    pred_lines = f.readlines()[1:]  # Skip header

# Read results  
with open('attached_assets/Pasted-Week-1-Date-Visitor-Home-Box-Score-09-04-2025-Dallas-Cowboys-20-Philadelphia-Eagles-24-Box-Score-1760725176797_1760725176798.txt', 'r') as f:
    result_lines = [line for line in f.readlines() if 'Box Score' in line and '09/' in line or '10/' in line]

# Parse predictions
predictions = []
for line in pred_lines:
    if not line.strip():
        continue
    parts = line.strip().split('\t')
    if len(parts) >= 7:
        date = parts[0].strip()
        away = parts[1].strip()
        home = parts[3].strip()
        pick = parts[4].strip()
        xgb = float(parts[5].strip().replace('%', ''))
        elo = float(parts[6].strip().replace('%', ''))
        predictions.append({
            'date': date,
            'away': away, 
            'home': home,
            'pick': pick,
            'xgb': xgb,
            'elo': elo
        })

# Parse results
results = []
for line in result_lines:
    parts = line.split('\t')
    if len(parts) >= 5:
        date = parts[0].strip()
        away = parts[1].strip()
        away_score = int(parts[2].strip())
        home = parts[3].strip()
        home_score = int(parts[4].strip())
        winner = home if home_score > away_score else away
        results.append({
            'date': date,
            'away': away,
            'home': home,
            'winner': winner
        })

# Match and calculate
matches = []
for pred in predictions:
    for res in results:
        if pred['date'] == res['date'] and pred['away'] == res['away'] and pred['home'] == res['home']:
            matches.append({**pred, **res})
            break

print(f"📊 NFL 2025 SEASON ACCURACY")
print("="*80)
print(f"Matched games: {len(matches)}")
print()

xgb_correct = sum(1 for m in matches if (m['home'] if m['xgb'] > 50 else m['away']) == m['winner'])
elo_correct = sum(1 for m in matches if (m['home'] if m['elo'] > 50 else m['away']) == m['winner'])  
main_correct = sum(1 for m in matches if m['pick'] == m['winner'])

total = len(matches)

print(f"XGB Pick:   {xgb_correct}/{total} = {xgb_correct/total*100:.1f}%")
print(f"Elo Pick:   {elo_correct}/{total} = {elo_correct/total*100:.1f}%")
print(f"Main Pick:  {main_correct}/{total} = {main_correct/total*100:.1f}%")
