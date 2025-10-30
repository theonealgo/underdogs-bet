import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('sports_predictions_original.db')

# Get first 148 games with predictions and results
query = """
    SELECT 
        g.game_id,
        g.game_date,
        g.home_team_id,
        g.away_team_id,
        g.home_score,
        g.away_score,
        p.xgboost_home_prob,
        p.catboost_home_prob,
        p.elo_home_prob,
        p.meta_home_prob,
        p.predicted_winner,
        CAST(SUBSTR(g.game_id, 10) AS INTEGER) as match_id
    FROM games g
    LEFT JOIN predictions p ON g.game_id = p.game_id
    WHERE g.sport='NHL' AND g.season=2025
    AND CAST(SUBSTR(g.game_id, 10) AS INTEGER) <= 148
    AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    ORDER BY CAST(SUBSTR(g.game_id, 10) AS INTEGER)
"""

games = pd.read_sql_query(query, conn)
conn.close()

print(f"Total games with predictions and results: {len(games)}")
print(f"Match IDs range: {games['match_id'].min()} to {games['match_id'].max()}")

# Calculate accuracy for different segments
def calculate_accuracy(df, model_col, threshold=0.5):
    correct = 0
    total = 0
    for _, row in df.iterrows():
        if pd.notna(row[model_col]):
            actual_winner = row['home_team_id'] if row['home_score'] > row['away_score'] else row['away_team_id']
            predicted_winner = row['home_team_id'] if row[model_col] > threshold else row['away_team_id']
            if predicted_winner == actual_winner:
                correct += 1
            total += 1
    return (correct / total * 100) if total > 0 else 0

# Split into segments
first_93 = games[games['match_id'] <= 93]
last_55 = games[games['match_id'] > 93]

print("\n" + "="*70)
print("ACCURACY BY SEGMENT")
print("="*70)

print(f"\nFIRST 93 GAMES (Match IDs 1-93):")
print(f"  XGBoost:  {calculate_accuracy(first_93, 'xgboost_home_prob'):.1f}%")
print(f"  CatBoost: {calculate_accuracy(first_93, 'catboost_home_prob'):.1f}%")
print(f"  Elo:      {calculate_accuracy(first_93, 'elo_home_prob'):.1f}%")
print(f"  Meta:     {calculate_accuracy(first_93, 'meta_home_prob'):.1f}%")

print(f"\nLAST 55 GAMES (Match IDs 94-148):")
print(f"  XGBoost:  {calculate_accuracy(last_55, 'xgboost_home_prob'):.1f}%")
print(f"  CatBoost: {calculate_accuracy(last_55, 'catboost_home_prob'):.1f}%")
print(f"  Elo:      {calculate_accuracy(last_55, 'elo_home_prob'):.1f}%")
print(f"  Meta:     {calculate_accuracy(last_55, 'meta_home_prob'):.1f}%")

print(f"\nOVERALL (All {len(games)} games):")
print(f"  XGBoost:  {calculate_accuracy(games, 'xgboost_home_prob'):.1f}%")
print(f"  CatBoost: {calculate_accuracy(games, 'catboost_home_prob'):.1f}%")
print(f"  Elo:      {calculate_accuracy(games, 'elo_home_prob'):.1f}%")
print(f"  Meta:     {calculate_accuracy(games, 'meta_home_prob'):.1f}%")

# Also check first 110 / last 38 split (user mentioned 110)
first_110 = games[games['match_id'] <= 110]
last_38 = games[games['match_id'] > 110]

print("\n" + "="*70)
print("ALTERNATIVE SPLIT (First 110 / Last 38)")
print("="*70)

print(f"\nFIRST 110 GAMES (Match IDs 1-110):")
print(f"  XGBoost:  {calculate_accuracy(first_110, 'xgboost_home_prob'):.1f}%")
print(f"  CatBoost: {calculate_accuracy(first_110, 'catboost_home_prob'):.1f}%")
print(f"  Elo:      {calculate_accuracy(first_110, 'elo_home_prob'):.1f}%")
print(f"  Meta:     {calculate_accuracy(first_110, 'meta_home_prob'):.1f}%")

print(f"\nLAST 38 GAMES (Match IDs 111-148):")
print(f"  XGBoost:  {calculate_accuracy(last_38, 'xgboost_home_prob'):.1f}%")
print(f"  CatBoost: {calculate_accuracy(last_38, 'catboost_home_prob'):.1f}%")
print(f"  Elo:      {calculate_accuracy(last_38, 'elo_home_prob'):.1f}%")
print(f"  Meta:     {calculate_accuracy(last_38, 'meta_home_prob'):.1f}%")

print("\n" + "="*70)
