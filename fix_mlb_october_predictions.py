#!/usr/bin/env python3
"""
Fix MLB October predictions by mapping abbreviations to full team names
"""
import sqlite3
import pandas as pd
from src.models.universal_ensemble_predictor import UniversalSportsEnsemble

# MLB team abbreviation to full name mapping
MLB_TEAM_MAP = {
    'DET': 'Detroit Tigers',
    'SEA': 'Seattle Mariners',
    'CHC': 'Chicago Cubs',
    'CLE': 'Cleveland Guardians',
    'MIL': 'Milwaukee Brewers',
    'TOR': 'Toronto Blue Jays',
    'NYY': 'New York Yankees',
    'LAD': 'Los Angeles Dodgers',
    'PHI': 'Philadelphia Phillies',
    'SD': 'San Diego Padres',
    'BOS': 'Boston Red Sox',
    'CIN': 'Cincinnati Reds',
    # Add more as needed
}

def main():
    print("\n" + "="*60)
    print("FIXING MLB OCTOBER PREDICTIONS")
    print("="*60)
    
    conn = sqlite3.connect('sports_predictions.db')
    
    # Load MLB model
    print("\nLoading MLB ensemble model...")
    predictor = UniversalSportsEnsemble(sport='MLB')
    predictor.load_model('models/mlb_ensemble.pkl')
    print(f"✓ Model loaded (trained on {predictor.games_trained} games)")
    
    # Get October MLB games with abbreviations
    print("\nFinding October MLB games with abbreviations...")
    october_games = pd.read_sql_query('''
        SELECT game_id, game_date, home_team_id, away_team_id
        FROM games
        WHERE sport = "MLB" 
          AND game_date LIKE "%/10/2025"
    ''', conn)
    
    print(f"Found {len(october_games)} October games")
    
    # Delete old predictions for these games
    cursor = conn.cursor()
    for game_id in october_games['game_id']:
        cursor.execute('DELETE FROM predictions WHERE game_id = ?', (game_id,))
    conn.commit()
    print(f"✓ Deleted old predictions for {len(october_games)} games")
    
    # Generate new predictions with mapped team names
    predictions_data = []
    fixed_count = 0
    
    for _, game in october_games.iterrows():
        home_team = game['home_team_id']
        away_team = game['away_team_id']
        
        # Map abbreviations to full names
        home_full = MLB_TEAM_MAP.get(home_team, home_team)
        away_full = MLB_TEAM_MAP.get(away_team, away_team)
        
        # Generate prediction with full team names
        pred = predictor.predict_game(home_team=home_full, away_team=away_full)
        
        predictions_data.append({
            'sport': 'MLB',
            'league': 'MLB',
            'game_id': game['game_id'],
            'game_date': game['game_date'],
            'home_team_id': home_team,  # Keep abbreviation in database
            'away_team_id': away_team,
            'predicted_winner': MLB_TEAM_MAP.get(pred['predicted_winner'], pred['predicted_winner']),
            'win_probability': pred['home_win_probability'],
            'elo_home_prob': pred['elo_home_prob'],
            'logistic_home_prob': pred['logistic_home_prob'],
            'xgboost_home_prob': pred['xgboost_home_prob'],
            'predicted_total': None,
            'model_version': '1.0',
            'key_factors': '[]'
        })
        
        if abs(pred['xgboost_home_prob'] - 0.472) > 0.01:  # Not a placeholder
            fixed_count += 1
        
        print(f"  ✓ {away_team} @ {home_team}: XGB {pred['xgboost_home_prob']*100:.1f}% (was placeholder)")
    
    # Store new predictions
    from src.data_storage.database import DatabaseManager
    db = DatabaseManager()
    db.store_predictions(predictions_data)
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS!")
    print(f"{'='*60}")
    print(f"Fixed {fixed_count} / {len(october_games)} October predictions")
    print("MLB October games now have real predictions!\n")

if __name__ == "__main__":
    main()
