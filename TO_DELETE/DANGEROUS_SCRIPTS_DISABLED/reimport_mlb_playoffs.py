#!/usr/bin/env python3
"""
Reimport MLB playoff schedule with full team names and regenerate predictions
"""
import sqlite3
import pandas as pd
from sports_schedules import get_mlb_schedule
from src.data_storage.database import DatabaseManager
from src.models.universal_ensemble_predictor import UniversalSportsEnsemble
from datetime import datetime

def main():
    print("\n" + "="*60)
    print("REIMPORTING MLB PLAYOFF SCHEDULE")
    print("="*60)
    
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    # Delete all old MLB games and predictions
    print("\n1. Deleting old MLB data...")
    cursor.execute('DELETE FROM predictions WHERE sport = "MLB"')
    cursor.execute('DELETE FROM games WHERE sport = "MLB"')
    conn.commit()
    print("   ✓ Deleted old MLB games and predictions")
    
    # Import new schedule
    print("\n2. Importing new MLB playoff schedule...")
    mlb_schedule = get_mlb_schedule()
    
    games_data = []
    for game in mlb_schedule:
        # Convert date format to DD/MM/YYYY
        date_obj = datetime.strptime(game['date'], '%Y-%m-%d')
        game_date = date_obj.strftime('%d/%m/%Y')
        
        games_data.append({
            'sport': 'MLB',
            'league': 'MLB',
            'game_id': f"MLB_{game['match_id']}",
            'season': 2025,
            'game_date': game_date,
            'home_team_id': game['home_team'],
            'away_team_id': game['away_team']
        })
    
    # Insert games
    db = DatabaseManager()
    games_df = pd.DataFrame(games_data)
    
    with db._get_connection() as conn:
        games_df.to_sql('games', conn, if_exists='append', index=False)
    
    print(f"   ✓ Imported {len(games_data)} MLB playoff games")
    
    # Load trained MLB model
    print("\n3. Loading MLB ensemble model...")
    predictor = UniversalSportsEnsemble(sport='MLB')
    predictor.load_model('models/mlb_ensemble.pkl')
    print(f"   ✓ Model loaded (trained on {predictor.games_trained} games)")
    
    # Generate predictions
    print("\n4. Generating predictions...")
    predictions_data = []
    
    for game_data in games_data:
        pred = predictor.predict_game(
            home_team=game_data['home_team_id'],
            away_team=game_data['away_team_id']
        )
        
        predictions_data.append({
            'sport': 'MLB',
            'league': 'MLB',
            'game_id': game_data['game_id'],
            'game_date': game_data['game_date'],
            'home_team_id': game_data['home_team_id'],
            'away_team_id': game_data['away_team_id'],
            'predicted_winner': pred['predicted_winner'],
            'win_probability': pred['home_win_probability'],
            'elo_home_prob': pred['elo_home_prob'],
            'logistic_home_prob': pred['logistic_home_prob'],
            'xgboost_home_prob': pred['xgboost_home_prob'],
            'predicted_total': None,
            'model_version': '1.0',
            'key_factors': '[]'
        })
    
    db.store_predictions(predictions_data)
    print(f"   ✓ Generated {len(predictions_data)} predictions")
    
    # Show sample
    print("\n5. Sample predictions:")
    sample = predictions_data[:5]
    for p in sample:
        away = p['away_team_id'][:20]
        home = p['home_team_id'][:20]
        xgb = p['xgboost_home_prob'] * 100
        elo = p['elo_home_prob'] * 100
        print(f"   {away:20} @ {home:20} - XGB: {xgb:5.1f}% Elo: {elo:5.1f}%")
    
    conn.close()
    
    print(f"\n{'='*60}")
    print("✅ MLB PLAYOFF SCHEDULE UPDATED!")
    print(f"{'='*60}")
    print(f"Imported {len(games_data)} games with full team names")
    print("Models will now recognize teams and make real predictions!\n")

if __name__ == "__main__":
    main()
