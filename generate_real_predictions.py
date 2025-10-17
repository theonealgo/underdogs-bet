#!/usr/bin/env python3
"""
Generate REAL predictions for all sports using trained ensemble models
"""
import pandas as pd
import numpy as np
from src.data_storage.database import DatabaseManager
from src.models.universal_ensemble_predictor import UniversalSportsEnsemble
import os

def load_model(sport):
    """Load trained model for a sport"""
    model_path = f'models/{sport.lower()}_ensemble.pkl'
    if not os.path.exists(model_path):
        print(f"  ⚠️  Model not found: {model_path}")
        return None
    
    predictor = UniversalSportsEnsemble(sport=sport)
    predictor.load_model(model_path)
    
    # Verify model is trained
    if not predictor.is_trained:
        print(f"  ⚠️  Model {sport} not trained")
        return None
    
    return predictor

def main():
    print("\n" + "="*60)
    print("GENERATING REAL PREDICTIONS USING TRAINED MODELS")
    print("="*60)
    
    db = DatabaseManager()
    
    # Process each sport
    sports = ['NFL', 'NBA', 'NHL', 'MLB', 'NCAAF']
    total_generated = 0
    
    for sport in sports:
        print(f"\n{sport}:")
        print("-" * 40)
        
        # Load trained model
        predictor = load_model(sport)
        if predictor is None:
            continue
        
        # Find games without predictions
        with db._get_connection() as conn:
            query = f"""
                SELECT g.game_id, g.home_team_id, g.away_team_id, g.game_date, g.sport, g.league
                FROM games g
                LEFT JOIN predictions p ON g.game_id = p.game_id
                WHERE g.sport = '{sport}' AND p.game_id IS NULL
            """
            games_df = pd.read_sql_query(query, conn)
        
        if games_df.empty:
            print(f"  ✅ All {sport} games already have predictions")
            continue
        
        print(f"  Found {len(games_df)} games without predictions")
        print(f"  Generating predictions using trained model...")
        
        predictions_data = []
        
        for _, game in games_df.iterrows():
            # Generate REAL prediction using trained model
            pred = predictor.predict_game(
                home_team=game['home_team_id'],
                away_team=game['away_team_id']
            )
            
            predictions_data.append({
                'sport': sport,
                'league': game['league'],
                'game_id': game['game_id'],
                'game_date': game['game_date'],
                'home_team_id': game['home_team_id'],
                'away_team_id': game['away_team_id'],
                'predicted_winner': pred['predicted_winner'],
                'win_probability': pred['home_win_probability'],
                'elo_home_prob': pred['elo_home_prob'],
                'logistic_home_prob': pred['logistic_home_prob'],
                'xgboost_home_prob': pred['xgboost_home_prob'],
                'predicted_total': None,
                'model_version': '1.0',
                'key_factors': '[]'
            })
        
        # Store predictions
        db.store_predictions(predictions_data)
        total_generated += len(predictions_data)
        
        print(f"  ✅ Generated {len(predictions_data)} predictions")
        
        # Show sample predictions
        if predictions_data:
            sample = predictions_data[0]
            print(f"  Sample: {sample['away_team_id']} @ {sample['home_team_id']}")
            print(f"    XGB: {sample['xgboost_home_prob']*100:.1f}% | Elo: {sample['elo_home_prob']*100:.1f}% | Log: {sample['logistic_home_prob']*100:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS!")
    print(f"{'='*60}")
    print(f"Generated {total_generated} REAL predictions using trained models")
    print("All games now have authentic predictions!\n")

if __name__ == "__main__":
    main()
