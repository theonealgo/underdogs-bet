#!/usr/bin/env python3
"""
Schedule Importer for PurePicks.COM
Imports schedules from sports_schedules.py into database
"""

import sys
import pandas as pd
from datetime import datetime
from sports_schedules import get_schedule, get_available_sports
from src.data_storage.database import DatabaseManager
from universal_sports_predictor import UniversalSportsPredictor

def parse_date(date_str):
    """Parse date from DD/MM/YYYY HH:MM format to YYYY-MM-DD"""
    try:
        dt = datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        return dt.strftime('%Y-%m-%d')
    except:
        return date_str

def import_schedule(sport, schedule_func):
    """Import schedule for a sport"""
    print(f"\n{'='*50}")
    print(f"Importing {sport} Schedule")
    print(f"{'='*50}")
    
    # Get schedule
    schedule = schedule_func()
    print(f"Found {len(schedule)} games")
    
    # Convert to DataFrame
    games_data = []
    for game in schedule:
        game_date = parse_date(game['date'])
        game_id = f"{sport}_{game['match_id']}"
        
        games_data.append({
            'sport': sport,
            'league': sport,
            'game_id': game_id,
            'game_date': game_date,
            'home_team_id': game['home_team'],
            'away_team_id': game['away_team'],
            'venue': game.get('venue', ''),
            'status': 'final' if game.get('result') else 'Scheduled'
        })
    
    games_df = pd.DataFrame(games_data)
    
    # Store in database
    db = DatabaseManager()
    db.store_games(games_df)
    print(f"✅ Stored {len(games_df)} {sport} games")
    
    # Generate predictions for future games
    future_games = games_df[games_df['status'] == 'Scheduled']
    print(f"\nGenerating predictions for {len(future_games)} upcoming games...")
    
    predictor = UniversalSportsPredictor(sport)
    
    # Load training data from completed games
    completed_games = games_df[games_df['status'] == 'final'].copy()
    if not completed_games.empty:
        # Extract results from schedule
        completed_data = []
        for game in schedule:
            if game.get('result'):
                try:
                    scores = game['result'].split('-')
                    if len(scores) == 2:
                        home_score = int(scores[1].strip())
                        away_score = int(scores[0].strip())
                        completed_data.append({
                            'Home': game['home_team'],
                            'Away': game['away_team'],
                            'Home_Score': home_score,
                            'Away_Score': away_score
                        })
                except:
                    pass
        
        if completed_data:
            training_df = pd.DataFrame(completed_data)
            print(f"Training models on {len(training_df)} completed games...")
            predictor.train(training_df)
            
            # Generate predictions
            predictions_data = []
            for _, game in future_games.iterrows():
                pred = predictor.predict(game['home_team_id'], game['away_team_id'])
                
                predictions_data.append({
                    'sport': sport,
                    'league': sport,
                    'game_id': game['game_id'],
                    'game_date': game['game_date'],
                    'home_team_id': game['home_team_id'],
                    'away_team_id': game['away_team_id'],
                    'predicted_winner': game['home_team_id'] if pred['home_win_prob'] > 0.5 else game['away_team_id'],
                    'win_probability': float(max(pred['home_win_prob'], 1 - pred['home_win_prob'])),
                    'elo_home_prob': float(pred['elo_home_prob']),
                    'logistic_home_prob': float(pred['logistic_home_prob']),
                    'xgboost_home_prob': float(pred['xgboost_home_prob']),
                    'predicted_total': None,
                    'model_version': '1.0',
                    'key_factors': '[]'
                })
            
            if predictions_data:
                pred_df = pd.DataFrame(predictions_data)
                db.store_predictions(pred_df)
                print(f"✅ Generated {len(pred_df)} predictions")
    
    return len(games_df)

if __name__ == "__main__":
    db = DatabaseManager()
    
    # Import all available schedules
    available_sports = get_available_sports()
    print(f"\nAvailable sports: {', '.join(available_sports)}")
    
    total_games = 0
    for sport in available_sports:
        try:
            count = import_schedule(sport, lambda s=sport: get_schedule(s))
            total_games += count
        except Exception as e:
            print(f"❌ Error importing {sport}: {e}")
    
    print(f"\n{'='*50}")
    print(f"✅ Import Complete: {total_games} total games")
    print(f"{'='*50}")
