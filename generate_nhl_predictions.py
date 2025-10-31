#!/usr/bin/env python3
"""
Generate NHL Predictions for 2025-26 Season
Uses trained ensemble models (XGBoost, CatBoost, Logistic)
"""

import sqlite3
import pandas as pd
import logging
from datetime import datetime
from src.models.nhl_predictor import NHLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions_original.db'

def get_nhl_games():
    """Get all 2025-26 season NHL games for predictions from nhlschedules.py"""
    from nhlschedules import get_nhl_2025_schedule
    
    schedule = get_nhl_2025_schedule()
    
    # Convert to DataFrame
    games = []
    for game in schedule:
        games.append({
            'game_id': f"nhl_{game['date']}_{game['home_team']}_{game['away_team']}",
            'game_date': game['date'],
            'home_team_id': game['home_team'],
            'away_team_id': game['away_team'],
            'home_score': game.get('home_score'),
            'away_score': game.get('away_score'),
            'status': 'final' if game.get('home_score') is not None else 'scheduled'
        })
    
    df = pd.DataFrame(games)
    logger.info(f"Loaded {len(df)} games for 2025-26 season from nhlschedules.py")
    return df

def get_historical_games():
    """Get historical games for feature engineering (2024 season + 2025 completed games)"""
    import pickle
    import os
    
    games = []
    
    # Load ONLY 2024 season historical data (matching training data!)
    if os.path.exists('nhl_historical_data.pkl'):
        with open('nhl_historical_data.pkl', 'rb') as f:
            all_games = pickle.load(f)
        
        # Filter to 2024 season ONLY (matching training)
        from datetime import datetime as dt
        for game in all_games:
            try:
                date_obj = dt.strptime(game['date'], '%Y-%m-%d')
                # Only 2024 season (matching training data exactly)
                if date_obj.year != 2024:
                    continue
            except:
                continue
                
            games.append({
                'game_id': f"nhl_{game['match_id']}",
                'game_date': game['date'],  # Keep original YYYY-MM-DD format
                'home_team_id': game['home_team'],
                'away_team_id': game['away_team'],
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'status': 'final'
            })
        logger.info(f"Loaded {len(games)} games from 2024 season historical data (matching training)")
    else:
        logger.warning("Historical data file not found")
    
    # DO NOT add 2025 games - use only 2024 for features (matching backtest)
    # This prevents future data leakage where later games influence earlier predictions
    
    df = pd.DataFrame(games)
    logger.info(f"Total {len(df)} games available for feature engineering (2024 ONLY, matching backtest)")
    return df

def main():
    """Main prediction generation function"""
    logger.info("=" * 70)
    logger.info("GENERATING NHL PREDICTIONS FOR 2025-26 SEASON")
    logger.info("=" * 70)
    
    # Load predictor with trained models
    predictor = NHLPredictor(model_dir='models')
    
    if not predictor.is_trained:
        logger.error("Models not trained! Run 'python train_nhl_models.py' first")
        return
    
    # Get games to predict
    upcoming_games = get_nhl_games()
    
    if len(upcoming_games) == 0:
        logger.warning("No games found for 2025-26 season")
        return
    
    # Get historical data for feature engineering
    historical_games = get_historical_games()
    
    # Generate predictions for all games
    logger.info("Generating predictions from ensemble models...")
    predictions = predictor.predict_multiple_games(upcoming_games, historical_games)
    
    # Save predictions to database
    logger.info("Saving predictions to database...")
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear old NHL predictions for 2025-26 season
    cursor.execute("DELETE FROM predictions WHERE sport='NHL'")
    
    predictions_saved = 0
    for pred in predictions:
        home_team = pred['home_team']
        away_team = pred['away_team']
        game_date = pred['game_date']
        
        # Convert DD/MM/YYYY to YYYY-MM-DD for matching with games table
        try:
            date_obj = datetime.strptime(game_date, '%d/%m/%Y')
            db_date = date_obj.strftime('%Y-%m-%d')
        except:
            db_date = game_date
        
        # Lookup correct game_id from games table using date+teams
        cursor.execute("""
            SELECT game_id FROM games 
            WHERE sport='NHL' AND season=2025
            AND game_date=? AND home_team_id=? AND away_team_id=?
        """, (db_date, home_team, away_team))
        
        result = cursor.fetchone()
        if not result:
            logger.warning(f"No matching game found for {home_team} vs {away_team} on {game_date}")
            continue
        
        game_id = result[0]
        
        # Get probabilities
        xgb_prob = float(pred['xgb_home_prob'])
        cat_prob = float(pred['catboost_home_prob'])
        elo_prob = float(pred['elo_home_prob'])
        meta_prob = float(pred['meta_home_prob'])
        predicted_winner = pred['predicted_winner']
        
        # Insert prediction with correct game_id
        cursor.execute("""
            INSERT INTO predictions (
                sport, league, game_id, game_date, home_team_id, away_team_id,
                predicted_winner, win_probability, 
                elo_home_prob, xgboost_home_prob, 
                catboost_home_prob, meta_home_prob,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'NHL', 'NHL', game_id, db_date, home_team, away_team,
            predicted_winner, meta_prob,
            elo_prob, xgb_prob, cat_prob, meta_prob,
            datetime.now().isoformat()
        ))
        
        predictions_saved += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"✓ Saved {predictions_saved} predictions to database")
    
    # Show sample predictions
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE PREDICTIONS (First 5 games)")
    logger.info("=" * 70)
    for i in range(min(5, len(predictions))):
        pred = predictions[i]
        logger.info(f"\n{pred['home_team']} vs {pred['away_team']}")
        logger.info(f"  XGBoost:  {pred['xgb_home_prob']:.1%}")
        logger.info(f"  CatBoost: {pred['catboost_home_prob']:.1%}")
        logger.info(f"  Elo:      {pred['elo_home_prob']:.1%}")
        logger.info(f"  Meta:     {pred['meta_home_prob']:.1%} → {pred['predicted_winner']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ NHL PREDICTION GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Generated predictions for {len(predictions)} games")
    logger.info("Predictions are now visible on the NHL predictions page")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
