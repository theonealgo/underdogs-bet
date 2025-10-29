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
    """Get all 2025-26 season NHL games for predictions"""
    conn = sqlite3.connect(DATABASE)
    
    query = """
        SELECT 
            game_id,
            game_date,
            home_team_id,
            away_team_id,
            home_score,
            away_score,
            status
        FROM games
        WHERE sport = 'NHL' 
        AND season = 2026
        ORDER BY game_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} games for 2025-26 season")
    return df

def get_historical_games():
    """Get historical games for feature engineering"""
    conn = sqlite3.connect(DATABASE)
    
    query = """
        SELECT 
            game_id,
            game_date,
            home_team_id,
            away_team_id,
            home_score,
            away_score,
            status
        FROM games
        WHERE sport = 'NHL' 
        AND home_score IS NOT NULL
        AND status = 'final'
        ORDER BY game_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} historical games for feature engineering")
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
        game_id = pred['game_id']
        home_team = pred['home_team']
        away_team = pred['away_team']
        game_date = pred['game_date']
        
        # Get probabilities
        xgb_prob = float(pred['xgb_home_prob'])
        cat_prob = float(pred['catboost_home_prob'])
        log_prob = float(pred['logistic_home_prob'])
        meta_prob = float(pred['meta_home_prob'])
        predicted_winner = pred['predicted_winner']
        
        # Insert prediction
        cursor.execute("""
            INSERT INTO predictions (
                sport, league, game_id, game_date, home_team_id, away_team_id,
                predicted_winner, win_probability, 
                elo_home_prob, xgboost_home_prob, 
                catboost_home_prob, logistic_home_prob, meta_home_prob,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'NHL', 'NHL', game_id, game_date, home_team, away_team,
            predicted_winner, meta_prob,
            0.50, xgb_prob, cat_prob, log_prob, meta_prob,  # elo_home_prob = 0.50 (not used for NHL)
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
        logger.info(f"  Logistic: {pred['logistic_home_prob']:.1%}")
        logger.info(f"  Meta:     {pred['meta_home_prob']:.1%} → {pred['predicted_winner']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ NHL PREDICTION GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Generated predictions for {len(predictions)} games")
    logger.info("Predictions are now visible on the NHL predictions page")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
