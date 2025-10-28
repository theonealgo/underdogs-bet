#!/usr/bin/env python3
"""
Generate NBA Predictions for 2025-26 Season
Uses trained ensemble models (Elo, XGBoost, CatBoost, Logistic)
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions_original.db'

def load_models():
    """Load trained NBA models"""
    logger.info("Loading trained models...")
    
    # Load ensemble model (contains Elo, XGBoost, Logistic)
    with open('models/nba_ensemble.pkl', 'rb') as f:
        ensemble = pickle.load(f)
    logger.info("✓ Loaded ensemble model (Elo, XGBoost, Logistic)")
    
    # Load CatBoost model
    with open('models/nba_catboost.pkl', 'rb') as f:
        catboost = pickle.load(f)
    logger.info("✓ Loaded CatBoost model")
    
    # Load feature names
    with open('models/nba_feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    logger.info(f"✓ Loaded {len(feature_names)} feature names")
    
    return ensemble, catboost, feature_names

def get_upcoming_games():
    """Get 2025-26 season games (not 2024 training data)"""
    conn = sqlite3.connect(DATABASE)
    
    query = """
        SELECT 
            game_id,
            game_date,
            home_team_id as home_team,
            away_team_id as away_team
        FROM games
        WHERE sport = 'NBA' 
        AND season = 2025
        AND home_score IS NULL
        ORDER BY game_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} upcoming games for 2025-26 season")
    return df

def generate_predictions():
    """Generate predictions for all upcoming NBA games"""
    logger.info("=" * 70)
    logger.info("GENERATING NBA PREDICTIONS FOR 2025-26 SEASON")
    logger.info("=" * 70)
    
    # Load models
    ensemble, catboost, feature_names = load_models()
    
    # Get upcoming games
    upcoming_games = get_upcoming_games()
    
    if len(upcoming_games) == 0:
        logger.warning("No upcoming games found")
        return
    
    # Generate features
    logger.info("Creating features...")
    features_df = ensemble.create_features(upcoming_games, is_training=False)
    
    # Scale features
    X = features_df[feature_names] if all(col in features_df.columns for col in feature_names) else features_df
    X_scaled = ensemble.scaler.transform(X)
    
    # Get predictions from each model
    logger.info("Generating predictions from individual models...")
    
    # Elo predictions
    elo_probs = []
    for idx, row in upcoming_games.iterrows():
        elo_prob = ensemble.elo_system.predict_game(row['home_team'], row['away_team'])
        elo_probs.append(elo_prob)
    elo_probs = np.array(elo_probs)
    
    # XGBoost predictions
    xgb_probs = ensemble.xgb_model.predict_proba(X_scaled)[:, 1]
    
    # Logistic predictions
    logistic_probs = ensemble.logistic_model.predict_proba(X_scaled)[:, 1]
    
    # CatBoost predictions
    catboost_probs = catboost.predict_proba(X)[:, 1]
    
    # Meta ensemble (weighted average of all 4 models)
    # Using equal weights for now, can be tuned based on validation performance
    meta_probs = (
        0.25 * elo_probs +
        0.25 * xgb_probs +
        0.25 * catboost_probs +
        0.25 * logistic_probs
    )
    
    # Save predictions to database
    logger.info("Saving predictions to database...")
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear old NBA predictions for 2025 season
    cursor.execute("DELETE FROM predictions WHERE sport='NBA'")
    
    predictions_saved = 0
    for idx, row in upcoming_games.iterrows():
        game_id = row['game_id']
        home_team = row['home_team']
        away_team = row['away_team']
        game_date = row['game_date']
        
        # Get probabilities
        elo_prob = elo_probs[idx]
        xgb_prob = xgb_probs[idx]
        cat_prob = catboost_probs[idx]
        log_prob = logistic_probs[idx]
        meta_prob = meta_probs[idx]
        
        # Determine winner
        predicted_winner = home_team if meta_prob > 0.5 else away_team
        
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
            'NBA', 'NBA', game_id, game_date, home_team, away_team,
            predicted_winner, meta_prob,
            elo_prob, xgb_prob, cat_prob, log_prob, meta_prob,
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
    for i in range(min(5, len(upcoming_games))):
        game = upcoming_games.iloc[i]
        logger.info(f"\n{game['home_team']} vs {game['away_team']}")
        logger.info(f"  Elo:      {elo_probs[i]:.1%}")
        logger.info(f"  XGBoost:  {xgb_probs[i]:.1%}")
        logger.info(f"  CatBoost: {catboost_probs[i]:.1%}")
        logger.info(f"  Logistic: {logistic_probs[i]:.1%}")
        logger.info(f"  Meta:     {meta_probs[i]:.1%} → {game['home_team'] if meta_probs[i] > 0.5 else game['away_team']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ NBA PREDICTION GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Generated predictions for {len(upcoming_games)} games")
    logger.info("Predictions are now visible on the NBA predictions page")
    logger.info("=" * 70)

def main():
    """Main function"""
    try:
        generate_predictions()
    except Exception as e:
        logger.error(f"Error generating predictions: {e}", exc_info=True)

if __name__ == "__main__":
    main()
