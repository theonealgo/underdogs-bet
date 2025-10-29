#!/usr/bin/env python3
"""
Train NHL Ensemble Models (XGBoost, CatBoost, Elo, Meta)
Uses 2024-25 season historical data for training
"""

import sqlite3
import pandas as pd
import logging
from src.models.nhl_predictor import NHLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions_original.db'

def load_nhl_training_data():
    """Load 3 seasons of NHL historical data (2022-25) from API"""
    import pickle
    import os
    
    # Check if historical data exists
    if os.path.exists('nhl_historical_data.pkl'):
        with open('nhl_historical_data.pkl', 'rb') as f:
            all_games = pickle.load(f)
        
        # Convert to DataFrame with date format conversion (YYYY-MM-DD -> DD/MM/YYYY)
        from datetime import datetime
        games = []
        for game in all_games:
            # Convert ISO date format to DD/MM/YYYY
            try:
                date_obj = datetime.strptime(game['date'], '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d/%m/%Y')
            except:
                formatted_date = game['date']
            
            games.append({
                'game_id': f"nhl_{game['match_id']}",
                'game_date': formatted_date,
                'home_team_id': game['home_team'],
                'away_team_id': game['away_team'],
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'status': 'final'
            })
        
        df = pd.DataFrame(games)
        
        logger.info(f"Loaded {len(df)} training games from 3 NHL seasons (2022-25) - API data")
        if len(df) > 0:
            logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        
        return df
    else:
        logger.error("Historical data file not found. Run 'python fetch_nhl_historical_data.py' first")
        return pd.DataFrame()

def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("TRAINING NHL ENSEMBLE MODELS")
    logger.info("=" * 70)
    
    # Load training data
    training_data = load_nhl_training_data()
    
    if len(training_data) < 50:
        logger.error(f"Insufficient training data: {len(training_data)} games (minimum 50 required)")
        logger.info("Please ensure 2024-25 season data is loaded into database")
        return
    
    logger.info(f"Using {len(training_data)} games for training")
    
    # Initialize NHL predictor
    predictor = NHLPredictor(model_dir='models')
    
    # Train all models (XGBoost, CatBoost, Elo)
    logger.info("\nTraining XGBoost, CatBoost, and Elo models...")
    results = predictor.train_models(training_data)
    
    if results['success']:
        logger.info("\n" + "="*70)
        logger.info("TRAINING RESULTS")
        logger.info("="*70)
        logger.info(f"  Training Games:        {results['training_games']}")
        logger.info(f"  Number of Features:    {results['num_features']}")
        logger.info(f"  Elo Teams:             {results['elo_teams']}")
        logger.info(f"  XGBoost Accuracy:      {results['xgb_accuracy']:.1%}")
        logger.info(f"  CatBoost Accuracy:     {results['catboost_accuracy']:.1%}")
        logger.info(f"  Elo Accuracy:          {results['elo_accuracy']:.1%}")
        logger.info(f"  XGBoost Total MAE:     {results['xgb_total_mae']:.2f} goals")
        logger.info(f"  CatBoost Total MAE:    {results['catboost_total_mae']:.2f} goals")
        logger.info("="*70)
        
        logger.info("\n✅ NHL MODEL TRAINING COMPLETE")
        logger.info("Models trained and saved:")
        logger.info("  1. XGBoost (winner + totals)")
        logger.info("  2. CatBoost (winner + totals)")
        logger.info("  3. Elo (winner ratings)")
        logger.info("  4. Meta Ensemble (average of all 3)")
        logger.info("\nNext step: Run 'python generate_nhl_predictions.py' to generate predictions")
    else:
        logger.error(f"Training failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
