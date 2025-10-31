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
    """Load NHL training data (2024 season + 2025 completed games only for speed)"""
    import pickle
    import os
    from datetime import datetime
    
    games = []
    
    # Load historical data and filter to 2024 season only
    if os.path.exists('nhl_historical_data.pkl'):
        with open('nhl_historical_data.pkl', 'rb') as f:
            all_games = pickle.load(f)
        
        # Use ONLY 2024 season - most recent data for best accuracy
        for game in all_games:
            try:
                date_obj = datetime.strptime(game['date'], '%Y-%m-%d')
                
                # Only 2024 season (most relevant for current predictions)
                if date_obj.year != 2024:
                    continue
                    
                formatted_date = date_obj.strftime('%d/%m/%Y')
            except:
                continue
            
            games.append({
                'game_id': f"nhl_{game['match_id']}",
                'game_date': formatted_date,
                'home_team_id': game['home_team'],
                'away_team_id': game['away_team'],
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'status': 'final'
            })
        logger.info(f"Loaded {len(games)} games from 2024 season (most recent full season)")
    else:
        logger.error("Historical data file not found. Run 'python fetch_nhl_historical_data.py' first")
        return pd.DataFrame()
    
    # DO NOT add 2025 games to training - they are our true test set!
    # Training on 2024 season only to prevent data leakage
    
    df = pd.DataFrame(games)
    logger.info(f"Total {len(df)} training games (2024 season only - no 2025 to prevent leakage)")
    if len(df) > 0:
        logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    return df

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
    
    # Train all models (XGBoost, CatBoost, Elo) on all available data (2024 + 2025 completed)
    logger.info("\nTraining XGBoost, CatBoost, and Elo models on all available data...")
    results = predictor.train_models(training_data)
    
    if results['success']:
        logger.info("\n" + "="*70)
        logger.info("TRAINING RESULTS (2024 SEASON)")
        logger.info("="*70)
        logger.info(f"  Season:                {results['season']}")
        logger.info(f"  Training Games:        {results['training_games']}")
        logger.info(f"  Number of Features:    {results['num_features']}")
        logger.info(f"  Elo Teams:             {results['elo_teams']}")
        logger.info(f"  XGBoost Accuracy:      {results['xgb_accuracy']:.1%}")
        logger.info(f"  CatBoost Accuracy:     {results['catboost_accuracy']:.1%}")
        logger.info(f"  Elo Accuracy:          {results['elo_accuracy']:.1%}")
        logger.info(f"  XGBoost Total MAE:     {results['xgb_total_mae']:.2f} goals")
        logger.info(f"  CatBoost Total MAE:    {results['catboost_total_mae']:.2f} goals")
        logger.info("="*70)
        
        # AUTOMATIC BACKTEST ON TRUE HELD-OUT 2025 GAMES
        logger.info("\n" + "="*70)
        logger.info("AUTOMATIC BACKTEST ON 2025 COMPLETED GAMES (TRUE HELD-OUT DATA)")
        logger.info("="*70)
        
        # Load 2025 completed games from database (NOT in training set)
        conn = sqlite3.connect(DATABASE)
        test_games = pd.read_sql_query("""
            SELECT game_id, game_date, home_team_id, away_team_id, home_score, away_score
            FROM games
            WHERE sport='NHL' AND season=2025 
            AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY game_date
        """, conn)
        conn.close()
        
        if len(test_games) == 0:
            logger.warning("No completed 2025 games found for backtest")
            logger.info("Models trained successfully. Backtest will run once 2025 games are completed.")
            return
        
        # Convert date format for compatibility
        test_games['game_date'] = pd.to_datetime(test_games['game_date'], format='%Y-%m-%d')
        
        logger.info(f"Testing on {len(test_games)} completed 2025 games (NEVER SEEN BEFORE)")
        logger.info(f"Date range: {test_games['game_date'].min().strftime('%Y-%m-%d')} to {test_games['game_date'].max().strftime('%Y-%m-%d')}")
        
        # Generate predictions for test games
        xgb_correct = 0
        cat_correct = 0
        elo_correct = 0
        meta_correct = 0
        
        for idx, game in test_games.iterrows():
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                actual_winner = game['home_team_id'] if game['home_score'] > game['away_score'] else game['away_team_id']
                
                # Get predictions
                pred = predictor.predict_game(
                    game['home_team_id'],
                    game['away_team_id'],
                    game['game_date'].strftime('%d/%m/%Y'),
                    training_data
                )
                
                # Check each model
                if pred['xgb_home_prob'] > 0.5:
                    xgb_winner = game['home_team_id']
                else:
                    xgb_winner = game['away_team_id']
                    
                if pred['catboost_home_prob'] > 0.5:
                    cat_winner = game['home_team_id']
                else:
                    cat_winner = game['away_team_id']
                    
                if pred['elo_home_prob'] > 0.5:
                    elo_winner = game['home_team_id']
                else:
                    elo_winner = game['away_team_id']
                    
                if pred['meta_home_prob'] > 0.5:
                    meta_winner = game['home_team_id']
                else:
                    meta_winner = game['away_team_id']
                
                if xgb_winner == actual_winner:
                    xgb_correct += 1
                if cat_winner == actual_winner:
                    cat_correct += 1
                if elo_winner == actual_winner:
                    elo_correct += 1
                if meta_winner == actual_winner:
                    meta_correct += 1
        
        total_games = len(test_games)
        logger.info(f"\n{total_games}-GAME BACKTEST ACCURACY (TRUE HELD-OUT 2025 DATA):")
        logger.info(f"  XGBoost:   {xgb_correct}/{total_games} = {xgb_correct/total_games:.1%}")
        logger.info(f"  CatBoost:  {cat_correct}/{total_games} = {cat_correct/total_games:.1%}")
        logger.info(f"  Elo:       {elo_correct}/{total_games} = {elo_correct/total_games:.1%}")
        logger.info(f"  Meta:      {meta_correct}/{total_games} = {meta_correct/total_games:.1%}")
        logger.info("="*70)
        
        logger.info("\n✅ NHL MODEL TRAINING COMPLETE")
        logger.info("Models trained and saved:")
        logger.info("  1. XGBoost (winner + totals)")
        logger.info("  2. CatBoost (winner + totals)")
        logger.info("  3. Elo (winner ratings)")
        logger.info("  4. Meta Ensemble (XGBoost + CatBoost, NO Elo)")
        logger.info("\nNext step: Run 'python generate_nhl_predictions.py' to generate predictions")
    else:
        logger.error(f"Training failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
