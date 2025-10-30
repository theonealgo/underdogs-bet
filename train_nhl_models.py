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
    
    # Train all models (XGBoost, CatBoost, Elo) on 2024 season ONLY
    logger.info("\nTraining XGBoost, CatBoost, and Elo models on 2024 season only...")
    results = predictor.train_models(training_data, season_year=2024)
    
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
        
        # AUTOMATIC 148-GAME BACKTEST
        logger.info("\n" + "="*70)
        logger.info("AUTOMATIC BACKTEST ON LAST 148 GAMES")
        logger.info("="*70)
        
        # Filter to 2024 games and take last 148
        training_data['game_date'] = pd.to_datetime(training_data['game_date'], format='%d/%m/%Y', dayfirst=True)
        season_2024 = training_data[training_data['game_date'].dt.year == 2024].copy()
        season_2024 = season_2024.sort_values('game_date')
        test_games = season_2024.tail(148)
        
        logger.info(f"Testing on {len(test_games)} most recent 2024 games")
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
        
        logger.info("\n148-GAME BACKTEST ACCURACY:")
        logger.info(f"  XGBoost:   {xgb_correct}/148 = {xgb_correct/148:.1%}")
        logger.info(f"  CatBoost:  {cat_correct}/148 = {cat_correct/148:.1%}")
        logger.info(f"  Elo:       {elo_correct}/148 = {elo_correct/148:.1%}")
        logger.info(f"  Meta:      {meta_correct}/148 = {meta_correct/148:.1%}")
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
