"""
Time-Series Cross-Validation for NFL Model
Implements walk-forward validation to get realistic accuracy estimates.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from src.models.universal_ensemble_predictor import UniversalSportsEnsemble
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_nfl_data():
    """Load NFL games and team stats"""
    conn = sqlite3.connect('sports_predictions.db')
    
    # Load games
    games_query = """
        SELECT 
            game_date as date,
            home_team_id as home_team,
            away_team_id as away_team,
            home_score,
            away_score,
            CASE WHEN home_score > away_score THEN 'H' ELSE 'A' END as result
        FROM games
        WHERE sport = 'NFL' AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY game_date ASC
    """
    games_df = pd.read_sql_query(games_query, conn)
    
    # Load team stats
    stats_query = """
        SELECT 
            sport,
            team_id as team,
            season,
            date,
            games_played,
            wins,
            losses,
            metrics,
            last_updated
        FROM team_stats
        WHERE sport = 'NFL'
        ORDER BY date ASC
    """
    stats_df = pd.read_sql_query(stats_query, conn)
    
    # Expand metrics
    metrics_df = pd.json_normalize(stats_df['metrics'].apply(eval))
    stats_df = pd.concat([stats_df.drop('metrics', axis=1), metrics_df], axis=1)
    
    conn.close()
    
    logger.info(f"Loaded {len(games_df)} NFL games and {len(stats_df)} team stats records")
    return games_df, stats_df

def time_series_cv(games_df, stats_df, min_train_size=100, test_size=20, step=10):
    """
    Walk-forward time-series cross-validation.
    
    Args:
        min_train_size: Minimum number of games to train on
        test_size: Number of games in each test fold
        step: Number of games to step forward each iteration
    """
    
    results = []
    n_games = len(games_df)
    
    # Convert dates to datetime for proper sorting
    games_df['date'] = pd.to_datetime(games_df['date'], format='%d/%m/%Y', errors='coerce')
    stats_df['date'] = pd.to_datetime(stats_df['date'], format='%d/%m/%Y', errors='coerce')
    
    games_df = games_df.sort_values('date').reset_index(drop=True)
    
    fold = 1
    for train_end in range(min_train_size, n_games - test_size, step):
        test_start = train_end
        test_end = min(train_end + test_size, n_games)
        
        # Split data
        train_df = games_df.iloc[:train_end].copy()
        test_df = games_df.iloc[test_start:test_end].copy()
        
        # Filter team stats to only include data before test period
        test_start_date = test_df['date'].min()
        train_stats = stats_df[stats_df['date'] < test_start_date].copy()
        
        logger.info(f"\nFold {fold}: Train on {len(train_df)} games, Test on {len(test_df)} games")
        logger.info(f"  Train period: {train_df['date'].min()} to {train_df['date'].max()}")
        logger.info(f"  Test period: {test_df['date'].min()} to {test_df['date'].max()}")
        
        # Train model
        try:
            predictor = UniversalSportsEnsemble(sport='NFL', k_factor=35)
            predictor.train(train_df, team_stats=train_stats)
            
            # Predict on test set
            test_features = predictor.create_features(test_df, team_stats=train_stats, is_training=False)
            
            if test_features.empty:
                logger.warning(f"  Fold {fold}: No features generated, skipping")
                fold += 1
                continue
            
            predictions = []
            actuals = []
            
            for idx, row in test_df.iterrows():
                features_row = test_features[test_features.index == idx]
                if features_row.empty:
                    continue
                
                pred_proba = predictor.predict_proba(features_row)[0]
                predictions.append(pred_proba)
                actuals.append(1 if row['result'] == 'H' else 0)
            
            if len(predictions) > 0:
                # Calculate metrics
                preds_binary = [1 if p >= 0.5 else 0 for p in predictions]
                acc = accuracy_score(actuals, preds_binary)
                logloss = log_loss(actuals, predictions)
                brier = brier_score_loss(actuals, predictions)
                
                results.append({
                    'fold': fold,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'accuracy': acc,
                    'log_loss': logloss,
                    'brier_score': brier
                })
                
                logger.info(f"  Accuracy: {acc:.3f} | LogLoss: {logloss:.4f} | Brier: {brier:.4f}")
        
        except Exception as e:
            logger.error(f"  Fold {fold} failed: {e}")
        
        fold += 1
    
    return pd.DataFrame(results)

def main():
    logger.info("="*70)
    logger.info("NFL TIME-SERIES CROSS-VALIDATION")
    logger.info("="*70)
    
    # Load data
    games_df, stats_df = load_nfl_data()
    
    # Run time-series CV
    results_df = time_series_cv(
        games_df, 
        stats_df,
        min_train_size=100,  # Start with 100 games for training
        test_size=20,         # Test on 20 games each fold
        step=10               # Move forward 10 games each iteration
    )
    
    if len(results_df) > 0:
        logger.info("\n" + "="*70)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("="*70)
        logger.info(f"\nFolds completed: {len(results_df)}")
        logger.info(f"\nMean Accuracy: {results_df['accuracy'].mean():.3f} (+/- {results_df['accuracy'].std():.3f})")
        logger.info(f"Mean LogLoss: {results_df['log_loss'].mean():.4f} (+/- {results_df['log_loss'].std():.4f})")
        logger.info(f"Mean Brier Score: {results_df['brier_score'].mean():.4f} (+/- {results_df['brier_score'].std():.4f})")
        
        logger.info("\nPer-Fold Results:")
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('nfl_timeseries_cv_results.csv', index=False)
        logger.info("\n✅ Results saved to nfl_timeseries_cv_results.csv")
    else:
        logger.warning("No results generated")

if __name__ == '__main__':
    main()
