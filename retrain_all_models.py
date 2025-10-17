"""
Retrain all sport models with comprehensive features
Uses historical game results and team_stats from database
"""

import pandas as pd
import sqlite3
import logging
from src.data_storage.database import DatabaseManager
from src.models.universal_ensemble_predictor import UniversalSportsEnsemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_historical_games(db_path: str, sport: str) -> pd.DataFrame:
    """Load historical games with results from database"""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT 
                game_date as date,
                home_team_id as home_team,
                away_team_id as away_team,
                CASE 
                    WHEN home_score > away_score THEN 'H'
                    WHEN away_score > home_score THEN 'A'
                    WHEN home_score = away_score THEN 'D'
                    ELSE NULL
                END as result,
                home_score,
                away_score
            FROM games
            WHERE sport = ?
            AND status = 'final'
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
            ORDER BY game_date ASC
        """
        df = pd.read_sql_query(query, conn, params=[sport])
    
    logger.info(f"{sport}: Loaded {len(df)} games with results")
    return df


def get_team_stats(db_path: str, sport: str) -> pd.DataFrame:
    """Load team stats from database for feature engineering"""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT 
                sport,
                team_id as team,
                date,
                metrics
            FROM team_stats
            WHERE sport = ?
            ORDER BY team_id, date ASC
        """
        df = pd.read_sql_query(query, conn, params=[sport])
    
    if len(df) > 0:
        # Parse JSON metrics
        import json
        df['metrics'] = df['metrics'].apply(json.loads)
        
        # Expand metrics into columns
        metrics_df = pd.json_normalize(df['metrics'])
        df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
        
        logger.info(f"{sport}: Loaded {len(df)} team stat records for {df['team'].nunique()} teams")
    else:
        logger.warning(f"{sport}: No team stats available")
    
    return df


def retrain_sport(sport: str, db_path: str):
    """Retrain model for a specific sport"""
    logger.info(f"\n{'='*70}")
    logger.info(f"RETRAINING {sport} MODEL")
    logger.info(f"{'='*70}")
    
    # Load historical games
    historical_games = get_historical_games(db_path, sport)
    
    if len(historical_games) == 0:
        logger.warning(f"{sport}: No historical games found, skipping")
        return
    
    # Load team stats
    team_stats = get_team_stats(db_path, sport)
    
    # Initialize predictor with sport-specific K-factor
    k_factors = {
        'NFL': 35,
        'NBA': 18,
        'NHL': 22,
        'MLB': 14,
        'NCAAF': 30,
        'NCAAB': 25
    }
    k_factor = k_factors.get(sport, 20)
    
    predictor = UniversalSportsEnsemble(sport=sport, k_factor=k_factor)
    
    # Train model
    logger.info(f"{sport}: Training with {len(historical_games)} games...")
    
    if len(team_stats) > 0:
        training_results = predictor.train(historical_games, team_stats=team_stats)
    else:
        training_results = predictor.train(historical_games)
    
    logger.info(f"{sport}: Training results: {training_results}")
    
    # Save model
    model_path = f'models/{sport.lower()}_ensemble.pkl'
    predictor.save_model(model_path)
    logger.info(f"{sport}: Model saved to {model_path}")
    
    return training_results


def main():
    """Retrain all sport models"""
    db_manager = DatabaseManager()
    
    # Sports to retrain
    sports = ['NFL', 'NBA', 'NHL', 'MLB', 'NCAAF']
    
    results = {}
    
    for sport in sports:
        try:
            result = retrain_sport(sport, db_manager.db_path)
            if result:
                results[sport] = result
        except Exception as e:
            logger.error(f"Error retraining {sport}: {str(e)}", exc_info=True)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("RETRAINING SUMMARY")
    logger.info(f"{'='*70}")
    
    for sport, result in results.items():
        accuracy = result.get('accuracy', 0)
        logger.info(f"{sport}: Accuracy = {accuracy:.1%}")
    
    logger.info(f"\nSuccessfully retrained {len(results)} / {len(sports)} sports")


if __name__ == "__main__":
    main()
