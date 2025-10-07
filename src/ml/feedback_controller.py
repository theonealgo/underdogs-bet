import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import json

class FeedbackController:
    """
    Manages the machine learning feedback loop by ingesting game results,
    calculating prediction errors, and triggering incremental model retraining.
    """
    
    def __init__(self, db_manager, predictor):
        self.db_manager = db_manager
        self.predictor = predictor
        self.logger = logging.getLogger(__name__)
        
    def ingest_completed_games(self, sport: str = 'MLB') -> Dict:
        """
        Ingest completed game results and update predictions with actual outcomes.
        
        Args:
            sport: Sport code (MLB, NBA, NFL, etc.)
            
        Returns:
            Dict with ingestion stats
        """
        try:
            self.logger.info(f"Ingesting completed {sport} games...")
            
            # Get completed games that haven't been processed
            with self.db_manager._get_connection() as conn:
                query = """
                    SELECT p.id, p.game_id, p.sport, p.game_date,
                           p.home_team_id, p.away_team_id, p.predicted_winner,
                           p.predicted_total, p.win_probability,
                           g.home_score, g.away_score, g.status
                    FROM predictions p
                    LEFT JOIN games g ON p.game_id = g.game_id
                    WHERE p.sport = ?
                    AND g.status = 'completed'
                    AND p.result_updated_at IS NULL
                """
                completed_df = pd.read_sql_query(query, conn, params=[sport])
            
            if completed_df.empty:
                self.logger.info(f"No new completed {sport} games to process")
                return {'success': True, 'processed': 0}
            
            # Process each completed game
            processed_count = 0
            for _, game in completed_df.iterrows():
                self._update_prediction_result(game)
                processed_count += 1
            
            self.logger.info(f"Processed {processed_count} completed {sport} games")
            
            return {
                'success': True,
                'processed': processed_count,
                'sport': sport
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting completed games: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _update_prediction_result(self, game_row: pd.Series):
        """Update prediction table with actual game result"""
        try:
            # Determine actual winner
            home_score = game_row['home_score']
            away_score = game_row['away_score']
            
            if home_score is None or away_score is None:
                return
            
            actual_winner = game_row['home_team_id'] if home_score > away_score else game_row['away_team_id']
            predicted_winner = game_row['predicted_winner']
            
            # Calculate errors
            win_correct = 1 if actual_winner == predicted_winner else 0
            actual_total = home_score + away_score
            predicted_total = game_row.get('predicted_total', 0)
            total_error = actual_total - predicted_total if predicted_total else 0
            total_abs_error = abs(total_error)
            
            # Update database
            with self.db_manager._get_connection() as conn:
                conn.execute("""
                    UPDATE predictions
                    SET actual_winner = ?,
                        actual_home_score = ?,
                        actual_away_score = ?,
                        actual_total = ?,
                        win_prediction_correct = ?,
                        total_prediction_error = ?,
                        total_absolute_error = ?,
                        result_updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    1 if actual_winner == game_row['home_team_id'] else 0,
                    int(home_score),
                    int(away_score),
                    actual_total,
                    win_correct,
                    total_error,
                    total_abs_error,
                    game_row['id']
                ))
                conn.commit()
                
            self.logger.debug(f"Updated prediction for game {game_row['game_id']}: "
                            f"Predicted {predicted_winner}, Actual {actual_winner}, "
                            f"Correct: {win_correct}")
            
        except Exception as e:
            self.logger.error(f"Error updating prediction result: {str(e)}")
    
    def trigger_incremental_retraining(self, sport: str = 'MLB', 
                                       window_days: int = 60,
                                       min_games: int = 100) -> Dict:
        """
        Trigger incremental model retraining with recent games weighted more heavily.
        
        Args:
            sport: Sport code
            window_days: Number of days to include in training window
            min_games: Minimum number of games required for retraining
            
        Returns:
            Dict with retraining results
        """
        try:
            self.logger.info(f"Triggering incremental retraining for {sport}...")
            
            # Get training data from sliding window - use predictions with actual results
            cutoff_date = datetime.now() - timedelta(days=window_days)
            
            with self.db_manager._get_connection() as conn:
                # Get completed predictions with actual results to use as training data
                query = """
                    SELECT p.*, g.home_score, g.away_score
                    FROM predictions p
                    LEFT JOIN games g ON p.game_id = g.game_id
                    WHERE p.sport = ?
                    AND p.game_date >= ?
                    AND p.result_updated_at IS NOT NULL
                    AND g.home_score IS NOT NULL
                    AND g.away_score IS NOT NULL
                    ORDER BY p.game_date DESC
                """
                training_data = pd.read_sql_query(
                    query, 
                    conn, 
                    params=[sport, cutoff_date.strftime('%Y-%m-%d')]
                )
            
            if len(training_data) < min_games:
                self.logger.warning(f"Insufficient training data for {sport}: {len(training_data)} games "
                                  f"(need {min_games})")
                return {'success': False, 'error': f'Insufficient training data ({len(training_data)} games, need {min_games})'}
            
            # Calculate sample weights - more recent games get higher weight
            training_data['days_ago'] = (
                datetime.now() - pd.to_datetime(training_data['game_date'])
            ).dt.days
            
            # Exponential decay: recent games weighted 3x more than oldest
            # Guard against division by zero when all games are on same date
            max_days = max(training_data['days_ago'].max(), 1)
            training_data['sample_weight'] = np.exp(-training_data['days_ago'] / max(max_days / 2, 1))
            
            # Get prediction errors to further weight training
            with self.db_manager._get_connection() as conn:
                error_query = """
                    SELECT game_date, total_absolute_error, win_prediction_correct
                    FROM predictions
                    WHERE sport = ?
                    AND result_updated_at IS NOT NULL
                    AND game_date >= ?
                """
                error_data = pd.read_sql_query(
                    error_query,
                    conn,
                    params=[sport, cutoff_date.strftime('%Y-%m-%d')]
                )
            
            # Retrain models with weighted data
            results = self.predictor.train_models(
                training_data,
                sample_weight=training_data['sample_weight'].values
            )
            
            if results:
                # Store retraining metrics
                self._store_retraining_metrics(sport, results, window_days, len(training_data))
                
                self.logger.info(f"Incremental retraining completed for {sport}")
                return {
                    'success': True,
                    'sport': sport,
                    'training_samples': len(training_data),
                    'window_days': window_days,
                    'metrics': results
                }
            else:
                return {'success': False, 'error': 'Training failed'}
                
        except Exception as e:
            self.logger.error(f"Error in incremental retraining: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _store_retraining_metrics(self, sport: str, results: Dict, 
                                   window_days: int, sample_count: int):
        """Store incremental learning metrics"""
        try:
            with self.db_manager._get_connection() as conn:
                # Store winner model metrics
                if 'winner_model' in results:
                    for metric_name, value in results['winner_model'].items():
                        conn.execute("""
                            INSERT INTO model_metrics 
                            (sport, model_type, metric_name, metric_value, model_version)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            sport,
                            'winner_incremental',
                            metric_name,
                            float(value),
                            f"window_{window_days}d_{sample_count}games"
                        ))
                
                # Store total model metrics
                if 'total_model' in results:
                    for metric_name, value in results['total_model'].items():
                        conn.execute("""
                            INSERT INTO model_metrics 
                            (sport, model_type, metric_name, metric_value, model_version)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            sport,
                            'total_incremental',
                            metric_name,
                            float(value),
                            f"window_{window_days}d_{sample_count}games"
                        ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing retraining metrics: {str(e)}")
    
    def run_daily_feedback_loop(self, sport: str = 'MLB') -> Dict:
        """
        Run complete daily feedback loop: ingest results + retrain models.
        
        Args:
            sport: Sport code
            
        Returns:
            Dict with feedback loop results
        """
        try:
            self.logger.info(f"Running daily feedback loop for {sport}...")
            
            # Step 1: Ingest completed game results
            ingest_result = self.ingest_completed_games(sport)
            
            # Step 2: Trigger incremental retraining if we processed games
            if ingest_result.get('processed', 0) > 0:
                retrain_result = self.trigger_incremental_retraining(sport)
            else:
                retrain_result = {'skipped': True, 'reason': 'No new completed games'}
            
            return {
                'success': True,
                'sport': sport,
                'ingestion': ingest_result,
                'retraining': retrain_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in daily feedback loop: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_learning_performance(self, sport: str = 'MLB', days: int = 30) -> Dict:
        """
        Get learning performance metrics over time.
        
        Args:
            sport: Sport code
            days: Number of days to analyze
            
        Returns:
            Dict with performance trends
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.db_manager._get_connection() as conn:
                # Get prediction accuracy over time
                query = """
                    SELECT 
                        DATE(result_updated_at) as date,
                        COUNT(*) as total_predictions,
                        SUM(win_prediction_correct) as correct_predictions,
                        AVG(win_prediction_correct) * 100 as accuracy,
                        AVG(total_absolute_error) as avg_total_error
                    FROM predictions
                    WHERE sport = ?
                    AND result_updated_at >= ?
                    GROUP BY DATE(result_updated_at)
                    ORDER BY date
                """
                performance_df = pd.read_sql_query(
                    query,
                    conn,
                    params=[sport, cutoff_date.strftime('%Y-%m-%d')]
                )
            
            if performance_df.empty:
                return {'error': 'No performance data available'}
            
            # Calculate improvement trend
            if len(performance_df) > 1:
                recent_accuracy = performance_df.tail(7)['accuracy'].mean()
                older_accuracy = performance_df.head(7)['accuracy'].mean()
                improvement = recent_accuracy - older_accuracy
            else:
                improvement = 0
            
            return {
                'sport': sport,
                'days_analyzed': days,
                'total_predictions': int(performance_df['total_predictions'].sum()),
                'overall_accuracy': float(performance_df['accuracy'].mean()),
                'recent_accuracy': float(performance_df.tail(7)['accuracy'].mean()),
                'accuracy_trend': 'improving' if improvement > 0 else 'declining',
                'improvement_pct': float(improvement),
                'daily_performance': performance_df.to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning performance: {str(e)}")
            return {'error': str(e)}
