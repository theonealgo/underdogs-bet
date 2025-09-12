import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import sqlite3
import numpy as np

from data_storage.database import DatabaseManager


class ResultTracker:
    """
    Tracks actual game results and compares them to predictions for learning
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
    def fetch_and_update_results(self, date: datetime = None, days_back: int = 1) -> Dict:
        """
        Fetch actual game results and update prediction accuracy
        
        Args:
            date: Date to fetch results for (defaults to yesterday)
            days_back: Number of days to look back for results
            
        Returns:
            Dictionary with update results
        """
        try:
            if date is None:
                date = datetime.now() - timedelta(days=1)  # Default to yesterday
            
            self.logger.info(f"Fetching game results for {date.date()}")
            
            results = {
                'success': True,
                'date': date.date().isoformat(),
                'games_processed': 0,
                'predictions_updated': 0,
                'accuracy_calculated': 0,
                'errors': []
            }
            
            # Fetch completed games for the date range
            completed_games = self._fetch_completed_games(date, days_back)
            
            if completed_games.empty:
                self.logger.info(f"No completed games found for {date.date()}")
                return results
            
            results['games_processed'] = len(completed_games)
            
            # Update predictions with actual results
            updated_predictions = self._update_predictions_with_results(completed_games)
            results['predictions_updated'] = updated_predictions
            
            # Calculate and store accuracy metrics
            accuracy_metrics = self._calculate_and_store_accuracy(date)
            results['accuracy_calculated'] = len(accuracy_metrics)
            
            # Store performance trends
            self._update_performance_trends(date)
            
            # Generate learning insights
            insights = self._generate_learning_insights(date)
            if insights:
                results['insights_generated'] = len(insights)
            
            self.logger.info(f"Result tracking completed: {updated_predictions} predictions updated")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fetch_and_update_results: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'date': date.date().isoformat() if date else None
            }
    
    def _fetch_completed_games(self, date: datetime, days_back: int) -> pd.DataFrame:
        """
        Fetch completed games from MLB API
        
        Args:
            date: Date to fetch games for
            days_back: Number of days to look back
            
        Returns:
            DataFrame with completed games
        """
        try:
            start_date = date - timedelta(days=days_back-1)
            end_date = date
            
            url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,  # MLB
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'hydrate': 'team,linescore,decisions'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            if 'dates' in data and data['dates']:
                for date_entry in data['dates']:
                    if 'games' in date_entry:
                        for game in date_entry['games']:
                            # Only process completed games
                            status = game.get('status', {})
                            if status.get('statusCode') in ['F', 'O']:  # Final or Other (completed)
                                game_info = self._parse_completed_game(game)
                                if game_info:
                                    games.append(game_info)
            
            df = pd.DataFrame(games)
            self.logger.info(f"Fetched {len(df)} completed games")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching completed games: {str(e)}")
            return pd.DataFrame()
    
    def _parse_completed_game(self, game: Dict) -> Optional[Dict]:
        """
        Parse a completed game from MLB API response
        
        Args:
            game: Game data from MLB API
            
        Returns:
            Parsed game info or None if parsing fails
        """
        try:
            game_pk = game.get('gamePk')
            if not game_pk:
                return None
            
            # Game date
            game_date = game.get('gameDate', '')
            if game_date:
                game_datetime = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                game_date = game_datetime.strftime('%Y-%m-%d')
            
            # Teams and scores
            teams = game.get('teams', {})
            home_team = teams.get('home', {})
            away_team = teams.get('away', {})
            
            home_team_id = home_team.get('team', {}).get('abbreviation', '')
            away_team_id = away_team.get('team', {}).get('abbreviation', '')
            home_score = home_team.get('score', 0)
            away_score = away_team.get('score', 0)
            
            if not home_team_id or not away_team_id:
                return None
            
            # Calculate total runs and winner
            total_runs = home_score + away_score
            home_win = 1 if home_score > away_score else 0
            actual_winner = 1 if home_win else 0  # 1 for home win, 0 for away win
            
            return {
                'game_id': str(game_pk),
                'game_date': game_date,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_score': home_score,
                'away_score': away_score,
                'total_runs': total_runs,
                'actual_winner': actual_winner,
                'home_win': home_win
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing completed game: {str(e)}")
            return None
    
    def _update_predictions_with_results(self, completed_games: pd.DataFrame) -> int:
        """
        Update predictions table with actual results
        
        Args:
            completed_games: DataFrame with completed game results
            
        Returns:
            Number of predictions updated
        """
        try:
            updated_count = 0
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                for _, game in completed_games.iterrows():
                    # Find matching prediction
                    query = """
                        SELECT id, predicted_winner, predicted_total, win_probability, total_confidence
                        FROM predictions 
                        WHERE game_id = ? AND sport = 'MLB' AND result_updated_at IS NULL
                    """
                    
                    cursor = conn.execute(query, (game['game_id'],))
                    prediction_row = cursor.fetchone()
                    
                    if prediction_row:
                        prediction_id, predicted_winner, predicted_total, win_prob, total_conf = prediction_row
                        
                        # Calculate accuracy metrics
                        win_correct = 1 if predicted_winner == game['actual_winner'] else 0
                        total_error = predicted_total - game['total_runs'] if predicted_total is not None else None
                        total_abs_error = abs(total_error) if total_error is not None else None
                        
                        # Update prediction with actual results
                        update_query = """
                            UPDATE predictions SET
                                actual_winner = ?,
                                actual_home_score = ?,
                                actual_away_score = ?,
                                actual_total = ?,
                                win_prediction_correct = ?,
                                total_prediction_error = ?,
                                total_absolute_error = ?,
                                result_updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """
                        
                        conn.execute(update_query, (
                            game['actual_winner'],
                            game['home_score'],
                            game['away_score'],
                            game['total_runs'],
                            win_correct,
                            total_error,
                            total_abs_error,
                            prediction_id
                        ))
                        
                        updated_count += 1
                        
                        self.logger.debug(f"Updated prediction for game {game['game_id']}: "
                                       f"Win correct: {win_correct}, Total error: {total_error}")
                
                conn.commit()
            
            self.logger.info(f"Updated {updated_count} predictions with actual results")
            return updated_count
            
        except Exception as e:
            self.logger.error(f"Error updating predictions with results: {str(e)}")
            return 0
    
    def _calculate_and_store_accuracy(self, date: datetime) -> List[Dict]:
        """
        Calculate and store accuracy metrics for the given date
        
        Args:
            date: Date to calculate accuracy for
            
        Returns:
            List of accuracy metrics calculated
        """
        try:
            metrics = []
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Calculate daily accuracy metrics
                query = """
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN win_prediction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                        AVG(win_probability) as avg_confidence,
                        AVG(total_absolute_error) as total_mae,
                        SQRT(AVG(total_prediction_error * total_prediction_error)) as total_rmse
                    FROM predictions 
                    WHERE sport = 'MLB' 
                    AND DATE(game_date) = DATE(?)
                    AND result_updated_at IS NOT NULL
                """
                
                cursor = conn.execute(query, (date.date(),))
                result = cursor.fetchone()
                
                if result and result[0] > 0:  # If we have predictions for this date
                    total_pred, correct_pred, avg_conf, mae, rmse = result
                    accuracy_rate = correct_pred / total_pred if total_pred > 0 else 0
                    
                    # Calculate confidence calibration (simplified)
                    calibration_query = """
                        SELECT AVG(win_probability), AVG(CAST(win_prediction_correct AS REAL))
                        FROM predictions 
                        WHERE sport = 'MLB' 
                        AND DATE(game_date) = DATE(?)
                        AND result_updated_at IS NOT NULL
                    """
                    
                    cal_cursor = conn.execute(calibration_query, (date.date(),))
                    cal_result = cal_cursor.fetchone()
                    
                    confidence_calibration = None
                    if cal_result and cal_result[0] and cal_result[1]:
                        # Difference between average confidence and actual accuracy
                        confidence_calibration = abs(cal_result[0] - cal_result[1])
                    
                    # Store accuracy metrics
                    insert_query = """
                        INSERT OR REPLACE INTO prediction_accuracy 
                        (sport, model_type, model_version, date_period, total_predictions, 
                         correct_predictions, accuracy_rate, avg_win_confidence, 
                         total_mae, total_rmse, confidence_calibration)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    conn.execute(insert_query, (
                        'MLB', 'xgboost', 'v1.0', date.date(), total_pred, 
                        correct_pred, accuracy_rate, avg_conf, mae, rmse, confidence_calibration
                    ))
                    
                    metrics.append({
                        'date': date.date().isoformat(),
                        'accuracy_rate': accuracy_rate,
                        'total_predictions': total_pred,
                        'correct_predictions': correct_pred,
                        'total_mae': mae,
                        'total_rmse': rmse,
                        'confidence_calibration': confidence_calibration
                    })
                    
                    conn.commit()
                    
                    self.logger.info(f"Stored accuracy metrics for {date.date()}: "
                                   f"{accuracy_rate:.3f} accuracy, {mae:.3f} MAE")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {str(e)}")
            return []
    
    def _update_performance_trends(self, date: datetime) -> bool:
        """
        Update performance trends for monitoring model degradation
        
        Args:
            date: Current date for trend calculation
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Calculate 7-day trend
                week_ago = date - timedelta(days=7)
                
                trend_query = """
                    SELECT 
                        AVG(accuracy_rate) as avg_accuracy,
                        AVG(total_mae) as avg_mae,
                        COUNT(*) as volume
                    FROM prediction_accuracy
                    WHERE sport = 'MLB' 
                    AND date_period >= DATE(?) 
                    AND date_period <= DATE(?)
                """
                
                cursor = conn.execute(trend_query, (week_ago.date(), date.date()))
                result = cursor.fetchone()
                
                if result and result[2] > 0:  # If we have data
                    avg_accuracy, avg_mae, volume = result
                    
                    # Compare to previous week
                    prev_week_start = week_ago - timedelta(days=7)
                    prev_week_end = week_ago
                    
                    prev_cursor = conn.execute(trend_query, (prev_week_start.date(), prev_week_end.date()))
                    prev_result = prev_cursor.fetchone()
                    
                    accuracy_trend = 0
                    mae_trend = 0
                    volume_trend = 0
                    
                    if prev_result and prev_result[2] > 0:
                        prev_accuracy, prev_mae, prev_volume = prev_result
                        accuracy_trend = avg_accuracy - prev_accuracy if prev_accuracy else 0
                        mae_trend = avg_mae - prev_mae if prev_mae else 0
                        volume_trend = volume - prev_volume if prev_volume else 0
                    
                    # Calculate improvement score (higher is better)
                    improvement_score = accuracy_trend * 100 - mae_trend * 10
                    
                    # Check if retraining should be triggered
                    retraining_triggered = 0
                    if avg_accuracy < 0.52 or accuracy_trend < -0.05:  # Accuracy below 52% or dropping significantly
                        retraining_triggered = 1
                        self.logger.warning(f"Performance degradation detected. Accuracy: {avg_accuracy:.3f}, Trend: {accuracy_trend:.3f}")
                    
                    # Store trend data
                    insert_query = """
                        INSERT INTO performance_trends 
                        (sport, model_type, trend_period, period_start, period_end,
                         accuracy_trend, mae_trend, volume_trend, improvement_score, retraining_triggered)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    conn.execute(insert_query, (
                        'MLB', 'xgboost', 'weekly', week_ago.date(), date.date(),
                        accuracy_trend, mae_trend, volume_trend, improvement_score, retraining_triggered
                    ))
                    
                    conn.commit()
                    
                    self.logger.info(f"Updated performance trends: accuracy_trend={accuracy_trend:.4f}, "
                                   f"mae_trend={mae_trend:.4f}, retraining_needed={retraining_triggered}")
                    
                    return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating performance trends: {str(e)}")
            return False
    
    def _generate_learning_insights(self, date: datetime) -> List[Dict]:
        """
        Generate insights about model performance and errors
        
        Args:
            date: Date to generate insights for
            
        Returns:
            List of learning insights
        """
        try:
            insights = []
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Look for systematic errors in the last week
                week_ago = date - timedelta(days=7)
                
                # Check for team-specific biases
                team_bias_query = """
                    SELECT 
                        home_team_id,
                        COUNT(*) as total_predictions,
                        AVG(CAST(win_prediction_correct AS REAL)) as accuracy,
                        AVG(win_probability) as avg_confidence
                    FROM predictions
                    WHERE sport = 'MLB' 
                    AND DATE(game_date) >= DATE(?) 
                    AND DATE(game_date) <= DATE(?)
                    AND result_updated_at IS NOT NULL
                    GROUP BY home_team_id
                    HAVING COUNT(*) >= 3
                """
                
                cursor = conn.execute(team_bias_query, (week_ago.date(), date.date()))
                team_results = cursor.fetchall()
                
                for team, total, accuracy, confidence in team_results:
                    if accuracy < 0.4:  # Very poor accuracy for this team
                        insight = {
                            'type': 'error_pattern',
                            'category': 'team_bias',
                            'description': f"Poor prediction accuracy for {team}: {accuracy:.2%} over {total} games",
                            'confidence': 0.8,
                            'team': team,
                            'accuracy': accuracy
                        }
                        insights.append(insight)
                        
                        # Store in database
                        self._store_learning_insight('MLB', 'xgboost', 'error_pattern', 'team_bias',
                                                   insight['description'], 0.8, 
                                                   f"Flag {team} predictions for review")
                
                # Check for total over/under estimation patterns
                total_pattern_query = """
                    SELECT 
                        AVG(total_prediction_error) as avg_error,
                        ABS(AVG(total_prediction_error)) as abs_avg_error,
                        COUNT(*) as total_predictions
                    FROM predictions
                    WHERE sport = 'MLB' 
                    AND DATE(game_date) >= DATE(?) 
                    AND DATE(game_date) <= DATE(?)
                    AND result_updated_at IS NOT NULL
                    AND total_prediction_error IS NOT NULL
                """
                
                cursor = conn.execute(total_pattern_query, (week_ago.date(), date.date()))
                total_result = cursor.fetchone()
                
                if total_result and total_result[2] > 5:  # At least 5 predictions
                    avg_error, abs_avg_error, count = total_result
                    
                    if abs(avg_error) > 0.75:  # Systematic over/under estimation
                        direction = "overestimating" if avg_error > 0 else "underestimating"
                        insight = {
                            'type': 'error_pattern',
                            'category': 'total_bias',
                            'description': f"Systematically {direction} totals by {abs(avg_error):.2f} runs",
                            'confidence': 0.9,
                            'error_magnitude': abs(avg_error)
                        }
                        insights.append(insight)
                        
                        # Store in database
                        self._store_learning_insight('MLB', 'xgboost', 'error_pattern', 'total_bias',
                                                   insight['description'], 0.9, 
                                                   "Adjust total prediction calibration")
                
                self.logger.info(f"Generated {len(insights)} learning insights")
                return insights
                
        except Exception as e:
            self.logger.error(f"Error generating learning insights: {str(e)}")
            return []
    
    def _store_learning_insight(self, sport: str, model_type: str, insight_type: str, 
                               category: str, description: str, confidence: float, action: str):
        """Store a learning insight in the database"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    INSERT INTO learning_insights 
                    (sport, model_type, insight_type, insight_category, insight_description, 
                     confidence_score, action_taken)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                
                conn.execute(query, (sport, model_type, insight_type, category, 
                                   description, confidence, action))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing learning insight: {str(e)}")
    
    def get_recent_accuracy(self, days: int = 7) -> Dict:
        """
        Get recent accuracy metrics
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT 
                        AVG(accuracy_rate) as avg_accuracy,
                        AVG(total_mae) as avg_mae,
                        SUM(total_predictions) as total_predictions,
                        SUM(correct_predictions) as total_correct
                    FROM prediction_accuracy
                    WHERE sport = 'MLB' 
                    AND date_period >= DATE(?) 
                    AND date_period <= DATE(?)
                """
                
                cursor = conn.execute(query, (start_date.date(), end_date.date()))
                result = cursor.fetchone()
                
                if result:
                    avg_acc, avg_mae, total_pred, total_correct = result
                    return {
                        'period_days': days,
                        'average_accuracy': avg_acc,
                        'total_predictions': total_pred or 0,
                        'total_correct': total_correct or 0,
                        'average_mae': avg_mae
                    }
                
            return {'period_days': days, 'no_data': True}
            
        except Exception as e:
            self.logger.error(f"Error getting recent accuracy: {str(e)}")
            return {'error': str(e)}
    
    def should_trigger_retraining(self) -> Tuple[bool, str]:
        """
        Check if model retraining should be triggered based on performance
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Check recent performance trends
                query = """
                    SELECT retraining_triggered, accuracy_trend, improvement_score
                    FROM performance_trends
                    WHERE sport = 'MLB' AND model_type = 'xgboost'
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                
                cursor = conn.execute(query)
                result = cursor.fetchone()
                
                if result:
                    retraining_triggered, accuracy_trend, improvement_score = result
                    
                    if retraining_triggered == 1:
                        return True, "Performance degradation detected"
                    
                    if accuracy_trend and accuracy_trend < -0.03:
                        return True, f"Accuracy trending down: {accuracy_trend:.4f}"
                    
                    if improvement_score and improvement_score < -5:
                        return True, f"Low improvement score: {improvement_score:.2f}"
                
                # Check if we have enough new data since last training
                recent_predictions_query = """
                    SELECT COUNT(*)
                    FROM predictions
                    WHERE sport = 'MLB' 
                    AND result_updated_at IS NOT NULL
                    AND DATE(result_updated_at) >= DATE('now', '-7 days')
                """
                
                cursor = conn.execute(recent_predictions_query)
                recent_count = cursor.fetchone()[0]
                
                if recent_count >= 50:  # Enough new data for retraining
                    return True, f"Sufficient new data available: {recent_count} predictions"
                
            return False, "No retraining trigger conditions met"
            
        except Exception as e:
            self.logger.error(f"Error checking retraining trigger: {str(e)}")
            return False, f"Error checking: {str(e)}"