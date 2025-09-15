import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import sqlite3

from data_storage.database import DatabaseManager
from models.prediction_models import MLBPredictor
from data_collectors.baseball_savant_scraper import BaseballSavantScraper
# Note: OddsShark dependency removed

class PredictionAPI:
    """
    API interface for MLB predictions and data management
    """
    
    def __init__(self, db_manager: DatabaseManager, predictor: MLBPredictor):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.predictor = predictor
        self.baseball_scraper = BaseballSavantScraper()
        # Note: OddsShark dependency removed
    
    def get_todays_predictions(self, date: datetime = None) -> List[Dict]:
        """
        Get predictions for today's games
        
        Args:
            date: Date to get predictions for (defaults to today)
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if date is None:
                date = datetime.now()
            
            self.logger.info(f"Getting predictions for {date.date()}")
            
            # Get today's games from database
            todays_games = self._get_todays_games(date)
            
            # If no games found, try to fetch from MLB API
            if todays_games.empty:
                self.logger.info("No games in database, fetching from MLB API...")
                try:
                    from data_collectors.mlb_schedule_collector import MLBScheduleCollector
                    schedule_collector = MLBScheduleCollector()
                    fresh_games = schedule_collector.get_todays_games(date.strftime('%Y-%m-%d'))
                    
                    if not fresh_games.empty:
                        # Store the games in database
                        self.db_manager.store_games(fresh_games)
                        todays_games = fresh_games
                        self.logger.info(f"Fetched {len(fresh_games)} games from MLB API")
                    
                except Exception as e:
                    self.logger.error(f"Error fetching from MLB API: {str(e)}")
            
            if todays_games.empty:
                self.logger.warning(f"No games found for {date.date()}")
                return []
            
            # Enhance game data before generating predictions
            enhanced_games = self._enhance_game_data(todays_games)
            
            # Check if models need training and train if necessary
            if not self.predictor.is_trained:
                self.logger.info("Models not trained, attempting to train...")
                training_result = self._ensure_models_trained()
                if not training_result:
                    self.logger.warning("Model training failed, using basic predictions")
            
            # Generate predictions
            predictions = self.predictor.predict_multiple_games(enhanced_games)
            
            # Enhance predictions with additional data
            enhanced_predictions = []
            for prediction in predictions:
                enhanced_prediction = self._enhance_prediction(prediction, date)
                enhanced_predictions.append(enhanced_prediction)
            
            # Store predictions in database
            if enhanced_predictions:
                self.db_manager.store_predictions(enhanced_predictions)
            
            self.logger.info(f"Generated {len(enhanced_predictions)} predictions for {date.date()}")
            return enhanced_predictions
            
        except Exception as e:
            self.logger.error(f"Error getting today's predictions: {str(e)}")
            return []
    
    def _ensure_models_trained(self) -> bool:
        """
        Ensure models are trained by getting training data and training if needed
        
        Returns:
            True if models are trained successfully, False otherwise
        """
        try:
            self.logger.info("Ensuring models are trained...")
            
            # Get training data from database
            training_data = self.db_manager.get_training_data()
            
            if training_data.empty:
                self.logger.warning("No training data available in database")
                return False
            
            self.logger.info(f"Found {len(training_data)} training records")
            
            # Train models using pregame features
            training_results = self.predictor.train_models(training_data)
            
            if training_results and len(training_results) > 0:
                self.logger.info("Model training completed successfully")
                return True
            else:
                self.logger.error("Model training failed or returned empty results")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return False
    
    def get_team_prediction(self, home_team: str, away_team: str, game_date: datetime = None) -> Dict:
        """
        Get prediction for a specific matchup
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Date of the game (defaults to today)
            
        Returns:
            Prediction dictionary
        """
        try:
            if game_date is None:
                game_date = datetime.now()
            
            self.logger.info(f"Getting prediction for {away_team} @ {home_team} on {game_date.date()}")
            
            # Create game data
            game_data = pd.DataFrame([{
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'game_pk': f"{away_team}_{home_team}_{game_date.strftime('%Y%m%d')}"
            }])
            
            # Add recent team performance data
            enhanced_game_data = self._enhance_game_data(game_data)
            
            # Generate prediction
            prediction = self.predictor.predict_game(enhanced_game_data)
            
            if prediction:
                # Enhance with additional context
                enhanced_prediction = self._enhance_prediction(prediction, game_date)
                return enhanced_prediction
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting team prediction: {str(e)}")
            return {}
    
    def get_historical_predictions(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Get historical predictions from database
        
        Args:
            start_date: Start date for query
            end_date: End date for query
            
        Returns:
            List of historical prediction dictionaries
        """
        try:
            query = """
                SELECT *
                FROM predictions
                WHERE game_date BETWEEN ? AND ?
                ORDER BY game_date DESC, created_at DESC
            """
            
            with self.db_manager._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[start_date.date(), end_date.date()])
            
            # Convert to list of dictionaries
            predictions = []
            for _, row in df.iterrows():
                prediction = dict(row)
                
                # Parse key_factors if it's a JSON string
                if 'key_factors' in prediction and isinstance(prediction['key_factors'], str):
                    try:
                        prediction['key_factors'] = json.loads(prediction['key_factors'])
                    except:
                        prediction['key_factors'] = []
                
                predictions.append(prediction)
            
            self.logger.info(f"Retrieved {len(predictions)} historical predictions")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error getting historical predictions: {str(e)}")
            return []
    
    def update_predictions_with_results(self, date: datetime = None) -> Dict:
        """
        Update predictions with actual game results
        
        Args:
            date: Date to update (defaults to yesterday)
            
        Returns:
            Dictionary with update results
        """
        try:
            if date is None:
                date = datetime.now() - timedelta(days=1)
            
            self.logger.info(f"Updating predictions with results for {date.date()}")
            
            # Get predictions for the date
            predictions = self.get_historical_predictions(date, date)
            
            if not predictions:
                return {'updated': 0, 'message': 'No predictions found for date'}
            
            # Get actual results from database
            actual_results = self.db_manager.get_historical_games(date, date)
            
            if actual_results.empty:
                return {'updated': 0, 'message': 'No game results found for date'}
            
            # Match predictions with results and calculate accuracy
            updated_count = 0
            correct_predictions = 0
            
            for prediction in predictions:
                # Find matching game in results
                matching_game = actual_results[
                    (actual_results['home_team'] == prediction['home_team']) &
                    (actual_results['away_team'] == prediction['away_team'])
                ]
                
                if not matching_game.empty:
                    game = matching_game.iloc[0]
                    
                    # Check if prediction was correct
                    actual_winner = game['home_team'] if game['home_win'] == 1 else game['away_team']
                    prediction_correct = prediction['predicted_winner'] == actual_winner
                    
                    if prediction_correct:
                        correct_predictions += 1
                    
                    updated_count += 1
            
            accuracy = correct_predictions / updated_count if updated_count > 0 else 0
            
            result = {
                'updated': updated_count,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'date': date.date().isoformat()
            }
            
            self.logger.info(f"Updated {updated_count} predictions with {accuracy:.1%} accuracy")
            return result
            
        except Exception as e:
            self.logger.error(f"Error updating predictions with results: {str(e)}")
            return {'error': str(e)}
    
    def get_model_performance_summary(self) -> Dict:
        """
        Get summary of model performance
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get recent predictions (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            recent_predictions = self.get_historical_predictions(start_date, end_date)
            
            if not recent_predictions:
                return {'message': 'No recent predictions available'}
            
            # Get model metrics from predictor
            model_metrics = self.predictor.get_model_metrics()
            
            # Calculate recent performance
            recent_performance = self._calculate_recent_performance(recent_predictions)
            
            summary = {
                'model_metrics': model_metrics,
                'recent_performance': recent_performance,
                'total_predictions_30_days': len(recent_predictions),
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting model performance summary: {str(e)}")
            return {'error': str(e)}
    
    def retrain_models(self, days_back: int = 90) -> Dict:
        """
        Retrain models with recent data
        
        Args:
            days_back: Number of days of data to use for training
            
        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info(f"Retraining models with {days_back} days of data")
            
            # Get training data
            training_data = self.db_manager.get_training_data(days_back)
            
            if training_data.empty:
                return {'error': 'No training data available'}
            
            # Train models
            results = self.predictor.train_models(training_data)
            
            if results:
                # Store model metrics
                self.db_manager.store_model_metrics(results)
                
                return {
                    'success': True,
                    'training_samples': results.get('training_samples', 0),
                    'feature_count': results.get('feature_count', 0),
                    'timestamp': results.get('timestamp'),
                    'winner_model_metrics': results.get('winner_model', {}),
                    'total_model_metrics': results.get('total_model', {})
                }
            else:
                return {'error': 'Model training failed'}
                
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")
            return {'error': str(e)}
    
    def _get_todays_games(self, date: datetime) -> pd.DataFrame:
        """
        Get today's games from the database
        
        Args:
            date: Date to get games for
            
        Returns:
            DataFrame with today's games
        """
        try:
            # Get games from the games table for the specific date
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT * FROM games 
                    WHERE sport = 'MLB' AND date(game_date) = ?
                    ORDER BY game_id
                """
                todays_games = pd.read_sql_query(query, conn, params=[date.strftime('%Y-%m-%d')])
                
                if not todays_games.empty:
                    self.logger.info(f"Found {len(todays_games)} games in database for {date.date()}")
                    return todays_games
            
            self.logger.info(f"No games found in database for {date.date()}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting today's games: {str(e)}")
            return pd.DataFrame()
    
    def _enhance_game_data(self, game_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance game data with recent team performance
        
        Args:
            game_data: Basic game data
            
        Returns:
            Enhanced DataFrame with team stats
        """
        try:
            enhanced_data = game_data.copy()
            
            # Ensure proper column names for feature engineering
            if 'home_team_id' in enhanced_data.columns and 'home_team' not in enhanced_data.columns:
                enhanced_data['home_team'] = enhanced_data['home_team_id']
            if 'away_team_id' in enhanced_data.columns and 'away_team' not in enhanced_data.columns:
                enhanced_data['away_team'] = enhanced_data['away_team_id']
            
            for idx, game in game_data.iterrows():
                # Handle both possible column name formats
                home_team = game.get('home_team') or game.get('home_team_id')
                away_team = game.get('away_team') or game.get('away_team_id')
                
                if not home_team or not away_team:
                    self.logger.warning(f"Missing team data in game: {game}")
                    continue
                    
                # Get recent performance for both teams
                home_performance = self.db_manager.get_team_recent_performance(home_team, days=30)
                away_performance = self.db_manager.get_team_recent_performance(away_team, days=30)
                
                # Add basic team stats to the game data using the current row index
                if not home_performance.empty:
                    enhanced_data.loc[idx, 'home_recent_wins'] = home_performance['team_won'].sum()
                    enhanced_data.loc[idx, 'home_recent_games'] = len(home_performance)
                    enhanced_data.loc[idx, 'home_avg_score'] = home_performance['team_score'].mean()
                
                if not away_performance.empty:
                    enhanced_data.loc[idx, 'away_recent_wins'] = away_performance['team_won'].sum()
                    enhanced_data.loc[idx, 'away_recent_games'] = len(away_performance)
                    enhanced_data.loc[idx, 'away_avg_score'] = away_performance['team_score'].mean()
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error enhancing game data: {str(e)}")
            return game_data
    
    def _enhance_prediction(self, prediction: Dict, game_date: datetime) -> Dict:
        """
        Enhance prediction with additional context and data
        
        Args:
            prediction: Basic prediction dictionary
            game_date: Date of the game
            
        Returns:
            Enhanced prediction dictionary
        """
        try:
            enhanced = prediction.copy()
            
            # Add game time (would come from schedule data in real implementation)
            enhanced['game_time'] = self._get_estimated_game_time(game_date)
            
            # Add weather info (placeholder - would come from weather API)
            enhanced['weather'] = self._get_weather_info(game_date)
            
            # Note: Betting odds context removed - no longer using OddsShark
            
            # Add model confidence score
            enhanced['model_score'] = self._calculate_model_score(prediction)
            
            # Add recent trends
            trends = self._get_recent_trends(
                prediction.get('home_team'),
                prediction.get('away_team')
            )
            enhanced['recent_trends'] = trends
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing prediction: {str(e)}")
            return prediction
    
    def _get_estimated_game_time(self, game_date: datetime) -> str:
        """Get estimated game time (placeholder implementation)"""
        try:
            # Standard game times based on day of week
            if game_date.weekday() < 5:  # Weekday
                return "7:10 PM"
            else:  # Weekend
                return "1:10 PM" if game_date.weekday() == 5 else "1:10 PM"  # Saturday or Sunday
        except:
            return "TBD"
    
    def _get_weather_info(self, game_date: datetime) -> Dict:
        """Get weather information (placeholder implementation)"""
        return {
            'condition': 'Clear',
            'temperature': 72,
            'wind_speed': 5,
            'wind_direction': 'SSW'
        }
    
    # Note: _get_odds_context method removed - no longer using OddsShark odds data
    
    def _calculate_model_score(self, prediction: Dict) -> float:
        """Calculate overall model confidence score"""
        try:
            win_confidence = prediction.get('win_probability', 0.5)
            total_confidence = prediction.get('total_confidence', 0.5)
            
            # Weighted average (winner prediction weighted more heavily)
            model_score = (win_confidence * 0.7) + (total_confidence * 0.3)
            
            return model_score
            
        except:
            return 0.5
    
    def _get_recent_trends(self, home_team: str, away_team: str) -> List[str]:
        """Get recent trends for the teams"""
        try:
            trends = []
            
            # Get recent performance for both teams
            home_performance = self.db_manager.get_team_recent_performance(home_team, days=10)
            away_performance = self.db_manager.get_team_recent_performance(away_team, days=10)
            
            if not home_performance.empty:
                home_wins = home_performance['team_won'].sum()
                home_games = len(home_performance)
                trends.append(f"{home_team} is {home_wins}-{home_games-home_wins} in last {home_games} games")
            
            if not away_performance.empty:
                away_wins = away_performance['team_won'].sum()
                away_games = len(away_performance)
                trends.append(f"{away_team} is {away_wins}-{away_games-away_wins} in last {away_games} games")
            
            return trends[:3]  # Limit to top 3 trends
            
        except Exception as e:
            self.logger.error(f"Error getting recent trends: {str(e)}")
            return []
    
    def _calculate_recent_performance(self, predictions: List[Dict]) -> Dict:
        """Calculate performance metrics for recent predictions"""
        try:
            if not predictions:
                return {}
            
            # Basic metrics
            total_predictions = len(predictions)
            avg_win_confidence = np.mean([p.get('win_probability', 0) for p in predictions])
            avg_total_confidence = np.mean([p.get('total_confidence', 0) for p in predictions])
            
            return {
                'total_predictions': total_predictions,
                'avg_win_confidence': avg_win_confidence,
                'avg_total_confidence': avg_total_confidence,
                'prediction_frequency': total_predictions / 30  # Per day over 30 days
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating recent performance: {str(e)}")
            return {}
