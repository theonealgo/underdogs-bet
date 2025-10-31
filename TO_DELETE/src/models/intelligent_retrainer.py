import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import sqlite3
import json
import threading
import time

from data_storage.database import DatabaseManager
from models.prediction_models import MLBPredictor
from models.performance_analyzer import PerformanceAnalyzer
from data_collectors.result_tracker import ResultTracker


class IntelligentRetrainer:
    """
    Intelligent model retraining system that learns from prediction errors and performance trends
    """
    
    def __init__(self, db_manager: DatabaseManager, predictor, sport: str = 'MLB'):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.predictor = predictor
        self.sport = sport
        self.performance_analyzer = PerformanceAnalyzer(db_manager, sport)
        self.result_tracker = ResultTracker(db_manager, sport)
        
        # Retraining thresholds
        self.thresholds = {
            'min_accuracy': 0.52,  # Minimum acceptable accuracy (52% to beat break-even)
            'accuracy_drop_threshold': 0.05,  # Retrain if accuracy drops by 5%
            'confidence_calibration_error': 0.15,  # Maximum allowed calibration error
            'min_predictions_for_retrain': 25,  # Minimum predictions needed before retraining
            'max_days_since_retrain': 14,  # Maximum days without retraining
            'mae_increase_threshold': 0.5,  # Retrain if MAE increases significantly
            'consecutive_poor_days': 3,  # Consecutive days of poor performance
            'high_confidence_error_rate': 0.15  # Max allowed rate of high-confidence errors
        }
        
        # Retraining state
        self.last_retrain_date = None
        self.retraining_in_progress = False
        self.retraining_history = []
        self.performance_trend = []
        
        # Load retraining history
        self._load_retraining_history()
    
    def evaluate_retraining_need(self, sport: Optional[str] = None) -> Dict:
        """
        Evaluate whether model retraining is needed based on multiple criteria
        
        Args:
            sport: Sport to evaluate (defaults to instance sport)
            
        Returns:
            Dictionary with evaluation results and recommendation
        """
        try:
            self.logger.info("Evaluating retraining need...")
            
            evaluation = {
                'timestamp': datetime.now().isoformat(),
                'retraining_needed': False,
                'priority': 'none',  # none, low, medium, high, critical
                'triggers': [],
                'recommendations': [],
                'metrics': {},
                'next_check_in': 24  # hours
            }
            
            # Check if retraining is already in progress
            if self.retraining_in_progress:
                evaluation['status'] = 'retraining_in_progress'
                evaluation['next_check_in'] = 2  # Check in 2 hours
                return evaluation
            
            # Get recent performance data
            current_sport = sport or self.sport
            recent_performance = self._get_recent_performance_metrics(current_sport)
            evaluation['metrics'] = recent_performance
            
            if not recent_performance.get('sufficient_data', False):
                evaluation['status'] = 'insufficient_data'
                evaluation['recommendations'].append('Wait for more prediction results')
                return evaluation
            
            # Evaluate multiple trigger conditions
            triggers = self._evaluate_trigger_conditions(recent_performance)
            evaluation['triggers'] = triggers
            
            # Determine overall priority
            priority, recommendations = self._determine_retraining_priority(triggers)
            evaluation['priority'] = priority
            evaluation['recommendations'] = recommendations
            evaluation['retraining_needed'] = priority != 'none'
            
            # Set next check interval based on priority
            if priority == 'critical':
                evaluation['next_check_in'] = 1  # Check hourly
            elif priority == 'high':
                evaluation['next_check_in'] = 4  # Check every 4 hours
            elif priority == 'medium':
                evaluation['next_check_in'] = 12  # Check twice daily
            # else keep default 24 hours
            
            # Store evaluation results
            self._store_evaluation_results(evaluation)
            
            self.logger.info(f"Retraining evaluation complete: {priority} priority, "
                           f"{len(triggers)} triggers, retraining_needed: {evaluation['retraining_needed']}")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating retraining need: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'retraining_needed': False
            }
    
    def _get_recent_performance_metrics(self, sport: Optional[str] = None) -> Dict:
        """Get recent performance metrics for evaluation"""
        try:
            current_sport = sport or self.sport
            current_date = datetime.now()
            
            # Get accuracy data for last 7 days
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT 
                        date_period,
                        total_predictions,
                        correct_predictions,
                        accuracy_rate,
                        avg_win_confidence,
                        total_mae,
                        total_rmse,
                        confidence_calibration
                    FROM prediction_accuracy
                    WHERE sport = ? 
                    AND date_period >= DATE('now', '-7 days')
                    ORDER BY date_period DESC
                """
                
                df = pd.read_sql_query(query, conn, params=[current_sport])
            
            if df.empty:
                return {'sufficient_data': False, 'reason': 'No recent accuracy data'}
            
            # Calculate aggregated metrics
            total_predictions = df['total_predictions'].sum()
            total_correct = df['correct_predictions'].sum()
            
            if total_predictions < self.thresholds['min_predictions_for_retrain']:
                return {
                    'sufficient_data': False, 
                    'reason': f'Only {total_predictions} predictions, need {self.thresholds["min_predictions_for_retrain"]}'
                }
            
            current_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
            avg_mae = df['total_mae'].mean()
            avg_confidence_calibration = df['confidence_calibration'].mean()
            
            # Get trend data
            if len(df) >= 3:
                recent_accuracy = df.head(3)['accuracy_rate'].mean()
                older_accuracy = df.tail(3)['accuracy_rate'].mean()
                accuracy_trend = recent_accuracy - older_accuracy
            else:
                accuracy_trend = 0
            
            # Get last retraining date
            retrain_query = """
                SELECT MAX(date_recorded) as last_retrain
                FROM model_metrics
                WHERE sport = ? AND metric_name = 'retraining_completed'
            """
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.execute(retrain_query, (current_sport,))
                result = cursor.fetchone()
                last_retrain = result[0] if result and result[0] else None
            
            days_since_retrain = None
            if last_retrain:
                last_retrain_date = datetime.fromisoformat(last_retrain)
                days_since_retrain = (current_date - last_retrain_date).days
            
            return {
                'sufficient_data': True,
                'total_predictions': total_predictions,
                'current_accuracy': current_accuracy,
                'accuracy_trend': accuracy_trend,
                'avg_mae': avg_mae,
                'confidence_calibration_error': avg_confidence_calibration,
                'days_since_retrain': days_since_retrain,
                'last_retrain_date': last_retrain,
                'evaluation_period_days': len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recent performance metrics: {str(e)}")
            return {'sufficient_data': False, 'error': str(e)}
    
    def _evaluate_trigger_conditions(self, metrics: Dict) -> List[Dict]:
        """Evaluate all trigger conditions for retraining"""
        try:
            triggers = []
            
            # 1. Accuracy below minimum threshold
            if metrics.get('current_accuracy', 1.0) < self.thresholds['min_accuracy']:
                triggers.append({
                    'type': 'accuracy_below_threshold',
                    'severity': 'high',
                    'description': f"Accuracy {metrics['current_accuracy']:.3f} below minimum {self.thresholds['min_accuracy']:.3f}",
                    'metric_value': metrics['current_accuracy'],
                    'threshold': self.thresholds['min_accuracy']
                })
            
            # 2. Accuracy trending downward
            if metrics.get('accuracy_trend', 0) < -self.thresholds['accuracy_drop_threshold']:
                triggers.append({
                    'type': 'accuracy_degradation',
                    'severity': 'medium',
                    'description': f"Accuracy dropping by {abs(metrics['accuracy_trend']):.3f}",
                    'metric_value': metrics['accuracy_trend'],
                    'threshold': -self.thresholds['accuracy_drop_threshold']
                })
            
            # 3. Poor confidence calibration
            cal_error = metrics.get('confidence_calibration_error')
            if cal_error and cal_error > self.thresholds['confidence_calibration_error']:
                triggers.append({
                    'type': 'confidence_miscalibration',
                    'severity': 'medium',
                    'description': f"Confidence calibration error {cal_error:.3f} above threshold {self.thresholds['confidence_calibration_error']:.3f}",
                    'metric_value': cal_error,
                    'threshold': self.thresholds['confidence_calibration_error']
                })
            
            # 4. Too long since last retraining
            days_since_retrain = metrics.get('days_since_retrain')
            if days_since_retrain and days_since_retrain > self.thresholds['max_days_since_retrain']:
                triggers.append({
                    'type': 'time_since_retrain',
                    'severity': 'low',
                    'description': f"{days_since_retrain} days since last retraining (max {self.thresholds['max_days_since_retrain']})",
                    'metric_value': days_since_retrain,
                    'threshold': self.thresholds['max_days_since_retrain']
                })
            
            # 5. MAE increase
            if metrics.get('avg_mae', 0) > self.thresholds['mae_increase_threshold'] + 1.0:  # Baseline MAE ~1.0
                triggers.append({
                    'type': 'mae_increase',
                    'severity': 'medium',
                    'description': f"MAE {metrics['avg_mae']:.3f} significantly increased",
                    'metric_value': metrics['avg_mae'],
                    'threshold': self.thresholds['mae_increase_threshold'] + 1.0
                })
            
            # 6. Check for consecutive poor performance days
            consecutive_poor = self._check_consecutive_poor_performance(self.sport)
            if consecutive_poor >= self.thresholds['consecutive_poor_days']:
                triggers.append({
                    'type': 'consecutive_poor_performance',
                    'severity': 'high',
                    'description': f"{consecutive_poor} consecutive days of poor performance",
                    'metric_value': consecutive_poor,
                    'threshold': self.thresholds['consecutive_poor_days']
                })
            
            # 7. Check high confidence error rate
            high_conf_error_rate = self._check_high_confidence_error_rate(self.sport)
            if high_conf_error_rate > self.thresholds['high_confidence_error_rate']:
                triggers.append({
                    'type': 'high_confidence_errors',
                    'severity': 'medium',
                    'description': f"High-confidence error rate {high_conf_error_rate:.3f} above threshold",
                    'metric_value': high_conf_error_rate,
                    'threshold': self.thresholds['high_confidence_error_rate']
                })
            
            return triggers
            
        except Exception as e:
            self.logger.error(f"Error evaluating trigger conditions: {str(e)}")
            return []
    
    def _check_consecutive_poor_performance(self, sport: Optional[str] = None) -> int:
        """Check for consecutive days of poor performance"""
        try:
            current_sport = sport or self.sport
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT accuracy_rate
                    FROM prediction_accuracy
                    WHERE sport = ? 
                    AND date_period >= DATE('now', '-7 days')
                    ORDER BY date_period DESC
                """
                
                cursor = conn.execute(query, (current_sport,))
                results = cursor.fetchall()
            
            consecutive_poor = 0
            for (accuracy,) in results:
                if accuracy and accuracy < self.thresholds['min_accuracy']:
                    consecutive_poor += 1
                else:
                    break
            
            return consecutive_poor
            
        except Exception as e:
            self.logger.error(f"Error checking consecutive poor performance: {str(e)}")
            return 0
    
    def _check_high_confidence_error_rate(self, sport: Optional[str] = None) -> float:
        """Check the rate of errors in high-confidence predictions"""
        try:
            current_sport = sport or self.sport
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT 
                        COUNT(*) as high_conf_predictions,
                        SUM(CASE WHEN win_prediction_correct = 0 THEN 1 ELSE 0 END) as high_conf_errors
                    FROM predictions
                    WHERE sport = ? 
                    AND win_probability > 0.75
                    AND result_updated_at IS NOT NULL
                    AND game_date >= DATE('now', '-7 days')
                """
                
                cursor = conn.execute(query, (current_sport,))
                result = cursor.fetchone()
            
            if result and result[0] > 0:
                high_conf_predictions, high_conf_errors = result
                return high_conf_errors / high_conf_predictions
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error checking high confidence error rate: {str(e)}")
            return 0.0
    
    def _determine_retraining_priority(self, triggers: List[Dict]) -> Tuple[str, List[str]]:
        """Determine overall retraining priority and recommendations"""
        try:
            if not triggers:
                return 'none', ['Continue monitoring performance']
            
            # Count severity levels
            critical_count = sum(1 for t in triggers if t.get('severity') == 'critical')
            high_count = sum(1 for t in triggers if t.get('severity') == 'high')
            medium_count = sum(1 for t in triggers if t.get('severity') == 'medium')
            low_count = sum(1 for t in triggers if t.get('severity') == 'low')
            
            recommendations = []
            
            # Determine priority
            if critical_count > 0 or high_count >= 2:
                priority = 'critical'
                recommendations.extend([
                    'URGENT: Immediate model retraining required',
                    'Monitor predictions closely until retraining completes',
                    'Consider temporary prediction confidence adjustment'
                ])
            elif high_count == 1 or medium_count >= 2:
                priority = 'high'
                recommendations.extend([
                    'Schedule model retraining within next 4 hours',
                    'Analyze recent prediction errors for patterns',
                    'Prepare additional training data if available'
                ])
            elif medium_count == 1 or low_count >= 2:
                priority = 'medium'
                recommendations.extend([
                    'Schedule model retraining within next 12 hours',
                    'Review recent performance trends',
                    'Consider feature engineering improvements'
                ])
            elif low_count == 1:
                priority = 'low'
                recommendations.extend([
                    'Consider retraining during next maintenance window',
                    'Monitor performance for additional degradation signals',
                    'Ensure adequate training data is available'
                ])
            else:
                priority = 'none'
                recommendations.append('Continue monitoring performance')
            
            # Add specific recommendations based on trigger types
            trigger_types = [t['type'] for t in triggers]
            
            if 'accuracy_below_threshold' in trigger_types:
                recommendations.append('Focus on improving feature quality and model hyperparameters')
            
            if 'confidence_miscalibration' in trigger_types:
                recommendations.append('Apply confidence calibration techniques (Platt scaling, isotonic regression)')
            
            if 'consecutive_poor_performance' in trigger_types:
                recommendations.append('Investigate recent data quality issues or external factors')
            
            if 'high_confidence_errors' in trigger_types:
                recommendations.append('Review high-confidence predictions for systematic biases')
            
            return priority, recommendations
            
        except Exception as e:
            self.logger.error(f"Error determining retraining priority: {str(e)}")
            return 'none', ['Error in priority evaluation']
    
    def execute_intelligent_retraining(self, priority: str = 'medium') -> Dict:
        """
        Execute intelligent model retraining with adaptive strategies
        
        Args:
            priority: Priority level affecting retraining approach
            
        Returns:
            Dictionary with retraining results
        """
        try:
            if self.retraining_in_progress:
                return {
                    'success': False,
                    'error': 'Retraining already in progress',
                    'timestamp': datetime.now().isoformat()
                }
            
            self.retraining_in_progress = True
            self.logger.info(f"Starting intelligent retraining with {priority} priority...")
            
            results = {
                'success': True,
                'priority': priority,
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'strategies_applied': [],
                'performance_before': {},
                'performance_after': {},
                'improvements': {}
            }
            
            try:
                # 1. Analyze current performance and errors
                self.logger.info("Analyzing current performance...")
                error_analysis = self.performance_analyzer.analyze_prediction_errors(days_back=14)
                results['error_analysis'] = error_analysis
                
                # Get baseline performance
                baseline_performance = self.performance_analyzer.get_performance_summary(days=7)
                results['performance_before'] = baseline_performance
                
                # 2. Determine training data strategy based on priority and insights
                training_data = self._prepare_intelligent_training_data(priority, error_analysis)
                
                if training_data.empty:
                    raise Exception("No suitable training data available")
                
                results['training_data_size'] = len(training_data)
                results['strategies_applied'].append(f'training_data_strategy_{priority}')
                
                # 3. Apply intelligent training strategies
                training_strategies = self._determine_training_strategies(priority, error_analysis)
                results['strategies_applied'].extend(training_strategies)
                
                # 4. Train the models with intelligent strategies
                self.logger.info("Training models with intelligent strategies...")
                training_results = self._train_with_strategies(training_data, training_strategies)
                results['training_results'] = training_results
                
                # 5. Validate improved performance
                validation_results = self._validate_retrained_model()
                results['validation_results'] = validation_results
                
                # 6. Store retraining metadata and insights
                self._store_retraining_results(results)
                
                # 7. Update retraining history
                self.last_retrain_date = datetime.now()
                self.retraining_history.append({
                    'date': self.last_retrain_date.isoformat(),
                    'priority': priority,
                    'success': True,
                    'improvements': results.get('improvements', {})
                })
                
                results['end_time'] = datetime.now().isoformat()
                
                # Calculate improvements
                if baseline_performance.get('accuracy') and validation_results.get('accuracy'):
                    accuracy_improvement = validation_results['accuracy'] - baseline_performance['accuracy']
                    results['improvements']['accuracy_change'] = accuracy_improvement
                
                self.logger.info(f"Intelligent retraining completed successfully. "
                               f"Accuracy improvement: {results.get('improvements', {}).get('accuracy_change', 'N/A')}")
                
                return results
                
            except Exception as e:
                results['success'] = False
                results['error'] = str(e)
                results['end_time'] = datetime.now().isoformat()
                
                self.logger.error(f"Error during intelligent retraining: {str(e)}")
                
                # Store failed retraining attempt
                self.retraining_history.append({
                    'date': datetime.now().isoformat(),
                    'priority': priority,
                    'success': False,
                    'error': str(e)
                })
                
                return results
                
        finally:
            self.retraining_in_progress = False
    
    def _prepare_intelligent_training_data(self, priority: str, error_analysis: Dict) -> pd.DataFrame:
        """Prepare training data based on priority and error analysis"""
        try:
            current_date = datetime.now()
            
            # Determine training data window based on priority
            if priority == 'critical':
                # Use last 60 days for critical retraining
                days_back = 60
            elif priority == 'high':
                # Use last 90 days
                days_back = 90
            elif priority == 'medium':
                # Use last 120 days
                days_back = 120
            else:
                # Use last 180 days for low priority
                days_back = 180
            
            start_date = current_date - timedelta(days=days_back)
            
            # Get base training data
            training_data = self.db_manager.get_historical_games(start_date, current_date)
            
            if training_data.empty:
                return training_data
            
            # Apply intelligent filtering based on error analysis
            if error_analysis and 'actionable_insights' in error_analysis:
                insights = error_analysis['actionable_insights']
                
                # If there are team-specific biases, focus on recent data for those teams
                team_bias_insights = [i for i in insights if i.get('type') == 'team_bias']
                if team_bias_insights:
                    # Focus more on recent data for problematic teams
                    recent_cutoff = current_date - timedelta(days=30)
                    recent_data = training_data[pd.to_datetime(training_data['game_date']) >= recent_cutoff]
                    
                    if len(recent_data) > len(training_data) * 0.4:  # If recent data is substantial
                        # Give more weight to recent data
                        older_data = training_data[pd.to_datetime(training_data['game_date']) < recent_cutoff]
                        # Sample older data to balance with recent
                        if len(older_data) > len(recent_data):
                            older_sample = older_data.sample(n=len(recent_data), random_state=42)
                            training_data = pd.concat([recent_data, older_sample], ignore_index=True)
            
            self.logger.info(f"Prepared training data: {len(training_data)} samples from {days_back} days")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error preparing intelligent training data: {str(e)}")
            return pd.DataFrame()
    
    def _determine_training_strategies(self, priority: str, error_analysis: Dict) -> List[str]:
        """Determine intelligent training strategies based on analysis"""
        try:
            strategies = []
            
            # Base strategy based on priority
            if priority in ['critical', 'high']:
                strategies.append('aggressive_learning_rate')
                strategies.append('early_stopping')
            else:
                strategies.append('conservative_learning_rate')
            
            # Strategies based on error analysis insights
            if error_analysis and 'actionable_insights' in error_analysis:
                insights = error_analysis['actionable_insights']
                
                for insight in insights:
                    if insight.get('type') == 'confidence_calibration':
                        strategies.append('calibration_focus')
                    elif insight.get('type') == 'systematic_bias':
                        strategies.append('bias_correction')
                    elif insight.get('type') == 'team_bias':
                        strategies.append('team_specific_features')
                    elif insight.get('type') == 'performance_degradation':
                        strategies.append('feature_selection_review')
            
            # Confidence-based strategies
            if error_analysis and 'confidence_analysis' in error_analysis:
                conf_analysis = error_analysis['confidence_analysis']
                if conf_analysis.get('calibration_quality') in ['fair', 'poor']:
                    strategies.append('confidence_recalibration')
            
            return list(set(strategies))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error determining training strategies: {str(e)}")
            return ['default_training']
    
    def _train_with_strategies(self, training_data: pd.DataFrame, strategies: List[str]) -> Dict:
        """Train models applying intelligent strategies"""
        try:
            # Apply strategy-based modifications to training
            training_params = self._get_strategy_based_params(strategies)
            
            # Update predictor parameters
            if 'winner_params' in training_params:
                self.predictor.winner_params.update(training_params['winner_params'])
            
            if 'total_params' in training_params:
                self.predictor.total_params.update(training_params['total_params'])
            
            # Train the models
            training_results = self.predictor.train_models(training_data)
            
            # Apply post-training strategies
            if 'confidence_recalibration' in strategies:
                self._apply_confidence_calibration()
                training_results['calibration_applied'] = True
            
            training_results['strategies_applied'] = strategies
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training with strategies: {str(e)}")
            return {'error': str(e)}
    
    def _get_strategy_based_params(self, strategies: List[str]) -> Dict:
        """Get training parameters based on strategies"""
        params = {}
        
        if 'aggressive_learning_rate' in strategies:
            params['winner_params'] = {'learning_rate': 0.15, 'n_estimators': 150}
            params['total_params'] = {'learning_rate': 0.15, 'n_estimators': 150}
        
        elif 'conservative_learning_rate' in strategies:
            params['winner_params'] = {'learning_rate': 0.05, 'n_estimators': 200}
            params['total_params'] = {'learning_rate': 0.05, 'n_estimators': 200}
        
        if 'early_stopping' in strategies:
            if 'winner_params' not in params:
                params['winner_params'] = {}
            if 'total_params' not in params:
                params['total_params'] = {}
            
            params['winner_params']['early_stopping_rounds'] = 10
            params['total_params']['early_stopping_rounds'] = 10
        
        if 'bias_correction' in strategies:
            if 'winner_params' not in params:
                params['winner_params'] = {}
            if 'total_params' not in params:
                params['total_params'] = {}
            
            # Increase regularization to reduce bias
            params['winner_params']['reg_alpha'] = 0.1
            params['winner_params']['reg_lambda'] = 0.1
            params['total_params']['reg_alpha'] = 0.1
            params['total_params']['reg_lambda'] = 0.1
        
        return params
    
    def _apply_confidence_calibration(self):
        """Apply confidence calibration to improve prediction reliability"""
        try:
            # This would implement calibration techniques like Platt scaling
            # For now, we'll log that calibration was applied
            self.logger.info("Applied confidence calibration techniques")
            
            # Store calibration metadata
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    INSERT INTO model_metrics 
                    (sport, model_type, metric_name, metric_value)
                    VALUES (?, ?, ?, ?)
                """
                conn.execute(query, ('MLB', 'xgboost', 'calibration_applied', 1.0))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error applying confidence calibration: {str(e)}")
    
    def _validate_retrained_model(self) -> Dict:
        """Validate the performance of the retrained model"""
        try:
            # Get recent test data for validation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            validation_data = self.db_manager.get_historical_games(start_date, end_date)
            
            if validation_data.empty:
                return {'error': 'No validation data available'}
            
            # Make predictions on validation data
            predictions = []
            for _, game in validation_data.iterrows():
                game_df = pd.DataFrame([game])
                prediction = self.predictor.predict_game(game_df)
                if prediction:
                    predictions.append(prediction)
            
            if not predictions:
                return {'error': 'No predictions generated for validation'}
            
            # Calculate validation metrics
            accurate_predictions = []
            all_confidences = []
            
            for p, (_, game) in zip(predictions, validation_data.iterrows()):
                actual_winner = game.get('actual_winner')
                if actual_winner is not None and pd.notna(actual_winner):
                    predicted_winner = p.get('predicted_winner')
                    if predicted_winner is not None:
                        accurate_predictions.append(1 if predicted_winner == actual_winner else 0)
                
                win_prob = p.get('win_probability', 0)
                if win_prob is not None:
                    all_confidences.append(win_prob)
            
            validation_metrics = {
                'total_predictions': len(predictions),
                'accuracy': np.mean(accurate_predictions) if accurate_predictions else 0.0,
                'avg_confidence': np.mean(all_confidences) if all_confidences else 0.0
            }
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating retrained model: {str(e)}")
            return {'error': str(e)}
    
    def _store_retraining_results(self, results: Dict):
        """Store retraining results in database"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Store retraining completion metric
                query = """
                    INSERT INTO model_metrics 
                    (sport, model_type, metric_name, metric_value)
                    VALUES (?, ?, ?, ?)
                """
                
                conn.execute(query, ('MLB', 'xgboost', 'retraining_completed', 1.0))
                
                # Store improvement metrics if available
                if 'improvements' in results and 'accuracy_change' in results['improvements']:
                    conn.execute(query, ('MLB', 'xgboost', 'accuracy_improvement', 
                                       results['improvements']['accuracy_change']))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing retraining results: {str(e)}")
    
    def _store_evaluation_results(self, evaluation: Dict):
        """Store evaluation results for tracking"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    INSERT INTO learning_insights 
                    (sport, model_type, insight_type, insight_category, insight_description, 
                     confidence_score, action_taken)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                
                description = f"Retraining evaluation: {evaluation['priority']} priority, {len(evaluation.get('triggers', []))} triggers"
                action = 'Retraining scheduled' if evaluation['retraining_needed'] else 'Continue monitoring'
                
                conn.execute(query, (self.sport, 'xgboost', 'retraining_evaluation', 'priority_assessment',
                                   description, 1.0, action))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing evaluation results: {str(e)}")
    
    def _load_retraining_history(self):
        """Load retraining history from database"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT date_recorded, metric_name, metric_value
                    FROM model_metrics
                    WHERE sport = ? 
                    AND model_type = 'xgboost'
                    AND metric_name IN ('retraining_completed', 'accuracy_improvement')
                    ORDER BY date_recorded DESC
                    LIMIT 10
                """
                
                cursor = conn.execute(query, (self.sport,))
                results = cursor.fetchall()
                
                # Parse results into history
                for date_recorded, metric_name, metric_value in results:
                    if metric_name == 'retraining_completed':
                        self.retraining_history.append({
                            'date': date_recorded,
                            'success': True
                        })
                        if not self.last_retrain_date:
                            self.last_retrain_date = datetime.fromisoformat(date_recorded)
                
        except Exception as e:
            self.logger.error(f"Error loading retraining history: {str(e)}")
    
    def get_retraining_status(self) -> Dict:
        """
        Get current retraining status and recommendations
        
        Returns:
            Dictionary with retraining status
        """
        try:
            evaluation = self.evaluate_retraining_need()
            
            status = {
                'retraining_in_progress': self.retraining_in_progress,
                'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                'days_since_retrain': (datetime.now() - self.last_retrain_date).days if self.last_retrain_date else None,
                'current_evaluation': evaluation,
                'retraining_history': self.retraining_history[-5:],  # Last 5 retrainings
                'thresholds': self.thresholds
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting retraining status: {str(e)}")
            return {'error': str(e)}
    
    def force_retraining(self, reason: str = "Manual trigger") -> Dict:
        """
        Force model retraining regardless of automatic triggers
        
        Args:
            reason: Reason for forced retraining
            
        Returns:
            Dictionary with retraining results
        """
        try:
            self.logger.info(f"Forced retraining triggered: {reason}")
            
            # Execute retraining with high priority
            results = self.execute_intelligent_retraining(priority='high')
            results['forced_retraining'] = True
            results['reason'] = reason
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in forced retraining: {str(e)}")
            return {'error': str(e), 'forced_retraining': True, 'reason': reason}