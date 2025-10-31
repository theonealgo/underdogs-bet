import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import sqlite3
import json
from collections import defaultdict

from data_storage.database import DatabaseManager


class PerformanceAnalyzer:
    """
    Advanced performance analysis and error pattern detection for MLB predictions
    """
    
    def __init__(self, db_manager: DatabaseManager, sport: str = 'MLB'):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.sport = sport
        
    def analyze_prediction_errors(self, days_back: int = 30, sport: Optional[str] = None) -> Dict:
        """
        Perform comprehensive error analysis on recent predictions
        
        Args:
            days_back: Number of days to analyze
            sport: Sport to analyze (defaults to instance sport)
            
        Returns:
            Dictionary with detailed error analysis
        """
        try:
            current_sport = sport or self.sport
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            self.logger.info(f"Analyzing {current_sport} prediction errors for last {days_back} days")
            
            analysis = {
                'analysis_period': {
                    'start_date': start_date.date().isoformat(),
                    'end_date': end_date.date().isoformat(),
                    'days': days_back
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Get predictions with results
            predictions_df = self._get_predictions_with_results(start_date, end_date, current_sport)
            
            if predictions_df.empty:
                analysis['error'] = 'No predictions with results found for analysis period'
                return analysis
            
            analysis['data_summary'] = {
                'total_predictions': len(predictions_df),
                'predictions_with_results': len(predictions_df[predictions_df['result_updated_at'].notna()]),
                'win_predictions': len(predictions_df[predictions_df['predicted_winner'].notna()]),
                'total_predictions': len(predictions_df[predictions_df['predicted_total'].notna()])
            }
            
            # Analyze win prediction errors
            win_analysis = self._analyze_win_prediction_errors(predictions_df)
            analysis['win_prediction_analysis'] = win_analysis
            
            # Analyze total prediction errors
            total_analysis = self._analyze_total_prediction_errors(predictions_df)
            analysis['total_prediction_analysis'] = total_analysis
            
            # Analyze confidence calibration
            confidence_analysis = self._analyze_confidence_calibration(predictions_df)
            analysis['confidence_analysis'] = confidence_analysis
            
            # Analyze team-specific patterns
            team_analysis = self._analyze_team_performance_patterns(predictions_df)
            analysis['team_analysis'] = team_analysis
            
            # Analyze temporal patterns
            temporal_analysis = self._analyze_temporal_patterns(predictions_df)
            analysis['temporal_analysis'] = temporal_analysis
            
            # Generate actionable insights
            insights = self._generate_actionable_insights(analysis)
            analysis['actionable_insights'] = insights
            
            # Store analysis results
            self._store_analysis_results(analysis)
            
            self.logger.info("Prediction error analysis completed successfully")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction errors: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_predictions_with_results(self, start_date: datetime, end_date: datetime, sport: Optional[str] = None) -> pd.DataFrame:
        """Get predictions with actual results for analysis"""
        try:
            current_sport = sport or self.sport
            query = """
                SELECT 
                    p.*,
                    g.status as game_status,
                    g.home_score as game_home_score,
                    g.away_score as game_away_score
                FROM predictions p
                LEFT JOIN games g ON p.game_id = g.game_id AND p.sport = g.sport
                WHERE p.sport = ?
                AND p.game_date >= DATE(?)
                AND p.game_date <= DATE(?)
                AND p.result_updated_at IS NOT NULL
                ORDER BY p.game_date DESC
            """
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[current_sport, start_date.date(), end_date.date()])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting predictions with results: {str(e)}")
            return pd.DataFrame()
    
    def _analyze_win_prediction_errors(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in win prediction errors"""
        try:
            win_df = df[df['predicted_winner'].notna() & df['actual_winner'].notna()].copy()
            
            if win_df.empty:
                return {'error': 'No win predictions with results'}
            
            # Basic accuracy metrics
            total_predictions = len(win_df)
            correct_predictions = len(win_df[win_df['win_prediction_correct'] == 1])
            accuracy = correct_predictions / total_predictions
            
            analysis = {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'error_rate': 1 - accuracy
            }
            
            # Analyze confidence vs accuracy correlation
            win_df['confidence_bin'] = pd.cut(win_df['win_probability'], 
                                            bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                            labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
            
            confidence_accuracy = win_df.groupby('confidence_bin').agg({
                'win_prediction_correct': ['count', 'sum', 'mean'],
                'win_probability': 'mean'
            }).round(3)
            
            analysis['confidence_breakdown'] = confidence_accuracy.to_dict()
            
            # Identify home field advantage bias
            home_predictions = win_df[win_df['predicted_winner'] == 1]  # Predicted home win
            away_predictions = win_df[win_df['predicted_winner'] == 0]  # Predicted away win
            
            if len(home_predictions) > 0:
                home_accuracy = len(home_predictions[home_predictions['win_prediction_correct'] == 1]) / len(home_predictions)
                analysis['home_prediction_accuracy'] = home_accuracy
            
            if len(away_predictions) > 0:
                away_accuracy = len(away_predictions[away_predictions['win_prediction_correct'] == 1]) / len(away_predictions)
                analysis['away_prediction_accuracy'] = away_accuracy
            
            # Identify most problematic predictions (high confidence, wrong prediction)
            wrong_high_confidence = win_df[
                (win_df['win_prediction_correct'] == 0) & 
                (win_df['win_probability'] > 0.75)
            ]
            
            if len(wrong_high_confidence) > 0:
                # Get examples as DataFrame to ensure proper typing
                examples_df = wrong_high_confidence[['game_date', 'home_team_id', 'away_team_id', 
                                                    'win_probability', 'predicted_winner', 'actual_winner']]
                examples_list = examples_df.head(5).to_dict('records')
                
                analysis['high_confidence_errors'] = {
                    'count': len(wrong_high_confidence),
                    'percentage_of_total': len(wrong_high_confidence) / total_predictions,
                    'average_confidence': wrong_high_confidence['win_probability'].mean(),
                    'examples': examples_list
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing win prediction errors: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_total_prediction_errors(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in total runs prediction errors"""
        try:
            total_df = df[df['predicted_total'].notna() & df['actual_total'].notna()].copy()
            
            if total_df.empty:
                return {'error': 'No total predictions with results'}
            
            # Calculate error metrics
            errors = total_df['total_prediction_error']
            abs_errors = total_df['total_absolute_error']
            
            analysis = {
                'total_predictions': len(total_df),
                'mean_error': float(errors.mean()),
                'mean_absolute_error': float(abs_errors.mean()),
                'root_mean_square_error': float(np.sqrt((errors ** 2).mean())),
                'median_absolute_error': float(abs_errors.median()),
                'std_error': float(errors.std())
            }
            
            # Analyze over/under patterns
            over_predictions = total_df[total_df['total_prediction_error'] > 0]
            under_predictions = total_df[total_df['total_prediction_error'] < 0]
            
            analysis['over_under_bias'] = {
                'over_predictions': len(over_predictions),
                'under_predictions': len(under_predictions),
                'over_percentage': len(over_predictions) / len(total_df),
                'under_percentage': len(under_predictions) / len(total_df),
                'avg_over_error': over_predictions['total_prediction_error'].mean() if len(over_predictions) > 0 else 0,
                'avg_under_error': under_predictions['total_prediction_error'].mean() if len(under_predictions) > 0 else 0
            }
            
            # Analyze error distribution by predicted total range
            total_df['predicted_total_bin'] = pd.cut(total_df['predicted_total'], 
                                                   bins=[0, 7, 8.5, 10, 12, 20], 
                                                   labels=['Under 7', '7-8.5', '8.5-10', '10-12', 'Over 12'])
            
            bin_analysis = total_df.groupby('predicted_total_bin').agg({
                'total_absolute_error': ['count', 'mean', 'std'],
                'total_prediction_error': 'mean'
            }).round(3)
            
            analysis['prediction_range_breakdown'] = bin_analysis.to_dict()
            
            # Identify worst predictions (highest absolute error)  
            worst_predictions_df = total_df.nlargest(5, 'total_absolute_error')
            worst_examples = worst_predictions_df[['game_date', 'home_team_id', 'away_team_id',
                                                  'predicted_total', 'actual_total', 
                                                  'total_absolute_error']]
            analysis['worst_predictions'] = worst_examples.to_dict('records')
            
            # Check for systematic biases by teams
            team_errors = defaultdict(list)
            for _, row in total_df.iterrows():
                for team in [row['home_team_id'], row['away_team_id']]:
                    team_errors[team].append(row['total_prediction_error'])
            
            team_bias_analysis = {}
            for team, errors_list in team_errors.items():
                if len(errors_list) >= 3:  # At least 3 predictions for this team
                    avg_error = np.mean(errors_list)
                    if abs(avg_error) > 0.5:  # Significant bias
                        team_bias_analysis[team] = {
                            'prediction_count': len(errors_list),
                            'average_error': avg_error,
                            'bias_direction': 'over' if avg_error > 0 else 'under'
                        }
            
            analysis['team_total_biases'] = team_bias_analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing total prediction errors: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_confidence_calibration(self, df: pd.DataFrame) -> Dict:
        """Analyze how well prediction confidence matches actual accuracy"""
        try:
            win_df = df[df['predicted_winner'].notna() & df['actual_winner'].notna()].copy()
            
            if win_df.empty:
                return {'error': 'No win predictions for confidence analysis'}
            
            # Create confidence bins
            confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            win_df['confidence_bin'] = pd.cut(win_df['win_probability'], bins=confidence_bins)
            
            calibration_data = []
            confidence_categories = win_df['confidence_bin'].cat.categories
            for bin_range in confidence_categories:
                bin_data = win_df[win_df['confidence_bin'] == bin_range]
                if len(bin_data) > 0:
                    avg_confidence = bin_data['win_probability'].mean()
                    actual_accuracy = bin_data['win_prediction_correct'].mean()
                    calibration_error = abs(avg_confidence - actual_accuracy)
                    
                    calibration_data.append({
                        'confidence_range': str(bin_range),
                        'prediction_count': len(bin_data),
                        'average_confidence': avg_confidence,
                        'actual_accuracy': actual_accuracy,
                        'calibration_error': calibration_error
                    })
            
            # Calculate overall calibration metrics
            overall_calibration_error = np.mean([item['calibration_error'] for item in calibration_data])
            
            # Calculate Brier score (calibration metric)
            brier_score = np.mean((win_df['win_probability'] - win_df['win_prediction_correct']) ** 2)
            
            analysis = {
                'calibration_by_bin': calibration_data,
                'overall_calibration_error': overall_calibration_error,
                'brier_score': brier_score,
                'total_predictions': len(win_df)
            }
            
            # Reliability metrics
            if overall_calibration_error < 0.05:
                analysis['calibration_quality'] = 'excellent'
            elif overall_calibration_error < 0.10:
                analysis['calibration_quality'] = 'good'
            elif overall_calibration_error < 0.15:
                analysis['calibration_quality'] = 'fair'
            else:
                analysis['calibration_quality'] = 'poor'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing confidence calibration: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_team_performance_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze prediction accuracy by team"""
        try:
            team_stats: Dict[str, Dict] = defaultdict(lambda: {
                'predictions': 0, 'correct': 0, 'total_errors': [], 'win_confidences': []
            })
            
            # Aggregate team statistics
            for _, row in df.iterrows():
                for team in [row['home_team_id'], row['away_team_id']]:
                    stats = team_stats[team]
                    stats['predictions'] += 1
                    
                    win_correct = row['win_prediction_correct']
                    if pd.notna(win_correct) and not pd.isna(win_correct):
                        stats['correct'] += int(win_correct)
                    
                    total_error = row['total_absolute_error']
                    if pd.notna(total_error) and not pd.isna(total_error):
                        stats['total_errors'].append(float(total_error))
                    
                    win_prob = row['win_probability']
                    if pd.notna(win_prob) and not pd.isna(win_prob):
                        stats['win_confidences'].append(float(win_prob))
            
            # Calculate team metrics
            team_analysis = {}
            for team, stats in team_stats.items():
                predictions_count = stats['predictions']
                if predictions_count >= 3:  # At least 3 predictions
                    correct_count = stats['correct']
                    win_accuracy = correct_count / predictions_count if predictions_count > 0 else 0
                    
                    total_errors_list = stats['total_errors']
                    avg_total_error = np.mean(total_errors_list) if len(total_errors_list) > 0 else None
                    
                    win_confidences_list = stats['win_confidences']
                    avg_confidence = np.mean(win_confidences_list) if len(win_confidences_list) > 0 else None
                    
                    team_analysis[team] = {
                        'prediction_count': stats['predictions'],
                        'win_accuracy': win_accuracy,
                        'avg_total_error': avg_total_error,
                        'avg_confidence': avg_confidence
                    }
            
            # Find best and worst predicted teams
            sorted_teams = sorted(team_analysis.items(), key=lambda x: x[1]['win_accuracy'], reverse=True)
            
            analysis = {
                'team_count': len(team_analysis),
                'best_predicted_teams': sorted_teams[:5],
                'worst_predicted_teams': sorted_teams[-5:],
                'all_team_stats': team_analysis
            }
            
            # Identify teams with systematic issues
            problem_teams = []
            for team, stats in team_analysis.items():
                if stats['win_accuracy'] < 0.4 and stats['prediction_count'] >= 5:
                    problem_teams.append({
                        'team': team,
                        'accuracy': stats['win_accuracy'],
                        'prediction_count': stats['prediction_count']
                    })
            
            analysis['problem_teams'] = problem_teams
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing team performance patterns: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze how prediction accuracy varies over time"""
        try:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['week'] = df['game_date'].dt.isocalendar().week
            df['month'] = df['game_date'].dt.month
            df['day_of_week'] = df['game_date'].dt.day_name()
            
            analysis = {}
            
            # Weekly accuracy trends
            weekly_stats = df.groupby('week').agg({
                'win_prediction_correct': ['count', 'sum', 'mean'],
                'total_absolute_error': 'mean'
            }).round(3)
            analysis['weekly_trends'] = weekly_stats.to_dict()
            
            # Daily patterns (day of week)
            daily_stats = df.groupby('day_of_week').agg({
                'win_prediction_correct': ['count', 'sum', 'mean'],
                'total_absolute_error': 'mean'
            }).round(3)
            analysis['daily_patterns'] = daily_stats.to_dict()
            
            # Monthly patterns
            if df['month'].nunique() > 1:
                monthly_stats = df.groupby('month').agg({
                    'win_prediction_correct': ['count', 'sum', 'mean'],
                    'total_absolute_error': 'mean'
                }).round(3)
                analysis['monthly_patterns'] = monthly_stats.to_dict()
            
            # Detect degradation over time
            df_sorted = df.sort_values('game_date')
            
            # Calculate rolling 10-game accuracy
            if len(df_sorted) >= 10:
                df_sorted['rolling_accuracy'] = df_sorted['win_prediction_correct'].rolling(window=10).mean()
                
                # Check for degradation (compare first 10% vs last 10% of period)
                early_period = df_sorted.head(max(10, len(df_sorted) // 10))
                late_period = df_sorted.tail(max(10, len(df_sorted) // 10))
                
                early_accuracy = early_period['win_prediction_correct'].mean()
                late_accuracy = late_period['win_prediction_correct'].mean()
                
                analysis['temporal_degradation'] = {
                    'early_period_accuracy': early_accuracy,
                    'late_period_accuracy': late_accuracy,
                    'accuracy_change': late_accuracy - early_accuracy,
                    'degradation_detected': (late_accuracy - early_accuracy) < -0.05
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {str(e)}")
            return {'error': str(e)}
    
    def _generate_actionable_insights(self, analysis: Dict) -> List[Dict]:
        """Generate actionable insights from error analysis"""
        try:
            insights = []
            
            # Win prediction insights
            if 'win_prediction_analysis' in analysis:
                win_analysis = analysis['win_prediction_analysis']
                
                if 'accuracy' in win_analysis and win_analysis['accuracy'] < 0.52:
                    insights.append({
                        'type': 'performance_alert',
                        'priority': 'high',
                        'issue': 'Low win prediction accuracy',
                        'description': f"Win accuracy ({win_analysis['accuracy']:.1%}) is below acceptable threshold (52%)",
                        'action': 'Immediate model retraining recommended',
                        'metric': win_analysis['accuracy']
                    })
                
                if 'high_confidence_errors' in win_analysis:
                    hce = win_analysis['high_confidence_errors']
                    if hce['count'] > 0 and hce['percentage_of_total'] > 0.1:
                        insights.append({
                            'type': 'confidence_calibration',
                            'priority': 'medium',
                            'issue': 'High confidence prediction errors',
                            'description': f"{hce['count']} high-confidence predictions ({hce['percentage_of_total']:.1%}) were incorrect",
                            'action': 'Review confidence calibration and feature importance',
                            'metric': hce['percentage_of_total']
                        })
            
            # Total prediction insights
            if 'total_prediction_analysis' in analysis:
                total_analysis = analysis['total_prediction_analysis']
                
                if 'over_under_bias' in total_analysis:
                    bias = total_analysis['over_under_bias']
                    if abs(bias['over_percentage'] - 0.5) > 0.15:  # More than 65% or less than 35%
                        bias_direction = 'over' if bias['over_percentage'] > 0.5 else 'under'
                        insights.append({
                            'type': 'systematic_bias',
                            'priority': 'medium',
                            'issue': f'Systematic {bias_direction}-prediction bias',
                            'description': f"{bias_direction.capitalize()} predictions: {bias['over_percentage' if bias_direction == 'over' else 'under_percentage']:.1%}",
                            'action': f'Adjust model calibration for {bias_direction}-prediction bias',
                            'metric': bias['over_percentage'] if bias_direction == 'over' else bias['under_percentage']
                        })
                
                if 'team_total_biases' in total_analysis and total_analysis['team_total_biases']:
                    team_biases = total_analysis['team_total_biases']
                    worst_bias = max(team_biases.items(), key=lambda x: abs(x[1]['average_error']))
                    
                    insights.append({
                        'type': 'team_bias',
                        'priority': 'low',
                        'issue': f'Team-specific total prediction bias',
                        'description': f"Team {worst_bias[0]} shows {worst_bias[1]['bias_direction']} bias of {abs(worst_bias[1]['average_error']):.2f} runs",
                        'action': 'Consider team-specific features or model adjustments',
                        'metric': abs(worst_bias[1]['average_error'])
                    })
            
            # Confidence calibration insights
            if 'confidence_analysis' in analysis:
                conf_analysis = analysis['confidence_analysis']
                
                if 'calibration_quality' in conf_analysis and conf_analysis['calibration_quality'] in ['fair', 'poor']:
                    insights.append({
                        'type': 'calibration_issue',
                        'priority': 'medium',
                        'issue': 'Poor confidence calibration',
                        'description': f"Confidence calibration quality: {conf_analysis['calibration_quality']}",
                        'action': 'Recalibrate prediction confidence using isotonic regression or Platt scaling',
                        'metric': conf_analysis.get('overall_calibration_error', 0)
                    })
            
            # Team performance insights
            if 'team_analysis' in analysis:
                team_analysis = analysis['team_analysis']
                
                if 'problem_teams' in team_analysis and team_analysis['problem_teams']:
                    problem_count = len(team_analysis['problem_teams'])
                    insights.append({
                        'type': 'team_performance',
                        'priority': 'low',
                        'issue': 'Poor prediction accuracy for specific teams',
                        'description': f"{problem_count} teams show consistently poor prediction accuracy",
                        'action': 'Investigate team-specific factors or data quality issues',
                        'metric': problem_count
                    })
            
            # Temporal insights
            if 'temporal_analysis' in analysis:
                temp_analysis = analysis['temporal_analysis']
                
                if 'temporal_degradation' in temp_analysis:
                    degradation = temp_analysis['temporal_degradation']
                    if degradation.get('degradation_detected', False):
                        insights.append({
                            'type': 'performance_degradation',
                            'priority': 'high',
                            'issue': 'Model performance degrading over time',
                            'description': f"Accuracy dropped {abs(degradation['accuracy_change']):.1%} from early to late period",
                            'action': 'Immediate model retraining with recent data',
                            'metric': abs(degradation['accuracy_change'])
                        })
            
            # Sort insights by priority
            priority_order = {'high': 3, 'medium': 2, 'low': 1}
            insights.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating actionable insights: {str(e)}")
            return []
    
    def _store_analysis_results(self, analysis: Dict) -> bool:
        """Store analysis results for tracking"""
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Store main analysis summary
                insert_query = """
                    INSERT INTO learning_insights 
                    (sport, model_type, insight_type, insight_category, insight_description, 
                     confidence_score, action_taken)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                
                # Store key insights
                if 'actionable_insights' in analysis:
                    for insight in analysis['actionable_insights']:
                        conn.execute(insert_query, (
                            'MLB', 'xgboost', 'performance_analysis', insight['type'],
                            insight['description'], insight.get('metric', 0.5),
                            insight['action']
                        ))
                
                # Store overall analysis summary
                summary = f"Analyzed {analysis.get('data_summary', {}).get('total_predictions', 0)} predictions"
                conn.execute(insert_query, (
                    'MLB', 'xgboost', 'analysis_summary', 'comprehensive_analysis',
                    summary, 1.0, 'Performance analysis completed'
                ))
                
                conn.commit()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing analysis results: {str(e)}")
            return False
    
    def get_performance_summary(self, days: int = 7) -> Dict:
        """
        Get a quick performance summary for dashboard display
        
        Args:
            days: Number of recent days to summarize
            
        Returns:
            Dictionary with key performance metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Get basic performance metrics
                query = """
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN win_prediction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                        AVG(win_probability) as avg_confidence,
                        AVG(total_absolute_error) as avg_total_error,
                        COUNT(CASE WHEN result_updated_at IS NOT NULL THEN 1 END) as predictions_with_results
                    FROM predictions 
                    WHERE sport = 'MLB' 
                    AND game_date >= DATE(?)
                    AND game_date <= DATE(?)
                """
                
                cursor = conn.execute(query, (start_date.date(), end_date.date()))
                result = cursor.fetchone()
                
                if result:
                    total, correct, avg_conf, avg_error, with_results = result
                    accuracy = correct / total if total > 0 else 0
                    
                    # Get trend information
                    trend_query = """
                        SELECT accuracy_trend, improvement_score, retraining_triggered
                        FROM performance_trends
                        WHERE sport = 'MLB' AND model_type = 'xgboost'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """
                    
                    trend_cursor = conn.execute(trend_query)
                    trend_result = trend_cursor.fetchone()
                    
                    summary = {
                        'period_days': days,
                        'total_predictions': total or 0,
                        'predictions_with_results': with_results or 0,
                        'accuracy': accuracy,
                        'average_confidence': avg_conf,
                        'average_total_error': avg_error,
                        'data_coverage': (with_results / total) if total > 0 else 0
                    }
                    
                    if trend_result:
                        trend, improvement, retraining = trend_result
                        summary.update({
                            'accuracy_trend': trend,
                            'improvement_score': improvement,
                            'retraining_needed': bool(retraining)
                        })
                    
                    # Performance status
                    if accuracy >= 0.55:
                        summary['status'] = 'excellent'
                    elif accuracy >= 0.52:
                        summary['status'] = 'good'
                    elif accuracy >= 0.50:
                        summary['status'] = 'fair'
                    else:
                        summary['status'] = 'poor'
                    
                    return summary
                
            return {'error': 'No data available'}
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {'error': str(e)}