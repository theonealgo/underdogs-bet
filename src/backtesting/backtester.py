import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from data_storage.database import DatabaseManager
from models.prediction_models import MLBPredictor

@dataclass
class BacktestResult:
    """Data class for storing backtest results"""
    date: datetime
    game_pk: str
    home_team: str
    away_team: str
    actual_winner: str
    predicted_winner: str
    win_prediction_correct: bool
    win_confidence: float
    actual_total: float
    predicted_total: float
    total_error: float
    total_absolute_error: float

class Backtester:
    """
    Backtesting framework for MLB prediction models
    """
    
    def __init__(self, db_manager: DatabaseManager, predictor: MLBPredictor):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.predictor = predictor
    
    def run_backtest(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        min_confidence: float = 0.6,
        retrain_frequency: int = 30
    ) -> Dict:
        """
        Run comprehensive backtest of prediction models
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            min_confidence: Minimum confidence threshold for predictions
            retrain_frequency: Days between model retraining
            
        Returns:
            Dictionary with backtest results and metrics
        """
        try:
            self.logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}")
            
            # Get historical data for the backtest period
            historical_data = self.db_manager.get_historical_games(start_date, end_date)
            
            if historical_data.empty:
                self.logger.warning("No historical data available for backtest period")
                return {}
            
            # Prepare data for backtesting
            backtest_data = self._prepare_backtest_data(historical_data)
            
            if backtest_data.empty:
                self.logger.warning("No valid backtest data after preparation")
                return {}
            
            # Run walk-forward backtesting
            results = self._walk_forward_backtest(
                backtest_data, 
                min_confidence, 
                retrain_frequency
            )
            
            # Calculate comprehensive metrics
            metrics = self._calculate_backtest_metrics(results)
            
            # Generate detailed analysis
            analysis = self._generate_backtest_analysis(results, metrics)
            
            final_results = {
                'backtest_period': {
                    'start_date': start_date.date().isoformat(),
                    'end_date': end_date.date().isoformat(),
                    'total_days': (end_date - start_date).days
                },
                'data_summary': {
                    'total_games': len(backtest_data),
                    'games_predicted': len(results),
                    'prediction_rate': len(results) / len(backtest_data) if len(backtest_data) > 0 else 0
                },
                'metrics': metrics,
                'analysis': analysis,
                'predictions': [result.__dict__ for result in results[-100:]],  # Last 100 predictions
                'daily_performance': self._calculate_daily_performance(results),
                **self._calculate_advanced_metrics(results)
            }
            
            self.logger.info(f"Backtest completed: {len(results)} predictions analyzed")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            return {}
    
    def _prepare_backtest_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare historical data for backtesting
        
        Args:
            historical_data: Raw historical game data
            
        Returns:
            Prepared DataFrame for backtesting
        """
        try:
            # Sort by date
            data = historical_data.sort_values('game_date').copy()
            
            # Remove games with missing essential data
            data = data.dropna(subset=['home_team', 'away_team', 'total_runs', 'home_win'])
            
            # Add derived columns for easier analysis
            data['actual_winner'] = data.apply(
                lambda row: row['home_team'] if row['home_win'] == 1 else row['away_team'], 
                axis=1
            )
            
            # Ensure game_date is datetime
            data['game_date'] = pd.to_datetime(data['game_date'])
            
            self.logger.info(f"Prepared {len(data)} games for backtesting")
            return data
            
        except Exception as e:
            self.logger.error(f"Error preparing backtest data: {str(e)}")
            return pd.DataFrame()
    
    def _walk_forward_backtest(
        self, 
        data: pd.DataFrame, 
        min_confidence: float,
        retrain_frequency: int
    ) -> List[BacktestResult]:
        """
        Perform walk-forward backtesting
        
        Args:
            data: Prepared backtest data
            min_confidence: Minimum confidence for predictions
            retrain_frequency: Days between retraining
            
        Returns:
            List of BacktestResult objects
        """
        results = []
        last_retrain_date = None
        
        try:
            # Initial training period (minimum 60 days)
            min_training_days = 60
            
            for idx, (_, game_row) in enumerate(data.iterrows()):
                game_date = pd.to_datetime(game_row['game_date'])
                
                # Get training data up to this game (excluding this game)
                training_cutoff = game_date - timedelta(days=1)
                training_start = game_date - timedelta(days=365)  # Use up to 1 year of history
                
                training_data = data[
                    (data['game_date'] >= training_start) & 
                    (data['game_date'] <= training_cutoff)
                ].copy()
                
                # Skip if insufficient training data
                if len(training_data) < min_training_days:
                    continue
                
                # Retrain model if needed
                if (last_retrain_date is None or 
                    (game_date - last_retrain_date).days >= retrain_frequency):
                    
                    self.logger.info(f"Retraining model for date {game_date.date()}")
                    self.predictor.train_models(training_data)
                    last_retrain_date = game_date
                
                # Make prediction for this game
                game_df = pd.DataFrame([game_row])
                prediction = self.predictor.predict_game(game_df)
                
                if not prediction:
                    continue
                
                # Apply confidence filter
                if prediction.get('win_probability', 0) < min_confidence:
                    continue
                
                # Create result object
                result = BacktestResult(
                    date=game_date,
                    game_pk=game_row.get('game_pk', ''),
                    home_team=game_row['home_team'],
                    away_team=game_row['away_team'],
                    actual_winner=game_row['actual_winner'],
                    predicted_winner=prediction['predicted_winner'],
                    win_prediction_correct=prediction['predicted_winner'] == game_row['actual_winner'],
                    win_confidence=prediction['win_probability'],
                    actual_total=float(game_row['total_runs']),
                    predicted_total=float(prediction['predicted_total']),
                    total_error=float(prediction['predicted_total']) - float(game_row['total_runs']),
                    total_absolute_error=abs(float(prediction['predicted_total']) - float(game_row['total_runs']))
                )
                
                results.append(result)
                
                # Log progress every 50 predictions
                if len(results) % 50 == 0:
                    self.logger.info(f"Processed {len(results)} predictions")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward backtest: {str(e)}")
            return results
    
    def _calculate_backtest_metrics(self, results: List[BacktestResult]) -> Dict:
        """
        Calculate comprehensive backtest metrics
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            Dictionary with calculated metrics
        """
        try:
            if not results:
                return {}
            
            # Winner prediction metrics
            correct_predictions = sum(r.win_prediction_correct for r in results)
            total_predictions = len(results)
            winner_accuracy = correct_predictions / total_predictions
            
            # Total prediction metrics
            total_errors = [r.total_error for r in results]
            total_absolute_errors = [r.total_absolute_error for r in results]
            
            totals_mae = np.mean(total_absolute_errors)
            totals_rmse = np.sqrt(np.mean([e**2 for e in total_errors]))
            totals_mape = np.mean([abs(e) / r.actual_total for r, e in zip(results, total_errors) if r.actual_total > 0]) * 100
            
            # Confidence-based metrics
            confidence_scores = [r.win_confidence for r in results]
            avg_confidence = np.mean(confidence_scores)
            
            # Calculate ROI (simplified betting simulation)
            roi_metrics = self._calculate_roi_metrics(results)
            
            metrics = {
                'total_predictions': total_predictions,
                'winner_accuracy': winner_accuracy,
                'correct_predictions': correct_predictions,
                'average_confidence': avg_confidence,
                'totals_mae': totals_mae,
                'totals_rmse': totals_rmse,
                'totals_mape': totals_mape,
                'total_error_std': np.std(total_errors),
                **roi_metrics
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest metrics: {str(e)}")
            return {}
    
    def _calculate_roi_metrics(self, results: List[BacktestResult]) -> Dict:
        """
        Calculate ROI metrics based on betting simulation
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            Dictionary with ROI metrics
        """
        try:
            # Simplified betting simulation
            # Assume standard -110 odds for all bets
            bet_amount = 100  # Standard bet amount
            wins = 0
            losses = 0
            total_wagered = 0
            total_won = 0
            
            for result in results:
                total_wagered += bet_amount
                
                if result.win_prediction_correct:
                    wins += 1
                    # Win $90.91 on a $100 bet at -110 odds
                    total_won += bet_amount + (bet_amount / 1.1)
                else:
                    losses += 1
                    # Lose the $100 bet
                    pass
            
            net_profit = total_won - total_wagered
            roi = (net_profit / total_wagered) * 100 if total_wagered > 0 else 0
            
            # Break-even rate (need 52.38% to break even at -110 odds)
            breakeven_rate = 52.38
            edge = (wins / len(results) * 100) - breakeven_rate if results else 0
            
            return {
                'roi': roi,
                'net_profit': net_profit,
                'total_wagered': total_wagered,
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / len(results)) * 100 if results else 0,
                'edge_over_breakeven': edge,
                'units_won': net_profit / bet_amount
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ROI metrics: {str(e)}")
            return {}
    
    def _calculate_daily_performance(self, results: List[BacktestResult]) -> List[Dict]:
        """
        Calculate daily performance metrics
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            List of daily performance dictionaries
        """
        try:
            if not results:
                return []
            
            # Group results by date
            daily_results = {}
            for result in results:
                date_str = result.date.date().isoformat()
                if date_str not in daily_results:
                    daily_results[date_str] = []
                daily_results[date_str].append(result)
            
            # Calculate metrics for each day
            daily_performance = []
            cumulative_correct = 0
            cumulative_total = 0
            
            for date_str in sorted(daily_results.keys()):
                day_results = daily_results[date_str]
                
                # Daily metrics
                day_correct = sum(r.win_prediction_correct for r in day_results)
                day_total = len(day_results)
                day_accuracy = day_correct / day_total if day_total > 0 else 0
                
                # Cumulative metrics
                cumulative_correct += day_correct
                cumulative_total += day_total
                cumulative_accuracy = cumulative_correct / cumulative_total if cumulative_total > 0 else 0
                
                # Total predictions
                day_totals_mae = np.mean([r.total_absolute_error for r in day_results])
                
                daily_performance.append({
                    'date': date_str,
                    'games_predicted': day_total,
                    'correct_predictions': day_correct,
                    'daily_accuracy': day_accuracy,
                    'cumulative_accuracy': cumulative_accuracy,
                    'totals_mae': day_totals_mae,
                    'avg_confidence': np.mean([r.win_confidence for r in day_results])
                })
            
            return daily_performance
            
        except Exception as e:
            self.logger.error(f"Error calculating daily performance: {str(e)}")
            return []
    
    def _generate_backtest_analysis(self, results: List[BacktestResult], metrics: Dict) -> Dict:
        """
        Generate detailed analysis of backtest results
        
        Args:
            results: List of BacktestResult objects
            metrics: Calculated metrics
            
        Returns:
            Dictionary with analysis insights
        """
        try:
            analysis = {}
            
            if not results:
                return analysis
            
            # Performance by team
            team_performance = self._analyze_team_performance(results)
            analysis['team_performance'] = team_performance
            
            # Performance by confidence level
            confidence_analysis = self._analyze_confidence_levels(results)
            analysis['confidence_analysis'] = confidence_analysis
            
            # Temporal analysis
            temporal_analysis = self._analyze_temporal_patterns(results)
            analysis['temporal_patterns'] = temporal_analysis
            
            # Model stability
            stability_analysis = self._analyze_model_stability(results)
            analysis['model_stability'] = stability_analysis
            
            # Key insights
            insights = self._generate_key_insights(results, metrics)
            analysis['key_insights'] = insights
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating backtest analysis: {str(e)}")
            return {}
    
    def _analyze_team_performance(self, results: List[BacktestResult]) -> Dict:
        """Analyze prediction performance by team"""
        try:
            team_stats = {}
            
            for result in results:
                for team in [result.home_team, result.away_team]:
                    if team not in team_stats:
                        team_stats[team] = {'correct': 0, 'total': 0}
                    
                    team_stats[team]['total'] += 1
                    if result.win_prediction_correct:
                        team_stats[team]['correct'] += 1
            
            # Calculate accuracy for each team
            for team in team_stats:
                total = team_stats[team]['total']
                correct = team_stats[team]['correct']
                team_stats[team]['accuracy'] = correct / total if total > 0 else 0
            
            # Sort by accuracy
            sorted_teams = sorted(
                team_stats.items(), 
                key=lambda x: x[1]['accuracy'], 
                reverse=True
            )
            
            return {
                'best_predicted_teams': sorted_teams[:5],
                'worst_predicted_teams': sorted_teams[-5:],
                'total_teams_analyzed': len(team_stats)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing team performance: {str(e)}")
            return {}
    
    def _analyze_confidence_levels(self, results: List[BacktestResult]) -> Dict:
        """Analyze prediction accuracy by confidence levels"""
        try:
            confidence_bins = [0.6, 0.7, 0.8, 0.9, 1.0]
            bin_stats = {f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}": {'correct': 0, 'total': 0} 
                        for i in range(len(confidence_bins)-1)}
            
            for result in results:
                confidence = result.win_confidence
                
                for i in range(len(confidence_bins)-1):
                    if confidence_bins[i] <= confidence < confidence_bins[i+1]:
                        bin_key = f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}"
                        bin_stats[bin_key]['total'] += 1
                        if result.win_prediction_correct:
                            bin_stats[bin_key]['correct'] += 1
                        break
            
            # Calculate accuracy for each bin
            for bin_key in bin_stats:
                total = bin_stats[bin_key]['total']
                correct = bin_stats[bin_key]['correct']
                bin_stats[bin_key]['accuracy'] = correct / total if total > 0 else 0
            
            return bin_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing confidence levels: {str(e)}")
            return {}
    
    def _analyze_temporal_patterns(self, results: List[BacktestResult]) -> Dict:
        """Analyze prediction patterns over time"""
        try:
            # Group by month
            monthly_performance = {}
            
            for result in results:
                month_key = result.date.strftime('%Y-%m')
                if month_key not in monthly_performance:
                    monthly_performance[month_key] = {'correct': 0, 'total': 0}
                
                monthly_performance[month_key]['total'] += 1
                if result.win_prediction_correct:
                    monthly_performance[month_key]['correct'] += 1
            
            # Calculate monthly accuracy
            for month in monthly_performance:
                total = monthly_performance[month]['total']
                correct = monthly_performance[month]['correct']
                monthly_performance[month]['accuracy'] = correct / total if total > 0 else 0
            
            return {
                'monthly_performance': monthly_performance,
                'total_months': len(monthly_performance)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {str(e)}")
            return {}
    
    def _analyze_model_stability(self, results: List[BacktestResult]) -> Dict:
        """Analyze model prediction stability"""
        try:
            # Calculate rolling accuracy (20-game windows)
            window_size = 20
            rolling_accuracies = []
            
            for i in range(len(results) - window_size + 1):
                window_results = results[i:i + window_size]
                window_accuracy = sum(r.win_prediction_correct for r in window_results) / window_size
                rolling_accuracies.append(window_accuracy)
            
            if rolling_accuracies:
                stability_metrics = {
                    'avg_rolling_accuracy': np.mean(rolling_accuracies),
                    'accuracy_volatility': np.std(rolling_accuracies),
                    'min_rolling_accuracy': min(rolling_accuracies),
                    'max_rolling_accuracy': max(rolling_accuracies)
                }
            else:
                stability_metrics = {}
            
            return stability_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing model stability: {str(e)}")
            return {}
    
    def _generate_key_insights(self, results: List[BacktestResult], metrics: Dict) -> List[str]:
        """Generate key insights from backtest results"""
        try:
            insights = []
            
            # Overall performance insight
            accuracy = metrics.get('winner_accuracy', 0)
            if accuracy > 0.55:
                insights.append(f"Strong predictive performance with {accuracy:.1%} accuracy")
            elif accuracy > 0.52:
                insights.append(f"Modest edge over random with {accuracy:.1%} accuracy")
            else:
                insights.append(f"Performance below expectations at {accuracy:.1%} accuracy")
            
            # ROI insight
            roi = metrics.get('roi', 0)
            if roi > 5:
                insights.append(f"Profitable betting strategy with {roi:.1f}% ROI")
            elif roi > 0:
                insights.append(f"Marginally profitable with {roi:.1f}% ROI")
            else:
                insights.append(f"Unprofitable strategy with {roi:.1f}% ROI")
            
            # Totals prediction insight
            mae = metrics.get('totals_mae', 0)
            if mae < 1.0:
                insights.append(f"Excellent totals prediction with {mae:.2f} average error")
            elif mae < 1.5:
                insights.append(f"Good totals prediction accuracy with {mae:.2f} average error")
            else:
                insights.append(f"Room for improvement in totals prediction ({mae:.2f} average error)")
            
            # Sample size insight
            total_predictions = metrics.get('total_predictions', 0)
            if total_predictions < 100:
                insights.append("Limited sample size - results should be interpreted cautiously")
            elif total_predictions > 500:
                insights.append("Large sample size provides statistical confidence in results")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating key insights: {str(e)}")
            return []
    
    def _calculate_advanced_metrics(self, results: List[BacktestResult]) -> Dict:
        """Calculate advanced backtest metrics"""
        try:
            if not results:
                return {}
            
            # Sharpe ratio (risk-adjusted returns)
            returns = []
            for result in results:
                if result.win_prediction_correct:
                    returns.append(0.91)  # Win $90.91 on $100 bet
                else:
                    returns.append(-1.0)  # Lose $100
            
            if returns:
                avg_return = np.mean(returns)
                return_std = np.std(returns)
                sharpe_ratio = avg_return / return_std if return_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            # Win streak analysis
            win_streaks = []
            current_streak = 0
            
            for result in results:
                if result.win_prediction_correct:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        win_streaks.append(current_streak)
                    current_streak = 0
            
            if current_streak > 0:
                win_streaks.append(current_streak)
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'longest_win_streak': max(win_streaks) if win_streaks else 0,
                'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
                'total_win_streaks': len(win_streaks)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {str(e)}")
            return {}
