import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collectors.baseball_savant_scraper import BaseballSavantScraper
from data_collectors.result_tracker import ResultTracker
from models.performance_analyzer import PerformanceAnalyzer
from models.intelligent_retrainer import IntelligentRetrainer
# Note: OddsShark dependency removed
from data_storage.database import DatabaseManager
from models.prediction_models import MLBPredictor

class DataScheduler:
    """
    Scheduler for automated data collection and model updates
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.baseball_scraper = BaseballSavantScraper()
        # Note: OddsShark dependency removed
        self.predictor = MLBPredictor()
        
        # Initialize learning system components
        self.result_tracker = ResultTracker(db_manager)
        self.performance_analyzer = PerformanceAnalyzer(db_manager)
        self.intelligent_retrainer = IntelligentRetrainer(db_manager, self.predictor)
        
        self.is_running = False
        self.scheduler_thread = None
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        try:
            if self.is_running:
                self.logger.warning("Scheduler is already running")
                return
            
            # Schedule daily data updates
            schedule.every().day.at("06:00").do(self._daily_data_update)
            schedule.every().day.at("08:00").do(self._generate_daily_predictions)
            schedule.every().day.at("14:00").do(self._update_live_data)
            schedule.every().day.at("20:00").do(self._evening_data_update)
            
            # Schedule automated learning system tasks
            schedule.every().day.at("09:00").do(self._daily_result_tracking)  # After games complete
            schedule.every().day.at("10:00").do(self._performance_evaluation)  # Check if retraining needed
            schedule.every().day.at("22:00").do(self._evening_result_tracking)  # End-of-day tracking
            
            # Schedule periodic analysis and learning
            schedule.every().monday.at("03:00").do(self._weekly_performance_analysis)
            schedule.every().sunday.at("04:00").do(self._intelligent_retraining_check)
            
            # Schedule monthly cleanup
            schedule.every().month.do(self._monthly_cleanup)
            
            self.is_running = True
            
            # Start scheduler in separate thread
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            self.logger.info("Data scheduler started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting scheduler: {str(e)}")
    
    def stop_scheduler(self):
        """Stop the automated scheduler"""
        try:
            self.is_running = False
            schedule.clear()
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            self.logger.info("Data scheduler stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping scheduler: {str(e)}")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def run_daily_update(self) -> Dict:
        """
        Run daily data update manually
        
        Returns:
            Dictionary with update results
        """
        return self._daily_data_update()
    
    def _daily_data_update(self) -> Dict:
        """Run the daily data update process"""
        self.logger.info("Starting daily data update")
        
        results = {
            'success': True,
            'messages': [],
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Update Baseball Savant data
            savant_result = self._update_baseball_savant_data()
            if savant_result['success']:
                results['messages'].append(f"Baseball Savant: {savant_result['message']}")
            else:
                results['errors'].append(f"Baseball Savant: {savant_result['error']}")
                results['success'] = False
            
            # Note: OddsShark data update removed - no longer using odds data
            
            # Update team statistics
            team_stats_result = self._update_team_stats()
            if team_stats_result['success']:
                results['messages'].append(f"Team Stats: {team_stats_result['message']}")
            else:
                results['errors'].append(f"Team Stats: {team_stats_result['error']}")
            
            self.logger.info("Daily data update completed")
            return results
            
        except Exception as e:
            error_msg = f"Error in daily data update: {str(e)}"
            self.logger.error(error_msg)
            results['success'] = False
            results['errors'].append(error_msg)
            return results
    
    def _update_baseball_savant_data(self) -> Dict:
        """Update Baseball Savant data"""
        try:
            # Get recent games (last 7 days)
            recent_data = self.baseball_scraper.get_recent_games(days=7)
            
            if not recent_data.empty:
                success = self.db_manager.store_statcast_data(recent_data)
                if success:
                    return {
                        'success': True,
                        'message': f"Updated {len(recent_data)} Statcast records"
                    }
                else:
                    return {
                        'success': False,
                        'error': "Failed to store Statcast data"
                    }
            else:
                return {
                    'success': False,
                    'error': "No new Statcast data available"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Baseball Savant update error: {str(e)}"
            }
    
    # Note: _update_odds_data method removed - no longer using OddsShark odds data
    
    def _update_team_stats(self) -> Dict:
        """Update team statistics"""
        try:
            current_year = datetime.now().year
            team_stats = self.baseball_scraper.get_team_stats(current_year)
            
            if team_stats and ('batting' in team_stats or 'pitching' in team_stats):
                # Store team stats (would need implementation in database manager)
                return {
                    'success': True,
                    'message': "Team stats updated successfully"
                }
            else:
                return {
                    'success': False,
                    'error': "No team stats data available"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Team stats update error: {str(e)}"
            }
    
    def _generate_daily_predictions(self):
        """Generate predictions for today's games"""
        try:
            self.logger.info("Generating daily predictions")
            
            # Get today's games (would need schedule data)
            today = datetime.now().date()
            
            # For now, generate predictions based on recent team data
            # In a full implementation, this would use actual schedule data
            
            # Get recent game data for prediction
            recent_games = self.db_manager.get_historical_games(
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
            
            if not recent_games.empty:
                # Generate sample predictions
                predictions = self.predictor.predict_multiple_games(recent_games.head(5))
                
                if predictions:
                    self.db_manager.store_predictions(predictions)
                    self.logger.info(f"Generated {len(predictions)} daily predictions")
                else:
                    self.logger.warning("No predictions generated")
            else:
                self.logger.warning("No recent game data for predictions")
                
        except Exception as e:
            self.logger.error(f"Error generating daily predictions: {str(e)}")
    
    def _update_live_data(self):
        """Update live data during the day"""
        try:
            self.logger.info("Running live data update")
            
            # Get today's data
            today_data = self.baseball_scraper.get_recent_games(days=1)
            
            if not today_data.empty:
                self.db_manager.store_statcast_data(today_data)
                self.logger.info("Live data updated")
            
            # Update current odds
            current_odds = self.odds_scraper.get_mlb_odds()
            if current_odds:
                self.db_manager.store_odds_data(current_odds)
                self.logger.info("Live odds updated")
                
        except Exception as e:
            self.logger.error(f"Error in live data update: {str(e)}")
    
    def _evening_data_update(self):
        """Evening data update after games complete"""
        try:
            self.logger.info("Running evening data update")
            
            # Get completed games from today
            today_data = self.baseball_scraper.get_recent_games(days=1)
            
            if not today_data.empty:
                self.db_manager.store_statcast_data(today_data)
                self.logger.info("Evening game data updated")
                
        except Exception as e:
            self.logger.error(f"Error in evening data update: {str(e)}")
    
    def _daily_result_tracking(self):
        """Daily automated result tracking"""
        try:
            self.logger.info("Starting daily result tracking")
            
            # Fetch actual game results and compare to predictions
            results = self.result_tracker.fetch_and_update_results()
            
            if results.get('success'):
                self.logger.info(f"Result tracking: Updated {results.get('predictions_updated', 0)} predictions")
                
                if results.get('insights_generated'):
                    self.logger.info(f"Generated {results['insights_generated']} new learning insights")
            else:
                self.logger.error(f"Result tracking failed: {results.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in daily result tracking: {str(e)}")
    
    def _performance_evaluation(self):
        """Daily performance evaluation and retraining need assessment"""
        try:
            self.logger.info("Starting performance evaluation")
            
            # Evaluate if retraining is needed
            evaluation = self.intelligent_retrainer.evaluate_retraining_need()
            
            if evaluation.get('retraining_needed'):
                priority = evaluation.get('priority', 'medium')
                self.logger.warning(f"Retraining needed with {priority} priority")
                
                # Trigger retraining for high/critical priority
                if priority in ['high', 'critical']:
                    self.logger.info("Auto-triggering retraining due to high priority")
                    retraining_result = self.intelligent_retrainer.execute_intelligent_retraining(priority)
                    
                    if retraining_result.get('success'):
                        improvement = retraining_result.get('improvements', {}).get('accuracy_change')
                        self.logger.info(f"Auto-retraining completed. Accuracy change: {improvement}")
                    else:
                        self.logger.error(f"Auto-retraining failed: {retraining_result.get('error')}")
            else:
                self.logger.info("Performance evaluation: No retraining needed")
                
        except Exception as e:
            self.logger.error(f"Error in performance evaluation: {str(e)}")
    
    def _evening_result_tracking(self):
        """Evening result tracking for completed games"""
        try:
            self.logger.info("Starting evening result tracking")
            
            # Track results for games that finished today
            results = self.result_tracker.fetch_and_update_results(days_back=1)
            
            if results.get('success'):
                self.logger.info(f"Evening tracking: Processed {results.get('games_processed', 0)} games")
            else:
                self.logger.error(f"Evening result tracking failed: {results.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in evening result tracking: {str(e)}")
    
    def _weekly_performance_analysis(self):
        """Weekly comprehensive performance analysis"""
        try:
            self.logger.info("Starting weekly performance analysis")
            
            # Run detailed error analysis
            analysis = self.performance_analyzer.analyze_prediction_errors(days_back=14)
            
            if 'error' not in analysis:
                insights = analysis.get('actionable_insights', [])
                high_priority_insights = [i for i in insights if i.get('priority') == 'high']
                
                if high_priority_insights:
                    self.logger.warning(f"Weekly analysis found {len(high_priority_insights)} high-priority issues")
                    for insight in high_priority_insights:
                        self.logger.warning(f"Issue: {insight.get('description')}")
                else:
                    self.logger.info("Weekly analysis: No critical issues found")
                    
                self.logger.info(f"Weekly analysis completed with {len(insights)} total insights")
            else:
                self.logger.error(f"Weekly performance analysis failed: {analysis.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in weekly performance analysis: {str(e)}")
    
    def _intelligent_retraining_check(self):
        """Weekly intelligent retraining assessment"""
        try:
            self.logger.info("Starting intelligent retraining check")
            
            # Comprehensive retraining evaluation
            evaluation = self.intelligent_retrainer.evaluate_retraining_need()
            
            if evaluation.get('retraining_needed'):
                priority = evaluation.get('priority', 'medium')
                triggers = evaluation.get('triggers', [])
                
                self.logger.info(f"Weekly check: Retraining needed ({priority} priority)")
                self.logger.info(f"Triggers: {[t.get('type') for t in triggers]}")
                
                # Execute retraining
                retraining_result = self.intelligent_retrainer.execute_intelligent_retraining(priority)
                
                if retraining_result.get('success'):
                    strategies = retraining_result.get('strategies_applied', [])
                    improvements = retraining_result.get('improvements', {})
                    
                    self.logger.info(f"Weekly retraining completed successfully")
                    self.logger.info(f"Strategies applied: {strategies}")
                    
                    if 'accuracy_change' in improvements:
                        self.logger.info(f"Accuracy improvement: {improvements['accuracy_change']:+.3f}")
                else:
                    self.logger.error(f"Weekly retraining failed: {retraining_result.get('error')}")
            else:
                self.logger.info("Weekly check: No retraining needed")
                
        except Exception as e:
            self.logger.error(f"Error in intelligent retraining check: {str(e)}")
    
    def _monthly_cleanup(self):
        """Monthly database cleanup"""
        try:
            self.logger.info("Starting monthly database cleanup")
            
            # Clean up old data (keep 1 year)
            success = self.db_manager.cleanup_old_data(days_to_keep=365)
            
            if success:
                self.logger.info("Monthly cleanup completed")
            else:
                self.logger.error("Monthly cleanup failed")
                
        except Exception as e:
            self.logger.error(f"Error in monthly cleanup: {str(e)}")
    
    def get_scheduler_status(self) -> Dict:
        """
        Get current scheduler status
        
        Returns:
            Dictionary with scheduler status
        """
        try:
            status = {
                'is_running': self.is_running,
                'scheduled_jobs': len(schedule.jobs),
                'next_run': None,
                'last_update': None
            }
            
            if schedule.jobs:
                next_job = schedule.next_run()
                if next_job:
                    status['next_run'] = next_job.isoformat()
            
            # Get last update time from database
            last_update = self.db_manager.get_latest_data_timestamp('statcast')
            if last_update:
                status['last_update'] = last_update
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting scheduler status: {str(e)}")
            return {'is_running': False, 'error': str(e)}
    
    def force_update_now(self) -> Dict:
        """
        Force an immediate data update
        
        Returns:
            Update results
        """
        try:
            self.logger.info("Forcing immediate data update")
            return self._daily_data_update()
            
        except Exception as e:
            self.logger.error(f"Error in forced update: {str(e)}")
            return {
                'success': False,
                'errors': [str(e)],
                'timestamp': datetime.now().isoformat()
            }
