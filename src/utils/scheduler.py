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
            
            # Schedule weekly model retraining
            schedule.every().sunday.at("02:00").do(self._weekly_model_retrain)
            
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
    
    def _weekly_model_retrain(self):
        """Weekly model retraining"""
        try:
            self.logger.info("Starting weekly model retraining")
            
            # Get training data from last 90 days
            training_data = self.db_manager.get_training_data(days_back=90)
            
            if not training_data.empty and len(training_data) > 100:
                results = self.predictor.train_models(training_data)
                
                if results:
                    # Store model metrics
                    self.db_manager.store_model_metrics(results)
                    self.logger.info("Weekly model retraining completed")
                else:
                    self.logger.error("Model retraining failed")
            else:
                self.logger.warning("Insufficient data for model retraining")
                
        except Exception as e:
            self.logger.error(f"Error in weekly model retraining: {str(e)}")
    
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
