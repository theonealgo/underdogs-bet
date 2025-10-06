"""
Sport-specific data management utility for multi-sport prediction system.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class SportDataManager:
    """
    Manages data collection across multiple sports.
    Provides unified interface for fetching schedules and updating data
    regardless of the sport.
    """
    
    def __init__(self):
        self._collectors = {}
        self._initialize_collectors()
    
    def _initialize_collectors(self):
        """Initialize all sport-specific data collectors."""
        try:
            # MLB collectors
            from src.data_collectors.mlb_schedule_collector import MLBScheduleCollector
            from src.data_collectors.baseball_savant_scraper import BaseballSavantScraper
            self._collectors['MLB'] = {
                'schedule': MLBScheduleCollector(),
                'data': BaseballSavantScraper()
            }
            
            # NBA collector
            from src.data_collectors.nba_collector import NBADataCollector
            self._collectors['NBA'] = {
                'collector': NBADataCollector()
            }
            
            # NFL collector
            from src.data_collectors.nfl_collector import NFLDataCollector
            self._collectors['NFL'] = {
                'collector': NFLDataCollector()
            }
            
            # NHL collector
            from src.data_collectors.nhl_collector import NHLDataCollector
            self._collectors['NHL'] = {
                'collector': NHLDataCollector()
            }
            
            # NCAA Football collector
            from src.data_collectors.ncaaf_collector import NCAAFDataCollector
            self._collectors['NCAA'] = {
                'collector': NCAAFDataCollector()
            }
            
            # NCAA Basketball collector
            from src.data_collectors.ncaab_collector import NCAABDataCollector
            self._collectors['NCAAB'] = {
                'collector': NCAABDataCollector()
            }
            
            # WNBA collector
            from src.data_collectors.wnba_collector import WNBADataCollector
            self._collectors['WNBA'] = {
                'collector': WNBADataCollector()
            }
            
        except Exception as e:
            logger.error(f"Error initializing collectors: {e}")
        
        # Initialize odds collector for all sports
        try:
            from src.data_collectors.odds_collector import OddsCollector
            self._odds_collector = OddsCollector()
        except ImportError:
            logger.warning("Odds collector not available")
            self._odds_collector = None
    
    def get_todays_games(self, sport_code: str, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get games for the specified sport and date.
        
        Args:
            sport_code: Sport code (MLB, NBA, NFL, etc.)
            date: Date in YYYY-MM-DD format. If None, uses today.
        
        Returns:
            DataFrame with games for the specified date
        """
        try:
            if sport_code == 'MLB':
                collector = self._collectors['MLB']['schedule']
                return collector.get_todays_games(date=date)
            elif sport_code in self._collectors:
                collector = self._collectors[sport_code]['collector']
                # For now, other sports only support today's games
                # TODO: Add date parameter support to other sport collectors
                if date is None:
                    return collector.get_todays_games()
                else:
                    logger.warning(f"{sport_code} collector doesn't support date filtering yet. Showing today's games.")
                    return collector.get_todays_games()
            else:
                logger.warning(f"No collector found for sport: {sport_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching {sport_code} games: {e}")
            return pd.DataFrame()
    
    def update_sport_data(self, sport_code: str, days: int = 7) -> Dict[str, Any]:
        """Update data for the specified sport."""
        result = {
            'success': False,
            'games_found': 0,
            'data_updated': False,
            'messages': [],
            'errors': []
        }
        
        try:
            # Get today's games
            todays_games = self.get_todays_games(sport_code)
            if not todays_games.empty:
                result['games_found'] = len(todays_games)
                result['messages'].append(f"Found {len(todays_games)} {sport_code} games for today")
            else:
                result['messages'].append(f"No {sport_code} games scheduled for today")
            
            # Update sport-specific historical data
            if sport_code == 'MLB':
                # Update Baseball Savant data for MLB
                try:
                    savant_scraper = self._collectors['MLB']['data']
                    savant_data = savant_scraper.get_recent_games(days=days)
                    if not savant_data.empty:
                        result['data_updated'] = True
                        result['messages'].append(f"Updated {len(savant_data)} Baseball Savant records")
                except Exception as e:
                    result['errors'].append(f"Baseball Savant update failed: {str(e)}")
            
            elif sport_code in self._collectors and 'collector' in self._collectors[sport_code]:
                # For other sports, try to get recent data if available
                try:
                    collector = self._collectors[sport_code]['collector']
                    if hasattr(collector, 'get_recent_games'):
                        recent_data = collector.get_recent_games(days=days)
                        if not recent_data.empty:
                            result['data_updated'] = True
                            result['messages'].append(f"Updated {len(recent_data)} {sport_code} records")
                except Exception as e:
                    result['errors'].append(f"{sport_code} data update failed: {str(e)}")
            
            # Update odds data for this sport
            if self._odds_collector:
                try:
                    odds_data = self._odds_collector.get_sport_odds(sport_code)
                    if not odds_data.empty:
                        # Import db_manager to save odds
                        from ..data_storage.database import DatabaseManager
                        db_manager = DatabaseManager()
                        if db_manager.save_odds(odds_data, sport_code):
                            result['messages'].append(f"Updated {len(odds_data)} odds records for {sport_code}")
                        else:
                            result['errors'].append("Failed to save odds data")
                except Exception as e:
                    result['errors'].append(f"Odds update failed: {str(e)}")
            
            result['success'] = True
            return result
            
        except Exception as e:
            result['errors'].append(f"Error updating {sport_code} data: {str(e)}")
            return result
    
    def get_supported_sports(self) -> list:
        """Get list of supported sports."""
        return list(self._collectors.keys())
    
    def is_sport_supported(self, sport_code: str) -> bool:
        """Check if a sport is supported."""
        return sport_code in self._collectors
    
    def get_sport_display_name(self, sport_code: str) -> str:
        """Get display name for a sport."""
        display_names = {
            'MLB': 'Major League Baseball',
            'NBA': 'National Basketball Association',
            'NFL': 'National Football League', 
            'NHL': 'National Hockey League',
            'NCAA': 'NCAA Football',
            'NCAAB': 'NCAA Basketball',
            'WNBA': 'Women\'s National Basketball Association'
        }
        return display_names.get(sport_code, sport_code)
    
    def get_sport_emoji(self, sport_code: str) -> str:
        """Get emoji for a sport."""
        emojis = {
            'MLB': '⚾',
            'NBA': '🏀',
            'NFL': '🏈',
            'NHL': '🏒',
            'NCAA': '🏈',
            'NCAAB': '🏀',
            'WNBA': '🏀'
        }
        return emojis.get(sport_code, '🏆')