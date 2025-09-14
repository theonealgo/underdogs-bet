import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, List
import os

try:
    from pybaseball import statcast, statcast_pitcher, statcast_batter, playerid_lookup
    from pybaseball import team_batting, team_pitching, schedule_and_record
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    logging.warning("pybaseball not available. Baseball Savant functionality will be limited.")

class BaseballSavantScraper:
    """
    Scraper for Baseball Savant data using pybaseball library
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not PYBASEBALL_AVAILABLE:
            self.logger.error("pybaseball library is required for Baseball Savant data")
            
    def get_recent_games(self, days: int = 7) -> pd.DataFrame:
        """
        Get Statcast data for recent games
        
        Args:
            days: Number of days back to fetch data
            
        Returns:
            DataFrame with recent game data
        """
        if not PYBASEBALL_AVAILABLE:
            self.logger.error("pybaseball not available")
            return pd.DataFrame()
            
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self.logger.info(f"Fetching Statcast data from {start_date.date()} to {end_date.date()}")
            
            # Get statcast data with rate limiting
            data = self._fetch_with_retry(
                statcast,
                start_dt=start_date.strftime('%Y-%m-%d'),
                end_dt=end_date.strftime('%Y-%m-%d')
            )
            
            if data is not None and not data.empty:
                # Clean and process the data
                processed_data = self._process_statcast_data(data)
                self.logger.info(f"Successfully fetched {len(processed_data)} records")
                return processed_data
            else:
                self.logger.warning("No data returned from Baseball Savant")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching recent games: {str(e)}")
            return pd.DataFrame()
    
    def get_team_stats(self, year: int = None) -> Dict[str, pd.DataFrame]:
        """
        Get team batting and pitching statistics
        
        Args:
            year: Year to fetch stats for (defaults to current year)
            
        Returns:
            Dictionary with 'batting' and 'pitching' DataFrames
        """
        if not PYBASEBALL_AVAILABLE:
            return {}
            
        if year is None:
            year = datetime.now().year
            
        try:
            self.logger.info(f"Fetching team stats for {year}")
            
            # Get team batting stats
            batting_stats = self._fetch_with_retry(team_batting, year)
            time.sleep(1)  # Rate limiting
            
            # Get team pitching stats
            pitching_stats = self._fetch_with_retry(team_pitching, year)
            
            return {
                'batting': batting_stats if batting_stats is not None else pd.DataFrame(),
                'pitching': pitching_stats if pitching_stats is not None else pd.DataFrame()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching team stats: {str(e)}")
            return {}
    
    def get_player_data(self, player_name: str, last_name: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Get player-specific Statcast data
        
        Args:
            player_name: Player's first name
            last_name: Player's last name
            days: Number of days back to fetch
            
        Returns:
            Dictionary with player data
        """
        if not PYBASEBALL_AVAILABLE:
            return {}
            
        try:
            # Look up player ID
            player_lookup = playerid_lookup(last_name, player_name)
            if player_lookup.empty:
                self.logger.warning(f"Player not found: {player_name} {last_name}")
                return {}
            
            player_id = player_lookup.iloc[0]['key_mlbam']
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get pitcher data
            pitcher_data = self._fetch_with_retry(
                statcast_pitcher, start_str, end_str, player_id
            )
            time.sleep(1)
            
            # Get batter data
            batter_data = self._fetch_with_retry(
                statcast_batter, start_str, end_str, player_id
            )
            
            return {
                'pitcher': pitcher_data if pitcher_data is not None else pd.DataFrame(),
                'batter': batter_data if batter_data is not None else pd.DataFrame()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching player data: {str(e)}")
            return {}
    
    def get_schedule_data(self, team: str, year: int = None) -> pd.DataFrame:
        """
        Get team schedule and record data
        
        Args:
            team: Team abbreviation
            year: Year to fetch (defaults to current year)
            
        Returns:
            DataFrame with schedule data
        """
        if not PYBASEBALL_AVAILABLE:
            return pd.DataFrame()
            
        if year is None:
            year = datetime.now().year
            
        try:
            schedule_data = self._fetch_with_retry(schedule_and_record, year, team)
            return schedule_data if schedule_data is not None else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching schedule data: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_with_retry(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch data with retry logic and rate limiting
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            DataFrame or None if failed
        """
        max_retries = 3
        delay = 6  # seconds between requests
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(delay * attempt)  # Exponential backoff
                
                data = func(*args, **kwargs)
                
                if data is not None and not data.empty:
                    time.sleep(delay)  # Rate limiting
                    return data
                else:
                    self.logger.warning(f"Empty data returned on attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All attempts failed for function {func.__name__}")
                    
        return None
    
    def _process_statcast_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean Statcast data
        
        Args:
            data: Raw Statcast DataFrame
            
        Returns:
            Processed DataFrame
        """
        try:
            # Create a copy to avoid modifying original
            processed = data.copy()
            
            # Convert date column
            if 'game_date' in processed.columns:
                processed['game_date'] = pd.to_datetime(processed['game_date'])
            
            # Clean numeric columns
            numeric_columns = [
                'release_speed', 'launch_speed', 'launch_angle', 'hit_distance_sc',
                'release_spin_rate', 'plate_x', 'plate_z'
            ]
            
            for col in numeric_columns:
                if col in processed.columns:
                    processed[col] = pd.to_numeric(processed[col], errors='coerce')
            
            # Add derived features
            processed = self._add_derived_features(processed)
            
            # Filter out invalid data
            processed = processed.dropna(subset=['game_date'])
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing Statcast data: {str(e)}")
            return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to Statcast data
        
        Args:
            data: Processed DataFrame
            
        Returns:
            DataFrame with additional features
        """
        try:
            # Add game-level aggregations
            if 'game_pk' in data.columns:
                # Calculate game totals
                game_stats = data.groupby('game_pk').agg({
                    'game_date': 'first',
                    'home_team': 'first',
                    'away_team': 'first',
                    'home_score': 'max',
                    'away_score': 'max'
                }).reset_index()
                
                # Handle NA values before calculations
                game_stats['home_score'] = game_stats['home_score'].fillna(0)
                game_stats['away_score'] = game_stats['away_score'].fillna(0)
                
                game_stats['total_runs'] = game_stats['home_score'] + game_stats['away_score']
                game_stats['home_win'] = (game_stats['home_score'] > game_stats['away_score']).astype(int)
                
                # Merge back to main data
                data = data.merge(
                    game_stats[['game_pk', 'total_runs', 'home_win']], 
                    on='game_pk', 
                    how='left'
                )
            
            # Add pitch-level features
            if 'launch_speed' in data.columns and 'launch_angle' in data.columns:
                # Fill NA values before boolean operations
                data['launch_speed'] = data['launch_speed'].fillna(0)
                data['launch_angle'] = data['launch_angle'].fillna(0)
                
                # Calculate expected batting average (simplified)
                data['hard_hit'] = (data['launch_speed'] >= 95).astype(int)
                data['barrel'] = (
                    (data['launch_speed'] >= 98) & 
                    (data['launch_angle'].between(26, 30))
                ).astype(int)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding derived features: {str(e)}")
            return data
    
    def get_daily_summary(self, date: datetime = None) -> Dict:
        """
        Get daily summary of MLB games for a specific date
        
        Args:
            date: Date to get summary for (defaults to today)
            
        Returns:
            Dictionary with daily summary
        """
        if date is None:
            date = datetime.now()
        
        try:
            # Get data for the specific date
            date_str = date.strftime('%Y-%m-%d')
            data = self.get_recent_games(days=1)
            
            if data.empty:
                return {}
            
            # Filter for the specific date
            day_data = data[data['game_date'].dt.date == date.date()]
            
            if day_data.empty:
                return {}
            
            # Calculate summary statistics
            summary = {
                'date': date_str,
                'total_games': day_data['game_pk'].nunique(),
                'total_pitches': len(day_data),
                'avg_velocity': day_data['release_speed'].mean(),
                'home_win_rate': day_data.groupby('game_pk')['home_win'].first().mean(),
                'avg_total_runs': day_data.groupby('game_pk')['total_runs'].first().mean()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting daily summary: {str(e)}")
            return {}
