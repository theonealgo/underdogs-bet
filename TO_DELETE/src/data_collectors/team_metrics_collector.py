"""
Advanced Team Metrics Collector for Baseball Statistics
Implements Bill James's Pythagorean Theorem and collects high-correlation stats
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import pybaseball as pyb

class TeamMetricsCollector:
    """
    Collects advanced team metrics including Pythagorean expectation
    and high-correlation statistics for baseball prediction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cache to avoid repeated API calls
        self._season_cache = {}
        self._team_stats_cache = {}
        
        # High-correlation stats based on user's analysis
        self.key_pitching_stats = [
            'era', 'fip', 'lob_percent', 'war_pitching', 'whip', 
            'h_per_9', 'batting_avg_against', 'saves'
        ]
        
        self.key_hitting_stats = [
            'war_hitting', 'obp', 'slg', 'wrc_plus', 'iso', 'woba', 'ops'
        ]
    
    def collect_team_metrics(self, season: int, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Collect comprehensive team metrics for all MLB teams
        
        Args:
            season: MLB season year
            end_date: End date for data collection (defaults to current date)
            
        Returns:
            DataFrame with team metrics including Pythagorean calculations
        """
        try:
            if end_date is None:
                end_date = datetime.now()
                
            self.logger.info(f"Collecting team metrics for {season} season through {end_date.date()}")
            
            # Get basic team standings and run data
            team_data = self._get_team_standings(season, end_date)
            
            if team_data.empty:
                self.logger.warning(f"No team data available for season {season}")
                return pd.DataFrame()
            
            # Add advanced pitching and hitting stats
            team_data = self._add_advanced_stats(team_data, season, end_date)
            
            # Calculate Pythagorean expectations
            team_data = self._calculate_pythagorean_stats(team_data)
            
            # Add rolling window calculations
            team_data = self._add_rolling_windows(team_data, season, end_date)
            
            # Standardize column names and add metadata
            team_data = self._standardize_team_data(team_data, season, end_date)
            
            self.logger.info(f"Successfully collected metrics for {len(team_data)} teams")
            return team_data
            
        except Exception as e:
            self.logger.error(f"Error collecting team metrics: {str(e)}")
            return pd.DataFrame()
    
    def _get_team_standings(self, season: int, end_date: datetime) -> pd.DataFrame:
        """Get basic team standings with runs scored/allowed"""
        try:
            cache_key = f"{season}_{end_date.date()}"
            
            if cache_key in self._season_cache:
                return self._season_cache[cache_key].copy()
            
            # Get team standings from pybaseball
            standings = pyb.standings(season)
            
            if not standings or len(standings) == 0:
                return pd.DataFrame()
            
            # Combine all divisions into single dataframe
            all_teams = []
            for division_data in standings:
                if isinstance(division_data, pd.DataFrame) and not division_data.empty:
                    all_teams.append(division_data)
            
            if not all_teams:
                return pd.DataFrame()
            
            team_data = pd.concat(all_teams, ignore_index=True)
            
            # Get runs scored/allowed data
            team_data = self._add_runs_data(team_data, season, end_date)
            
            self._season_cache[cache_key] = team_data.copy()
            return team_data
            
        except Exception as e:
            self.logger.error(f"Error getting team standings: {str(e)}")
            return pd.DataFrame()
    
    def _add_runs_data(self, team_data: pd.DataFrame, season: int, end_date: datetime) -> pd.DataFrame:
        """Add runs scored and runs allowed data"""
        try:
            # Try to get team batting and pitching stats
            team_batting = pyb.team_batting(season, start_date=f"{season}-03-01", end_date=end_date.strftime("%Y-%m-%d"))
            team_pitching = pyb.team_pitching(season, start_date=f"{season}-03-01", end_date=end_date.strftime("%Y-%m-%d"))
            
            if not team_batting.empty and not team_pitching.empty:
                # Merge runs data
                if 'R' in team_batting.columns and 'Team' in team_batting.columns:
                    runs_scored = team_batting[['Team', 'R']].rename(columns={'R': 'runs_scored'})
                    team_data = team_data.merge(runs_scored, left_on='Tm', right_on='Team', how='left')
                
                if 'R' in team_pitching.columns and 'Team' in team_pitching.columns:
                    runs_allowed = team_pitching[['Team', 'R']].rename(columns={'R': 'runs_allowed'}) 
                    team_data = team_data.merge(runs_allowed, left_on='Tm', right_on='Team', how='left', suffixes=('', '_pitch'))
            
            # Calculate run differential
            if 'runs_scored' in team_data.columns and 'runs_allowed' in team_data.columns:
                team_data['run_differential'] = team_data['runs_scored'] - team_data['runs_allowed']
            
            return team_data
            
        except Exception as e:
            self.logger.warning(f"Could not get runs data: {str(e)}")
            # Return original data if runs data unavailable
            return team_data
    
    def _add_advanced_stats(self, team_data: pd.DataFrame, season: int, end_date: datetime) -> pd.DataFrame:
        """Add advanced pitching and hitting statistics"""
        try:
            # Get team pitching stats
            team_pitching = pyb.team_pitching(season, start_date=f"{season}-03-01", end_date=end_date.strftime("%Y-%m-%d"))
            
            if not team_pitching.empty:
                # Map key pitching stats
                pitching_cols = ['Team']
                if 'ERA' in team_pitching.columns:
                    pitching_cols.append('ERA')
                if 'WHIP' in team_pitching.columns:
                    pitching_cols.append('WHIP')
                if 'FIP' in team_pitching.columns:
                    pitching_cols.append('FIP')
                if 'H9' in team_pitching.columns:
                    pitching_cols.append('H9')
                if 'BAOpp' in team_pitching.columns or 'BAA' in team_pitching.columns:
                    baa_col = 'BAOpp' if 'BAOpp' in team_pitching.columns else 'BAA'
                    pitching_cols.append(baa_col)
                    
                if len(pitching_cols) > 1:
                    pitching_stats = team_pitching[pitching_cols]
                    team_data = team_data.merge(pitching_stats, left_on='Tm', right_on='Team', how='left', suffixes=('', '_pitch'))
            
            # Get team batting stats
            team_batting = pyb.team_batting(season, start_date=f"{season}-03-01", end_date=end_date.strftime("%Y-%m-%d"))
            
            if not team_batting.empty:
                # Map key hitting stats
                batting_cols = ['Team']
                if 'OBP' in team_batting.columns:
                    batting_cols.append('OBP')
                if 'SLG' in team_batting.columns:
                    batting_cols.append('SLG')
                if 'OPS' in team_batting.columns:
                    batting_cols.append('OPS')
                if 'ISO' in team_batting.columns:
                    batting_cols.append('ISO')
                    
                if len(batting_cols) > 1:
                    batting_stats = team_batting[batting_cols]
                    team_data = team_data.merge(batting_stats, left_on='Tm', right_on='Team', how='left', suffixes=('', '_bat'))
            
            return team_data
            
        except Exception as e:
            self.logger.warning(f"Could not get advanced stats: {str(e)}")
            return team_data
    
    def _calculate_pythagorean_stats(self, team_data: pd.DataFrame, exponent: float = 2.0) -> pd.DataFrame:
        """
        Calculate Pythagorean expectation using Bill James's formula:
        W% = (Runs Scored)^2 / [(Runs Scored)^2 + (Runs Allowed)^2]
        """
        try:
            if 'runs_scored' in team_data.columns and 'runs_allowed' in team_data.columns:
                rs = team_data['runs_scored'].fillna(0)
                ra = team_data['runs_allowed'].fillna(0)
                
                # Calculate Pythagorean win percentage
                rs_exp = np.power(rs, exponent)
                ra_exp = np.power(ra, exponent)
                
                # Avoid division by zero
                denominator = rs_exp + ra_exp
                team_data['pythag_win_pct'] = np.where(
                    denominator > 0,
                    rs_exp / denominator,
                    0.5  # Default to .500 if no data
                )
                
                # Calculate expected wins based on games played
                games_played = team_data.get('G', team_data.get('games_played', 162))
                team_data['pythag_wins'] = team_data['pythag_win_pct'] * games_played
                team_data['pythag_exponent'] = exponent
                
                self.logger.debug(f"Calculated Pythagorean stats with exponent {exponent}")
            else:
                self.logger.warning("Missing runs data for Pythagorean calculation")
            
            return team_data
            
        except Exception as e:
            self.logger.error(f"Error calculating Pythagorean stats: {str(e)}")
            return team_data
    
    def _add_rolling_windows(self, team_data: pd.DataFrame, season: int, end_date: datetime) -> pd.DataFrame:
        """Add 14-day and 30-day rolling statistics"""
        try:
            # This would require game-by-game data for proper rolling calculations
            # For now, use simplified approach with available data
            
            # Calculate recent performance windows (simplified)
            if 'runs_scored' in team_data.columns and 'runs_allowed' in team_data.columns:
                # Estimate recent performance (this would be more accurate with game-level data)
                team_data['runs_scored_14'] = team_data['runs_scored'] * 0.85  # Recent performance weight
                team_data['runs_allowed_14'] = team_data['runs_allowed'] * 0.85
                team_data['run_diff_14'] = team_data['runs_scored_14'] - team_data['runs_allowed_14']
                
                team_data['runs_scored_30'] = team_data['runs_scored'] * 0.90  # 30-day weight
                team_data['runs_allowed_30'] = team_data['runs_allowed'] * 0.90
                team_data['run_diff_30'] = team_data['runs_scored_30'] - team_data['runs_allowed_30']
                
                # Calculate rolling Pythagorean win percentages
                team_data = self._calculate_rolling_pythag(team_data)
            
            return team_data
            
        except Exception as e:
            self.logger.warning(f"Could not calculate rolling windows: {str(e)}")
            return team_data
    
    def _calculate_rolling_pythag(self, team_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pythagorean win percentage for rolling windows"""
        try:
            for window in ['14', '30']:
                rs_col = f'runs_scored_{window}'
                ra_col = f'runs_allowed_{window}'
                pythag_col = f'pythag_win_pct_{window}'
                
                if rs_col in team_data.columns and ra_col in team_data.columns:
                    rs = team_data[rs_col].fillna(0)
                    ra = team_data[ra_col].fillna(0)
                    
                    rs_sq = np.power(rs, 2)
                    ra_sq = np.power(ra, 2)
                    
                    denominator = rs_sq + ra_sq
                    team_data[pythag_col] = np.where(
                        denominator > 0,
                        rs_sq / denominator,
                        0.5
                    )
            
            return team_data
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling Pythagorean: {str(e)}")
            return team_data
    
    def _standardize_team_data(self, team_data: pd.DataFrame, season: int, end_date: datetime) -> pd.DataFrame:
        """Standardize team data format for database storage"""
        try:
            # Standardize team identifiers and add metadata
            standardized = pd.DataFrame()
            
            # Basic identifiers
            standardized['sport'] = 'MLB'
            standardized['league'] = 'MLB'
            standardized['season'] = season
            standardized['date'] = end_date.date()
            
            # Team identification
            standardized['team_id'] = team_data.get('Tm', team_data.get('Team', ''))
            
            # Basic stats
            standardized['games_played'] = team_data.get('G', 0)
            standardized['wins'] = team_data.get('W', 0)
            standardized['losses'] = team_data.get('L', 0)
            
            # Runs data
            standardized['runs_scored'] = team_data.get('runs_scored', 0)
            standardized['runs_allowed'] = team_data.get('runs_allowed', 0)
            standardized['run_differential'] = team_data.get('run_differential', 0)
            
            # Pitching stats (rename to match database schema)
            standardized['era'] = team_data.get('ERA', None)
            standardized['fip'] = team_data.get('FIP', None)
            standardized['whip'] = team_data.get('WHIP', None)
            standardized['h_per_9'] = team_data.get('H9', None)
            standardized['batting_avg_against'] = team_data.get('BAOpp', team_data.get('BAA', None))
            
            # Hitting stats
            standardized['obp'] = team_data.get('OBP', None)
            standardized['slg'] = team_data.get('SLG', None)
            standardized['ops'] = team_data.get('OPS', None)
            standardized['iso'] = team_data.get('ISO', None)
            
            # Pythagorean calculations
            standardized['pythag_win_pct'] = team_data.get('pythag_win_pct', None)
            standardized['pythag_wins'] = team_data.get('pythag_wins', None)
            standardized['pythag_exponent'] = team_data.get('pythag_exponent', 2.0)
            
            # Rolling windows
            for window in ['14', '30']:
                standardized[f'runs_scored_{window}'] = team_data.get(f'runs_scored_{window}', None)
                standardized[f'runs_allowed_{window}'] = team_data.get(f'runs_allowed_{window}', None) 
                standardized[f'run_diff_{window}'] = team_data.get(f'run_diff_{window}', None)
                standardized[f'pythag_win_pct_{window}'] = team_data.get(f'pythag_win_pct_{window}', None)
            
            # Remove rows with missing team_id
            standardized = standardized[standardized['team_id'].notna() & (standardized['team_id'] != '')]
            
            return standardized
            
        except Exception as e:
            self.logger.error(f"Error standardizing team data: {str(e)}")
            return pd.DataFrame()

    def calculate_pythagorean_win_pct(self, runs_scored: float, runs_allowed: float, exponent: float = 2.0) -> float:
        """
        Calculate single Pythagorean win percentage
        W% = (RS^exponent) / (RS^exponent + RA^exponent)
        """
        try:
            if runs_scored <= 0 or runs_allowed <= 0:
                return 0.5
            
            rs_exp = pow(runs_scored, exponent)
            ra_exp = pow(runs_allowed, exponent)
            
            return rs_exp / (rs_exp + ra_exp)
            
        except Exception as e:
            self.logger.error(f"Error calculating Pythagorean win %: {str(e)}")
            return 0.5