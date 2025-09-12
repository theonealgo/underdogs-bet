import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from src.interfaces.base_feature_engineer import BaseFeatureEngineer


class MLBFeatureEngineer(BaseFeatureEngineer):
    """
    MLB-specific feature engineering for baseball prediction models.
    
    Creates features specific to baseball analytics including:
    - Pitching and batting advanced metrics
    - Statcast-derived features (exit velocity, launch angle, etc.)
    - Situational baseball factors (leverage, WPA, etc.)
    - Weather and ballpark effects
    - Bullpen and rotation strength
    """
    
    def __init__(self):
        super().__init__('MLB')
        self.logger = logging.getLogger(__name__)
        
        # MLB-specific feature definitions
        self.required_features = [
            # Basic game features
            'home_team_encoded', 'away_team_encoded', 'month', 'day_of_week', 'is_weekend',
            'is_playoffs', 'days_rest_home', 'days_rest_away',
            
            # Team performance features
            'home_win_pct_season', 'away_win_pct_season', 'home_runs_per_game', 'away_runs_per_game',
            'home_runs_allowed_per_game', 'away_runs_allowed_per_game',
            
            # Baseball-specific pitching features
            'home_era', 'away_era', 'home_whip', 'away_whip', 'home_strikeouts_per_9', 'away_strikeouts_per_9',
            'home_walks_per_9', 'away_walks_per_9', 'home_hr_per_9', 'away_hr_per_9',
            
            # Baseball-specific batting features
            'home_batting_avg', 'away_batting_avg', 'home_obp', 'away_obp', 'home_slg', 'away_slg',
            'home_ops', 'away_ops', 'home_woba', 'away_woba', 'home_wrc_plus', 'away_wrc_plus',
            
            # Advanced Statcast features
            'home_exit_velocity', 'away_exit_velocity', 'home_hard_hit_rate', 'away_hard_hit_rate',
            'home_barrel_rate', 'away_barrel_rate', 'home_launch_angle', 'away_launch_angle',
            
            # Situational features
            'home_clutch_performance', 'away_clutch_performance', 'home_bullpen_era', 'away_bullpen_era',
            'ballpark_factor', 'weather_impact', 'rest_advantage',
            
            # Matchup features
            'pitching_matchup', 'hitting_vs_pitching', 'bullpen_advantage',
            'home_field_advantage', 'total_prediction', 'h2h_home_wins', 'h2h_total_games'
        ]
        
        self.feature_descriptions = {
            'home_runs_per_game': 'Average runs scored per game by home team',
            'away_runs_per_game': 'Average runs scored per game by away team',
            'home_era': 'Earned run average for home team pitching staff',
            'away_era': 'Earned run average for away team pitching staff',
            'home_whip': 'Walks plus hits per inning pitched for home team',
            'away_whip': 'Walks plus hits per inning pitched for away team',
            'home_ops': 'On-base plus slugging percentage for home team',
            'away_ops': 'On-base plus slugging percentage for away team',
            'home_woba': 'Weighted on-base average for home team',
            'away_woba': 'Weighted on-base average for away team',
            'home_exit_velocity': 'Average exit velocity off the bat for home team',
            'away_exit_velocity': 'Average exit velocity off the bat for away team',
            'home_barrel_rate': 'Percentage of batted balls with ideal launch conditions',
            'away_barrel_rate': 'Percentage of batted balls with ideal launch conditions',
            'ballpark_factor': 'Park factor affecting offensive production',
            'weather_impact': 'Weather conditions impact on game (wind, temperature)',
            'pitching_matchup': 'Quality of pitching matchup (starter vs opposing offense)',
            'home_field_advantage': 'Estimated home field advantage in runs',
            'days_rest_home': 'Days of rest for home team since last game',
            'days_rest_away': 'Days of rest for away team since last game'
        }
    
    def create_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame,
                       additional_data: Optional[Dict] = None,
                       include_targets: bool = True) -> pd.DataFrame:
        """
        Create comprehensive MLB features for prediction models.
        
        Args:
            games_df: DataFrame with MLB game information
            team_stats_df: DataFrame with MLB team statistics
            additional_data: Optional additional data (weather, Statcast, etc.)
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if games_df.empty:
                self.logger.warning("Empty games DataFrame provided")
                return games_df
            
            self.logger.info(f"Creating MLB features for {len(games_df)} games")
            features_df = games_df.copy()
            
            # Basic game features
            features_df = self._add_basic_features(features_df)
            
            # MLB-specific team performance features
            features_df = self._add_team_performance_features(features_df, team_stats_df)
            
            # Baseball-specific advanced features
            features_df = self._add_baseball_advanced_features(features_df, team_stats_df)
            
            # Statcast features
            features_df = self._add_statcast_features(features_df, additional_data)
            
            # Situational features (weather, ballpark, etc.)
            features_df = self._add_situational_features(features_df, additional_data)
            
            # Rolling/trend features
            features_df = self._add_rolling_features(features_df, team_stats_df)
            
            # Head-to-head and matchup features
            features_df = self._add_matchup_features(features_df, team_stats_df)
            
            # Target variables
            if include_targets:
                features_df = self._add_target_variables(features_df)
            
            # Handle missing values
            features_df = self.handle_missing_values(features_df)
            
            self.logger.info(f"Created {len(features_df.columns)} MLB features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating MLB features: {str(e)}")
            return games_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic MLB game features"""
        try:
            # Date features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['month'] = df['game_date'].dt.month
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                
                # MLB season features (April-October)
                df['is_playoffs'] = ((df['month'] >= 10) | (df['month'] <= 2)).astype(int)
                df['is_early_season'] = (df['month'] <= 5).astype(int)
                df['is_late_season'] = (df['month'] >= 8).astype(int)
                
                # Day/night games
                if 'game_time' in df.columns:
                    df['is_day_game'] = df['game_time'].str.contains('PM', case=False, na=False).astype(int)
                else:
                    df['is_day_game'] = 0  # Default to night
            
            # Team encoding
            if 'home_team_id' in df.columns and 'away_team_id' in df.columns:
                all_teams = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
                team_map = {team: idx for idx, team in enumerate(sorted(all_teams))}
                
                df['home_team_encoded'] = df['home_team_id'].map(lambda x: team_map.get(x, -1)).fillna(-1).astype(int)
                df['away_team_encoded'] = df['away_team_id'].map(lambda x: team_map.get(x, -1)).fillna(-1).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basic MLB features: {str(e)}")
            return df
    
    def _add_team_performance_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance and recent form features"""
        try:
            if team_stats_df.empty:
                self.logger.warning("No team stats provided for performance features")
                return df
            
            # Get latest team stats
            latest_stats = team_stats_df.groupby('team_id').last().reset_index()
            
            for prefix in ['home', 'away']:
                team_col = f'{prefix}_team_id'
                
                if team_col in df.columns:
                    # Merge team stats
                    team_data = latest_stats.rename(columns={
                        'team_id': team_col,
                        'wins': f'{prefix}_wins',
                        'losses': f'{prefix}_losses',
                        'win_percentage': f'{prefix}_win_pct_season',
                        'runs_per_game': f'{prefix}_runs_per_game',
                        'runs_allowed_per_game': f'{prefix}_runs_allowed_per_game'
                    })
                    
                    df = df.merge(team_data[[team_col, f'{prefix}_wins', f'{prefix}_losses', 
                                           f'{prefix}_win_pct_season', f'{prefix}_runs_per_game', 
                                           f'{prefix}_runs_allowed_per_game']], 
                                 on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding team performance features: {str(e)}")
            return df
    
    def _add_baseball_advanced_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add MLB-specific advanced baseball statistics"""
        try:
            if team_stats_df.empty:
                return df
            
            # Get latest advanced stats
            latest_stats = team_stats_df.groupby('team_id').last().reset_index()
            
            for prefix in ['home', 'away']:
                team_col = f'{prefix}_team_id'
                
                if team_col in df.columns:
                    team_data = latest_stats.copy()
                    
                    # Pitching statistics
                    team_data[f'{prefix}_era'] = team_data.get('team_era', 4.00)
                    team_data[f'{prefix}_whip'] = team_data.get('team_whip', 1.30)
                    team_data[f'{prefix}_strikeouts_per_9'] = team_data.get('strikeouts_per_9', 8.5)
                    team_data[f'{prefix}_walks_per_9'] = team_data.get('walks_per_9', 3.2)
                    team_data[f'{prefix}_hr_per_9'] = team_data.get('home_runs_per_9', 1.2)
                    
                    # Batting statistics
                    team_data[f'{prefix}_batting_avg'] = team_data.get('batting_average', 0.250)
                    team_data[f'{prefix}_obp'] = team_data.get('on_base_percentage', 0.320)
                    team_data[f'{prefix}_slg'] = team_data.get('slugging_percentage', 0.420)
                    team_data[f'{prefix}_ops'] = team_data[f'{prefix}_obp'] + team_data[f'{prefix}_slg']
                    
                    # Advanced batting metrics
                    team_data[f'{prefix}_woba'] = team_data.get('weighted_on_base_average', 0.320)
                    team_data[f'{prefix}_wrc_plus'] = team_data.get('weighted_runs_created_plus', 100)
                    
                    # Bullpen statistics
                    team_data[f'{prefix}_bullpen_era'] = team_data.get('bullpen_era', 4.20)
                    
                    # Situational performance
                    team_data[f'{prefix}_clutch_performance'] = team_data.get('clutch_performance', 0.0)
                    
                    # Merge advanced stats
                    merge_cols = [team_col] + [col for col in team_data.columns if col.startswith(prefix)]
                    team_data = team_data.rename(columns={'team_id': team_col})
                    
                    df = df.merge(team_data[merge_cols], on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding baseball advanced features: {str(e)}")
            return df
    
    def _add_statcast_features(self, df: pd.DataFrame, additional_data: Optional[Dict] = None) -> pd.DataFrame:
        """Add Statcast-derived features"""
        try:
            # Statcast data if available
            if additional_data and 'statcast' in additional_data:
                statcast_data = additional_data['statcast']
                
                for prefix in ['home', 'away']:
                    team_col = f'{prefix}_team_id'
                    if team_col in df.columns:
                        df[f'{prefix}_exit_velocity'] = df[team_col].map(
                            lambda x: statcast_data.get(x, {}).get('exit_velocity', 89.0)
                        )
                        df[f'{prefix}_hard_hit_rate'] = df[team_col].map(
                            lambda x: statcast_data.get(x, {}).get('hard_hit_rate', 35.0)
                        )
                        df[f'{prefix}_barrel_rate'] = df[team_col].map(
                            lambda x: statcast_data.get(x, {}).get('barrel_rate', 6.0)
                        )
                        df[f'{prefix}_launch_angle'] = df[team_col].map(
                            lambda x: statcast_data.get(x, {}).get('launch_angle', 12.0)
                        )
            else:
                # Default Statcast values
                for prefix in ['home', 'away']:
                    df[f'{prefix}_exit_velocity'] = 89.0  # MLB average
                    df[f'{prefix}_hard_hit_rate'] = 35.0
                    df[f'{prefix}_barrel_rate'] = 6.0
                    df[f'{prefix}_launch_angle'] = 12.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding Statcast features: {str(e)}")
            return df
    
    def _add_situational_features(self, df: pd.DataFrame, additional_data: Optional[Dict] = None) -> pd.DataFrame:
        """Add situational features like weather and ballpark factors"""
        try:
            # Calculate rest days (MLB teams play almost daily)
            if 'game_date' in df.columns:
                df = df.sort_values(['home_team_id', 'game_date'])
                
                # Calculate rest days for home team
                df['home_prev_game_date'] = df.groupby('home_team_id')['game_date'].shift(1)
                df['days_rest_home'] = (df['game_date'] - df['home_prev_game_date']).dt.days
                df['days_rest_home'] = df['days_rest_home'].fillna(1)  # Default 1 day rest
                
                # Calculate rest days for away team  
                df = df.sort_values(['away_team_id', 'game_date'])
                df['away_prev_game_date'] = df.groupby('away_team_id')['game_date'].shift(1)
                df['days_rest_away'] = (df['game_date'] - df['away_prev_game_date']).dt.days
                df['days_rest_away'] = df['days_rest_away'].fillna(1)  # Default 1 day rest
                
                # Rest advantage
                df['rest_advantage'] = df['days_rest_home'] - df['days_rest_away']
            
            # Ballpark factors
            ballpark_factors = {
                'COL': 1.2,  # Coors Field (hitter friendly)
                'TEX': 1.1,  # Globe Life Park (hitter friendly)
                'SD': 0.9,   # Petco Park (pitcher friendly)
                'SEA': 0.9,  # T-Mobile Park (pitcher friendly)
            }
            
            if 'home_team_id' in df.columns:
                df['ballpark_factor'] = df['home_team_id'].map(
                    lambda x: ballpark_factors.get(x, 1.0)
                )
            else:
                df['ballpark_factor'] = 1.0
            
            # Weather impact
            if additional_data and 'weather' in additional_data:
                weather_data = additional_data['weather']
                df['temperature'] = weather_data.get('temperature', 75)
                df['wind_speed'] = weather_data.get('wind_speed', 5)
                df['wind_direction'] = weather_data.get('wind_direction', 'calm')
                
                # Weather impact calculation
                df['weather_impact'] = (
                    (df['temperature'] - 75) * 0.01 +  # Temperature effect
                    np.where(df['wind_direction'] == 'out', df['wind_speed'] * 0.02, 0) -  # Wind out helps
                    np.where(df['wind_direction'] == 'in', df['wind_speed'] * 0.02, 0)     # Wind in hurts
                )
            else:
                df['weather_impact'] = 0.0
            
            # Home field advantage (MLB typically ~0.1-0.2 runs)
            df['home_field_advantage'] = 0.15
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding situational features: {str(e)}")
            return df
    
    def _add_rolling_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling/trend features"""
        try:
            # Use team stats as proxy for recent form
            for prefix in ['home', 'away']:
                if f'{prefix}_runs_per_game' in df.columns:
                    # Recent form indicators (simplified)
                    df[f'{prefix}_runs_per_game_l10'] = df[f'{prefix}_runs_per_game']
                    df[f'{prefix}_runs_allowed_per_game_l10'] = df[f'{prefix}_runs_allowed_per_game']
                    df[f'{prefix}_era_l10'] = df[f'{prefix}_era']
                    df[f'{prefix}_ops_l10'] = df[f'{prefix}_ops']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {str(e)}")
            return df
    
    def _add_matchup_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features"""
        try:
            # Pitching vs batting matchup
            if 'home_era' in df.columns and 'away_ops' in df.columns:
                df['pitching_matchup'] = (4.00 - df['home_era']) - (df['away_ops'] - 0.750)  # Normalized
            else:
                df['pitching_matchup'] = 0
            
            # Hitting vs pitching matchup
            if 'home_ops' in df.columns and 'away_era' in df.columns:
                df['hitting_vs_pitching'] = (df['home_ops'] - 0.750) - (4.00 - df['away_era'])  # Normalized
            else:
                df['hitting_vs_pitching'] = 0
            
            # Bullpen advantage
            if 'home_bullpen_era' in df.columns and 'away_bullpen_era' in df.columns:
                df['bullpen_advantage'] = df['away_bullpen_era'] - df['home_bullpen_era']
            else:
                df['bullpen_advantage'] = 0
            
            # Head-to-head record (simplified)
            df['h2h_home_wins'] = 0
            df['h2h_total_games'] = 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding matchup features: {str(e)}")
            return df
    
    def _add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variables for training"""
        try:
            # Winner target (1 if home wins, 0 if away wins)
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df['target_winner'] = (df['home_score'] > df['away_score']).astype(int)
                
                # Total runs target
                df['target_total'] = df['home_score'] + df['away_score']
                
                # Run line target (typically -1.5 for home team)
                df['target_run_line'] = (df['home_score'] - df['away_score'] > 1.5).astype(int)
            else:
                # For inference, these will be filled later
                df['target_winner'] = None
                df['target_total'] = None
                df['target_run_line'] = None
            
            # Estimated total for o/u predictions
            if 'home_runs_per_game' in df.columns and 'away_runs_per_game' in df.columns:
                df['total_prediction'] = (
                    df['home_runs_per_game'] + df['away_runs_per_game'] +
                    df.get('ballpark_factor', 1.0) - 1.0 +  # Park adjustment
                    df.get('weather_impact', 0.0)  # Weather adjustment
                )
            else:
                df['total_prediction'] = 8.5  # MLB average
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding target variables: {str(e)}")
            return df
    
    def create_rolling_features(self, team_stats_df: pd.DataFrame,
                               window_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """Create rolling/moving average features"""
        if window_sizes is None:
            window_sizes = [7, 14, 30]  # MLB plays daily
        
        try:
            df = team_stats_df.copy()
            
            if 'date' in df.columns:
                df = df.sort_values(['team_id', 'date'])
                
                # Rolling averages for key stats
                for window in window_sizes:
                    for stat in ['runs_per_game', 'runs_allowed_per_game', 'team_era', 'batting_average', 'on_base_percentage']:
                        if stat in df.columns:
                            df[f'{stat}_rolling_{window}'] = df.groupby('team_id')[stat].transform(
                                lambda x: x.rolling(window=window, min_periods=1).mean()
                            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating rolling features: {str(e)}")
            return team_stats_df
    
    def create_matchup_features(self, home_team_id: str, away_team_id: str,
                               team_stats_df: pd.DataFrame,
                               h2h_data: Optional[pd.DataFrame] = None) -> Dict:
        """Create features specific to team matchup"""
        try:
            features = {}
            
            # Get team stats
            home_stats = team_stats_df[team_stats_df['team_id'] == home_team_id]
            away_stats = team_stats_df[team_stats_df['team_id'] == away_team_id]
            
            if not home_stats.empty and not away_stats.empty:
                home_latest = home_stats.iloc[-1]
                away_latest = away_stats.iloc[-1]
                
                # Pitching vs hitting matchup
                features['pitching_advantage'] = (
                    away_latest.get('team_era', 4.0) - home_latest.get('team_era', 4.0)
                )
                
                # Offensive power matchup
                features['offensive_advantage'] = (
                    home_latest.get('on_base_percentage', 0.32) + home_latest.get('slugging_percentage', 0.42) -
                    away_latest.get('on_base_percentage', 0.32) - away_latest.get('slugging_percentage', 0.42)
                )
                
                # Speed vs defense
                features['speed_vs_defense'] = (
                    home_latest.get('stolen_bases_per_game', 0.8) - 
                    away_latest.get('caught_stealing_allowed', 0.3)
                )
            
            # Head-to-head history
            if h2h_data is not None and not h2h_data.empty:
                home_wins = len(h2h_data[h2h_data['winner_id'] == home_team_id])
                total_games = len(h2h_data)
                features['h2h_home_win_pct'] = home_wins / max(total_games, 1)
                features['h2h_total_games'] = total_games
                
                # Recent head-to-head performance
                recent_h2h = h2h_data.tail(10)
                if not recent_h2h.empty:
                    recent_home_wins = len(recent_h2h[recent_h2h['winner_id'] == home_team_id])
                    features['h2h_recent_home_win_pct'] = recent_home_wins / len(recent_h2h)
            else:
                features['h2h_home_win_pct'] = 0.5
                features['h2h_total_games'] = 0
                features['h2h_recent_home_win_pct'] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating matchup features: {str(e)}")
            return {}
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by importance categories"""
        return {
            'offense': [
                'home_runs_per_game', 'away_runs_per_game', 'home_batting_avg', 'away_batting_avg',
                'home_obp', 'away_obp', 'home_slg', 'away_slg', 'home_ops', 'away_ops',
                'home_woba', 'away_woba', 'home_wrc_plus', 'away_wrc_plus'
            ],
            'defense': [
                'home_runs_allowed_per_game', 'away_runs_allowed_per_game', 'home_era', 'away_era',
                'home_whip', 'away_whip', 'home_strikeouts_per_9', 'away_strikeouts_per_9',
                'home_walks_per_9', 'away_walks_per_9', 'home_hr_per_9', 'away_hr_per_9'
            ],
            'statcast': [
                'home_exit_velocity', 'away_exit_velocity', 'home_hard_hit_rate', 'away_hard_hit_rate',
                'home_barrel_rate', 'away_barrel_rate', 'home_launch_angle', 'away_launch_angle'
            ],
            'situational': [
                'is_playoffs', 'is_day_game', 'days_rest_home', 'days_rest_away', 
                'ballpark_factor', 'weather_impact', 'home_field_advantage', 'is_weekend'
            ],
            'bullpen': [
                'home_bullpen_era', 'away_bullpen_era', 'bullpen_advantage'
            ],
            'form': [
                'home_win_pct_season', 'away_win_pct_season', 'home_clutch_performance', 'away_clutch_performance'
            ],
            'matchup': [
                'pitching_matchup', 'hitting_vs_pitching', 'h2h_home_wins', 'h2h_total_games'
            ]
        }