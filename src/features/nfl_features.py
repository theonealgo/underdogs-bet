import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from src.interfaces.base_feature_engineer import BaseFeatureEngineer


class NFLFeatureEngineer(BaseFeatureEngineer):
    """
    NFL-specific feature engineering for football prediction models.
    
    Creates features specific to football analytics including:
    - Rushing and passing efficiency metrics
    - Defensive statistics and turnover rates
    - Red zone and third down conversion rates
    - Weather and field conditions
    - Injury and roster strength factors
    """
    
    def __init__(self):
        super().__init__('NFL')
        self.logger = logging.getLogger(__name__)
        
        # NFL-specific feature definitions
        self.required_features = [
            # Basic game features
            'home_team_encoded', 'away_team_encoded', 'month', 'day_of_week', 'is_weekend',
            'is_playoffs', 'days_rest_home', 'days_rest_away', 'week_of_season',
            
            # Team performance features
            'home_win_pct_season', 'away_win_pct_season', 'home_points_per_game', 'away_points_per_game',
            'home_points_allowed_per_game', 'away_points_allowed_per_game',
            
            # Football-specific offensive features
            'home_passing_yards_per_game', 'away_passing_yards_per_game', 'home_rushing_yards_per_game', 'away_rushing_yards_per_game',
            'home_total_yards_per_game', 'away_total_yards_per_game', 'home_turnover_differential', 'away_turnover_differential',
            
            # Football-specific defensive features
            'home_passing_yards_allowed', 'away_passing_yards_allowed', 'home_rushing_yards_allowed', 'away_rushing_yards_allowed',
            'home_sacks_per_game', 'away_sacks_per_game', 'home_interceptions_per_game', 'away_interceptions_per_game',
            
            # Efficiency metrics
            'home_red_zone_pct', 'away_red_zone_pct', 'home_third_down_pct', 'away_third_down_pct',
            'home_time_of_possession', 'away_time_of_possession',
            
            # Matchup features
            'offensive_matchup_advantage', 'defensive_matchup_advantage', 'turnover_differential',
            'home_field_advantage', 'total_prediction', 'h2h_home_wins', 'h2h_total_games'
        ]
        
        self.feature_descriptions = {
            'home_points_per_game': 'Average points scored per game by home team',
            'away_points_per_game': 'Average points scored per game by away team',
            'home_passing_yards_per_game': 'Average passing yards per game by home team',
            'away_passing_yards_per_game': 'Average passing yards per game by away team',
            'home_rushing_yards_per_game': 'Average rushing yards per game by home team',
            'away_rushing_yards_per_game': 'Average rushing yards per game by away team',
            'home_turnover_differential': 'Turnovers forced minus turnovers committed by home team',
            'away_turnover_differential': 'Turnovers forced minus turnovers committed by away team',
            'home_red_zone_pct': 'Red zone touchdown conversion percentage for home team',
            'away_red_zone_pct': 'Red zone touchdown conversion percentage for away team',
            'home_third_down_pct': 'Third down conversion percentage for home team',
            'away_third_down_pct': 'Third down conversion percentage for away team',
            'week_of_season': 'Week number in NFL season (1-18 regular season, 19+ playoffs)',
            'days_rest_home': 'Days of rest for home team since last game',
            'days_rest_away': 'Days of rest for away team since last game',
            'home_field_advantage': 'Estimated home field advantage points',
            'offensive_matchup_advantage': 'Home offense vs away defense matchup rating',
            'defensive_matchup_advantage': 'Home defense vs away offense matchup rating'
        }
    
    def create_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame,
                       additional_data: Optional[Dict] = None,
                       include_targets: bool = True) -> pd.DataFrame:
        """
        Create comprehensive NFL features for prediction models.
        
        Args:
            games_df: DataFrame with NFL game information
            team_stats_df: DataFrame with NFL team statistics
            additional_data: Optional additional data (weather, injuries, etc.)
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if games_df.empty:
                self.logger.warning("Empty games DataFrame provided")
                return games_df
            
            self.logger.info(f"Creating NFL features for {len(games_df)} games")
            features_df = games_df.copy()
            
            # Basic game features
            features_df = self._add_basic_features(features_df)
            
            # NFL-specific team performance features
            features_df = self._add_team_performance_features(features_df, team_stats_df)
            
            # Football-specific advanced features
            features_df = self._add_football_advanced_features(features_df, team_stats_df)
            
            # Situational features (rest, weather, etc.)
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
            
            self.logger.info(f"Created {len(features_df.columns)} NFL features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating NFL features: {str(e)}")
            return games_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic NFL game features"""
        try:
            # Date features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['month'] = df['game_date'].dt.month
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                
                # NFL season features (Sept-Feb)
                df['is_playoffs'] = ((df['month'] <= 2) | (df['month'] == 12)).astype(int)
                
                # Week of season (estimated)
                df['week_of_season'] = np.select([
                    df['month'] == 9,   # September
                    df['month'] == 10,  # October  
                    df['month'] == 11,  # November
                    df['month'] == 12,  # December
                    df['month'] == 1,   # January (playoffs)
                    df['month'] == 2    # February (playoffs)
                ], [1, 5, 9, 13, 19, 22], default=1)
            
            # Team encoding
            if 'home_team_id' in df.columns and 'away_team_id' in df.columns:
                all_teams = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
                team_map = {team: idx for idx, team in enumerate(sorted(all_teams))}
                
                # Use replace instead of map to avoid type checker issues
                df['home_team_encoded'] = df['home_team_id'].replace(team_map).fillna(-1).astype(int)
                df['away_team_encoded'] = df['away_team_id'].replace(team_map).fillna(-1).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basic NFL features: {str(e)}")
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
                        'points_per_game': f'{prefix}_points_per_game',
                        'points_against_per_game': f'{prefix}_points_allowed_per_game'
                    })
                    
                    df = df.merge(team_data[[team_col, f'{prefix}_wins', f'{prefix}_losses', 
                                           f'{prefix}_win_pct_season', f'{prefix}_points_per_game', 
                                           f'{prefix}_points_allowed_per_game']], 
                                 on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding team performance features: {str(e)}")
            return df
    
    def _add_football_advanced_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add NFL-specific advanced football statistics"""
        try:
            if team_stats_df.empty:
                return df
            
            # Get latest advanced stats
            latest_stats = team_stats_df.groupby('team_id').last().reset_index()
            
            for prefix in ['home', 'away']:
                team_col = f'{prefix}_team_id'
                
                if team_col in df.columns:
                    team_data = latest_stats.copy()
                    
                    # Offensive statistics
                    team_data[f'{prefix}_passing_yards_per_game'] = team_data.get('passing_yards_per_game', 250)
                    team_data[f'{prefix}_rushing_yards_per_game'] = team_data.get('rushing_yards_per_game', 120)
                    team_data[f'{prefix}_total_yards_per_game'] = (
                        team_data[f'{prefix}_passing_yards_per_game'] + 
                        team_data[f'{prefix}_rushing_yards_per_game']
                    )
                    
                    # Defensive statistics (yards allowed)
                    team_data[f'{prefix}_passing_yards_allowed'] = team_data.get('passing_yards_allowed_per_game', 250)
                    team_data[f'{prefix}_rushing_yards_allowed'] = team_data.get('rushing_yards_allowed_per_game', 120)
                    
                    # Defensive pressure and turnovers
                    team_data[f'{prefix}_sacks_per_game'] = team_data.get('sacks_per_game', 2.5)
                    team_data[f'{prefix}_interceptions_per_game'] = team_data.get('interceptions_per_game', 1.0)
                    
                    # Turnover differential
                    if 'turnovers_forced_per_game' in team_data.columns and 'turnovers_per_game' in team_data.columns:
                        team_data[f'{prefix}_turnover_differential'] = (
                            team_data['turnovers_forced_per_game'].fillna(1.5) - 
                            team_data['turnovers_per_game'].fillna(1.5)
                        )
                    else:
                        team_data[f'{prefix}_turnover_differential'] = 0.0
                    
                    # Efficiency metrics
                    team_data[f'{prefix}_red_zone_pct'] = team_data.get('red_zone_percentage', 60.0)
                    team_data[f'{prefix}_third_down_pct'] = team_data.get('third_down_percentage', 40.0)
                    team_data[f'{prefix}_time_of_possession'] = team_data.get('time_of_possession_minutes', 30.0)
                    
                    # Merge advanced stats
                    merge_cols = [team_col] + [col for col in team_data.columns if col.startswith(prefix)]
                    team_data = team_data.rename(columns={'team_id': team_col})
                    
                    df = df.merge(team_data[merge_cols], on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding football advanced features: {str(e)}")
            return df
    
    def _add_situational_features(self, df: pd.DataFrame, additional_data: Optional[Dict] = None) -> pd.DataFrame:
        """Add situational features like rest days and weather"""
        try:
            # Calculate rest days (NFL typically plays once per week)
            if 'game_date' in df.columns:
                df = df.sort_values(['home_team_id', 'game_date'])
                
                # Calculate rest days for home team
                df['home_prev_game_date'] = df.groupby('home_team_id')['game_date'].shift(1)
                df['days_rest_home'] = (df['game_date'] - df['home_prev_game_date']).dt.days
                df['days_rest_home'] = df['days_rest_home'].fillna(7)  # Default 1 week rest
                
                # Calculate rest days for away team  
                df = df.sort_values(['away_team_id', 'game_date'])
                df['away_prev_game_date'] = df.groupby('away_team_id')['game_date'].shift(1)
                df['days_rest_away'] = (df['game_date'] - df['away_prev_game_date']).dt.days
                df['days_rest_away'] = df['days_rest_away'].fillna(7)  # Default 1 week rest
                
                # Rest advantage
                df['rest_advantage'] = df['days_rest_home'] - df['days_rest_away']
                
                # Thursday/Monday games (short rest)
                df['short_rest_home'] = (df['days_rest_home'] <= 5).astype(int)
                df['short_rest_away'] = (df['days_rest_away'] <= 5).astype(int)
            
            # Home field advantage (NFL typically ~2-3 points)
            df['home_field_advantage'] = 2.5
            
            # Weather features (if provided)
            if additional_data and 'weather' in additional_data:
                weather_data = additional_data['weather']
                df['temperature'] = weather_data.get('temperature', 70)
                df['wind_speed'] = weather_data.get('wind_speed', 5)
                df['is_dome'] = weather_data.get('is_dome', 0)
                df['precipitation'] = weather_data.get('precipitation', 0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding situational features: {str(e)}")
            return df
    
    def _add_rolling_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling/trend features"""
        try:
            # This would ideally use game-by-game data for rolling calculations
            # For now, use team stats as proxy for recent form
            
            for prefix in ['home', 'away']:
                if f'{prefix}_points_per_game' in df.columns:
                    # Recent form indicators (simplified)
                    df[f'{prefix}_points_per_game_l4'] = df[f'{prefix}_points_per_game']
                    df[f'{prefix}_points_allowed_per_game_l4'] = df[f'{prefix}_points_allowed_per_game']
                    df[f'{prefix}_yards_per_game_l4'] = df.get(f'{prefix}_total_yards_per_game', 370)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {str(e)}")
            return df
    
    def _add_matchup_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features"""
        try:
            # Offensive matchup advantage (home offense vs away defense)
            if 'home_total_yards_per_game' in df.columns and 'away_rushing_yards_allowed' in df.columns:
                home_offense_rating = df['home_total_yards_per_game'] / 370  # NFL average
                away_defense_rating = 370 / (df['away_passing_yards_allowed'] + df['away_rushing_yards_allowed'])
                df['offensive_matchup_advantage'] = home_offense_rating - away_defense_rating
            else:
                df['offensive_matchup_advantage'] = 0
            
            # Defensive matchup advantage (home defense vs away offense)
            if 'home_passing_yards_allowed' in df.columns and 'away_total_yards_per_game' in df.columns:
                home_defense_rating = 370 / (df['home_passing_yards_allowed'] + df['home_rushing_yards_allowed'])
                away_offense_rating = df['away_total_yards_per_game'] / 370
                df['defensive_matchup_advantage'] = home_defense_rating - away_offense_rating
            else:
                df['defensive_matchup_advantage'] = 0
            
            # Combined turnover differential
            if 'home_turnover_differential' in df.columns and 'away_turnover_differential' in df.columns:
                df['turnover_differential'] = df['home_turnover_differential'] - df['away_turnover_differential']
            
            # Head-to-head record (simplified - would use historical data)
            df['h2h_home_wins'] = 0
            df['h2h_total_games'] = 0
            
            # Total prediction (estimated total points based on offensive and defensive capabilities)
            if 'home_points_per_game' in df.columns and 'away_points_per_game' in df.columns:
                home_expected = df['home_points_per_game'].fillna(22.0)
                away_expected = df['away_points_per_game'].fillna(22.0)
                df['total_prediction'] = home_expected + away_expected
            else:
                df['total_prediction'] = 44.0  # NFL average total
            
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
                
                # Total points target
                df['target_total'] = df['home_score'] + df['away_score']
            else:
                # For inference, these will be filled later
                df['target_winner'] = None
                df['target_total'] = None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding target variables: {str(e)}")
            return df
    
    def create_rolling_features(self, team_stats_df: pd.DataFrame,
                               window_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """Create rolling/moving average features"""
        if window_sizes is None:
            window_sizes = [3, 5, 8]  # NFL seasons are shorter
        
        try:
            df = team_stats_df.copy()
            
            # Sort by team and date
            if 'date' in df.columns:
                df = df.sort_values(['team_id', 'date'])
                
                # Rolling averages for key stats
                for window in window_sizes:
                    for stat in ['points_per_game', 'points_against_per_game', 'passing_yards_per_game', 'rushing_yards_per_game']:
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
                
                # Offensive vs defensive matchup
                home_offense = home_latest.get('points_per_game', 22)
                away_defense = away_latest.get('points_against_per_game', 22)
                features['offensive_matchup'] = home_offense - away_defense
                
                # Turnover battle
                home_to_diff = home_latest.get('turnover_differential', 0)
                away_to_diff = away_latest.get('turnover_differential', 0)
                features['turnover_advantage'] = home_to_diff - away_to_diff
                
                # Yards per play efficiency
                home_ypp = home_latest.get('yards_per_play', 5.5)
                away_ypp_allowed = away_latest.get('yards_per_play_allowed', 5.5)
                features['efficiency_matchup'] = home_ypp - away_ypp_allowed
            
            # Head-to-head history
            if h2h_data is not None and not h2h_data.empty:
                home_wins = len(h2h_data[h2h_data['winner_id'] == home_team_id])
                total_games = len(h2h_data)
                features['h2h_home_win_pct'] = home_wins / max(total_games, 1)
                features['h2h_total_games'] = total_games
            else:
                features['h2h_home_win_pct'] = 0.5
                features['h2h_total_games'] = 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating matchup features: {str(e)}")
            return {}
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by importance categories"""
        return {
            'offense': [
                'home_points_per_game', 'away_points_per_game', 'home_passing_yards_per_game',
                'away_passing_yards_per_game', 'home_rushing_yards_per_game', 'away_rushing_yards_per_game',
                'home_red_zone_pct', 'away_red_zone_pct', 'home_third_down_pct', 'away_third_down_pct'
            ],
            'defense': [
                'home_points_allowed_per_game', 'away_points_allowed_per_game', 'home_passing_yards_allowed',
                'away_passing_yards_allowed', 'home_rushing_yards_allowed', 'away_rushing_yards_allowed',
                'home_sacks_per_game', 'away_sacks_per_game', 'home_interceptions_per_game', 'away_interceptions_per_game'
            ],
            'special': [
                'home_turnover_differential', 'away_turnover_differential', 'turnover_differential'
            ],
            'situational': [
                'is_playoffs', 'week_of_season', 'days_rest_home', 'days_rest_away', 'home_field_advantage',
                'short_rest_home', 'short_rest_away', 'is_weekend'
            ],
            'form': [
                'home_win_pct_season', 'away_win_pct_season'
            ],
            'matchup': [
                'offensive_matchup_advantage', 'defensive_matchup_advantage', 'h2h_home_wins', 'h2h_total_games'
            ]
        }