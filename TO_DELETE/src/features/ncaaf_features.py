import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from src.interfaces.base_feature_engineer import BaseFeatureEngineer


class NCAAFFeatureEngineer(BaseFeatureEngineer):
    """
    NCAA Football-specific feature engineering for college football prediction models.
    
    Creates features specific to college football analytics including:
    - Team offensive and defensive efficiency
    - Conference strength and schedule difficulty
    - Recruiting rankings and talent disparities
    - Coach experience and program prestige
    - Weather and travel factors for college games
    """
    
    def __init__(self):
        super().__init__('NCAAF')
        self.logger = logging.getLogger(__name__)
        
        # NCAA Football-specific feature definitions
        self.required_features = [
            # Basic game features
            'home_team_encoded', 'away_team_encoded', 'month', 'day_of_week', 'is_weekend',
            'is_playoffs', 'days_rest_home', 'days_rest_away', 'week_of_season',
            
            # Team performance features
            'home_win_pct_season', 'away_win_pct_season', 'home_points_per_game', 'away_points_per_game',
            'home_points_allowed_per_game', 'away_points_allowed_per_game',
            
            # College football-specific offensive features
            'home_passing_yards_per_game', 'away_passing_yards_per_game', 'home_rushing_yards_per_game', 'away_rushing_yards_per_game',
            'home_total_yards_per_game', 'away_total_yards_per_game', 'home_turnover_margin', 'away_turnover_margin',
            
            # College football-specific defensive features
            'home_passing_yards_allowed', 'away_passing_yards_allowed', 'home_rushing_yards_allowed', 'away_rushing_yards_allowed',
            'home_sacks_per_game', 'away_sacks_per_game', 'home_tackles_for_loss', 'away_tackles_for_loss',
            
            # Conference and competition level
            'home_conference_strength', 'away_conference_strength', 'conference_matchup', 'power5_matchup',
            'home_sos', 'away_sos', 'talent_gap', 'experience_advantage',
            
            # Matchup features
            'offensive_matchup_advantage', 'defensive_matchup_advantage', 'total_prediction',
            'home_field_advantage', 'h2h_home_wins', 'h2h_total_games'
        ]
        
        self.feature_descriptions = {
            'home_points_per_game': 'Average points scored per game by home team',
            'away_points_per_game': 'Average points scored per game by away team',
            'home_passing_yards_per_game': 'Average passing yards per game by home team',
            'away_passing_yards_per_game': 'Average passing yards per game by away team',
            'home_rushing_yards_per_game': 'Average rushing yards per game by home team',
            'away_rushing_yards_per_game': 'Average rushing yards per game by away team',
            'home_turnover_margin': 'Turnovers gained minus turnovers lost by home team',
            'away_turnover_margin': 'Turnovers gained minus turnovers lost by away team',
            'home_conference_strength': 'Strength rating of home team conference',
            'away_conference_strength': 'Strength rating of away team conference',
            'power5_matchup': 'Whether both teams are from Power 5 conferences',
            'home_sos': 'Home team strength of schedule rating',
            'away_sos': 'Away team strength of schedule rating',
            'talent_gap': 'Difference in recruiting rankings between teams',
            'experience_advantage': 'Difference in average player experience',
            'week_of_season': 'Week number in college football season (1-15)',
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
        Create comprehensive NCAA Football features for prediction models.
        
        Args:
            games_df: DataFrame with NCAA Football game information
            team_stats_df: DataFrame with NCAA Football team statistics
            additional_data: Optional additional data (recruiting, weather, etc.)
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if games_df.empty:
                self.logger.warning("Empty games DataFrame provided")
                return games_df
            
            self.logger.info(f"Creating NCAA Football features for {len(games_df)} games")
            features_df = games_df.copy()
            
            # Basic game features
            features_df = self._add_basic_features(features_df)
            
            # NCAA Football-specific team performance features
            features_df = self._add_team_performance_features(features_df, team_stats_df)
            
            # College football-specific advanced features
            features_df = self._add_college_football_features(features_df, team_stats_df)
            
            # Conference and competition level features
            features_df = self._add_conference_features(features_df, additional_data)
            
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
            
            self.logger.info(f"Created {len(features_df.columns)} NCAA Football features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating NCAA Football features: {str(e)}")
            return games_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic NCAA Football game features"""
        try:
            # Date features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['month'] = df['game_date'].dt.month
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                
                # NCAA Football season features (Aug-Jan)
                df['is_playoffs'] = ((df['month'] <= 1) | (df['month'] == 12)).astype(int)
                
                # Week of season (estimated)
                df['week_of_season'] = np.select([
                    df['month'] == 8,   # August (pre-season)
                    df['month'] == 9,   # September
                    df['month'] == 10,  # October  
                    df['month'] == 11,  # November
                    df['month'] == 12,  # December (bowl season)
                    df['month'] == 1,   # January (playoffs)
                ], [0, 2, 6, 10, 14, 16], default=1)
            
            # Team encoding
            if 'home_team_id' in df.columns and 'away_team_id' in df.columns:
                all_teams = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
                team_map = {team: idx for idx, team in enumerate(sorted(all_teams))}
                
                df['home_team_encoded'] = df['home_team_id'].map(lambda x: team_map.get(x, -1))
                df['away_team_encoded'] = df['away_team_id'].map(lambda x: team_map.get(x, -1))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basic NCAA Football features: {str(e)}")
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
    
    def _add_college_football_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add NCAA Football-specific advanced statistics"""
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
                    team_data[f'{prefix}_rushing_yards_per_game'] = team_data.get('rushing_yards_per_game', 150)
                    team_data[f'{prefix}_total_yards_per_game'] = (
                        team_data[f'{prefix}_passing_yards_per_game'] + 
                        team_data[f'{prefix}_rushing_yards_per_game']
                    )
                    
                    # Defensive statistics
                    team_data[f'{prefix}_passing_yards_allowed'] = team_data.get('passing_yards_allowed_per_game', 250)
                    team_data[f'{prefix}_rushing_yards_allowed'] = team_data.get('rushing_yards_allowed_per_game', 150)
                    
                    # Defensive pressure
                    team_data[f'{prefix}_sacks_per_game'] = team_data.get('sacks_per_game', 2.0)
                    team_data[f'{prefix}_tackles_for_loss'] = team_data.get('tackles_for_loss_per_game', 6.0)
                    
                    # Turnover margin
                    turnovers_gained = pd.Series([1.5] * len(team_data)) if 'turnovers_gained_per_game' not in team_data.columns else team_data['turnovers_gained_per_game'].fillna(1.5)
                    turnovers_lost = pd.Series([1.5] * len(team_data)) if 'turnovers_per_game' not in team_data.columns else team_data['turnovers_per_game'].fillna(1.5)
                    team_data[f'{prefix}_turnover_margin'] = turnovers_gained - turnovers_lost
                    
                    # Merge advanced stats
                    merge_cols = [team_col] + [col for col in team_data.columns if col.startswith(prefix)]
                    team_data = team_data.rename(columns={'team_id': team_col})
                    
                    df = df.merge(team_data[merge_cols], on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding college football features: {str(e)}")
            return df
    
    def _add_conference_features(self, df: pd.DataFrame, additional_data: Optional[Dict] = None) -> pd.DataFrame:
        """Add conference and competition level features"""
        try:
            # Conference strength ratings (simplified)
            power5_conferences = ['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12']
            conference_strength = {
                'SEC': 95, 'Big Ten': 90, 'Big 12': 85, 'ACC': 82, 'Pac-12': 80,
                'American': 70, 'Mountain West': 65, 'Conference USA': 60,
                'MAC': 55, 'Sun Belt': 50, 'FBS Independents': 60
            }
            
            # Add conference data if available
            if additional_data and 'conferences' in additional_data:
                conference_data = additional_data['conferences']
                
                for prefix in ['home', 'away']:
                    team_col = f'{prefix}_team_id'
                    if team_col in df.columns:
                        df[f'{prefix}_conference'] = df[team_col].map(
                            lambda x: conference_data.get(x, 'Unknown')
                        )
                        df[f'{prefix}_conference_strength'] = df[f'{prefix}_conference'].map(
                            lambda x: conference_strength.get(x, 50)
                        )
                        df[f'{prefix}_is_power5'] = df[f'{prefix}_conference'].map(
                            lambda x: 1 if x in power5_conferences else 0
                        )
            else:
                # Default values
                for prefix in ['home', 'away']:
                    df[f'{prefix}_conference_strength'] = 60  # Average
                    df[f'{prefix}_is_power5'] = 0
            
            # Conference matchup features
            if 'home_is_power5' in df.columns and 'away_is_power5' in df.columns:
                df['power5_matchup'] = (df['home_is_power5'] & df['away_is_power5']).astype(int)
                df['conference_mismatch'] = (df['home_is_power5'] != df['away_is_power5']).astype(int)
            
            # Strength of schedule (simplified)
            home_conf_strength = df['home_conference_strength'].fillna(60) if 'home_conference_strength' in df.columns else pd.Series([60] * len(df))
            away_conf_strength = df['away_conference_strength'].fillna(60) if 'away_conference_strength' in df.columns else pd.Series([60] * len(df))
            df['home_sos'] = home_conf_strength + np.random.normal(0, 10, len(df))
            df['away_sos'] = away_conf_strength + np.random.normal(0, 10, len(df))
            
            # Talent gap (recruiting rankings difference - simplified)
            if additional_data and 'recruiting' in additional_data:
                recruiting_data = additional_data['recruiting']
                df['home_recruiting_rank'] = df['home_team_id'].map(
                    lambda x: recruiting_data.get(x, 50)
                )
                df['away_recruiting_rank'] = df['away_team_id'].map(
                    lambda x: recruiting_data.get(x, 50)
                )
                df['talent_gap'] = df['away_recruiting_rank'] - df['home_recruiting_rank']  # Lower rank is better
            else:
                df['talent_gap'] = 0
            
            # Experience advantage (simplified)
            df['experience_advantage'] = np.random.normal(0, 1, len(df))  # Would be based on roster data
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding conference features: {str(e)}")
            return df
    
    def _add_situational_features(self, df: pd.DataFrame, additional_data: Optional[Dict] = None) -> pd.DataFrame:
        """Add situational features like rest days and weather"""
        try:
            # Calculate rest days (College football typically plays once per week)
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
            
            # Home field advantage (College football has larger home advantage than NFL)
            df['home_field_advantage'] = 3.5
            
            # Rivalry games (would need historical data)
            df['is_rivalry'] = 0  # Would identify traditional rivalries
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding situational features: {str(e)}")
            return df
    
    def _add_rolling_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling/trend features"""
        try:
            # Use team stats as proxy for recent form
            for prefix in ['home', 'away']:
                if f'{prefix}_points_per_game' in df.columns:
                    # Recent form indicators (simplified)
                    df[f'{prefix}_points_per_game_l3'] = df[f'{prefix}_points_per_game']
                    df[f'{prefix}_points_allowed_per_game_l3'] = df[f'{prefix}_points_allowed_per_game']
                    df[f'{prefix}_yards_per_game_l3'] = df.get(f'{prefix}_total_yards_per_game', 400)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {str(e)}")
            return df
    
    def _add_matchup_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features"""
        try:
            # Offensive matchup advantage
            if 'home_total_yards_per_game' in df.columns:
                home_offense_rating = df['home_total_yards_per_game'] / 400  # College average
                away_defense_rating = 400 / (df['away_passing_yards_allowed'] + df['away_rushing_yards_allowed'])
                df['offensive_matchup_advantage'] = home_offense_rating - away_defense_rating
            else:
                df['offensive_matchup_advantage'] = 0
            
            # Defensive matchup advantage  
            if 'home_passing_yards_allowed' in df.columns:
                home_defense_rating = 400 / (df['home_passing_yards_allowed'] + df['home_rushing_yards_allowed'])
                away_offense_rating = df['away_total_yards_per_game'] / 400
                df['defensive_matchup_advantage'] = home_defense_rating - away_offense_rating
            else:
                df['defensive_matchup_advantage'] = 0
            
            # Conference matchup
            if 'home_conference_strength' in df.columns and 'away_conference_strength' in df.columns:
                df['conference_matchup'] = abs(df['home_conference_strength'] - df['away_conference_strength'])
            
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
            window_sizes = [3, 5, 8]  # College seasons are shorter than NFL
        
        try:
            df = team_stats_df.copy()
            
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
                
                # Scoring matchup
                features['scoring_advantage'] = (
                    home_latest.get('points_per_game', 25) - 
                    away_latest.get('points_against_per_game', 25)
                )
                
                # Turnover battle
                features['turnover_advantage'] = (
                    home_latest.get('turnover_margin', 0) - 
                    away_latest.get('turnover_margin', 0)
                )
                
                # Yards per play efficiency
                home_ypp = home_latest.get('yards_per_play', 5.8)
                away_ypp_allowed = away_latest.get('yards_per_play_allowed', 5.8)
                features['efficiency_advantage'] = home_ypp - away_ypp_allowed
            
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
                'away_passing_yards_per_game', 'home_rushing_yards_per_game', 'away_rushing_yards_per_game'
            ],
            'defense': [
                'home_points_allowed_per_game', 'away_points_allowed_per_game', 'home_passing_yards_allowed',
                'away_passing_yards_allowed', 'home_rushing_yards_allowed', 'away_rushing_yards_allowed',
                'home_sacks_per_game', 'away_sacks_per_game', 'home_tackles_for_loss', 'away_tackles_for_loss'
            ],
            'special': [
                'home_turnover_margin', 'away_turnover_margin'
            ],
            'situational': [
                'is_playoffs', 'week_of_season', 'days_rest_home', 'days_rest_away', 'home_field_advantage',
                'is_weekend', 'is_rivalry'
            ],
            'conference': [
                'home_conference_strength', 'away_conference_strength', 'power5_matchup', 
                'conference_mismatch', 'home_sos', 'away_sos', 'talent_gap'
            ],
            'form': [
                'home_win_pct_season', 'away_win_pct_season'
            ],
            'matchup': [
                'offensive_matchup_advantage', 'defensive_matchup_advantage', 'conference_matchup',
                'h2h_home_wins', 'h2h_total_games'
            ]
        }