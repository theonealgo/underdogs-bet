import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from src.interfaces.base_feature_engineer import BaseFeatureEngineer


class NCAABFeatureEngineer(BaseFeatureEngineer):
    """
    NCAA Basketball-specific feature engineering for college basketball prediction models.
    
    Creates features specific to college basketball analytics including:
    - Team offensive and defensive efficiency ratings
    - Tempo/pace metrics and possession-based statistics
    - Conference strength and tournament implications
    - Coaching experience and program prestige factors
    - March Madness and tournament-specific dynamics
    """
    
    def __init__(self):
        super().__init__('NCAAB')
        self.logger = logging.getLogger(__name__)
        
        # NCAA Basketball-specific feature definitions
        self.required_features = [
            # Basic game features
            'home_team_encoded', 'away_team_encoded', 'month', 'day_of_week', 'is_weekend',
            'is_tournament', 'days_rest_home', 'days_rest_away',
            
            # Team performance features
            'home_win_pct_season', 'away_win_pct_season', 'home_points_per_game', 'away_points_per_game',
            'home_points_allowed_per_game', 'away_points_allowed_per_game',
            
            # Basketball-specific features
            'home_field_goal_pct', 'away_field_goal_pct', 'home_three_point_pct', 'away_three_point_pct',
            'home_free_throw_pct', 'away_free_throw_pct', 'home_rebounds_per_game', 'away_rebounds_per_game',
            'home_assists_per_game', 'away_assists_per_game', 'home_turnovers_per_game', 'away_turnovers_per_game',
            
            # Advanced basketball metrics
            'home_offensive_efficiency', 'away_offensive_efficiency', 'home_defensive_efficiency', 'away_defensive_efficiency',
            'home_tempo', 'away_tempo', 'home_effective_fg_pct', 'away_effective_fg_pct',
            
            # Conference and competition level
            'home_conference_strength', 'away_conference_strength', 'conference_matchup', 'major_conference_matchup',
            'home_rpi', 'away_rpi', 'home_kenpom_rating', 'away_kenpom_rating',
            
            # Matchup features
            'tempo_differential', 'efficiency_differential', 'experience_advantage',
            'home_court_advantage', 'total_prediction', 'h2h_home_wins', 'h2h_total_games'
        ]
        
        self.feature_descriptions = {
            'home_points_per_game': 'Average points scored per game by home team',
            'away_points_per_game': 'Average points scored per game by away team',
            'home_field_goal_pct': 'Field goal shooting percentage for home team',
            'away_field_goal_pct': 'Field goal shooting percentage for away team',
            'home_three_point_pct': 'Three-point shooting percentage for home team',
            'away_three_point_pct': 'Three-point shooting percentage for away team',
            'home_offensive_efficiency': 'Points per 100 possessions for home team',
            'away_offensive_efficiency': 'Points per 100 possessions for away team',
            'home_defensive_efficiency': 'Points allowed per 100 possessions by home team',
            'away_defensive_efficiency': 'Points allowed per 100 possessions by away team',
            'home_tempo': 'Possessions per 40 minutes for home team',
            'away_tempo': 'Possessions per 40 minutes for away team',
            'home_rpi': 'RPI (Rating Percentage Index) for home team',
            'away_rpi': 'RPI (Rating Percentage Index) for away team',
            'home_kenpom_rating': 'KenPom efficiency rating for home team',
            'away_kenpom_rating': 'KenPom efficiency rating for away team',
            'is_tournament': 'Whether game is part of March Madness tournament',
            'major_conference_matchup': 'Whether both teams are from major conferences',
            'tempo_differential': 'Difference in preferred game tempo between teams',
            'efficiency_differential': 'Difference in offensive efficiency between teams',
            'days_rest_home': 'Days of rest for home team since last game',
            'days_rest_away': 'Days of rest for away team since last game',
            'home_court_advantage': 'Estimated home court advantage points'
        }
    
    def create_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame,
                       additional_data: Optional[Dict] = None,
                       include_targets: bool = True) -> pd.DataFrame:
        """
        Create comprehensive NCAA Basketball features for prediction models.
        
        Args:
            games_df: DataFrame with NCAA Basketball game information
            team_stats_df: DataFrame with NCAA Basketball team statistics
            additional_data: Optional additional data (conference, ratings, etc.)
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if games_df.empty:
                self.logger.warning("Empty games DataFrame provided")
                return games_df
            
            self.logger.info(f"Creating NCAA Basketball features for {len(games_df)} games")
            features_df = games_df.copy()
            
            # Basic game features
            features_df = self._add_basic_features(features_df)
            
            # NCAA Basketball-specific team performance features
            features_df = self._add_team_performance_features(features_df, team_stats_df)
            
            # Basketball-specific advanced features
            features_df = self._add_basketball_advanced_features(features_df, team_stats_df)
            
            # Conference and competition level features
            features_df = self._add_conference_features(features_df, additional_data)
            
            # Situational features (rest, tournament, etc.)
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
            
            self.logger.info(f"Created {len(features_df.columns)} NCAA Basketball features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating NCAA Basketball features: {str(e)}")
            return games_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic NCAA Basketball game features"""
        try:
            # Date features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['month'] = df['game_date'].dt.month
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                
                # NCAA Basketball season features (Nov-Apr)
                df['is_tournament'] = ((df['month'] >= 3) & (df['month'] <= 4)).astype(int)
                df['is_conference_tournament'] = (df['month'] == 3).astype(int)
                df['is_march_madness'] = ((df['month'] == 3) | (df['month'] == 4)).astype(int)
                
                # Season phase
                df['season_phase'] = np.select([
                    df['month'].isin([11, 12]),    # Non-conference
                    df['month'].isin([1, 2]),      # Conference play  
                    df['month'] == 3,              # Conference tournaments
                    df['month'] == 4               # NCAA Tournament
                ], ['non_conference', 'conference', 'conf_tournament', 'ncaa_tournament'], default='other')
            
            # Team encoding
            if 'home_team_id' in df.columns and 'away_team_id' in df.columns:
                all_teams = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
                team_map = {team: idx for idx, team in enumerate(sorted(all_teams))}
                
                df['home_team_encoded'] = df['home_team_id'].map(lambda x: team_map.get(x, -1))
                df['away_team_encoded'] = df['away_team_id'].map(lambda x: team_map.get(x, -1))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basic NCAA Basketball features: {str(e)}")
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
    
    def _add_basketball_advanced_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add NCAA Basketball-specific advanced statistics"""
        try:
            if team_stats_df.empty:
                return df
            
            # Get latest advanced stats
            latest_stats = team_stats_df.groupby('team_id').last().reset_index()
            
            for prefix in ['home', 'away']:
                team_col = f'{prefix}_team_id'
                
                if team_col in df.columns:
                    team_data = latest_stats.copy()
                    
                    # Shooting statistics
                    team_data[f'{prefix}_field_goal_pct'] = team_data.get('field_goal_percentage', 45.0)
                    team_data[f'{prefix}_three_point_pct'] = team_data.get('three_point_percentage', 33.0)
                    team_data[f'{prefix}_free_throw_pct'] = team_data.get('free_throw_percentage', 72.0)
                    
                    # Other team statistics
                    team_data[f'{prefix}_rebounds_per_game'] = team_data.get('rebounds_per_game', 35.0)
                    team_data[f'{prefix}_assists_per_game'] = team_data.get('assists_per_game', 14.0)
                    team_data[f'{prefix}_turnovers_per_game'] = team_data.get('turnovers_per_game', 13.0)
                    
                    # Calculate advanced metrics
                    # Estimate possessions per game (simplified formula)
                    estimated_possessions = 70  # College basketball average
                    team_data[f'{prefix}_possessions_per_game'] = estimated_possessions
                    
                    # Offensive efficiency (points per 100 possessions)
                    points_pg = pd.Series([70] * len(team_data)) if 'points_per_game' not in team_data.columns else team_data['points_per_game'].fillna(70)
                    team_data[f'{prefix}_offensive_efficiency'] = (points_pg / estimated_possessions) * 100
                    
                    # Defensive efficiency (points allowed per 100 possessions)
                    points_allowed_pg = pd.Series([70] * len(team_data)) if 'points_against_per_game' not in team_data.columns else team_data['points_against_per_game'].fillna(70)
                    team_data[f'{prefix}_defensive_efficiency'] = (points_allowed_pg / estimated_possessions) * 100
                    
                    # Tempo (possessions per 40 minutes)
                    team_data[f'{prefix}_tempo'] = estimated_possessions
                    
                    # Effective field goal percentage (accounting for 3-pointers)
                    fg_pct = team_data[f'{prefix}_field_goal_pct'] / 100
                    three_pct = team_data[f'{prefix}_three_point_pct'] / 100
                    # Simplified effective FG% (would need more detailed stats)
                    team_data[f'{prefix}_effective_fg_pct'] = (fg_pct + 0.1 * three_pct) * 100
                    
                    # Merge advanced stats
                    merge_cols = [team_col] + [col for col in team_data.columns if col.startswith(prefix)]
                    team_data = team_data.rename(columns={'team_id': team_col})
                    
                    df = df.merge(team_data[merge_cols], on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basketball advanced features: {str(e)}")
            return df
    
    def _add_conference_features(self, df: pd.DataFrame, additional_data: Optional[Dict] = None) -> pd.DataFrame:
        """Add conference and competition level features"""
        try:
            # Major conference strength ratings
            major_conferences = ['ACC', 'Big East', 'Big Ten', 'Big 12', 'Pac-12', 'SEC']
            conference_strength = {
                'ACC': 90, 'Big East': 88, 'Big Ten': 87, 'Big 12': 86, 'SEC': 85, 'Pac-12': 83,
                'American': 75, 'Atlantic 10': 72, 'Mountain West': 70, 'WCC': 68,
                'Conference USA': 60, 'MAC': 55, 'Sun Belt': 50
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
                            lambda x: conference_strength.get(x, 60)
                        )
                        df[f'{prefix}_is_major_conference'] = df[f'{prefix}_conference'].map(
                            lambda x: 1 if x in major_conferences else 0
                        )
            else:
                # Default values
                for prefix in ['home', 'away']:
                    df[f'{prefix}_conference_strength'] = 65  # Average
                    df[f'{prefix}_is_major_conference'] = 0
            
            # Conference matchup features
            if 'home_is_major_conference' in df.columns and 'away_is_major_conference' in df.columns:
                df['major_conference_matchup'] = (df['home_is_major_conference'] & df['away_is_major_conference']).astype(int)
                df['conference_mismatch'] = (df['home_is_major_conference'] != df['away_is_major_conference']).astype(int)
            
            # Advanced ratings (simplified)
            if additional_data and 'ratings' in additional_data:
                ratings_data = additional_data['ratings']
                
                for prefix in ['home', 'away']:
                    team_col = f'{prefix}_team_id'
                    df[f'{prefix}_rpi'] = df[team_col].map(
                        lambda x: ratings_data.get(x, {}).get('rpi', 0.5)
                    )
                    df[f'{prefix}_kenpom_rating'] = df[team_col].map(
                        lambda x: ratings_data.get(x, {}).get('kenpom', 0.0)
                    )
            else:
                # Default ratings
                for prefix in ['home', 'away']:
                    df[f'{prefix}_rpi'] = 0.5
                    df[f'{prefix}_kenpom_rating'] = 0.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding conference features: {str(e)}")
            return df
    
    def _add_situational_features(self, df: pd.DataFrame, additional_data: Optional[Dict] = None) -> pd.DataFrame:
        """Add situational features like rest days and tournament context"""
        try:
            # Calculate rest days (college basketball can have varying schedules)
            if 'game_date' in df.columns:
                df = df.sort_values(['home_team_id', 'game_date'])
                
                # Calculate rest days for home team
                df['home_prev_game_date'] = df.groupby('home_team_id')['game_date'].shift(1)
                df['days_rest_home'] = (df['game_date'] - df['home_prev_game_date']).dt.days
                df['days_rest_home'] = df['days_rest_home'].fillna(3)  # Default rest
                
                # Calculate rest days for away team  
                df = df.sort_values(['away_team_id', 'game_date'])
                df['away_prev_game_date'] = df.groupby('away_team_id')['game_date'].shift(1)
                df['days_rest_away'] = (df['game_date'] - df['away_prev_game_date']).dt.days
                df['days_rest_away'] = df['days_rest_away'].fillna(3)  # Default rest
                
                # Rest advantage
                df['rest_advantage'] = df['days_rest_home'] - df['days_rest_away']
            
            # Home court advantage (college basketball has significant home advantage)
            df['home_court_advantage'] = 4.0  # College average higher than pro
            
            # Tournament and season context
            if 'is_tournament' in df.columns:
                # Neutral site games (tournaments)
                df['is_neutral_site'] = df['is_tournament']
                df.loc[df['is_neutral_site'] == 1, 'home_court_advantage'] = 0  # No home advantage on neutral court
            
            # Experience advantage (would use actual roster data)
            df['experience_advantage'] = np.random.normal(0, 0.5, len(df))
            
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
                    df[f'{prefix}_points_per_game_l5'] = df[f'{prefix}_points_per_game']
                    df[f'{prefix}_points_allowed_per_game_l5'] = df[f'{prefix}_points_allowed_per_game']
                    df[f'{prefix}_field_goal_pct_l5'] = df[f'{prefix}_field_goal_pct']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {str(e)}")
            return df
    
    def _add_matchup_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features"""
        try:
            # Tempo differential (how different team paces interact)
            if 'home_tempo' in df.columns and 'away_tempo' in df.columns:
                df['tempo_differential'] = abs(df['home_tempo'] - df['away_tempo'])
            
            # Efficiency differential
            if 'home_offensive_efficiency' in df.columns and 'away_defensive_efficiency' in df.columns:
                df['efficiency_differential'] = df['home_offensive_efficiency'] - df['away_defensive_efficiency']
            
            # Conference matchup strength
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
            window_sizes = [5, 10, 15]  # College basketball season lengths
        
        try:
            df = team_stats_df.copy()
            
            if 'date' in df.columns:
                df = df.sort_values(['team_id', 'date'])
                
                # Rolling averages for key stats
                for window in window_sizes:
                    for stat in ['points_per_game', 'points_against_per_game', 'field_goal_percentage', 'three_point_percentage']:
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
                    home_latest.get('points_per_game', 70) - 
                    away_latest.get('points_against_per_game', 70)
                )
                
                # Shooting matchup
                features['shooting_advantage'] = (
                    home_latest.get('field_goal_percentage', 45) - 
                    away_latest.get('field_goal_percentage_allowed', 45)
                )
                
                # Tempo matchup
                home_tempo = home_latest.get('possessions_per_game', 70)
                away_tempo = away_latest.get('possessions_per_game', 70)
                features['tempo_advantage'] = home_tempo - away_tempo
            
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
                'home_points_per_game', 'away_points_per_game', 'home_field_goal_pct', 'away_field_goal_pct',
                'home_three_point_pct', 'away_three_point_pct', 'home_assists_per_game', 'away_assists_per_game',
                'home_offensive_efficiency', 'away_offensive_efficiency'
            ],
            'defense': [
                'home_points_allowed_per_game', 'away_points_allowed_per_game', 
                'home_defensive_efficiency', 'away_defensive_efficiency'
            ],
            'pace': [
                'home_tempo', 'away_tempo', 'tempo_differential'
            ],
            'situational': [
                'is_tournament', 'is_march_madness', 'days_rest_home', 'days_rest_away', 
                'home_court_advantage', 'is_weekend', 'is_neutral_site'
            ],
            'conference': [
                'home_conference_strength', 'away_conference_strength', 'major_conference_matchup',
                'conference_mismatch', 'home_rpi', 'away_rpi', 'home_kenpom_rating', 'away_kenpom_rating'
            ],
            'form': [
                'home_win_pct_season', 'away_win_pct_season'
            ],
            'matchup': [
                'efficiency_differential', 'conference_matchup', 'experience_advantage',
                'h2h_home_wins', 'h2h_total_games'
            ]
        }