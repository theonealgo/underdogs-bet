import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from src.interfaces.base_feature_engineer import BaseFeatureEngineer


class NBAFeatureEngineer(BaseFeatureEngineer):
    """
    NBA-specific feature engineering for basketball prediction models.
    
    Creates features specific to basketball analytics including:
    - Team offensive/defensive efficiency
    - Pace and possession-based statistics
    - Rest days and schedule strength
    - Home court advantage
    - Player availability and injury effects
    """
    
    def __init__(self):
        super().__init__('NBA')
        self.logger = logging.getLogger(__name__)
        
        # NBA-specific feature definitions
        self.required_features = [
            # Basic game features
            'home_team_encoded', 'away_team_encoded', 'month', 'day_of_week', 'is_weekend',
            'is_playoffs', 'days_rest_home', 'days_rest_away',
            
            # Team performance features
            'home_win_pct_l10', 'away_win_pct_l10', 'home_win_pct_season', 'away_win_pct_season',
            'home_avg_points_l5', 'away_avg_points_l5', 'home_avg_points_allowed_l5', 'away_avg_points_allowed_l5',
            
            # Basketball-specific features
            'home_offensive_rating', 'away_offensive_rating', 'home_defensive_rating', 'away_defensive_rating',
            'home_pace', 'away_pace', 'home_true_shooting_pct', 'away_true_shooting_pct',
            'home_rebounding_pct', 'away_rebounding_pct', 'home_turnover_rate', 'away_turnover_rate',
            
            # Matchup features
            'pace_matchup_differential', 'offensive_rating_differential', 'h2h_home_wins', 'h2h_total_games',
            'home_court_advantage', 'total_prediction'
        ]
        
        self.feature_descriptions = {
            'home_offensive_rating': 'Points scored per 100 possessions by home team',
            'away_offensive_rating': 'Points scored per 100 possessions by away team', 
            'home_defensive_rating': 'Points allowed per 100 possessions by home team',
            'away_defensive_rating': 'Points allowed per 100 possessions by away team',
            'home_pace': 'Average possessions per game for home team',
            'away_pace': 'Average possessions per game for away team',
            'home_true_shooting_pct': 'True shooting percentage accounting for 2pt, 3pt, and FT',
            'away_true_shooting_pct': 'True shooting percentage accounting for 2pt, 3pt, and FT',
            'pace_matchup_differential': 'Difference in team pace preferences',
            'days_rest_home': 'Days of rest for home team since last game',
            'days_rest_away': 'Days of rest for away team since last game',
            'is_playoffs': 'Whether game is in playoffs (different dynamics)',
            'home_court_advantage': 'Estimated home court advantage points',
            'total_prediction': 'Predicted total points for over/under betting'
        }
    
    def create_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame,
                       additional_data: Optional[Dict] = None,
                       include_targets: bool = True) -> pd.DataFrame:
        """
        Create comprehensive NBA features for prediction models.
        
        Args:
            games_df: DataFrame with NBA game information
            team_stats_df: DataFrame with NBA team statistics
            additional_data: Optional additional data (player injuries, etc.)
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if games_df.empty:
                self.logger.warning("Empty games DataFrame provided")
                return games_df
            
            self.logger.info(f"Creating NBA features for {len(games_df)} games")
            features_df = games_df.copy()
            
            # Basic game features
            features_df = self._add_basic_features(features_df)
            
            # NBA-specific team performance features
            features_df = self._add_team_performance_features(features_df, team_stats_df)
            
            # Basketball-specific advanced features
            features_df = self._add_basketball_advanced_features(features_df, team_stats_df)
            
            # Situational features (rest, schedule, etc.)
            features_df = self._add_situational_features(features_df)
            
            # Rolling/trend features
            features_df = self._add_rolling_features(features_df, team_stats_df)
            
            # Head-to-head and matchup features
            features_df = self._add_matchup_features(features_df, team_stats_df)
            
            # Target variables
            if include_targets:
                features_df = self._add_target_variables(features_df)
            
            # Handle missing values
            features_df = self.handle_missing_values(features_df)
            
            self.logger.info(f"Created {len(features_df.columns)} NBA features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating NBA features: {str(e)}")
            return games_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic NBA game features"""
        try:
            # Date features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['month'] = df['game_date'].dt.month
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                
                # NBA season features
                df['is_playoffs'] = ((df['month'] >= 4) & (df['month'] <= 6)).astype(int)
                df['season_stage'] = np.select([
                    df['month'].isin([10, 11, 12]),  # Early season
                    df['month'].isin([1, 2, 3]),     # Mid season  
                    df['month'] == 4,                # Late season
                    df['month'].isin([5, 6])         # Playoffs
                ], ['early', 'mid', 'late', 'playoffs'], default='other')
            
            # Team encoding
            if 'home_team_id' in df.columns and 'away_team_id' in df.columns:
                all_teams = pd.concat([df['home_team_id'], df['away_team_id']]).unique()
                team_map = {team: idx for idx, team in enumerate(sorted(all_teams))}
                
                df['home_team_encoded'] = df['home_team_id'].map(lambda x: team_map.get(x, 0))
                df['away_team_encoded'] = df['away_team_id'].map(lambda x: team_map.get(x, 0))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basic NBA features: {str(e)}")
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
                        'points_per_game': f'{prefix}_avg_points',
                        'points_against_per_game': f'{prefix}_avg_points_allowed'
                    })
                    
                    df = df.merge(team_data[[team_col, f'{prefix}_wins', f'{prefix}_losses', 
                                           f'{prefix}_win_pct_season', f'{prefix}_avg_points', 
                                           f'{prefix}_avg_points_allowed']], 
                                 on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding team performance features: {str(e)}")
            return df
    
    def _add_basketball_advanced_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add NBA-specific advanced basketball statistics"""
        try:
            if team_stats_df.empty:
                return df
            
            # Get latest advanced stats
            latest_stats = team_stats_df.groupby('team_id').last().reset_index()
            
            for prefix in ['home', 'away']:
                team_col = f'{prefix}_team_id'
                
                if team_col in df.columns:
                    # Calculate advanced metrics from basic stats
                    team_data = latest_stats.copy()
                    
                    # Estimate possessions per game (simplified)
                    if 'field_goal_attempts_per_game' in team_data.columns:
                        fga = team_data['field_goal_attempts_per_game'].fillna(85)
                        if 'turnovers_per_game' in team_data.columns:
                            to = team_data['turnovers_per_game'].fillna(14)
                        else:
                            to = pd.Series([14] * len(team_data), index=team_data.index)
                        if 'offensive_rebounds_per_game' in team_data.columns:
                            oreb = team_data['offensive_rebounds_per_game'].fillna(10)
                        else:
                            oreb = pd.Series([10] * len(team_data), index=team_data.index)
                        if 'free_throw_attempts_per_game' in team_data.columns:
                            fta = team_data['free_throw_attempts_per_game'].fillna(20)
                        else:
                            fta = pd.Series([20] * len(team_data), index=team_data.index)
                        team_data['possessions_per_game'] = (
                            fga + to - oreb + 0.44 * fta
                        )
                    else:
                        team_data['possessions_per_game'] = 100  # NBA average
                    
                    # Offensive Rating (points per 100 possessions)
                    team_data[f'{prefix}_offensive_rating'] = (
                        team_data.get('points_per_game', 110) / 
                        team_data['possessions_per_game'] * 100
                    )
                    
                    # Defensive Rating (points allowed per 100 possessions)
                    team_data[f'{prefix}_defensive_rating'] = (
                        team_data.get('points_against_per_game', 110) / 
                        team_data['possessions_per_game'] * 100
                    )
                    
                    # Pace (possessions per game)
                    team_data[f'{prefix}_pace'] = team_data['possessions_per_game']
                    
                    # True Shooting Percentage
                    if 'field_goal_percentage' in team_data.columns and 'three_point_percentage' in team_data.columns:
                        if 'points_per_game' in team_data.columns:
                            ppg = team_data['points_per_game'].fillna(110)
                        else:
                            ppg = pd.Series([110] * len(team_data), index=team_data.index)
                        if 'field_goal_attempts_per_game' in team_data.columns:
                            fga = team_data['field_goal_attempts_per_game'].fillna(85)
                        else:
                            fga = pd.Series([85] * len(team_data), index=team_data.index)
                        if 'free_throw_attempts_per_game' in team_data.columns:
                            fta = team_data['free_throw_attempts_per_game'].fillna(20)
                        else:
                            fta = pd.Series([20] * len(team_data), index=team_data.index)
                        team_data[f'{prefix}_true_shooting_pct'] = (
                            ppg / (2 * (fga + 0.44 * fta))
                        )
                    else:
                        team_data[f'{prefix}_true_shooting_pct'] = 0.55  # NBA average
                    
                    # Rebounding percentage (estimate)
                    if 'rebounds_per_game' in team_data.columns:
                        team_data[f'{prefix}_rebounding_pct'] = min(team_data['rebounds_per_game'] / 50, 1.0)
                    else:
                        team_data[f'{prefix}_rebounding_pct'] = 0.5
                    
                    # Turnover rate (turnovers per 100 possessions)
                    if 'turnovers_per_game' in team_data.columns:
                        team_data[f'{prefix}_turnover_rate'] = (
                            team_data['turnovers_per_game'] / team_data['possessions_per_game'] * 100
                        )
                    else:
                        team_data[f'{prefix}_turnover_rate'] = 14.0  # NBA average
                    
                    # Merge advanced stats
                    merge_cols = [team_col] + [col for col in team_data.columns if col.startswith(prefix)]
                    team_data = team_data.rename(columns={'team_id': team_col})
                    
                    df = df.merge(team_data[merge_cols], on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basketball advanced features: {str(e)}")
            return df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational features like rest days and schedule context"""
        try:
            # Sort by team and date to calculate rest days
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
                
                # Back-to-back games
                df['home_back_to_back'] = (df['days_rest_home'] == 1).astype(int)
                df['away_back_to_back'] = (df['days_rest_away'] == 1).astype(int)
            
            # Home court advantage (NBA typically ~2-4 points)
            df['home_court_advantage'] = 3.0  # NBA average home advantage
            
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
                if f'{prefix}_avg_points' in df.columns:
                    # Recent form indicators (simplified)
                    df[f'{prefix}_avg_points_l5'] = df[f'{prefix}_avg_points']  # Would be last 5 games
                    df[f'{prefix}_avg_points_allowed_l5'] = df[f'{prefix}_avg_points_allowed']
                    df[f'{prefix}_win_pct_l10'] = df[f'{prefix}_win_pct_season']  # Would be last 10 games
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {str(e)}")
            return df
    
    def _add_matchup_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features"""
        try:
            # Pace differential (how different team tempos interact)
            if 'home_pace' in df.columns and 'away_pace' in df.columns:
                df['pace_matchup_differential'] = abs(df['home_pace'] - df['away_pace'])
            
            # Offensive vs Defensive matchups
            if 'home_offensive_rating' in df.columns and 'away_defensive_rating' in df.columns:
                df['offensive_rating_differential'] = df['home_offensive_rating'] - df['away_defensive_rating']
            
            # Head-to-head record (simplified - would use historical data)
            df['h2h_home_wins'] = 0  # Would calculate from historical matchups
            df['h2h_total_games'] = 0
            
            # Style matchup indicators
            if 'home_turnover_rate' in df.columns and 'away_turnover_rate' in df.columns:
                df['turnover_rate_matchup'] = abs(df['home_turnover_rate'] - df['away_turnover_rate'])
            
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
            window_sizes = [5, 10, 20]
        
        try:
            df = team_stats_df.copy()
            
            # Sort by team and date
            if 'date' in df.columns:
                df = df.sort_values(['team_id', 'date'])
                
                # Rolling averages for key stats
                for window in window_sizes:
                    for stat in ['points_per_game', 'points_against_per_game', 'field_goal_percentage', 'rebounds_per_game']:
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
                
                # Pace matchup
                home_pace = home_latest.get('possessions_per_game', 100)
                away_pace = away_latest.get('possessions_per_game', 100)
                features['pace_differential'] = abs(home_pace - away_pace)
                
                # Scoring matchup
                features['scoring_differential'] = (
                    home_latest.get('points_per_game', 110) - 
                    away_latest.get('points_against_per_game', 110)
                )
                
                # Style matchup
                features['rebounding_advantage'] = (
                    home_latest.get('rebounds_per_game', 45) - 
                    away_latest.get('rebounds_per_game', 45)
                )
            
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
                'home_offensive_rating', 'away_offensive_rating', 'home_true_shooting_pct', 
                'away_true_shooting_pct', 'home_avg_points_l5', 'away_avg_points_l5'
            ],
            'defense': [
                'home_defensive_rating', 'away_defensive_rating', 'home_avg_points_allowed_l5',
                'away_avg_points_allowed_l5'
            ],
            'pace': [
                'home_pace', 'away_pace', 'pace_matchup_differential'
            ],
            'situational': [
                'is_playoffs', 'days_rest_home', 'days_rest_away', 'home_court_advantage',
                'home_back_to_back', 'away_back_to_back', 'is_weekend'
            ],
            'form': [
                'home_win_pct_l10', 'away_win_pct_l10', 'home_win_pct_season', 'away_win_pct_season'
            ],
            'matchup': [
                'offensive_rating_differential', 'h2h_home_wins', 'h2h_total_games'
            ]
        }