import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from src.interfaces.base_feature_engineer import BaseFeatureEngineer


class NHLFeatureEngineer(BaseFeatureEngineer):
    """
    NHL-specific feature engineering for hockey prediction models.
    
    Creates features specific to hockey analytics including:
    - Goals for/against rates and shot metrics
    - Power play and penalty kill efficiency
    - Goaltender performance and save percentages
    - Rest days and travel factors
    - Special situations and game flow metrics
    """
    
    def __init__(self):
        super().__init__('NHL')
        self.logger = logging.getLogger(__name__)
        
        # NHL-specific feature definitions
        self.required_features = [
            # Basic game features
            'home_team_encoded', 'away_team_encoded', 'month', 'day_of_week', 'is_weekend',
            'is_playoffs', 'days_rest_home', 'days_rest_away',
            
            # Team performance features
            'home_win_pct_l10', 'away_win_pct_l10', 'home_win_pct_season', 'away_win_pct_season',
            'home_goals_per_game', 'away_goals_per_game', 'home_goals_against_per_game', 'away_goals_against_per_game',
            
            # Hockey-specific features
            'home_shots_per_game', 'away_shots_per_game', 'home_shots_against_per_game', 'away_shots_against_per_game',
            'home_shooting_pct', 'away_shooting_pct', 'home_save_pct', 'away_save_pct',
            'home_powerplay_pct', 'away_powerplay_pct', 'home_penalty_kill_pct', 'away_penalty_kill_pct',
            
            # Matchup features
            'goal_differential', 'shot_differential', 'special_teams_advantage', 'goaltending_advantage',
            'h2h_home_wins', 'h2h_total_games', 'home_ice_advantage', 'total_prediction'
        ]
        
        self.feature_descriptions = {
            'home_goals_per_game': 'Average goals scored per game by home team',
            'away_goals_per_game': 'Average goals scored per game by away team',
            'home_goals_against_per_game': 'Average goals allowed per game by home team',
            'away_goals_against_per_game': 'Average goals allowed per game by away team',
            'home_shots_per_game': 'Average shots on goal per game by home team',
            'away_shots_per_game': 'Average shots on goal per game by away team',
            'home_shooting_pct': 'Goals per shot percentage for home team',
            'away_shooting_pct': 'Goals per shot percentage for away team',
            'home_save_pct': 'Goaltender save percentage for home team',
            'away_save_pct': 'Goaltender save percentage for away team',
            'home_powerplay_pct': 'Power play success rate for home team',
            'away_powerplay_pct': 'Power play success rate for away team',
            'home_penalty_kill_pct': 'Penalty kill success rate for home team',
            'away_penalty_kill_pct': 'Penalty kill success rate for away team',
            'days_rest_home': 'Days of rest for home team since last game',
            'days_rest_away': 'Days of rest for away team since last game',
            'is_playoffs': 'Whether game is in playoffs (different dynamics)',
            'home_ice_advantage': 'Estimated home ice advantage goals',
            'goal_differential': 'Expected goal differential based on team stats',
            'shot_differential': 'Expected shot differential based on team stats',
            'special_teams_advantage': 'Combined power play/penalty kill advantage',
            'goaltending_advantage': 'Difference in goaltender save percentages'
        }
    
    def create_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame,
                       additional_data: Optional[Dict] = None,
                       include_targets: bool = True) -> pd.DataFrame:
        """
        Create comprehensive NHL features for prediction models.
        
        Args:
            games_df: DataFrame with NHL game information
            team_stats_df: DataFrame with NHL team statistics
            additional_data: Optional additional data (goalie stats, injuries, etc.)
            include_targets: Whether to include target variables
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if games_df.empty:
                self.logger.warning("Empty games DataFrame provided")
                return games_df
            
            self.logger.info(f"Creating NHL features for {len(games_df)} games")
            features_df = games_df.copy()
            
            # Basic game features
            features_df = self._add_basic_features(features_df)
            
            # NHL-specific team performance features
            features_df = self._add_team_performance_features(features_df, team_stats_df)
            
            # Hockey-specific advanced features
            features_df = self._add_hockey_advanced_features(features_df, team_stats_df)
            
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
            
            self.logger.info(f"Created {len(features_df.columns)} NHL features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating NHL features: {str(e)}")
            return games_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic NHL game features"""
        try:
            # Date features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['month'] = df['game_date'].dt.month
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                
                # NHL season features (season runs Oct-June)
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
            self.logger.error(f"Error adding basic NHL features: {str(e)}")
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
                        'goals_per_game': f'{prefix}_goals_per_game',
                        'goals_against_per_game': f'{prefix}_goals_against_per_game'
                    })
                    
                    df = df.merge(team_data[[team_col, f'{prefix}_wins', f'{prefix}_losses', 
                                           f'{prefix}_win_pct_season', f'{prefix}_goals_per_game', 
                                           f'{prefix}_goals_against_per_game']], 
                                 on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding team performance features: {str(e)}")
            return df
    
    def _add_hockey_advanced_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add NHL-specific advanced hockey statistics"""
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
                    if 'shots_per_game' in team_data.columns:
                        team_data[f'{prefix}_shots_per_game'] = team_data['shots_per_game']
                    else:
                        team_data[f'{prefix}_shots_per_game'] = 30.0  # NHL average
                    
                    if 'shots_against_per_game' in team_data.columns:
                        team_data[f'{prefix}_shots_against_per_game'] = team_data['shots_against_per_game']
                    else:
                        team_data[f'{prefix}_shots_against_per_game'] = 30.0  # NHL average
                    
                    # Shooting percentage (goals per shot)
                    goals_pg = team_data.get('goals_per_game', 3.0)
                    shots_pg = team_data[f'{prefix}_shots_per_game']
                    team_data[f'{prefix}_shooting_pct'] = goals_pg / max(shots_pg, 1) * 100
                    
                    # Save percentage (based on goals against and shots against)
                    goals_against_pg = team_data.get('goals_against_per_game', 3.0)
                    shots_against_pg = team_data[f'{prefix}_shots_against_per_game']
                    saves_per_game = shots_against_pg - goals_against_pg
                    team_data[f'{prefix}_save_pct'] = saves_per_game / max(shots_against_pg, 1) * 100
                    
                    # Special teams (use actual stats if available, otherwise estimate)
                    if 'power_play_percentage' in team_data.columns:
                        team_data[f'{prefix}_powerplay_pct'] = team_data['power_play_percentage']
                    else:
                        team_data[f'{prefix}_powerplay_pct'] = 20.0  # NHL average
                    
                    if 'penalty_kill_percentage' in team_data.columns:
                        team_data[f'{prefix}_penalty_kill_pct'] = team_data['penalty_kill_percentage']
                    else:
                        team_data[f'{prefix}_penalty_kill_pct'] = 80.0  # NHL average
                    
                    # Merge advanced stats
                    merge_cols = [team_col] + [col for col in team_data.columns if col.startswith(prefix)]
                    team_data = team_data.rename(columns={'team_id': team_col})
                    
                    df = df.merge(team_data[merge_cols], on=team_col, how='left')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding hockey advanced features: {str(e)}")
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
                df['days_rest_home'] = df['days_rest_home'].fillna(2)  # Default rest
                
                # Calculate rest days for away team  
                df = df.sort_values(['away_team_id', 'game_date'])
                df['away_prev_game_date'] = df.groupby('away_team_id')['game_date'].shift(1)
                df['days_rest_away'] = (df['game_date'] - df['away_prev_game_date']).dt.days
                df['days_rest_away'] = df['days_rest_away'].fillna(2)  # Default rest
                
                # Rest advantage
                df['rest_advantage'] = df['days_rest_home'] - df['days_rest_away']
                
                # Back-to-back games (common in NHL)
                df['home_back_to_back'] = (df['days_rest_home'] == 1).astype(int)
                df['away_back_to_back'] = (df['days_rest_away'] == 1).astype(int)
            
            # Home ice advantage (NHL typically ~0.3-0.5 goals)
            df['home_ice_advantage'] = 0.4  # NHL average home advantage
            
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
                if f'{prefix}_goals_per_game' in df.columns:
                    # Recent form indicators (simplified)
                    df[f'{prefix}_goals_per_game_l5'] = df[f'{prefix}_goals_per_game']
                    df[f'{prefix}_goals_against_per_game_l5'] = df[f'{prefix}_goals_against_per_game']
                    df[f'{prefix}_win_pct_l10'] = df[f'{prefix}_win_pct_season']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling features: {str(e)}")
            return df
    
    def _add_matchup_features(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features"""
        try:
            # Goal differential (expected goals based on team stats)
            if 'home_goals_per_game' in df.columns and 'away_goals_against_per_game' in df.columns:
                df['expected_home_goals'] = (df['home_goals_per_game'] + df['away_goals_against_per_game']) / 2
                df['expected_away_goals'] = (df['away_goals_per_game'] + df['home_goals_against_per_game']) / 2
                df['goal_differential'] = df['expected_home_goals'] - df['expected_away_goals']
            
            # Shot differential
            if 'home_shots_per_game' in df.columns and 'away_shots_against_per_game' in df.columns:
                df['shot_differential'] = (df['home_shots_per_game'] - df['away_shots_per_game'])
            
            # Special teams matchup
            if 'home_powerplay_pct' in df.columns and 'away_penalty_kill_pct' in df.columns:
                df['special_teams_advantage'] = (
                    (df['home_powerplay_pct'] - df['away_penalty_kill_pct']) +
                    (df['home_penalty_kill_pct'] - df['away_powerplay_pct'])
                ) / 2
            
            # Goaltending advantage
            if 'home_save_pct' in df.columns and 'away_save_pct' in df.columns:
                df['goaltending_advantage'] = df['home_save_pct'] - df['away_save_pct']
            
            # Head-to-head record (simplified - would use historical data)
            df['h2h_home_wins'] = 0  # Would calculate from historical matchups
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
                
                # Total goals target
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
                    for stat in ['goals_per_game', 'goals_against_per_game', 'shots_per_game', 'save_percentage']:
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
                
                # Goal matchup
                features['goal_differential'] = (
                    home_latest.get('goals_per_game', 3.0) - 
                    away_latest.get('goals_against_per_game', 3.0)
                )
                
                # Special teams matchup
                home_pp = home_latest.get('power_play_percentage', 20.0)
                away_pk = away_latest.get('penalty_kill_percentage', 80.0)
                features['pp_vs_pk_advantage'] = home_pp - (100 - away_pk)
                
                # Goaltending matchup
                features['save_pct_differential'] = (
                    home_latest.get('save_percentage', 91.0) - 
                    away_latest.get('save_percentage', 91.0)
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
                'home_goals_per_game', 'away_goals_per_game', 'home_shots_per_game', 
                'away_shots_per_game', 'home_shooting_pct', 'away_shooting_pct'
            ],
            'defense': [
                'home_goals_against_per_game', 'away_goals_against_per_game', 
                'home_shots_against_per_game', 'away_shots_against_per_game',
                'home_save_pct', 'away_save_pct'
            ],
            'special': [
                'home_powerplay_pct', 'away_powerplay_pct', 'home_penalty_kill_pct', 
                'away_penalty_kill_pct', 'special_teams_advantage'
            ],
            'situational': [
                'is_playoffs', 'days_rest_home', 'days_rest_away', 'home_ice_advantage',
                'home_back_to_back', 'away_back_to_back', 'is_weekend'
            ],
            'form': [
                'home_win_pct_l10', 'away_win_pct_l10', 'home_win_pct_season', 'away_win_pct_season'
            ],
            'matchup': [
                'goal_differential', 'shot_differential', 'goaltending_advantage', 
                'h2h_home_wins', 'h2h_total_games'
            ]
        }