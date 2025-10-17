"""
Universal Sports Ensemble Prediction System
Works for any sport: MLB, NFL, NBA, NHL, NCAA, WNBA, etc.
Combines Elo Ratings, GLMNet, and XGBoost models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import Dict, Tuple, List, Optional
import logging
import pickle
import os


class UniversalEloRatingSystem:
    """Elo rating system that works for any sport"""
    
    def __init__(self, sport: str, k_factor: float = 20, initial_rating: float = 1500):
        """
        Initialize Elo rating system
        
        Args:
            sport: Sport name (MLB, NFL, NBA, NHL, etc.)
            k_factor: How much ratings change after each game
            initial_rating: Starting rating for all teams
        """
        self.sport = sport
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
        self.logger = logging.getLogger(__name__)
    
    def get_rating(self, team: str) -> float:
        """Get team's current Elo rating"""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability for team A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team: str, away_team: str, home_won: bool):
        """Update Elo ratings after a game"""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected scores
        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = 1 - home_expected
        
        # Actual scores
        home_actual = 1.0 if home_won else 0.0
        away_actual = 1.0 - home_actual
        
        # Update ratings
        self.ratings[home_team] = home_rating + self.k_factor * (home_actual - home_expected)
        self.ratings[away_team] = away_rating + self.k_factor * (away_actual - away_expected)
    
    def predict_game(self, home_team: str, away_team: str) -> float:
        """Predict home team win probability"""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        return self.expected_score(home_rating, away_rating)


class UniversalSportsEnsemble:
    """
    Universal sports predictor using Elo + GLMNet + XGBoost
    Works for: MLB, NFL, NBA, NHL, NCAA Football, NCAA Basketball, WNBA
    """
    
    # Sport-specific Elo K-factors (optimized for each sport)
    SPORT_K_FACTORS = {
        'NFL': 35,      # 17 games, high variance per game
        'NBA': 18,      # 82 games, lower variance per game
        'NHL': 22,      # 82 games, more randomness than NBA
        'MLB': 14,      # 162 games, low variance per game
        'NCAAF': 30,    # ~12 games, high variance, recruiting matters
        'NCAAB': 24,    # ~30 games, tournament volatility
        'WNBA': 20      # 40 games, moderate variance
    }
    
    # Sport-specific XGBoost hyperparameters (tuned for each sport)
    SPORT_XGB_PARAMS = {
        'NFL': {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        },
        'NBA': {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'gamma': 0.5,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5
        },
        'NHL': {
            'n_estimators': 175,
            'max_depth': 5,
            'learning_rate': 0.04,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'gamma': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.8
        },
        'MLB': {
            'n_estimators': 250,
            'max_depth': 3,
            'learning_rate': 0.02,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'gamma': 0.3,
            'reg_alpha': 0.01,
            'reg_lambda': 0.3
        },
        'NCAAF': {
            'n_estimators': 160,
            'max_depth': 6,
            'learning_rate': 0.06,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 1.2,
            'reg_alpha': 0.15,
            'reg_lambda': 1.2
        },
        'NCAAB': {
            'n_estimators': 180,
            'max_depth': 5,
            'learning_rate': 0.04,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.7,
            'reg_alpha': 0.08,
            'reg_lambda': 0.7
        },
        'WNBA': {
            'n_estimators': 140,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'gamma': 0.6,
            'reg_alpha': 0.08,
            'reg_lambda': 0.6
        }
    }
    
    def __init__(self, sport: str, k_factor: float = None):
        """
        Initialize ensemble predictor
        
        Args:
            sport: Sport code (MLB, NFL, NBA, NHL, etc.)
            k_factor: Elo k-factor (if None, uses sport-specific optimized value)
        """
        self.sport = sport
        self.logger = logging.getLogger(__name__)
        
        # Use sport-specific K-factor if not provided
        if k_factor is None:
            k_factor = self.SPORT_K_FACTORS.get(sport, 20)
        
        # Initialize models
        self.elo_system = UniversalEloRatingSystem(sport, k_factor=k_factor)
        
        # Logistic Regression (simpler, more reliable)
        from sklearn.linear_model import LogisticRegression
        self.logistic_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        # Use sport-specific XGBoost parameters
        xgb_params = self.SPORT_XGB_PARAMS.get(sport, {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1.0
        })
        
        self.xgb_model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            **xgb_params
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Sport-specific ensemble weights (NFL needs higher XGBoost weight)
        if sport == 'NFL':
            self.ensemble_weights = {
                'elo': 0.35,
                'logistic': 0.15,
                'xgboost': 0.50  # NFL benefits more from XGBoost with proper features
            }
        else:
            self.ensemble_weights = {
                'elo': 0.35,
                'logistic': 0.15,
                'xgboost': 0.50
            }
    
    def create_features(self, df: pd.DataFrame, is_training: bool = True, team_stats: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create features for prediction with sport-specific enhancements
        
        Args:
            df: DataFrame with home_team, away_team columns
            is_training: If True, updates Elo ratings based on results
            team_stats: Optional team statistics for advanced feature engineering
            
        Returns:
            DataFrame with features
        """
        features_list = []
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get Elo ratings
            home_elo = self.elo_system.get_rating(home_team)
            away_elo = self.elo_system.get_rating(away_team)
            elo_diff = home_elo - away_elo
            
            # Create enhanced features
            features = {
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'elo_ratio': home_elo / away_elo if away_elo > 0 else 1.0,
                'home_elo_squared': home_elo ** 2,
                'away_elo_squared': away_elo ** 2,
                'elo_diff_squared': elo_diff ** 2,
                'home_advantage': 100,  # Stronger home advantage signal
                'elo_product': home_elo * away_elo,
            }
            
            # Add sport-specific features if team stats provided
            if team_stats is not None and not team_stats.empty:
                features = self._add_sport_specific_features(features, home_team, away_team, team_stats, row)
            
            # Update Elo if training
            if is_training and 'result' in df.columns:
                result = row['result']
                if result in ['H', 'A']:
                    home_won = result == 'H'
                    self.elo_system.update_ratings(home_team, away_team, home_won)
                    features['target'] = 1 if home_won else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _add_sport_specific_features(self, features: dict, home_team: str, away_team: str, team_stats: pd.DataFrame, game_row: pd.Series) -> dict:
        """
        Add sport-specific advanced features with chronological filtering to prevent target leakage.
        Computes rolling averages, lag variables, and fatigue metrics using ONLY prior games.
        """
        
        # Get game date for chronological filtering
        game_date = game_row.get('game_date') if 'game_date' in game_row else None
        
        # CRITICAL: Filter to only use team stats from games BEFORE this game (prevent target leakage)
        if game_date is not None and 'date' in team_stats.columns:
            try:
                # Ensure date columns are datetime
                if not pd.api.types.is_datetime64_any_dtype(team_stats['date']):
                    team_stats = team_stats.copy()
                    team_stats['date'] = pd.to_datetime(team_stats['date'], errors='coerce')
                
                if not pd.api.types.is_datetime64_any_dtype(pd.Series([game_date])):
                    game_date = pd.to_datetime(game_date, errors='coerce')
                
                # Filter to only prior games
                prior_stats = team_stats[team_stats['date'] < game_date].copy()
            except:
                prior_stats = team_stats.copy()
        else:
            prior_stats = team_stats.copy()
        
        if prior_stats.empty:
            return features  # No prior data available
        
        # Compute chronological team stats for home and away teams
        home_prior = prior_stats[prior_stats['team_id'] == home_team].sort_values('date') if 'date' in prior_stats.columns else prior_stats[prior_stats['team_id'] == home_team]
        away_prior = prior_stats[prior_stats['team_id'] == away_team].sort_values('date') if 'date' in prior_stats.columns else prior_stats[prior_stats['team_id'] == away_team]
        
        if home_prior.empty or away_prior.empty:
            return features
        
        # NFL-specific comprehensive features
        if self.sport == 'NFL':
            features.update(self._nfl_features(home_prior, away_prior, home_team, away_team, game_date))
        
        # NBA-specific comprehensive features
        elif self.sport == 'NBA':
            features.update(self._nba_features(home_prior, away_prior, home_team, away_team, game_date))
        
        # NHL-specific comprehensive features
        elif self.sport == 'NHL':
            features.update(self._nhl_features(home_prior, away_prior, home_team, away_team, game_date))
        
        # MLB-specific comprehensive features
        elif self.sport == 'MLB':
            features.update(self._mlb_features(home_prior, away_prior, home_team, away_team, game_date))
        
        # NCAAF-specific comprehensive features
        elif self.sport == 'NCAAF':
            features.update(self._ncaaf_features(home_prior, away_prior, home_team, away_team, game_date))
        
        return features
    
    def _compute_rolling_stats(self, team_prior: pd.DataFrame, stat_cols: list, windows: list = [3, 5]) -> dict:
        """Compute rolling average statistics for multiple windows"""
        rolling_features = {}
        
        for window in windows:
            for col in stat_cols:
                if col in team_prior.columns:
                    rolling_val = team_prior[col].tail(window).mean()
                    rolling_features[f'{col}_last{window}'] = rolling_val if not pd.isna(rolling_val) else 0
        
        return rolling_features
    
    def _compute_lag_features(self, team_prior: pd.DataFrame, stat_cols: list, lags: list = [1, 2]) -> dict:
        """Compute lag features (previous game stats)"""
        lag_features = {}
        
        for lag in lags:
            for col in stat_cols:
                if col in team_prior.columns and len(team_prior) >= lag:
                    lag_val = team_prior[col].iloc[-lag]
                    lag_features[f'{col}_lag{lag}'] = lag_val if not pd.isna(lag_val) else 0
        
        return lag_features
    
    def _compute_rest_days(self, team_prior: pd.DataFrame, current_date) -> int:
        """Compute days of rest since last game"""
        if 'date' in team_prior.columns and not team_prior.empty and current_date is not None:
            try:
                last_game_date = team_prior['date'].iloc[-1]
                if pd.notna(last_game_date) and pd.notna(current_date):
                    return (pd.to_datetime(current_date) - pd.to_datetime(last_game_date)).days
            except:
                pass
        return 7  # Default 1 week rest
    
    def _nfl_features(self, home_prior: pd.DataFrame, away_prior: pd.DataFrame, home_team: str, away_team: str, game_date) -> dict:
        """NFL-specific comprehensive features"""
        features = {}
        
        # Season averages (all prior games)
        home_season = home_prior.tail(10)  # Last 10 games max for season stats
        away_season = away_prior.tail(10)
        
        # Offensive stats
        features['home_pass_ypg'] = home_season['passing_yards_per_game'].mean() if 'passing_yards_per_game' in home_season else 250
        features['away_pass_ypg'] = away_season['passing_yards_per_game'].mean() if 'passing_yards_per_game' in away_season else 250
        features['home_rush_ypg'] = home_season['rushing_yards_per_game'].mean() if 'rushing_yards_per_game' in home_season else 120
        features['away_rush_ypg'] = away_season['rushing_yards_per_game'].mean() if 'rushing_yards_per_game' in away_season else 120
        
        # Defensive stats
        features['home_pass_yds_allowed'] = home_season['pass_yards_allowed_per_game'].mean() if 'pass_yards_allowed_per_game' in home_season else 250
        features['away_pass_yds_allowed'] = away_season['pass_yards_allowed_per_game'].mean() if 'pass_yards_allowed_per_game' in away_season else 250
        features['home_rush_yds_allowed'] = home_season['rush_yards_allowed_per_game'].mean() if 'rush_yards_allowed_per_game' in home_season else 120
        features['away_rush_yds_allowed'] = away_season['rush_yards_allowed_per_game'].mean() if 'rush_yards_allowed_per_game' in away_season else 120
        
        # Rolling averages (3-game, 5-game)
        nfl_stat_cols = ['offensive_epa', 'defensive_epa', 'net_epa']
        features.update({f'home_{k}': v for k, v in self._compute_rolling_stats(home_prior, nfl_stat_cols, [3, 5]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_rolling_stats(away_prior, nfl_stat_cols, [3, 5]).items()})
        
        # Lag features (previous 2 games)
        features.update({f'home_{k}': v for k, v in self._compute_lag_features(home_prior, nfl_stat_cols, [1, 2]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_lag_features(away_prior, nfl_stat_cols, [1, 2]).items()})
        
        # Rest/Fatigue metrics
        features['home_rest_days'] = self._compute_rest_days(home_prior, game_date)
        features['away_rest_days'] = self._compute_rest_days(away_prior, game_date)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        features['home_short_rest'] = 1 if features['home_rest_days'] <= 4 else 0  # Thursday/Monday games
        features['away_short_rest'] = 1 if features['away_rest_days'] <= 4 else 0
        
        # Matchup features
        features['off_matchup'] = features['home_pass_ypg'] + features['home_rush_ypg'] - (features['away_pass_yds_allowed'] + features['away_rush_yds_allowed'])
        features['def_matchup'] = (features['away_pass_ypg'] + features['away_rush_ypg']) - (features['home_pass_yds_allowed'] + features['home_rush_yds_allowed'])
        
        return features
    
    def _nba_features(self, home_prior: pd.DataFrame, away_prior: pd.DataFrame, home_team: str, away_team: str, game_date) -> dict:
        """NBA-specific comprehensive features"""
        features = {}
        
        home_season = home_prior.tail(15)
        away_season = away_prior.tail(15)
        
        # Scoring stats
        features['home_ppg'] = home_season['points_per_game'].mean() if 'points_per_game' in home_season else 110
        features['away_ppg'] = away_season['points_per_game'].mean() if 'points_per_game' in away_season else 110
        features['home_papg'] = home_season['points_allowed_per_game'].mean() if 'points_allowed_per_game' in home_season else 110
        features['away_papg'] = away_season['points_allowed_per_game'].mean() if 'points_allowed_per_game' in away_season else 110
        
        # Rolling averages (3, 5, 10 game)
        nba_cols = ['points_per_game', 'points_allowed_per_game']
        features.update({f'home_{k}': v for k, v in self._compute_rolling_stats(home_prior, nba_cols, [3, 5, 10]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_rolling_stats(away_prior, nba_cols, [3, 5, 10]).items()})
        
        # Lag features (previous 2 games)
        features.update({f'home_{k}': v for k, v in self._compute_lag_features(home_prior, nba_cols, [1, 2]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_lag_features(away_prior, nba_cols, [1, 2]).items()})
        
        # Rest/fatigue (NBA back-to-backs matter)
        features['home_rest_days'] = self._compute_rest_days(home_prior, game_date)
        features['away_rest_days'] = self._compute_rest_days(away_prior, game_date)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        features['home_back_to_back'] = 1 if features['home_rest_days'] <= 1 else 0
        features['away_back_to_back'] = 1 if features['away_rest_days'] <= 1 else 0
        
        # Matchup
        features['off_def_matchup'] = (features['home_ppg'] - features['away_papg']) - (features['away_ppg'] - features['home_papg'])
        
        return features
    
    def _nhl_features(self, home_prior: pd.DataFrame, away_prior: pd.DataFrame, home_team: str, away_team: str, game_date) -> dict:
        """NHL-specific comprehensive features"""
        features = {}
        
        home_season = home_prior.tail(15)
        away_season = away_prior.tail(15)
        
        # Scoring stats
        features['home_gpg'] = home_season['goals_per_game'].mean() if 'goals_per_game' in home_season else 3.0
        features['away_gpg'] = away_season['goals_per_game'].mean() if 'goals_per_game' in away_season else 3.0
        features['home_gapg'] = home_season['goals_against_per_game'].mean() if 'goals_against_per_game' in home_season else 3.0
        features['away_gapg'] = away_season['goals_against_per_game'].mean() if 'goals_against_per_game' in away_season else 3.0
        
        # Rolling averages
        nhl_cols = ['goals_per_game', 'goals_against_per_game']
        features.update({f'home_{k}': v for k, v in self._compute_rolling_stats(home_prior, nhl_cols, [3, 5, 10]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_rolling_stats(away_prior, nhl_cols, [3, 5, 10]).items()})
        
        # Lag features (previous 2 games)
        features.update({f'home_{k}': v for k, v in self._compute_lag_features(home_prior, nhl_cols, [1, 2]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_lag_features(away_prior, nhl_cols, [1, 2]).items()})
        
        # Rest/fatigue
        features['home_rest_days'] = self._compute_rest_days(home_prior, game_date)
        features['away_rest_days'] = self._compute_rest_days(away_prior, game_date)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        features['home_back_to_back'] = 1 if features['home_rest_days'] <= 1 else 0
        features['away_back_to_back'] = 1 if features['away_rest_days'] <= 1 else 0
        
        # Matchup
        features['off_def_matchup'] = (features['home_gpg'] - features['away_gapg']) - (features['away_gpg'] - features['home_gapg'])
        
        return features
    
    def _mlb_features(self, home_prior: pd.DataFrame, away_prior: pd.DataFrame, home_team: str, away_team: str, game_date) -> dict:
        """MLB-specific comprehensive features"""
        features = {}
        
        home_season = home_prior.tail(20)
        away_season = away_prior.tail(20)
        
        # Pitching stats
        features['home_era'] = home_season['era'].mean() if 'era' in home_season else 4.0
        features['away_era'] = away_season['era'].mean() if 'era' in away_season else 4.0
        features['home_whip'] = home_season['whip'].mean() if 'whip' in home_season else 1.3
        features['away_whip'] = away_season['whip'].mean() if 'whip' in away_season else 1.3
        
        # Batting stats
        features['home_ops'] = home_season['ops'].mean() if 'ops' in home_season else 0.750
        features['away_ops'] = away_season['ops'].mean() if 'ops' in away_season else 0.750
        
        # Rolling averages (5, 10, 15 game for baseball)
        mlb_cols = ['era', 'ops', 'runs_per_game']
        features.update({f'home_{k}': v for k, v in self._compute_rolling_stats(home_prior, mlb_cols, [5, 10, 15]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_rolling_stats(away_prior, mlb_cols, [5, 10, 15]).items()})
        
        # Lag features (previous 2 games)
        features.update({f'home_{k}': v for k, v in self._compute_lag_features(home_prior, mlb_cols, [1, 2]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_lag_features(away_prior, mlb_cols, [1, 2]).items()})
        
        # Rest (less important in MLB)
        features['home_rest_days'] = self._compute_rest_days(home_prior, game_date)
        features['away_rest_days'] = self._compute_rest_days(away_prior, game_date)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        
        # Matchup
        features['pitching_matchup'] = features['away_era'] - features['home_era']  # Lower is better
        features['batting_matchup'] = features['home_ops'] - features['away_ops']
        
        return features
    
    def _ncaaf_features(self, home_prior: pd.DataFrame, away_prior: pd.DataFrame, home_team: str, away_team: str, game_date) -> dict:
        """NCAA Football-specific comprehensive features"""
        features = {}
        
        home_season = home_prior.tail(8)  # Shorter season
        away_season = away_prior.tail(8)
        
        # Offensive stats
        features['home_ppg'] = home_season['points_per_game'].mean() if 'points_per_game' in home_season else 28
        features['away_ppg'] = away_season['points_per_game'].mean() if 'points_per_game' in away_season else 28
        
        # Defensive stats
        features['home_papg'] = home_season['points_allowed_per_game'].mean() if 'points_allowed_per_game' in home_season else 24
        features['away_papg'] = away_season['points_allowed_per_game'].mean() if 'points_allowed_per_game' in away_season else 24
        
        # Rolling averages (3-game for college)
        ncaaf_cols = ['points_per_game', 'points_allowed_per_game', 'total_yards_per_game']
        features.update({f'home_{k}': v for k, v in self._compute_rolling_stats(home_prior, ncaaf_cols, [3]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_rolling_stats(away_prior, ncaaf_cols, [3]).items()})
        
        # Lag features (previous 2 games)
        features.update({f'home_{k}': v for k, v in self._compute_lag_features(home_prior, ncaaf_cols, [1, 2]).items()})
        features.update({f'away_{k}': v for k, v in self._compute_lag_features(away_prior, ncaaf_cols, [1, 2]).items()})
        
        # Rest
        features['home_rest_days'] = self._compute_rest_days(home_prior, game_date)
        features['away_rest_days'] = self._compute_rest_days(away_prior, game_date)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        
        # Matchup
        features['off_def_matchup'] = (features['home_ppg'] - features['away_papg']) - (features['away_ppg'] - features['home_papg'])
        
        return features
    
    def train(self, historical_df: pd.DataFrame, team_stats: pd.DataFrame = None) -> Dict:
        """
        Train all models on historical data
        
        Args:
            historical_df: DataFrame with home_team, away_team, result columns
                          result should be 'H' (home win) or 'A' (away win)
            team_stats: Optional team statistics for advanced feature engineering
        
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info(f"Training {self.sport} ensemble on {len(historical_df)} games")
            
            # Create features with Elo updates and sport-specific stats
            features_df = self.create_features(historical_df, is_training=True, team_stats=team_stats)
            
            # Filter only games with results
            features_df = features_df.dropna(subset=['target'])
            
            if len(features_df) == 0:
                self.logger.warning("No training data with results")
                return {'error': 'No training data'}
            
            # Prepare training data with all available features
            # Start with base Elo features
            feature_cols = ['home_elo', 'away_elo', 'elo_diff', 'elo_ratio', 
                          'home_elo_squared', 'away_elo_squared', 'elo_diff_squared',
                          'home_advantage', 'elo_product']
            
            # Add sport-specific features if they exist
            sport_feature_cols = [col for col in features_df.columns 
                                if col not in feature_cols + ['target'] 
                                and not col.endswith('_id') 
                                and not col.startswith('game_')]
            
            if sport_feature_cols:
                feature_cols.extend(sport_feature_cols)
                self.logger.info(f"Using {len(sport_feature_cols)} sport-specific features for {self.sport}")
            
            # Store feature columns for prediction
            self.feature_cols = feature_cols
            
            X = features_df[feature_cols].fillna(0).values  # Fill NaN with 0 for missing stats
            y = features_df['target'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Logistic Regression
            self.logger.info("Training Logistic Regression...")
            self.logistic_model.fit(X_scaled, y)
            logistic_score = self.logistic_model.score(X_scaled, y)
            
            # Train XGBoost
            self.logger.info("Training XGBoost...")
            self.xgb_model.fit(X_scaled, y)
            xgb_score = self.xgb_model.score(X_scaled, y)
            
            self.is_trained = True
            
            results = {
                'sport': self.sport,
                'games_trained': len(features_df),
                'features_used': len(feature_cols),
                'feature_names': feature_cols,
                'logistic_accuracy': float(logistic_score),
                'xgboost_accuracy': float(xgb_score),
                'teams': len(self.elo_system.ratings)
            }
            
            self.logger.info(f"Training complete: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training {self.sport} models: {str(e)}")
            raise
    
    def predict_game(self, home_team: str, away_team: str, team_stats: pd.DataFrame = None) -> Dict:
        """
        Predict game outcome using ensemble
        
        Args:
            home_team: Home team name
            away_team: Away team name
            team_stats: Optional team statistics for advanced features
            
        Returns:
            Dictionary with predictions from each model
        """
        # Create features
        game_df = pd.DataFrame([{
            'home_team': home_team,
            'away_team': away_team
        }])
        
        features_df = self.create_features(game_df, is_training=False, team_stats=team_stats)
        
        # Use stored feature columns if available, otherwise use base features
        if hasattr(self, 'feature_cols'):
            feature_cols = self.feature_cols
        else:
            feature_cols = ['home_elo', 'away_elo', 'elo_diff', 'elo_ratio', 
                          'home_elo_squared', 'away_elo_squared', 'elo_diff_squared',
                          'home_advantage', 'elo_product']
        
        # Ensure all required features exist
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0  # Fill missing features with 0
        
        X = features_df[feature_cols].fillna(0).values
        
        # Elo prediction
        elo_prob = self.elo_system.predict_game(home_team, away_team)
        
        # ML predictions
        if self.is_trained:
            X_scaled = self.scaler.transform(X)
            logistic_prob = self.logistic_model.predict_proba(X_scaled)[0][1]
            xgb_prob = self.xgb_model.predict_proba(X_scaled)[0][1]
        else:
            logistic_prob = elo_prob
            xgb_prob = elo_prob
        
        # Blended prediction
        blended_prob = (
            self.ensemble_weights['elo'] * elo_prob +
            self.ensemble_weights['logistic'] * logistic_prob +
            self.ensemble_weights['xgboost'] * xgb_prob
        )
        
        predicted_winner = home_team if blended_prob > 0.5 else away_team
        
        return {
            'sport': self.sport,
            'home_team': home_team,
            'away_team': away_team,
            'elo_home_prob': float(elo_prob),
            'logistic_home_prob': float(logistic_prob),
            'xgboost_home_prob': float(xgb_prob),
            'blended_home_prob': float(blended_prob),
            'predicted_winner': predicted_winner,
            'home_win_probability': float(blended_prob),
            'confidence': abs(blended_prob - 0.5) * 2
        }
    
    def predict_multiple_games(self, games_df: pd.DataFrame) -> List[Dict]:
        """
        Predict multiple games
        
        Args:
            games_df: DataFrame with home_team, away_team columns
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for idx, row in games_df.iterrows():
            pred = self.predict_game(row['home_team'], row['away_team'])
            predictions.append(pred)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'sport': self.sport,
            'elo_ratings': self.elo_system.ratings,
            'k_factor': self.elo_system.k_factor,
            'logistic_model': self.logistic_model,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'ensemble_weights': self.ensemble_weights,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.sport = model_data['sport']
        self.elo_system.ratings = model_data['elo_ratings']
        self.elo_system.k_factor = model_data['k_factor']
        
        # Handle backward compatibility with old glmnet models
        if 'logistic_model' in model_data:
            self.logistic_model = model_data['logistic_model']
        elif 'glmnet_model' in model_data:
            self.logistic_model = model_data['glmnet_model']
            
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.ensemble_weights = model_data['ensemble_weights']
        
        # Update old ensemble weights if using glmnet key
        if 'glmnet' in self.ensemble_weights:
            self.ensemble_weights['logistic'] = self.ensemble_weights.pop('glmnet')
            
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")
