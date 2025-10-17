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
    """Elo rating system that works for any sport with advanced features"""
    
    def __init__(self, sport: str, k_factor: float = 20, initial_rating: float = 1500, use_time_decay: bool = False):
        """
        Initialize Elo rating system
        
        Args:
            sport: Sport name (MLB, NFL, NBA, NHL, etc.)
            k_factor: How much ratings change after each game
            initial_rating: Starting rating for all teams
            use_time_decay: If True, apply time decay to weight recent games higher
        """
        self.sport = sport
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.use_time_decay = use_time_decay
        self.ratings = {}
        # Offensive/Defensive split ratings for better matchup analysis
        self.offensive_ratings = {}
        self.defensive_ratings = {}
        self.logger = logging.getLogger(__name__)
    
    def get_rating(self, team: str) -> float:
        """Get team's current Elo rating"""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]
    
    def get_offensive_rating(self, team: str) -> float:
        """Get team's offensive Elo rating"""
        if team not in self.offensive_ratings:
            self.offensive_ratings[team] = self.initial_rating
        return self.offensive_ratings[team]
    
    def get_defensive_rating(self, team: str) -> float:
        """Get team's defensive Elo rating"""
        if team not in self.defensive_ratings:
            self.defensive_ratings[team] = self.initial_rating
        return self.defensive_ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability for team A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team: str, away_team: str, home_won: bool, home_score: int = None, away_score: int = None):
        """Update Elo ratings after a game with margin of victory adjustment"""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected scores
        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = 1 - home_expected
        
        # Actual scores
        home_actual = 1.0 if home_won else 0.0
        away_actual = 1.0 - home_actual
        
        # Margin of victory multiplier (if scores provided)
        mov_multiplier = 1.0
        if home_score is not None and away_score is not None:
            score_diff = abs(home_score - away_score)
            rating_diff = abs(home_rating - away_rating)
            # Logarithmic scaling: bigger wins matter more, but diminishing returns
            mov_multiplier = np.log(score_diff + 1) * 2.2 / (rating_diff / 400 + 1.0)
            mov_multiplier = min(mov_multiplier, 2.5)  # Cap at 2.5x
        
        # Update ratings with MoV adjustment
        k_effective = self.k_factor * mov_multiplier
        self.ratings[home_team] = home_rating + k_effective * (home_actual - home_expected)
        self.ratings[away_team] = away_rating + k_effective * (away_actual - away_expected)
        
        # Update offensive/defensive split ratings (if scores provided)
        if home_score is not None and away_score is not None:
            self._update_split_ratings(home_team, away_team, home_score, away_score)
    
    def _update_split_ratings(self, home_team: str, away_team: str, home_score: int, away_score: int):
        """Update offensive and defensive Elo ratings based on points scored/allowed"""
        # Get current split ratings
        home_off = self.get_offensive_rating(home_team)
        away_off = self.get_offensive_rating(away_team)
        home_def = self.get_defensive_rating(home_team)
        away_def = self.get_defensive_rating(away_team)
        
        # Expected points (normalized to 0-1 scale)
        # Higher defensive rating = fewer points allowed
        home_off_expected = self.expected_score(home_off, away_def)
        away_off_expected = self.expected_score(away_off, home_def)
        
        # Actual performance (normalized)
        total_points = home_score + away_score
        if total_points > 0:
            home_off_actual = home_score / total_points
            away_off_actual = away_score / total_points
        else:
            home_off_actual = 0.5
            away_off_actual = 0.5
        
        # Update offensive ratings
        k_split = self.k_factor * 0.8  # Slightly lower K for split ratings
        self.offensive_ratings[home_team] = home_off + k_split * (home_off_actual - home_off_expected)
        self.offensive_ratings[away_team] = away_off + k_split * (away_off_actual - away_off_expected)
        
        # Update defensive ratings (inverse logic - lower score allowed = better)
        home_def_actual = 1 - away_off_actual  # Good defense = opponent scores less
        away_def_actual = 1 - home_off_actual
        home_def_expected = 1 - away_off_expected
        away_def_expected = 1 - home_off_expected
        
        self.defensive_ratings[home_team] = home_def + k_split * (home_def_actual - home_def_expected)
        self.defensive_ratings[away_team] = away_def + k_split * (away_def_actual - away_def_expected)
    
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
            'n_estimators': 120,
            'max_depth': 3,
            'learning_rate': 0.03,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 5,
            'gamma': 1,
            'reg_alpha': 1.0,
            'reg_lambda': 10.0
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
        
        # Logistic Regression (simpler, more reliable) with strong regularization
        from sklearn.linear_model import LogisticRegression
        self.logistic_model = LogisticRegression(
            penalty='l2',
            C=0.1,  # Strong regularization to prevent overfitting
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
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
        
        # Sport-specific ensemble weights
        if sport == 'NFL':
            # NFL: Favor Elo (77% accuracy) over overfitted ML models
            # After regularization tuning, rebalance based on CV performance
            self.ensemble_weights = {
                'elo': 0.60,      # Best performer on real games
                'xgboost': 0.30,  # Heavily regularized
                'logistic': 0.10  # Heavily regularized
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
            
            # Get offensive/defensive split Elo ratings
            home_off_elo = self.elo_system.get_offensive_rating(home_team)
            away_off_elo = self.elo_system.get_offensive_rating(away_team)
            home_def_elo = self.elo_system.get_defensive_rating(home_team)
            away_def_elo = self.elo_system.get_defensive_rating(away_team)
            
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
                # Offensive/Defensive split Elo features
                'home_off_elo': home_off_elo,
                'away_off_elo': away_off_elo,
                'home_def_elo': home_def_elo,
                'away_def_elo': away_def_elo,
                'off_elo_diff': home_off_elo - away_off_elo,
                'def_elo_diff': home_def_elo - away_def_elo,
                # Matchup-specific Elo (offense vs defense)
                'off_def_matchup_elo': (home_off_elo - away_def_elo) - (away_off_elo - home_def_elo),
            }
            
            # Add sport-specific features if team stats provided
            if team_stats is not None and not team_stats.empty:
                features = self._add_sport_specific_features(features, home_team, away_team, team_stats, row)
            
            # Update Elo if training
            if is_training and 'result' in df.columns:
                result = row['result']
                if result in ['H', 'A']:
                    home_won = result == 'H'
                    # Get scores if available for MoV and split Elo updates
                    home_score = row.get('home_score') if 'home_score' in row else None
                    away_score = row.get('away_score') if 'away_score' in row else None
                    self.elo_system.update_ratings(home_team, away_team, home_won, home_score, away_score)
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
        # Column name is 'team' (renamed during loading from team_id)
        team_col = 'team' if 'team' in prior_stats.columns else 'team_id'
        home_prior = prior_stats[prior_stats[team_col] == home_team].sort_values('date') if 'date' in prior_stats.columns else prior_stats[prior_stats[team_col] == home_team]
        away_prior = prior_stats[prior_stats[team_col] == away_team].sort_values('date') if 'date' in prior_stats.columns else prior_stats[prior_stats[team_col] == away_team]
        
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
    
    def _extract_metrics(self, team_prior: pd.DataFrame) -> pd.DataFrame:
        """Extract metrics from JSON column if it exists"""
        if 'metrics' in team_prior.columns:
            import json
            metrics_df = team_prior.copy()
            # Parse JSON metrics into separate columns
            for idx, row in metrics_df.iterrows():
                if pd.notna(row['metrics']):
                    try:
                        metrics = json.loads(row['metrics'])
                        for key, value in metrics.items():
                            metrics_df.at[idx, key] = value
                    except:
                        pass
            return metrics_df
        return team_prior
    
    def _nfl_features(self, home_prior: pd.DataFrame, away_prior: pd.DataFrame, home_team: str, away_team: str, game_date) -> dict:
        """
        NFL-specific comprehensive features based on actual available data.
        
        1.1 Deep Lag Features: 3/5/10-game windows + home/away splits
        1.2 Advanced Metrics: Point differentials, scoring efficiency
        1.3 Opponent-Adjusted: Performance vs opponent Elo
        1.4 Contextual: Rest days, home streaks
        1.5 Chronological Filtering: Already enforced by parent method
        """
        features = {}
        
        # Extract metrics from JSON column
        home_data = self._extract_metrics(home_prior)
        away_data = self._extract_metrics(away_prior)
        
        # ========== 1.1 DEEP LAG FEATURES ==========
        # Multiple rolling windows (3, 5, 10-game) for points scored/allowed/differential
        
        if 'points_scored' in home_data.columns and 'points_allowed' in home_data.columns:
            # Ensure point_differential exists
            if 'point_differential' not in home_data.columns:
                home_data['point_differential'] = home_data['points_scored'] - home_data['points_allowed']
            if 'point_differential' not in away_data.columns:
                away_data['point_differential'] = away_data['points_scored'] - away_data['points_allowed']
            
            # Create win column if not exists
            if 'win' in home_data.columns:
                home_data['win_num'] = home_data['win'].astype(float)
            if 'win' in away_data.columns:
                away_data['win_num'] = away_data['win'].astype(float)
            
            # Rolling windows: 3, 5, 10 games
            for window in [3, 5, 10]:
                # Points scored rolling average
                features[f'home_pts_scored_l{window}'] = home_data['points_scored'].tail(window).mean() if len(home_data) >= window else home_data['points_scored'].mean()
                features[f'away_pts_scored_l{window}'] = away_data['points_scored'].tail(window).mean() if len(away_data) >= window else away_data['points_scored'].mean()
                
                # Points allowed rolling average
                features[f'home_pts_allowed_l{window}'] = home_data['points_allowed'].tail(window).mean() if len(home_data) >= window else home_data['points_allowed'].mean()
                features[f'away_pts_allowed_l{window}'] = away_data['points_allowed'].tail(window).mean() if len(away_data) >= window else away_data['points_allowed'].mean()
                
                # Point differential rolling average
                features[f'home_pt_diff_l{window}'] = home_data['point_differential'].tail(window).mean() if len(home_data) >= window else home_data['point_differential'].mean()
                features[f'away_pt_diff_l{window}'] = away_data['point_differential'].tail(window).mean() if len(away_data) >= window else away_data['point_differential'].mean()
                
                # Win percentage rolling
                if 'win_num' in home_data.columns:
                    features[f'home_win_pct_l{window}'] = home_data['win_num'].tail(window).mean() if len(home_data) >= window else home_data['win_num'].mean()
                if 'win_num' in away_data.columns:
                    features[f'away_win_pct_l{window}'] = away_data['win_num'].tail(window).mean() if len(away_data) >= window else away_data['win_num'].mean()
            
            # Home/Away splits (last 5 home games, last 5 away games)
            if 'is_home' in home_data.columns:
                home_home_games = home_data[home_data['is_home'] == 1].tail(5)
                home_away_games = home_data[home_data['is_home'] == 0].tail(5)
                
                if not home_home_games.empty:
                    features['home_pts_scored_l5_home'] = home_home_games['points_scored'].mean()
                    features['home_pt_diff_l5_home'] = home_home_games['point_differential'].mean()
                    if 'win_num' in home_home_games.columns:
                        features['home_win_pct_l5_home'] = home_home_games['win_num'].mean()
                
                if not home_away_games.empty:
                    features['home_pts_scored_l5_away'] = home_away_games['points_scored'].mean()
            
            if 'is_home' in away_data.columns:
                away_home_games = away_data[away_data['is_home'] == 1].tail(5)
                away_away_games = away_data[away_data['is_home'] == 0].tail(5)
                
                if not away_home_games.empty:
                    features['away_pts_scored_l5_home'] = away_home_games['points_scored'].mean()
                
                if not away_away_games.empty:
                    features['away_pts_scored_l5_away'] = away_away_games['points_scored'].mean()
                    features['away_pt_diff_l5_away'] = away_away_games['point_differential'].mean()
                    if 'win_num' in away_away_games.columns:
                        features['away_win_pct_l5_away'] = away_away_games['win_num'].mean()
            
            # Differential features (home advantage over away)
            features['pt_diff_advantage_l5'] = features.get('home_pt_diff_l5', 0) - features.get('away_pt_diff_l5', 0)
            features['win_pct_diff_l5'] = features.get('home_win_pct_l5', 0.5) - features.get('away_win_pct_l5', 0.5)
        
        # ========== 1.3 OPPONENT-ADJUSTED FEATURES ==========
        # Average opponent Elo from recent games
        if 'opponent' in home_data.columns:
            home_recent_opponents = home_data['opponent'].tail(5).tolist()
            home_opp_elo_avg = np.mean([self.elo_system.get_rating(opp) for opp in home_recent_opponents if pd.notna(opp)])
            features['home_avg_opp_elo_l5'] = home_opp_elo_avg if not np.isnan(home_opp_elo_avg) else 1500
        
        if 'opponent' in away_data.columns:
            away_recent_opponents = away_data['opponent'].tail(5).tolist()
            away_opp_elo_avg = np.mean([self.elo_system.get_rating(opp) for opp in away_recent_opponents if pd.notna(opp)])
            features['away_avg_opp_elo_l5'] = away_opp_elo_avg if not np.isnan(away_opp_elo_avg) else 1500
        
        # Strength of schedule differential
        features['sos_diff'] = features.get('home_avg_opp_elo_l5', 1500) - features.get('away_avg_opp_elo_l5', 1500)
        
        # ========== 1.4 CONTEXTUAL FEATURES ==========
        # Rest days since last game
        features['home_rest_days'] = self._compute_rest_days(home_data, game_date)
        features['away_rest_days'] = self._compute_rest_days(away_data, game_date)
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        features['home_short_rest'] = 1 if features['home_rest_days'] <= 4 else 0  # Thursday/Monday games
        features['away_short_rest'] = 1 if features['away_rest_days'] <= 4 else 0
        
        # Home win streak (last 5 home games)
        if 'is_home' in home_data.columns and 'win_num' in home_data.columns:
            home_home_games = home_data[home_data['is_home'] == 1].tail(5)
            if not home_home_games.empty and 'win_num' in home_home_games.columns:
                # Consecutive wins at home
                home_streak = 0
                for win in reversed(home_home_games['win_num'].tolist()):
                    if win == 1:
                        home_streak += 1
                    else:
                        break
                features['home_home_win_streak'] = home_streak
        
        # ========== 1.2 ADVANCED METRICS ==========
        # Scoring efficiency (points per game differential)
        features['scoring_efficiency_diff'] = features.get('home_pts_scored_l5', 20) - features.get('away_pts_scored_l5', 20)
        features['defensive_efficiency_diff'] = features.get('away_pts_allowed_l5', 20) - features.get('home_pts_allowed_l5', 20)
        
        # Matchup features (offense vs defense)
        features['off_def_matchup'] = features.get('home_pts_scored_l5', 20) - features.get('away_pts_allowed_l5', 20) - (features.get('away_pts_scored_l5', 20) - features.get('home_pts_allowed_l5', 20))
        
        return features
    
    def _compute_win_pct(self, team_prior: pd.DataFrame, window: int = 5) -> float:
        """Compute win percentage over last N games"""
        if 'result' not in team_prior.columns or len(team_prior) == 0:
            return 0.5  # Default 50% if no data
        
        recent_games = team_prior.tail(window)
        if len(recent_games) == 0:
            return 0.5
        
        wins = (recent_games['result'] == 'W').sum()
        return wins / len(recent_games)
    
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
            
            # Store training metadata as instance variables
            self.games_trained = len(features_df)
            self.logistic_accuracy = float(logistic_score)
            self.xgboost_accuracy = float(xgb_score)
            
            results = {
                'sport': self.sport,
                'games_trained': self.games_trained,
                'features_used': len(feature_cols),
                'feature_names': feature_cols,
                'logistic_accuracy': self.logistic_accuracy,
                'xgboost_accuracy': self.xgboost_accuracy,
                'teams': len(self.elo_system.ratings)
            }
            
            self.logger.info(f"Training complete: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training {self.sport} models: {str(e)}")
            raise
    
    def _nfl_confidence_weighted_blend(self, elo_prob: float, xgb_prob: float, logistic_prob: float) -> float:
        """
        NFL-specific confidence-based weighting system.
        
        Logic:
        1. High Elo Favor (Elo ≥75% AND XGB ≤55%): Default to Elo pick
           - If Elo is extremely confident, don't let a weak XGB flip it
        2. Moderate Elo/Coin Flip (Elo 55-75%, XGB 45-55%): Allow XGB contrarian pick
           - Sweet spot for finding value/upsets
        3. Otherwise: Use weighted ensemble
        
        Args:
            elo_prob: Elo home team win probability
            xgb_prob: XGBoost home team win probability
            logistic_prob: Logistic home team win probability
            
        Returns:
            Confidence-weighted blended probability
        """
        # Convert to percentages for easier logic
        elo_pct = elo_prob * 100
        xgb_pct = xgb_prob * 100
        
        # Rule 1: High Elo Favor - Don't override strong Elo with weak XGB
        # If Elo is very confident (≥75%) and XGB is weak/neutral (≤55%), trust Elo
        if elo_pct >= 75 and xgb_pct <= 55:
            # Strong home favor from Elo, weak opposition from XGB -> trust Elo heavily
            return elo_prob * 0.85 + xgb_prob * 0.10 + logistic_prob * 0.05
        
        elif elo_pct <= 25 and xgb_pct >= 45:
            # Strong away favor from Elo (inverse), weak opposition from XGB -> trust Elo heavily
            return elo_prob * 0.85 + xgb_prob * 0.10 + logistic_prob * 0.05
        
        # Rule 2: Moderate/Coin Flip Range - Sweet spot for upsets
        # Elo moderately confident (55-75%), XGB neutral (45-55%) -> let XGB find value
        elif (55 <= elo_pct <= 75 or 25 <= elo_pct <= 45) and (45 <= xgb_pct <= 55):
            # This is where XGB can identify hidden patterns Elo misses
            return elo_prob * 0.50 + xgb_prob * 0.40 + logistic_prob * 0.10
        
        # Default: Use standard NFL ensemble weights (60/30/10)
        else:
            return elo_prob * 0.60 + xgb_prob * 0.30 + logistic_prob * 0.10
    
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
        
        # Blended prediction with confidence-based weighting for NFL
        if self.sport == 'NFL':
            blended_prob = self._nfl_confidence_weighted_blend(elo_prob, xgb_prob, logistic_prob)
        else:
            # Standard weighted average for other sports
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
            'is_trained': self.is_trained,
            'feature_cols': getattr(self, 'feature_cols', None),
            'games_trained': getattr(self, 'games_trained', None),
            'logistic_accuracy': getattr(self, 'logistic_accuracy', None),
            'xgboost_accuracy': getattr(self, 'xgboost_accuracy', None)
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
        self.is_trained = model_data.get('is_trained', True)
        
        # Load training metadata
        self.feature_cols = model_data.get('feature_cols', None)
        self.games_trained = model_data.get('games_trained', None)
        self.logistic_accuracy = model_data.get('logistic_accuracy', None)
        self.xgboost_accuracy = model_data.get('xgboost_accuracy', None)
        
        # Update old ensemble weights if using glmnet key
        if 'glmnet' in self.ensemble_weights:
            self.ensemble_weights['logistic'] = self.ensemble_weights.pop('glmnet')
            
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")
