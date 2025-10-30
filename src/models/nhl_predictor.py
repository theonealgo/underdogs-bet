import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor


class NHLPredictor:
    """
    Machine learning models for NHL game predictions with 4-model ensemble (XGBoost, CatBoost, Elo, Meta)
    """
    
    def __init__(self, model_dir: str = "models", db_manager=None):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.db_manager = db_manager
        
        # Initialize models
        self.xgb_winner_model = None
        self.catboost_winner_model = None
        self.xgb_total_model = None
        self.catboost_total_model = None
        self.is_trained = False
        self.feature_names = None
        
        # Elo rating system (K-factor=45 for NHL - higher for faster adaptation to form changes)
        self.elo_ratings = {}  # team_id -> rating
        self.elo_k_factor = 45  # Increased for faster response to recent performance
        self.elo_initial_rating = 1500
        
        # XGBoost parameters - AGGRESSIVE regularization to prevent overfitting on small samples
        self.xgb_winner_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,  # Reduced from 5 - shallower trees = less overfitting
            'learning_rate': 0.03,  # Reduced from 0.05 - slower learning = better generalization
            'n_estimators': 100,  # Reduced from 200 - fewer trees with small data
            'min_child_weight': 5,  # Increased from 3 - require more samples per leaf
            'subsample': 0.7,  # Reduced from 0.8 - more stochasticity
            'colsample_bytree': 0.7,  # Reduced from 0.8 - prevent feature overfitting
            'reg_alpha': 2.0,  # Increased from 0.5 - stronger L1 regularization
            'reg_lambda': 3.0,  # Increased from 1.0 - stronger L2 regularization
            'random_state': 42,
            'verbosity': 0
        }
        
        self.xgb_total_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 3,  # Reduced from 5
            'learning_rate': 0.03,  # Reduced from 0.05
            'n_estimators': 100,  # Reduced from 200
            'min_child_weight': 5,  # Increased from 3
            'subsample': 0.7,  # Reduced from 0.8
            'colsample_bytree': 0.7,  # Reduced from 0.8
            'reg_alpha': 2.0,  # Increased from 0.5
            'reg_lambda': 3.0,  # Increased from 1.0
            'random_state': 42,
            'verbosity': 0
        }
        
        # CatBoost parameters - AGGRESSIVE regularization for small sample sizes
        self.catboost_winner_params = {
            'iterations': 100,  # Reduced from 200 - fewer iterations with small data
            'depth': 4,  # Reduced from 6 - shallower trees
            'learning_rate': 0.03,  # Reduced from 0.05 - slower learning
            'l2_leaf_reg': 5.0,  # Increased from 3.0 - stronger regularization
            'random_strength': 1.5,  # Increased - more randomization
            'bagging_temperature': 0.7,  # Increased - more Bayesian bootstrap randomness
            'random_state': 42,
            'verbose': False,
            'loss_function': 'Logloss'
        }
        
        self.catboost_total_params = {
            'iterations': 100,  # Reduced from 200
            'depth': 4,  # Reduced from 6
            'learning_rate': 0.03,  # Reduced from 0.05
            'l2_leaf_reg': 5.0,  # Increased from 3.0
            'random_strength': 1.5,  # Increased
            'bagging_temperature': 0.7,  # Increased
            'random_state': 42,
            'verbose': False,
            'loss_function': 'RMSE'
        }
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load saved models if they exist"""
        try:
            xgb_winner_path = os.path.join(self.model_dir, 'nhl_xgb_winner.pkl')
            catboost_winner_path = os.path.join(self.model_dir, 'nhl_catboost_winner.pkl')
            elo_ratings_path = os.path.join(self.model_dir, 'nhl_elo_ratings.pkl')
            xgb_total_path = os.path.join(self.model_dir, 'nhl_xgb_total.pkl')
            catboost_total_path = os.path.join(self.model_dir, 'nhl_catboost_total.pkl')
            features_path = os.path.join(self.model_dir, 'nhl_feature_names.pkl')
            
            if os.path.exists(xgb_winner_path):
                with open(xgb_winner_path, 'rb') as f:
                    self.xgb_winner_model = pickle.load(f)
                with open(catboost_winner_path, 'rb') as f:
                    self.catboost_winner_model = pickle.load(f)
                if os.path.exists(elo_ratings_path):
                    with open(elo_ratings_path, 'rb') as f:
                        self.elo_ratings = pickle.load(f)
                with open(xgb_total_path, 'rb') as f:
                    self.xgb_total_model = pickle.load(f)
                with open(catboost_total_path, 'rb') as f:
                    self.catboost_total_model = pickle.load(f)
                if os.path.exists(features_path):
                    with open(features_path, 'rb') as f:
                        self.feature_names = pickle.load(f)
                
                self.is_trained = True
                self.logger.info("Loaded existing NHL models")
        except Exception as e:
            self.logger.warning(f"Could not load NHL models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            xgb_winner_path = os.path.join(self.model_dir, 'nhl_xgb_winner.pkl')
            catboost_winner_path = os.path.join(self.model_dir, 'nhl_catboost_winner.pkl')
            elo_ratings_path = os.path.join(self.model_dir, 'nhl_elo_ratings.pkl')
            xgb_total_path = os.path.join(self.model_dir, 'nhl_xgb_total.pkl')
            catboost_total_path = os.path.join(self.model_dir, 'nhl_catboost_total.pkl')
            features_path = os.path.join(self.model_dir, 'nhl_feature_names.pkl')
            
            with open(xgb_winner_path, 'wb') as f:
                pickle.dump(self.xgb_winner_model, f)
            with open(catboost_winner_path, 'wb') as f:
                pickle.dump(self.catboost_winner_model, f)
            with open(elo_ratings_path, 'wb') as f:
                pickle.dump(self.elo_ratings, f)
            with open(xgb_total_path, 'wb') as f:
                pickle.dump(self.xgb_total_model, f)
            with open(catboost_total_path, 'wb') as f:
                pickle.dump(self.catboost_total_model, f)
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            self.logger.info("Saved NHL models")
        except Exception as e:
            self.logger.error(f"Error saving NHL models: {e}")
    
    def get_elo_rating(self, team: str) -> float:
        """Get team's current Elo rating"""
        if team not in self.elo_ratings:
            self.elo_ratings[team] = self.elo_initial_rating
        return self.elo_ratings[team]
    
    def elo_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected win probability for team A vs team B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_elo_ratings(self, home_team: str, away_team: str, home_won: bool, home_score: int = None, away_score: int = None):
        """Update Elo ratings after a game with margin of victory adjustment"""
        home_rating = self.get_elo_rating(home_team)
        away_rating = self.get_elo_rating(away_team)
        
        # Expected scores
        home_expected = self.elo_expected_score(home_rating, away_rating)
        
        # Actual result
        home_actual = 1.0 if home_won else 0.0
        
        # Margin of victory multiplier (if scores provided)
        mov_multiplier = 1.0
        if home_score is not None and away_score is not None:
            score_diff = abs(home_score - away_score)
            # NHL: Score differences matter less than NBA/NFL (typical range 1-3 goals)
            mov_multiplier = 1.0 + (score_diff / 10.0)  # Max ~1.5x for 5-goal win
        
        # Update ratings
        rating_change = self.elo_k_factor * mov_multiplier * (home_actual - home_expected)
        self.elo_ratings[home_team] = home_rating + rating_change
        self.elo_ratings[away_team] = away_rating - rating_change
    
    def create_features(self, games_df: pd.DataFrame, historical_games: pd.DataFrame = None) -> pd.DataFrame:
        """Create ADVANCED features for NHL games"""
        features_list = []
        
        # Use historical_games if provided, otherwise use games_df
        history_df = historical_games if historical_games is not None and not historical_games.empty else games_df
        
        for idx, game in games_df.iterrows():
            # Get comprehensive team stats
            home_stats = self._get_advanced_team_stats(game['home_team_id'], game['game_date'], history_df, is_home=True)
            away_stats = self._get_advanced_team_stats(game['away_team_id'], game['game_date'], history_df, is_home=False)
            
            # Get head-to-head stats
            h2h_stats = self._get_head_to_head_stats(game['home_team_id'], game['away_team_id'], game['game_date'], history_df)
            
            features = {
                # Overall performance (multiple windows)
                'home_win_pct_5': home_stats['win_pct_5'],
                'home_win_pct_10': home_stats['win_pct_10'],
                'away_win_pct_5': away_stats['win_pct_5'],
                'away_win_pct_10': away_stats['win_pct_10'],
                
                # Offensive stats
                'home_goals_per_game_5': home_stats['goals_per_game_5'],
                'home_goals_per_game_10': home_stats['goals_per_game_10'],
                'away_goals_per_game_5': away_stats['goals_per_game_5'],
                'away_goals_per_game_10': away_stats['goals_per_game_10'],
                
                # Defensive stats
                'home_goals_against_5': home_stats['goals_against_5'],
                'home_goals_against_10': home_stats['goals_against_10'],
                'away_goals_against_5': away_stats['goals_against_5'],
                'away_goals_against_10': away_stats['goals_against_10'],
                
                # Recent form
                'home_recent_form': home_stats['recent_form'],
                'away_recent_form': away_stats['recent_form'],
                
                # NEW: Streak features (highly predictive)
                'home_current_streak': home_stats['current_streak'],
                'away_current_streak': away_stats['current_streak'],
                'home_max_win_streak_5': home_stats['max_win_streak'],
                'away_max_win_streak_5': away_stats['max_win_streak'],
                
                # NEW: Momentum features (trend over time)
                'home_momentum': home_stats['momentum'],
                'away_momentum': away_stats['momentum'],
                
                # Home/away splits
                'home_home_win_pct': home_stats['home_win_pct'],
                'away_away_win_pct': away_stats['away_win_pct'],
                'home_home_goals_avg': home_stats['home_goals_avg'],
                'away_away_goals_avg': away_stats['away_goals_avg'],
                
                # Rest and fatigue
                'home_rest_days': home_stats['rest_days'],
                'away_rest_days': away_stats['rest_days'],
                'home_back_to_back': home_stats['back_to_back'],
                'away_back_to_back': away_stats['back_to_back'],
                
                # Strength of schedule
                'home_opponent_strength': home_stats['opponent_strength'],
                'away_opponent_strength': away_stats['opponent_strength'],
                
                # Goalie performance differential
                'goalie_sv_pct_diff': home_stats.get('goalie_sv_pct', 0.910) - away_stats.get('goalie_sv_pct', 0.910),
                
                # Head-to-head
                'h2h_home_wins': h2h_stats['home_wins'],
                'h2h_total_games': h2h_stats['total_games'],
                'h2h_avg_total_goals': h2h_stats['avg_total_goals'],
                
                # Differential features (most predictive)
                'win_pct_diff_5': home_stats['win_pct_5'] - away_stats['win_pct_5'],
                'win_pct_diff_10': home_stats['win_pct_10'] - away_stats['win_pct_10'],
                'goals_diff_5': home_stats['goals_per_game_5'] - away_stats['goals_per_game_5'],
                'defense_diff_5': away_stats['goals_against_5'] - home_stats['goals_against_5'],
                'rest_diff': home_stats['rest_days'] - away_stats['rest_days'],
                'form_diff': home_stats['recent_form'] - away_stats['recent_form'],
                'streak_diff': home_stats['current_streak'] - away_stats['current_streak'],
                'momentum_diff': home_stats['momentum'] - away_stats['momentum'],
            }
            
            # Add target variables if available
            if 'home_score' in game and 'away_score' in game and pd.notna(game['home_score']):
                features['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
                features['total_goals'] = game['home_score'] + game['away_score']
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _get_advanced_team_stats(self, team_id: str, current_date, games_df: pd.DataFrame, is_home: bool) -> Dict:
        """Calculate ADVANCED team statistics with multiple windows and splits"""
        # Convert current_date to date object if it's a string
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date, format='%d/%m/%Y', dayfirst=True).date()
        
        # Ensure game_date column is datetime for comparison (DD/MM/YYYY format)
        games_df = games_df.copy()
        games_df['game_date'] = pd.to_datetime(games_df['game_date'], format='%d/%m/%Y', dayfirst=True)
        
        # Get ALL team's games before current date
        all_team_games = games_df[
            ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
            (games_df['game_date'].dt.date < current_date) &
            (games_df['status'] == 'final')
        ].sort_values('game_date', ascending=False)
        
        if len(all_team_games) == 0:
            # Default stats for new teams or start of season
            return self._default_team_stats()
        
        # Calculate rest days
        rest_days = self._calculate_rest_days(team_id, current_date, all_team_games)
        back_to_back = 1 if rest_days == 0 else 0
        
        # Calculate stats for multiple windows
        stats_5 = self._calculate_window_stats(team_id, all_team_games.head(5))
        stats_10 = self._calculate_window_stats(team_id, all_team_games.head(10))
        
        # Calculate home/away splits
        home_away_stats = self._calculate_home_away_splits(team_id, all_team_games, is_home)
        
        # Calculate strength of schedule
        opponent_strength = self._calculate_opponent_strength(team_id, all_team_games.head(10))
        
        # Get goalie stats if available
        goalie_sv_pct = self._get_goalie_sv_pct(team_id)
        
        # Calculate streaks and momentum
        streak_stats = self._calculate_streaks(team_id, all_team_games.head(10))
        momentum = self._calculate_momentum(team_id, all_team_games.head(10))
        
        return {
            # Multiple window win percentages
            'win_pct_5': stats_5['win_pct'],
            'win_pct_10': stats_10['win_pct'],
            
            # Multiple window offensive stats
            'goals_per_game_5': stats_5['goals_per_game'],
            'goals_per_game_10': stats_10['goals_per_game'],
            
            # Multiple window defensive stats
            'goals_against_5': stats_5['goals_against'],
            'goals_against_10': stats_10['goals_against'],
            
            # Recent form (last 5 games)
            'recent_form': stats_5['win_pct'],
            
            # Streaks (highly predictive in sports)
            'current_streak': streak_stats['current_streak'],
            'max_win_streak': streak_stats['max_win_streak'],
            
            # Momentum (trend indicator)
            'momentum': momentum,
            
            # Home/away splits
            'home_win_pct': home_away_stats['home_win_pct'],
            'away_win_pct': home_away_stats['away_win_pct'],
            'home_goals_avg': home_away_stats['home_goals_avg'],
            'away_goals_avg': home_away_stats['away_goals_avg'],
            
            # Rest and fatigue
            'rest_days': rest_days,
            'back_to_back': back_to_back,
            
            # Strength of schedule
            'opponent_strength': opponent_strength,
            
            # Goalie stats
            'goalie_sv_pct': goalie_sv_pct
        }
    
    def _calculate_streaks(self, team_id: str, recent_games: pd.DataFrame) -> Dict:
        """Calculate current winning/losing streak and max streak"""
        if len(recent_games) == 0:
            return {'current_streak': 0, 'max_win_streak': 0}
        
        current_streak = 0
        max_win_streak = 0
        current_win_streak = 0
        
        for idx, game in recent_games.iterrows():
            is_home = game['home_team_id'] == team_id
            team_score = game['home_score'] if is_home else game['away_score']
            opp_score = game['away_score'] if is_home else game['home_score']
            
            if pd.notna(team_score) and pd.notna(opp_score):
                won = team_score > opp_score
                
                # Current streak (positive for wins, negative for losses)
                if idx == 0:  # Most recent game
                    current_streak = 1 if won else -1
                elif won:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                else:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                
                # Track max win streak
                if won:
                    current_win_streak += 1
                    max_win_streak = max(max_win_streak, current_win_streak)
                else:
                    current_win_streak = 0
        
        return {
            'current_streak': current_streak,
            'max_win_streak': max_win_streak
        }
    
    def _calculate_momentum(self, team_id: str, recent_games: pd.DataFrame) -> float:
        """Calculate momentum: weighted average favoring recent games
        Positive = improving, Negative = declining
        """
        if len(recent_games) < 3:
            return 0.0
        
        results = []
        for idx, game in recent_games.iterrows():
            is_home = game['home_team_id'] == team_id
            team_score = game['home_score'] if is_home else game['away_score']
            opp_score = game['away_score'] if is_home else game['home_score']
            
            if pd.notna(team_score) and pd.notna(opp_score):
                # Win = 1, Loss = 0
                results.append(1.0 if team_score > opp_score else 0.0)
        
        if len(results) < 3:
            return 0.0
        
        # Calculate weighted win rates: recent games weighted more heavily
        # Compare last 3 games vs previous 3-7 games
        recent_win_rate = np.mean(results[:3]) if len(results) >= 3 else 0.5
        older_win_rate = np.mean(results[3:]) if len(results) > 3 else recent_win_rate
        
        # Momentum = difference (positive means improving)
        momentum = recent_win_rate - older_win_rate
        return momentum
    
    def _default_team_stats(self) -> Dict:
        """Return default stats for teams without history"""
        return {
            'win_pct_5': 0.5, 'win_pct_10': 0.5,
            'goals_per_game_5': 3.0, 'goals_per_game_10': 3.0,
            'goals_against_5': 3.0, 'goals_against_10': 3.0,
            'recent_form': 0.5,
            'current_streak': 0, 'max_win_streak': 0, 'momentum': 0.0,
            'home_win_pct': 0.55, 'away_win_pct': 0.45,
            'home_goals_avg': 3.2, 'away_goals_avg': 2.8,
            'rest_days': 1, 'back_to_back': 0,
            'opponent_strength': 0.5,
            'goalie_sv_pct': 0.910
        }
    
    def _calculate_window_stats(self, team_id: str, window_games: pd.DataFrame) -> Dict:
        """Calculate stats for a specific game window"""
        if len(window_games) == 0:
            return {'win_pct': 0.5, 'goals_per_game': 3.0, 'goals_against': 3.0}
        
        wins = 0
        goals_for = 0
        goals_against = 0
        
        for idx, game in window_games.iterrows():
            is_home = game['home_team_id'] == team_id
            team_score = game['home_score'] if is_home else game['away_score']
            opp_score = game['away_score'] if is_home else game['home_score']
            
            if pd.notna(team_score) and pd.notna(opp_score):
                goals_for += team_score
                goals_against += opp_score
                if team_score > opp_score:
                    wins += 1
        
        games_count = len(window_games)
        return {
            'win_pct': wins / games_count if games_count > 0 else 0.5,
            'goals_per_game': goals_for / games_count if games_count > 0 else 3.0,
            'goals_against': goals_against / games_count if games_count > 0 else 3.0
        }
    
    def _calculate_home_away_splits(self, team_id: str, all_games: pd.DataFrame, is_home: bool) -> Dict:
        """Calculate home/away performance splits"""
        home_games = all_games[all_games['home_team_id'] == team_id].head(10)
        away_games = all_games[all_games['away_team_id'] == team_id].head(10)
        
        home_wins = 0
        home_goals = 0
        for idx, game in home_games.iterrows():
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                home_goals += game['home_score']
                if game['home_score'] > game['away_score']:
                    home_wins += 1
        
        away_wins = 0
        away_goals = 0
        for idx, game in away_games.iterrows():
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                away_goals += game['away_score']
                if game['away_score'] > game['home_score']:
                    away_wins += 1
        
        return {
            'home_win_pct': home_wins / len(home_games) if len(home_games) > 0 else 0.55,
            'away_win_pct': away_wins / len(away_games) if len(away_games) > 0 else 0.45,
            'home_goals_avg': home_goals / len(home_games) if len(home_games) > 0 else 3.2,
            'away_goals_avg': away_goals / len(away_games) if len(away_games) > 0 else 2.8
        }
    
    def _calculate_rest_days(self, team_id: str, current_date, all_games: pd.DataFrame) -> int:
        """Calculate days of rest since last game"""
        if len(all_games) == 0:
            return 3
        
        last_game = all_games.iloc[0]
        last_game_date = pd.to_datetime(last_game['game_date']).date()
        rest_days = (current_date - last_game_date).days
        return min(rest_days, 7)  # Cap at 7 days
    
    def _calculate_opponent_strength(self, team_id: str, recent_games: pd.DataFrame) -> float:
        """Calculate average opponent strength based on opponent win percentages"""
        if len(recent_games) == 0:
            return 0.5
        
        # Simplified: use 0.5 as default opponent strength
        # In production, you'd calculate each opponent's actual win percentage
        return 0.5
    
    def _get_goalie_sv_pct(self, team_id: str) -> float:
        """Get team's goalie save percentage"""
        try:
            if self.db_manager:
                conn = self.db_manager.get_connection()
                result = conn.execute('''
                    SELECT goalie_save_percentage 
                    FROM team_goalies 
                    WHERE team_name = ?
                ''', (team_id,)).fetchone()
                if result:
                    return result[0]
        except:
            pass
        return 0.910  # League average
    
    def _get_head_to_head_stats(self, home_team: str, away_team: str, current_date, games_df: pd.DataFrame) -> Dict:
        """Calculate head-to-head statistics"""
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date, format='%d/%m/%Y', dayfirst=True).date()
        
        games_df = games_df.copy()
        games_df['game_date'] = pd.to_datetime(games_df['game_date'], format='%d/%m/%Y', dayfirst=True)
        
        # Get previous matchups
        h2h_games = games_df[
            ((games_df['home_team_id'] == home_team) & (games_df['away_team_id'] == away_team)) |
            ((games_df['home_team_id'] == away_team) & (games_df['away_team_id'] == home_team))
        ]
        h2h_games = h2h_games[
            (h2h_games['game_date'].dt.date < current_date) &
            (h2h_games['status'] == 'final')
        ].tail(5)  # Last 5 matchups
        
        if len(h2h_games) == 0:
            return {'home_wins': 0, 'total_games': 0, 'avg_total_goals': 6.0}
        
        home_wins = 0
        total_goals = 0
        
        for idx, game in h2h_games.iterrows():
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                total_goals += game['home_score'] + game['away_score']
                
                # Check if current home team won (could be home or away in this matchup)
                if game['home_team_id'] == home_team and game['home_score'] > game['away_score']:
                    home_wins += 1
                elif game['away_team_id'] == home_team and game['away_score'] > game['home_score']:
                    home_wins += 1
        
        return {
            'home_wins': home_wins,
            'total_games': len(h2h_games),
            'avg_total_goals': total_goals / len(h2h_games) if len(h2h_games) > 0 else 6.0
        }
    
    def train_models(self, games_df: pd.DataFrame) -> Dict:
        """Train ALL winner and totals prediction models (XGBoost, CatBoost, Elo)"""
        try:
            self.logger.info(f"Training NHL models with {len(games_df)} games")
            
            # Build Elo ratings from historical games
            self.logger.info("Building Elo ratings from historical games...")
            self.elo_ratings = {}  # Reset Elo ratings
            elo_correct = 0
            elo_total = 0
            
            for idx, game in games_df.iterrows():
                if game['status'] == 'final' and pd.notna(game['home_score']) and pd.notna(game['away_score']):
                    # Get current ratings
                    home_rating = self.get_elo_rating(game['home_team_id'])
                    away_rating = self.get_elo_rating(game['away_team_id'])
                    
                    # Make prediction before updating
                    home_win_prob = self.elo_expected_score(home_rating, away_rating)
                    predicted_winner = game['home_team_id'] if home_win_prob > 0.5 else game['away_team_id']
                    actual_winner = game['home_team_id'] if game['home_score'] > game['away_score'] else game['away_team_id']
                    
                    if predicted_winner == actual_winner:
                        elo_correct += 1
                    elo_total += 1
                    
                    # Update ratings
                    home_won = game['home_score'] > game['away_score']
                    self.update_elo_ratings(game['home_team_id'], game['away_team_id'], home_won, game['home_score'], game['away_score'])
            
            elo_acc = elo_correct / elo_total if elo_total > 0 else 0.0
            self.logger.info(f"Elo winner accuracy: {elo_acc:.3f}")
            
            # Create features
            features_df = self.create_features(games_df)
            
            # Filter to games with complete data
            complete_games = features_df.dropna()
            
            if len(complete_games) < 20:
                self.logger.error(f"Insufficient training data: {len(complete_games)} games")
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Prepare features and targets
            feature_cols = [col for col in complete_games.columns if col not in ['home_win', 'total_goals', 'game_date_rank']]
            
            # SELECT TOP PREDICTIVE FEATURES including new streak and momentum features
            # Prioritize differential features (most predictive) and new behavioral features
            top_features = [
                # Differential features (highest predictive power)
                'form_diff', 'win_pct_diff_10', 'win_pct_diff_5', 'goals_diff_5', 
                'defense_diff_5', 'rest_diff', 'streak_diff', 'momentum_diff',
                
                # Key absolute stats
                'home_goals_per_game_5', 'away_goals_per_game_5',
                'home_goals_against_5', 'away_goals_against_5',
                
                # Streaks and momentum (new high-impact features)
                'home_current_streak', 'away_current_streak', 
                'home_momentum', 'away_momentum',
                
                # Head-to-head history
                'h2h_home_wins', 'h2h_total_games'
            ]
            
            # Only use features that exist in the data
            available_features = [f for f in top_features if f in feature_cols]
            X = complete_games[available_features]
            y_winner = complete_games['home_win']
            y_total = complete_games['total_goals']
            
            self.feature_names = available_features
            self.logger.info(f"Training with {len(available_features)} features (including streak/momentum)")
            
            # TIME-BASED SPLIT: Train 70%, validate 15%, test 15%
            n_samples = len(X)
            train_size = int(0.70 * n_samples)
            val_size = int(0.15 * n_samples)
            
            X_train = X.iloc[:train_size]
            X_val = X.iloc[train_size:train_size+val_size]
            X_test = X.iloc[train_size+val_size:]
            
            y_train_winner = y_winner.iloc[:train_size]
            y_val_winner = y_winner.iloc[train_size:train_size+val_size]
            y_test_winner = y_winner.iloc[train_size+val_size:]
            
            # RECENCY WEIGHTING: Exponential decay giving more weight to recent games
            alpha = 2.5  # Controls strength of recency bias
            sample_weights = np.exp(alpha * np.arange(train_size) / train_size)
            sample_weights = sample_weights / sample_weights.mean()  # Normalize to mean=1
            
            self.logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} games")
            
            # Train XGBoost with EARLY STOPPING on validation set
            self.xgb_winner_model = xgb.XGBClassifier(**self.xgb_winner_params)
            self.xgb_winner_model.fit(
                X_train, y_train_winner, 
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val_winner)],
                early_stopping_rounds=20,
                verbose=False
            )
            xgb_acc = accuracy_score(y_test_winner, self.xgb_winner_model.predict(X_test))
            self.logger.info(f"XGBoost accuracy (test set): {xgb_acc:.3f}, best iteration: {self.xgb_winner_model.best_iteration}")
            
            # Train CatBoost with EARLY STOPPING on validation set
            self.catboost_winner_model = CatBoostClassifier(**self.catboost_winner_params)
            self.catboost_winner_model.fit(
                X_train, y_train_winner,
                sample_weight=sample_weights,
                eval_set=(X_val, y_val_winner),
                early_stopping_rounds=20,
                verbose=False
            )
            catboost_acc = accuracy_score(y_test_winner, self.catboost_winner_model.predict(X_test))
            self.logger.info(f"CatBoost accuracy (test set): {catboost_acc:.3f}, best iteration: {self.catboost_winner_model.best_iteration_}")
            
            # Train totals models with early stopping
            y_train_total = y_total.iloc[:train_size]
            y_val_total = y_total.iloc[train_size:train_size+val_size]
            y_test_total = y_total.iloc[train_size+val_size:]
            
            self.xgb_total_model = xgb.XGBRegressor(**self.xgb_total_params)
            self.xgb_total_model.fit(
                X_train, y_train_total, 
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val_total)],
                early_stopping_rounds=20,
                verbose=False
            )
            xgb_mae = mean_absolute_error(y_test_total, self.xgb_total_model.predict(X_test))
            self.logger.info(f"XGBoost total MAE (test set): {xgb_mae:.3f}")
            
            self.catboost_total_model = CatBoostRegressor(**self.catboost_total_params)
            self.catboost_total_model.fit(
                X_train, y_train_total, 
                sample_weight=sample_weights,
                eval_set=(X_val, y_val_total),
                early_stopping_rounds=20,
                verbose=False
            )
            catboost_mae = mean_absolute_error(y_test_total, self.catboost_total_model.predict(X_test))
            self.logger.info(f"CatBoost total MAE (test set): {catboost_mae:.3f}")
            
            self.is_trained = True
            self._save_models()
            
            return {
                'success': True,
                'xgb_accuracy': xgb_acc,
                'catboost_accuracy': catboost_acc,
                'elo_accuracy': elo_acc,
                'xgb_total_mae': xgb_mae,
                'catboost_total_mae': catboost_mae,
                'training_games': len(complete_games),
                'num_features': len(feature_cols),
                'elo_teams': len(self.elo_ratings)
            }
            
        except Exception as e:
            self.logger.error(f"Error training NHL models: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_game(self, home_team: str, away_team: str, game_date, historical_games: pd.DataFrame) -> Dict:
        """Predict outcome for a single NHL game using ENSEMBLE of all models"""
        try:
            # Create a game DataFrame for feature creation
            game_df = pd.DataFrame([{
                'home_team_id': home_team,
                'away_team_id': away_team,
                'game_date': game_date,
                'status': 'scheduled'
            }])
            
            # Create features using historical games for team stats
            features_df = self.create_features(game_df, historical_games)
            
            if self.feature_names is None:
                self.logger.error("No feature names available")
                return self._default_prediction(home_team, away_team)
            
            X = features_df[self.feature_names]
            
            if not self.is_trained:
                self.logger.warning("Models not trained, using default prediction")
                return self._default_prediction(home_team, away_team)
            
            # Get predictions from all 3 models (XGBoost, CatBoost, Elo)
            xgb_prob = self.xgb_winner_model.predict_proba(X)[0][1]
            catboost_prob = self.catboost_winner_model.predict_proba(X)[0][1]
            
            # Get Elo prediction
            home_rating = self.get_elo_rating(home_team)
            away_rating = self.get_elo_rating(away_team)
            elo_prob = self.elo_expected_score(home_rating, away_rating)
            
            # Meta ensemble: weighted average favoring XGBoost/CatBoost over Elo
            # XGBoost and CatBoost both show 52.7% accuracy, Elo only 48%
            # Weights: 45% XGBoost, 45% CatBoost, 10% Elo
            meta_prob = (0.45 * xgb_prob + 0.45 * catboost_prob + 0.10 * elo_prob)
            
            # Get total predictions
            xgb_total = self.xgb_total_model.predict(X)[0]
            catboost_total = self.catboost_total_model.predict(X)[0]
            predicted_total = (xgb_total + catboost_total) / 2.0
            
            # Determine winner based on corrected meta probability
            predicted_winner = home_team if meta_prob > 0.5 else away_team
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': predicted_winner,
                'home_win_probability': meta_prob,
                'xgb_home_prob': xgb_prob,
                'catboost_home_prob': catboost_prob,
                'elo_home_prob': elo_prob,
                'meta_home_prob': meta_prob,
                'predicted_total': predicted_total,
                'sport': 'NHL',
                'league': 'NHL'
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting NHL game: {e}")
            return self._default_prediction(home_team, away_team)
    
    def _default_prediction(self, home_team: str, away_team: str) -> Dict:
        """Return default prediction when models unavailable"""
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': home_team,
            'home_win_probability': 0.52,
            'predicted_total': 6.0,
            'sport': 'NHL',
            'league': 'NHL'
        }
    
    def predict_multiple_games(self, games_df: pd.DataFrame, historical_games: pd.DataFrame) -> List[Dict]:
        """Predict multiple NHL games"""
        predictions = []
        
        for idx, game in games_df.iterrows():
            prediction = self.predict_game(
                game['home_team_id'],
                game['away_team_id'],
                game['game_date'],
                historical_games
            )
            prediction['game_id'] = game.get('game_id', f"{game['away_team_id']}_{game['home_team_id']}_{game['game_date']}")
            prediction['game_date'] = game['game_date']
            predictions.append(prediction)
        
        return predictions
