"""
NFL Ensemble Prediction System
Combines Elo Ratings, GLMNet, and XGBoost models for game predictions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import Dict, Tuple, List
import logging

class EloRatingSystem:
    """Elo rating system for team strength calculation"""
    
    def __init__(self, k_factor: float = 20, initial_rating: float = 1500):
        """
        Initialize Elo rating system
        
        Args:
            k_factor: How much ratings change after each game (higher = more volatile)
            initial_rating: Starting rating for all teams
        """
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
        """
        Update Elo ratings after a game
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_won: True if home team won
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Expected scores
        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = 1 - home_expected
        
        # Actual scores (1 for win, 0 for loss)
        home_actual = 1.0 if home_won else 0.0
        away_actual = 1.0 - home_actual
        
        # Update ratings
        self.ratings[home_team] = home_rating + self.k_factor * (home_actual - home_expected)
        self.ratings[away_team] = away_rating + self.k_factor * (away_actual - away_expected)
    
    def predict_game(self, home_team: str, away_team: str) -> float:
        """
        Predict home team win probability
        
        Returns:
            Probability of home team winning (0-1)
        """
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        return self.expected_score(home_rating, away_rating)


class NFLEnsemblePredictor:
    """
    NFL game predictor using ensemble of Elo, GLMNet, and XGBoost
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.elo_system = EloRatingSystem(k_factor=20)
        
        # GLMNet (Logistic Regression with L1/L2 regularization)
        self.glmnet_model = LogisticRegressionCV(
            cv=5,
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[0.1, 0.5, 0.9],
            max_iter=1000,
            random_state=42
        )
        
        # XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Ensemble weights (can be optimized)
        self.ensemble_weights = {
            'elo': 0.3,
            'glmnet': 0.35,
            'xgboost': 0.35
        }
    
    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create features for prediction
        
        Features include:
        - Elo ratings for both teams
        - Home field advantage
        - Team performance metrics (if available)
        - Head-to-head history
        
        Args:
            df: DataFrame with game data
            is_training: Whether this is for training (updates Elo) or prediction
            
        Returns:
            DataFrame with features
        """
        features_list = []
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get current Elo ratings
            home_elo = self.elo_system.get_rating(home_team)
            away_elo = self.elo_system.get_rating(away_team)
            elo_diff = home_elo - away_elo
            
            # Create feature dict
            features = {
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'home_advantage': 1,  # Home field advantage indicator
            }
            
            # If training, update Elo ratings based on result
            if is_training and 'result' in df.columns:
                result = row['result']
                home_won = result == 'H' or result == 'Home' or result == 1
                self.elo_system.update_ratings(home_team, away_team, home_won)
                
                # Add target variable
                features['target'] = 1 if home_won else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train(self, historical_df: pd.DataFrame) -> Dict:
        """
        Train all models on historical data
        
        Args:
            historical_df: DataFrame with columns: home_team, away_team, result
                          result should be 'H'/'A' or 1/0 (1 = home win)
        
        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info(f"Training NFL ensemble on {len(historical_df)} games")
            
            # Create features with Elo updates
            features_df = self.create_features(historical_df, is_training=True)
            
            # Prepare training data
            X = features_df[['home_elo', 'away_elo', 'elo_diff', 'home_advantage']].values
            y = features_df['target'].values
            
            # Scale features for GLMNet and XGBoost
            X_scaled = self.scaler.fit_transform(X)
            
            # Train GLMNet
            self.logger.info("Training GLMNet model...")
            self.glmnet_model.fit(X_scaled, y)
            glmnet_score = self.glmnet_model.score(X_scaled, y)
            
            # Train XGBoost
            self.logger.info("Training XGBoost model...")
            self.xgb_model.fit(X_scaled, y)
            xgb_score = self.xgb_model.score(X_scaled, y)
            
            self.is_trained = True
            
            results = {
                'games_trained': len(historical_df),
                'glmnet_accuracy': glmnet_score,
                'xgboost_accuracy': xgb_score,
                'elo_teams': len(self.elo_system.ratings)
            }
            
            self.logger.info(f"Training complete: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise
    
    def predict_game(self, home_team: str, away_team: str) -> Dict:
        """
        Predict game outcome using ensemble
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with predictions from each model and ensemble
        """
        # Create features for this game
        game_df = pd.DataFrame([{
            'home_team': home_team,
            'away_team': away_team
        }])
        
        features_df = self.create_features(game_df, is_training=False)
        X = features_df[['home_elo', 'away_elo', 'elo_diff', 'home_advantage']].values
        
        # Elo prediction
        elo_prob = self.elo_system.predict_game(home_team, away_team)
        
        # GLMNet and XGBoost predictions (only if trained)
        if self.is_trained:
            X_scaled = self.scaler.transform(X)
            glmnet_prob = self.glmnet_model.predict_proba(X_scaled)[0][1]
            xgb_prob = self.xgb_model.predict_proba(X_scaled)[0][1]
        else:
            # Use Elo as fallback if models not trained
            glmnet_prob = elo_prob
            xgb_prob = elo_prob
        
        # Blended ensemble prediction (weighted average)
        blended_prob = (
            self.ensemble_weights['elo'] * elo_prob +
            self.ensemble_weights['glmnet'] * glmnet_prob +
            self.ensemble_weights['xgboost'] * xgb_prob
        )
        
        # Determine predicted winner
        predicted_winner = home_team if blended_prob > 0.5 else away_team
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'elo_home_prob': float(elo_prob),
            'glmnet_home_prob': float(glmnet_prob),
            'xgboost_home_prob': float(xgb_prob),
            'blended_home_prob': float(blended_prob),
            'predicted_winner': predicted_winner,
            'confidence': abs(blended_prob - 0.5) * 2  # 0-1 scale
        }
    
    def predict_multiple_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict multiple games
        
        Args:
            games_df: DataFrame with columns: home_team, away_team
            
        Returns:
            DataFrame with original data plus predictions
        """
        predictions = []
        
        for idx, row in games_df.iterrows():
            pred = self.predict_game(row['home_team'], row['away_team'])
            predictions.append(pred)
        
        pred_df = pd.DataFrame(predictions)
        
        # Merge with original data
        result_df = games_df.copy()
        result_df = result_df.merge(
            pred_df,
            on=['home_team', 'away_team'],
            how='left'
        )
        
        return result_df
