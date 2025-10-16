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
    
    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create features for prediction
        
        Args:
            df: DataFrame with home_team, away_team columns
            is_training: If True, updates Elo ratings based on results
            
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
            
            # Update Elo if training
            if is_training and 'result' in df.columns:
                result = row['result']
                if result in ['H', 'A']:
                    home_won = result == 'H'
                    self.elo_system.update_ratings(home_team, away_team, home_won)
                    features['target'] = 1 if home_won else 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train(self, historical_df: pd.DataFrame) -> Dict:
        """
        Train all models on historical data
        
        Args:
            historical_df: DataFrame with home_team, away_team, result columns
                          result should be 'H' (home win) or 'A' (away win)
        
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info(f"Training {self.sport} ensemble on {len(historical_df)} games")
            
            # Create features with Elo updates
            features_df = self.create_features(historical_df, is_training=True)
            
            # Filter only games with results
            features_df = features_df.dropna(subset=['target'])
            
            if len(features_df) == 0:
                self.logger.warning("No training data with results")
                return {'error': 'No training data'}
            
            # Prepare training data with all features
            feature_cols = ['home_elo', 'away_elo', 'elo_diff', 'elo_ratio', 
                          'home_elo_squared', 'away_elo_squared', 'elo_diff_squared',
                          'home_advantage', 'elo_product']
            X = features_df[feature_cols].values
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
                'logistic_accuracy': float(logistic_score),
                'xgboost_accuracy': float(xgb_score),
                'teams': len(self.elo_system.ratings)
            }
            
            self.logger.info(f"Training complete: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training {self.sport} models: {str(e)}")
            raise
    
    def predict_game(self, home_team: str, away_team: str) -> Dict:
        """
        Predict game outcome using ensemble
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with predictions from each model
        """
        # Create features
        game_df = pd.DataFrame([{
            'home_team': home_team,
            'away_team': away_team
        }])
        
        features_df = self.create_features(game_df, is_training=False)
        feature_cols = ['home_elo', 'away_elo', 'elo_diff', 'elo_ratio', 
                      'home_elo_squared', 'away_elo_squared', 'elo_diff_squared',
                      'home_advantage', 'elo_product']
        X = features_df[feature_cols].values
        
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
