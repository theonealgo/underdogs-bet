import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb


class NHLPredictor:
    """
    Machine learning models for NHL game predictions
    """
    
    def __init__(self, model_dir: str = "models", db_manager=None):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.db_manager = db_manager
        
        # Initialize models
        self.winner_model = None
        self.total_model = None
        self.is_trained = False
        self.feature_names = None
        
        # Model parameters
        self.winner_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        self.total_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load saved models if they exist"""
        try:
            winner_path = os.path.join(self.model_dir, 'nhl_winner_model.pkl')
            total_path = os.path.join(self.model_dir, 'nhl_total_model.pkl')
            features_path = os.path.join(self.model_dir, 'nhl_feature_names.pkl')
            
            if os.path.exists(winner_path) and os.path.exists(total_path):
                with open(winner_path, 'rb') as f:
                    self.winner_model = pickle.load(f)
                with open(total_path, 'rb') as f:
                    self.total_model = pickle.load(f)
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
            winner_path = os.path.join(self.model_dir, 'nhl_winner_model.pkl')
            total_path = os.path.join(self.model_dir, 'nhl_total_model.pkl')
            features_path = os.path.join(self.model_dir, 'nhl_feature_names.pkl')
            
            with open(winner_path, 'wb') as f:
                pickle.dump(self.winner_model, f)
            with open(total_path, 'wb') as f:
                pickle.dump(self.total_model, f)
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            self.logger.info("Saved NHL models")
        except Exception as e:
            self.logger.error(f"Error saving NHL models: {e}")
    
    def create_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for NHL games"""
        features_list = []
        
        for idx, game in games_df.iterrows():
            # Get team stats from recent games
            home_stats = self._get_team_stats(game['home_team_id'], game['game_date'], games_df)
            away_stats = self._get_team_stats(game['away_team_id'], game['game_date'], games_df)
            
            features = {
                # Home team features
                'home_win_pct': home_stats['win_pct'],
                'home_goals_per_game': home_stats['goals_per_game'],
                'home_goals_against_per_game': home_stats['goals_against_per_game'],
                'home_recent_form': home_stats['recent_form'],
                
                # Away team features
                'away_win_pct': away_stats['win_pct'],
                'away_goals_per_game': away_stats['goals_per_game'],
                'away_goals_against_per_game': away_stats['goals_against_per_game'],
                'away_recent_form': away_stats['recent_form'],
                
                # Differential features
                'win_pct_diff': home_stats['win_pct'] - away_stats['win_pct'],
                'goals_diff': home_stats['goals_per_game'] - away_stats['goals_per_game'],
                'defense_diff': away_stats['goals_against_per_game'] - home_stats['goals_against_per_game'],
                'form_diff': home_stats['recent_form'] - away_stats['recent_form'],
            }
            
            # Add target variables if available
            if 'home_score' in game and 'away_score' in game and pd.notna(game['home_score']):
                features['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
                features['total_goals'] = game['home_score'] + game['away_score']
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _get_team_stats(self, team_id: str, current_date, games_df: pd.DataFrame) -> Dict:
        """Calculate team statistics from recent games"""
        # Get team's games before current date
        team_games = games_df[
            ((games_df['home_team_id'] == team_id) | (games_df['away_team_id'] == team_id)) &
            (games_df['game_date'] < current_date) &
            (games_df['status'] == 'final')
        ].sort_values('game_date', ascending=False).head(10)  # Last 10 games
        
        if len(team_games) == 0:
            # Default stats for new teams or start of season
            return {
                'win_pct': 0.5,
                'goals_per_game': 3.0,
                'goals_against_per_game': 3.0,
                'recent_form': 0.5
            }
        
        wins = 0
        goals_for = 0
        goals_against = 0
        recent_wins = 0
        
        for idx, game in team_games.iterrows():
            is_home = game['home_team_id'] == team_id
            
            if is_home:
                team_score = game['home_score']
                opp_score = game['away_score']
            else:
                team_score = game['away_score']
                opp_score = game['home_score']
            
            if pd.notna(team_score) and pd.notna(opp_score):
                goals_for += team_score
                goals_against += opp_score
                
                if team_score > opp_score:
                    wins += 1
                    if idx < 5:  # Last 5 games for recent form
                        recent_wins += 1
        
        games_count = len(team_games)
        recent_count = min(5, games_count)
        
        return {
            'win_pct': wins / games_count if games_count > 0 else 0.5,
            'goals_per_game': goals_for / games_count if games_count > 0 else 3.0,
            'goals_against_per_game': goals_against / games_count if games_count > 0 else 3.0,
            'recent_form': recent_wins / recent_count if recent_count > 0 else 0.5
        }
    
    def train_models(self, games_df: pd.DataFrame) -> Dict:
        """Train winner and totals prediction models"""
        try:
            self.logger.info(f"Training NHL models with {len(games_df)} games")
            
            # Create features
            features_df = self.create_features(games_df)
            
            # Filter to games with complete data
            complete_games = features_df.dropna()
            
            if len(complete_games) < 20:
                self.logger.error(f"Insufficient training data: {len(complete_games)} games")
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Prepare features and targets
            feature_cols = [col for col in complete_games.columns if col not in ['home_win', 'total_goals']]
            X = complete_games[feature_cols]
            y_winner = complete_games['home_win']
            y_total = complete_games['total_goals']
            
            self.feature_names = feature_cols
            
            # Train winner model
            X_train, X_test, y_train, y_test = train_test_split(X, y_winner, test_size=0.2, random_state=42)
            self.winner_model = xgb.XGBClassifier(**self.winner_params)
            self.winner_model.fit(X_train, y_train)
            
            winner_acc = accuracy_score(y_test, self.winner_model.predict(X_test))
            self.logger.info(f"Winner model accuracy: {winner_acc:.3f}")
            
            # Train totals model
            X_train, X_test, y_train, y_test = train_test_split(X, y_total, test_size=0.2, random_state=42)
            self.total_model = xgb.XGBRegressor(**self.total_params)
            self.total_model.fit(X_train, y_train)
            
            total_mae = mean_absolute_error(y_test, self.total_model.predict(X_test))
            self.logger.info(f"Total model MAE: {total_mae:.3f}")
            
            self.is_trained = True
            self._save_models()
            
            return {
                'success': True,
                'winner_accuracy': winner_acc,
                'total_mae': total_mae,
                'training_games': len(complete_games)
            }
            
        except Exception as e:
            self.logger.error(f"Error training NHL models: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_game(self, home_team: str, away_team: str, game_date, historical_games: pd.DataFrame) -> Dict:
        """Predict outcome for a single NHL game"""
        try:
            # Create a game DataFrame for feature creation
            game_df = pd.DataFrame([{
                'home_team_id': home_team,
                'away_team_id': away_team,
                'game_date': game_date,
                'status': 'scheduled'
            }])
            
            # Create features
            features_df = self.create_features(game_df)
            
            if self.feature_names is None:
                self.logger.error("No feature names available")
                return self._default_prediction(home_team, away_team)
            
            X = features_df[self.feature_names]
            
            if not self.is_trained:
                self.logger.warning("Models not trained, using default prediction")
                return self._default_prediction(home_team, away_team)
            
            # Predict winner
            home_win_prob = self.winner_model.predict_proba(X)[0][1]
            predicted_winner = home_team if home_win_prob > 0.5 else away_team
            
            # Predict total
            predicted_total = self.total_model.predict(X)[0]
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': predicted_winner,
                'home_win_probability': home_win_prob,
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
