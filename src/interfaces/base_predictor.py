from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import date
import pickle
import os

class BasePredictor(ABC):
    """
    Abstract base class for sport-specific prediction models.
    
    All sport predictors must implement these methods to ensure
    consistent prediction capabilities across different sports.
    """
    
    def __init__(self, sport: str):
        """
        Initialize the predictor.
        
        Args:
            sport: Sport name (e.g., 'MLB', 'NBA', 'NFL', 'NHL', 'CFB', 'CBB')
        """
        self.sport = sport
        self.winner_model = None
        self.total_model = None
        self.scaler = None
        self.feature_names = []
        self.model_version = "1.0"
        self.is_trained = False
    
    @abstractmethod
    def train_models(self, features_df: pd.DataFrame, 
                    validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train both winner and total prediction models.
        
        Args:
            features_df: DataFrame with features and targets
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics for both models
        """
        pass
    
    @abstractmethod
    def predict_winner(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict game winners.
        
        Args:
            features_df: DataFrame with game features
            
        Returns:
            DataFrame with columns:
            - game_id: Game identifier
            - predicted_winner: Predicted winner (1=home, 0=away)
            - win_probability: Probability of home team winning
            - confidence: Model confidence in prediction
        """
        pass
    
    @abstractmethod
    def predict_total(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict game totals (points/runs/goals).
        
        Args:
            features_df: DataFrame with game features
            
        Returns:
            DataFrame with columns:
            - game_id: Game identifier
            - predicted_total: Predicted total points/runs/goals
            - total_confidence: Model confidence in prediction
            - over_under_rec: Optional recommendation vs betting line (None if no line available)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, model_type: str = 'winner') -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Args:
            model_type: Type of model ('winner' or 'total')
            
        Returns:
            DataFrame with feature names and importance scores
        """
        pass
    
    def predict_game(self, home_team_id: str, away_team_id: str,
                    game_date: date, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict outcome for a specific game.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            game_date: Date of the game
            features_df: DataFrame with features for this game
            
        Returns:
            Dictionary with complete prediction information
        """
        # Get predictions from both models
        winner_pred = self.predict_winner(features_df)
        total_pred = self.predict_total(features_df)
        
        # Combine results
        result = {
            'game_date': game_date,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'predicted_winner': winner_pred.iloc[0]['predicted_winner'],
            'win_probability': winner_pred.iloc[0]['win_probability'],
            'predicted_total': total_pred.iloc[0]['predicted_total'],
            'winner_confidence': winner_pred.iloc[0]['confidence'],
            'total_confidence': total_pred.iloc[0]['total_confidence'],
            'model_version': self.model_version,
            'sport': self.sport
        }
        
        return result
    
    def save_models(self, filepath: str) -> bool:
        """
        Save trained models to disk.
        
        Args:
            filepath: Base filepath to save models (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save models
            if self.winner_model is not None:
                with open(f"{filepath}_winner.pkl", 'wb') as f:
                    pickle.dump(self.winner_model, f)
            
            if self.total_model is not None:
                with open(f"{filepath}_total.pkl", 'wb') as f:
                    pickle.dump(self.total_model, f)
            
            if self.scaler is not None:
                with open(f"{filepath}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata = {
                'sport': self.sport,
                'model_version': self.model_version,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            filepath: Base filepath to load models from (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load models
            if os.path.exists(f"{filepath}_winner.pkl"):
                with open(f"{filepath}_winner.pkl", 'rb') as f:
                    self.winner_model = pickle.load(f)
            
            if os.path.exists(f"{filepath}_total.pkl"):
                with open(f"{filepath}_total.pkl", 'rb') as f:
                    self.total_model = pickle.load(f)
            
            if os.path.exists(f"{filepath}_scaler.pkl"):
                with open(f"{filepath}_scaler.pkl", 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load metadata
            if os.path.exists(f"{filepath}_metadata.pkl"):
                with open(f"{filepath}_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                    self.model_version = metadata.get('model_version', '1.0')
                    self.feature_names = metadata.get('feature_names', [])
                    self.is_trained = metadata.get('is_trained', False)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def evaluate_model(self, features_df: pd.DataFrame, 
                      model_type: str = 'winner') -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            features_df: DataFrame with features and actual results
            model_type: Type of model to evaluate ('winner' or 'total')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_type == 'winner':
            return self._evaluate_winner_model(features_df)
        else:
            return self._evaluate_total_model(features_df)
    
    def _evaluate_winner_model(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate winner prediction model (to be implemented by subclasses)."""
        return {}
    
    def _evaluate_total_model(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate total prediction model (to be implemented by subclasses)."""
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained models.
        
        Returns:
            Dictionary with model information
        """
        return {
            'sport': self.sport,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'has_winner_model': self.winner_model is not None,
            'has_total_model': self.total_model is not None,
            'has_scaler': self.scaler is not None
        }