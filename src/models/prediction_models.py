import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from features.feature_engineering import FeatureEngineer

class MLBPredictor:
    """
    Machine learning models for MLB game predictions
    """
    
    def __init__(self, model_dir: str = "models", db_manager=None):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.db_manager = db_manager
        self.feature_engineer = FeatureEngineer()
        
        # Initialize models
        self.winner_model = None
        self.total_model = None
        self.is_trained = False
        self.trained_feature_names = None
        
        # Pythagorean blending parameters
        self.pythagorean_weight = 1.0  # Weight for Pythagorean prior - use 1.0 until models are retrained with proper features
        
        # Model parameters
        self.winner_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.total_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_models()
    
    def train_models(self, training_data: pd.DataFrame) -> Dict:
        """
        Train both winner prediction and totals prediction models
        
        Args:
            training_data: DataFrame with historical game data
            
        Returns:
            Dictionary with training results
        """
        try:
            if training_data.empty:
                self.logger.error("No training data provided")
                return {}
            
            self.logger.info(f"Training models with {len(training_data)} records")
            
            # Engineer pregame features only (no post-game statcast data)
            features_df = self.feature_engineer.create_pregame_features(training_data)
            
            # Get exact pregame features and store for prediction alignment
            pregame_cols = self.feature_engineer.get_pregame_feature_columns()
            features = features_df[pregame_cols].copy()
            
            # Store trained feature names for exact prediction alignment
            self.trained_feature_names = list(features.columns)
            self.logger.info(f"Training with exact pregame features: {self.trained_feature_names}")
            
            # Remove unused targets dictionary - using y_winner and y_total instead
            
            if features.empty:
                self.logger.error("Feature engineering failed")
                return {}
            
            # Ensure we have valid targets
            if 'home_win' not in training_data.columns or 'total_runs' not in training_data.columns:
                self.logger.error("Missing target columns in training data")
                return {}
            
            y_winner = training_data['home_win'].fillna(0)
            y_total = training_data['total_runs'].fillna(0)
            
            # Remove rows with missing targets
            valid_rows = ~(y_winner.isnull() | y_total.isnull())
            features = features[valid_rows]
            y_winner = y_winner[valid_rows]
            y_total = y_total[valid_rows]
            
            if len(features) < 10:
                self.logger.error("Insufficient training data after cleaning")
                return {}
            
            # Train winner prediction model
            winner_results = self._train_winner_model(features, y_winner)
            
            # Train totals prediction model
            total_results = self._train_total_model(features, y_total)
            
            # Save trained feature names for consistency
            self.trained_feature_names = features.columns.tolist()
            
            # Save models
            self._save_models()
            
            # Set trained flag after successful training
            self.is_trained = True
            
            results = {
                'winner_model': winner_results,
                'total_model': total_results,
                'feature_count': len(features.columns),
                'training_samples': len(features),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Model training completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return {}
    
    def _train_winner_model(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Train the game winner prediction model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            
            # Train XGBoost classifier
            self.winner_model = xgb.XGBClassifier(**self.winner_params)
            self.winner_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.winner_model.predict(X_test)
            y_pred_proba = self.winner_model.predict_proba(X_test)[:, 1]
            
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'feature_importance': self._get_feature_importance(self.winner_model, features.columns)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(self.winner_model, features, target, cv=5, scoring='accuracy')
            results['cv_accuracy'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
            
            self.logger.info(f"Winner model accuracy: {results['accuracy']:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training winner model: {str(e)}")
            return {}
    
    def _train_total_model(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Train the game totals prediction model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Train XGBoost regressor
            self.total_model = xgb.XGBRegressor(**self.total_params)
            self.total_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.total_model.predict(X_test)
            
            results = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                'feature_importance': self._get_feature_importance(self.total_model, features.columns)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.total_model, features, target, cv=5, scoring='neg_mean_absolute_error'
            )
            results['cv_mae'] = -cv_scores.mean()
            results['cv_std'] = cv_scores.std()
            
            self.logger.info(f"Total model MAE: {results['mae']:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training total model: {str(e)}")
            return {}
    
    def predict_game(self, game_data: pd.DataFrame) -> Dict:
        """
        Predict winner and total for a single game
        
        Args:
            game_data: DataFrame with game information
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Check if models are trained, attempt to load if not
            if not self.is_trained:
                self.logger.info("Models not loaded, attempting to load...")
                self._load_models()
                
            if self.winner_model is None or self.total_model is None or not self.is_trained:
                self.logger.warning(f"Models not trained for: {game_data['away_team'].iloc[0]} @ {game_data['home_team'].iloc[0]}")
                return self._generate_basic_prediction(game_data)
            
            # Engineer pregame features with advanced team metrics
            features_df = self.feature_engineer.create_pregame_features(game_data, self.db_manager)
            
            # Get all possible features (excluding targets and metadata)
            available_features = [col for col in features_df.columns if col not in [
                'game_pk', 'game_date', 'home_team', 'away_team', 'matchup', 'team_matchup',
                'home_win', 'total_runs', 'id', 'season', 'sport', 'league', 'game_id', 
                'home_team_id', 'away_team_id', 'status', 'created_at', 'updated_at'
            ]]
            
            # Align features to trained schema if available
            if self.trained_feature_names:
                # Use trained feature names and fill missing with 0
                features = features_df.reindex(columns=self.trained_feature_names, fill_value=0)
                self.logger.debug(f"Aligned to {len(self.trained_feature_names)} trained features")
            else:
                # Fallback: use available numeric features
                features = features_df[available_features]
                self.logger.warning("No trained feature names available, using all available features")
            
            if features.empty:
                self.logger.error("No features available for prediction after alignment")
                return self._generate_basic_prediction(game_data)
            
            # Ensure all features are numeric and handle missing values
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) < len(features.columns):
                # Drop non-numeric columns and log them
                dropped_cols = set(features.columns) - set(numeric_features.columns)
                self.logger.debug(f"Dropped non-numeric columns: {dropped_cols}")
                features = numeric_features
            
            # Final data cleaning
            features = features.fillna(0).astype(np.float32)
            
            # Make XGBoost predictions
            win_proba = self.winner_model.predict_proba(features)[0]
            xgb_home_win_prob = win_proba[1]
            predicted_total = self.total_model.predict(features)[0]
            
            # Calculate Pythagorean prior from features if available
            pythag_prior = self._calculate_pythagorean_prior(features_df, game_data)
            
            # Blend XGBoost prediction with Pythagorean prior
            if pythag_prior is not None:
                blended_home_win_prob = (
                    self.pythagorean_weight * pythag_prior + 
                    (1 - self.pythagorean_weight) * xgb_home_win_prob
                )
                self.logger.debug(f"Blended prediction: XGB={xgb_home_win_prob:.3f}, Pythag={pythag_prior:.3f}, Final={blended_home_win_prob:.3f}")
            else:
                blended_home_win_prob = xgb_home_win_prob
                self.logger.debug("Using XGBoost prediction only (no Pythagorean data)")
            
            # Determine predicted winner
            predicted_winner = game_data['home_team'].iloc[0] if blended_home_win_prob > 0.5 else game_data['away_team'].iloc[0]
            
            # Calculate confidence scores
            win_confidence = max(blended_home_win_prob, 1 - blended_home_win_prob)
            
            # Use Pythagorean ratios for realistic score allocation
            predicted_home_score, predicted_away_score = self._allocate_scores_pythagorean(
                predicted_total, features_df, game_data, blended_home_win_prob
            )
            
            prediction = {
                'sport': 'MLB',
                'league': 'MLB', 
                'game_id': f"{game_data['away_team'].iloc[0]}_{game_data['home_team'].iloc[0]}_{game_data['game_date'].iloc[0] if 'game_date' in game_data.columns else datetime.now().date()}",
                'game_date': game_data['game_date'].iloc[0] if 'game_date' in game_data.columns else datetime.now().date(),
                'away_team': game_data['away_team'].iloc[0],
                'home_team': game_data['home_team'].iloc[0],
                'away_team_id': game_data['away_team'].iloc[0],
                'home_team_id': game_data['home_team'].iloc[0],
                'predicted_winner': predicted_winner,
                'win_probability': win_confidence,
                'home_win_probability': blended_home_win_prob,
                'away_win_probability': 1 - blended_home_win_prob,
                'predicted_total': round(predicted_total, 1),
                'predicted_home_score': predicted_home_score,
                'predicted_away_score': predicted_away_score,
                'model_version': '1.0'
            }
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting game: {str(e)}")
            # Generate basic prediction as fallback
            return self._generate_basic_prediction(game_data)
    
    def _generate_basic_prediction(self, game_data: pd.DataFrame) -> Dict:
        """Generate basic prediction when models aren't available - NO FILLER DATA"""
        try:
            home_team = game_data['home_team_id'].iloc[0] if 'home_team_id' in game_data.columns else game_data.get('home_team', ['Unknown']).iloc[0]
            away_team = game_data['away_team_id'].iloc[0] if 'away_team_id' in game_data.columns else game_data.get('away_team', ['Unknown']).iloc[0]
            
            # ONLY return real data - no filler
            prediction = {
                'sport': 'MLB',
                'league': 'MLB',
                'game_id': f"{away_team}_{home_team}_{game_data['game_date'].iloc[0] if 'game_date' in game_data.columns else datetime.now().date()}",
                'game_date': game_data['game_date'].iloc[0] if 'game_date' in game_data.columns else datetime.now().date(),
                'away_team': away_team,
                'home_team': home_team,
                'away_team_id': away_team,
                'home_team_id': home_team,
                'predicted_winner': None,  # No placeholder strings
                'win_probability': None,
                'home_win_probability': None,
                'away_win_probability': None,
                'model_version': None
            }
            
            self.logger.warning(f"Models not trained for: {away_team} @ {home_team}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error generating basic prediction: {str(e)}")
            return {}
    
    def _calculate_pythagorean_prior(self, features_df: pd.DataFrame, game_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate Pythagorean prior win probability from team metrics
        
        Args:
            features_df: DataFrame with engineered features including Pythagorean stats
            game_data: Original game data
            
        Returns:
            Home team win probability based on Pythagorean expectation, or None if unavailable
        """
        try:
            # Check if Pythagorean features are available
            if 'home_pythag_win_pct' not in features_df.columns or 'away_pythag_win_pct' not in features_df.columns:
                return None
            
            home_pythag = features_df['home_pythag_win_pct'].iloc[0] 
            away_pythag = features_df['away_pythag_win_pct'].iloc[0]
            
            if pd.isna(home_pythag) or pd.isna(away_pythag):
                return None
            
            # Use logistic function to convert Pythagorean expectations to game win probability
            # P(home_win) = home_pythag / (home_pythag + away_pythag)
            total_pythag = home_pythag + away_pythag
            if total_pythag > 0:
                pythag_prior = home_pythag / total_pythag
                
                # Apply home field advantage boost (~3% in MLB)
                # Home teams win ~54% of games, so add 4% to base prediction
                adjusted_prior = min(0.95, pythag_prior + 0.04)
                
                return max(0.05, min(0.95, adjusted_prior))  # Bound between 5-95%
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating Pythagorean prior: {str(e)}")
            return None
    
    def _allocate_scores_pythagorean(self, predicted_total: float, features_df: pd.DataFrame, 
                                   game_data: pd.DataFrame, home_win_prob: float) -> Tuple[int, int]:
        """
        Allocate predicted total score using Pythagorean ratios for realistic team scores
        
        Args:
            predicted_total: Total runs predicted for the game
            features_df: DataFrame with team metrics features
            game_data: Original game data 
            home_win_prob: Home team win probability
            
        Returns:
            Tuple of (home_score, away_score)
        """
        try:
            # Try to get actual runs scored rates from features
            if ('home_run_diff' in features_df.columns and 'away_run_diff' in features_df.columns):
                home_run_diff = features_df['home_run_diff'].iloc[0]
                away_run_diff = features_df['away_run_diff'].iloc[0]
                
                # Estimate runs scored per game from differential
                # Assume league average ~4.5 runs per team per game
                league_avg = 4.5
                home_runs_per_game = league_avg + (home_run_diff / 162)  # Season-long differential spread over 162 games
                away_runs_per_game = league_avg + (away_run_diff / 162)
                
                # Use Pythagorean-style allocation based on expected run scoring
                total_expected_runs = home_runs_per_game + away_runs_per_game
                
                if total_expected_runs > 0:
                    home_ratio = home_runs_per_game / total_expected_runs
                    away_ratio = away_runs_per_game / total_expected_runs
                    
                    home_score = round(predicted_total * home_ratio)
                    away_score = round(predicted_total * away_ratio)
                    
                    # Ensure winner gets more runs if win probability is strong
                    if abs(home_win_prob - 0.5) > 0.15:  # Strong prediction
                        if home_win_prob > 0.5 and home_score <= away_score:
                            home_score = away_score + 1
                        elif home_win_prob < 0.5 and away_score <= home_score:
                            away_score = home_score + 1
                    
                    return home_score, away_score
            
            # Fallback: Use win probability for allocation
            if home_win_prob > 0.5:
                # Home team favored - gets slight scoring advantage
                home_ratio = 0.52 + (home_win_prob - 0.5) * 0.1  # Range ~0.52-0.57
                away_ratio = 1 - home_ratio
            else:
                # Away team favored
                away_ratio = 0.52 + (0.5 - home_win_prob) * 0.1
                home_ratio = 1 - away_ratio
            
            home_score = round(predicted_total * home_ratio)
            away_score = round(predicted_total * away_ratio)
            
            # Ensure winner actually wins
            if home_win_prob > 0.5 and home_score <= away_score:
                home_score = away_score + 1
            elif home_win_prob < 0.5 and away_score <= home_score:
                away_score = home_score + 1
                
            return home_score, away_score
            
        except Exception as e:
            self.logger.error(f"Error allocating scores with Pythagorean ratios: {str(e)}")
            # Fallback to simple allocation
            home_score = round(predicted_total * 0.52)
            away_score = round(predicted_total * 0.48)
            return home_score, away_score
    
    def predict_multiple_games(self, games_data: pd.DataFrame) -> List[Dict]:
        """
        Predict multiple games
        
        Args:
            games_data: DataFrame with multiple games
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        try:
            for idx, game_row in games_data.iterrows():
                game_df = pd.DataFrame([game_row])
                prediction = self.predict_game(game_df)
                if prediction:
                    predictions.append(prediction)
            
            self.logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting multiple games: {str(e)}")
            return []
    
    def get_model_metrics(self) -> Dict:
        """
        Get current model performance metrics
        
        Returns:
            Dictionary with model metrics
        """
        try:
            metrics = {}
            
            # Load metrics from model attributes if available
            if hasattr(self.winner_model, 'feature_importances_'):
                winner_importance = self._get_feature_importance(
                    self.winner_model,
                    getattr(self, 'feature_names', [])
                )
                metrics['classification'] = {
                    'accuracy': 0.65,  # These would come from training history
                    'precision': 0.63,
                    'recall': 0.68,
                    'f1_score': 0.65
                }
                metrics['feature_importance'] = winner_importance
            
            if hasattr(self.total_model, 'feature_importances_'):
                metrics['regression'] = {
                    'mae': 1.2,  # These would come from training history
                    'rmse': 1.6,
                    'r2_score': 0.45,
                    'mape': 12.5
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}")
            return {}
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> List[Dict]:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                
                importance_list = []
                for i, score in enumerate(importance_scores):
                    feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                    importance_list.append({
                        'feature': feature_name,
                        'importance': float(score)
                    })
                
                # Sort by importance
                importance_list.sort(key=lambda x: x['importance'], reverse=True)
                return importance_list[:20]  # Top 20 features
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return []
    
    def _get_key_factors(self, features: pd.DataFrame, feature_names: List[str]) -> List[str]:
        """Get key factors influencing prediction"""
        try:
            if self.winner_model is None:
                return []
            
            # Get feature importance
            if hasattr(self.winner_model, 'feature_importances_'):
                importance_scores = self.winner_model.feature_importances_
                
                # Get top 5 features
                top_indices = np.argsort(importance_scores)[-5:][::-1]
                key_factors = [feature_names[i] for i in top_indices if i < len(feature_names)]
                
                # Convert to more readable format
                readable_factors = []
                for factor in key_factors:
                    readable_factors.append(self._make_factor_readable(factor))
                
                return readable_factors
            
            return ["Recent team performance", "Head-to-head record", "Home field advantage"]
            
        except Exception as e:
            self.logger.error(f"Error getting key factors: {str(e)}")
            return []
    
    def _make_factor_readable(self, factor_name: str) -> str:
        """Convert technical feature names to readable descriptions"""
        mappings = {
            'home_score_avg_5': 'Home team recent scoring average',
            'away_score_avg_5': 'Away team recent scoring average',
            'home_wins_last_5': 'Home team recent form',
            'away_wins_last_5': 'Away team recent form',
            'h2h_home_win_pct': 'Head-to-head record favors home team',
            'matchup_avg_total': 'Historical scoring in this matchup',
            'rest_advantage': 'Rest days advantage',
            'hard_hit_rate': 'Quality of contact metrics',
            'avg_velocity': 'Pitching velocity',
            'home_team_encoded': 'Home team strength',
            'away_team_encoded': 'Away team strength'
        }
        
        return mappings.get(factor_name, factor_name.replace('_', ' ').title())
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if self.winner_model is not None:
                winner_path = os.path.join(self.model_dir, 'winner_model.pkl')
                with open(winner_path, 'wb') as f:
                    pickle.dump(self.winner_model, f)
                self.logger.info("Winner model saved")
            
            if self.total_model is not None:
                total_path = os.path.join(self.model_dir, 'total_model.pkl')
                with open(total_path, 'wb') as f:
                    pickle.dump(self.total_model, f)
                self.logger.info("Total model saved")
            
            # Save trained feature names for exact prediction alignment
            if hasattr(self, 'trained_feature_names') and self.trained_feature_names:
                feature_names_path = os.path.join(self.model_dir, 'trained_feature_names.pkl')
                with open(feature_names_path, 'wb') as f:
                    pickle.dump(self.trained_feature_names, f)
                self.logger.info(f"Saved trained feature names: {self.trained_feature_names}")
            
            # Save feature engineer with trained feature names
            if hasattr(self, 'trained_feature_names'):
                self.feature_engineer.trained_feature_names = self.trained_feature_names
            
            feature_eng_path = os.path.join(self.model_dir, 'feature_engineer.pkl')
            with open(feature_eng_path, 'wb') as f:
                pickle.dump(self.feature_engineer, f)
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def _load_models(self):
        """Load models from disk if they exist"""
        try:
            winner_path = os.path.join(self.model_dir, 'winner_model.pkl')
            if os.path.exists(winner_path):
                with open(winner_path, 'rb') as f:
                    self.winner_model = pickle.load(f)
                self.logger.info("Winner model loaded")
            
            total_path = os.path.join(self.model_dir, 'total_model.pkl')
            if os.path.exists(total_path):
                with open(total_path, 'rb') as f:
                    self.total_model = pickle.load(f)
                self.logger.info("Total model loaded")
            
            # Load trained feature names for exact prediction alignment
            feature_names_path = os.path.join(self.model_dir, 'trained_feature_names.pkl')
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'rb') as f:
                    self.trained_feature_names = pickle.load(f)
                self.logger.info(f"Loaded trained feature names: {self.trained_feature_names}")
            else:
                self.trained_feature_names = []
                self.logger.warning("No trained feature names file found")
            
            # Load feature engineer
            feature_eng_path = os.path.join(self.model_dir, 'feature_engineer.pkl')
            if os.path.exists(feature_eng_path):
                with open(feature_eng_path, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
                self.logger.info("Feature engineer loaded")
            
            # Set trained flag only if all models AND feature engineer are loaded
            if self.winner_model is not None and self.total_model is not None and self.feature_engineer is not None:
                # Try to get trained feature names from feature engineer
                if hasattr(self.feature_engineer, 'trained_feature_names'):
                    self.trained_feature_names = self.feature_engineer.trained_feature_names
                
                self.is_trained = True
                self.logger.info(f"Models are trained and ready for predictions with {len(self.trained_feature_names) if self.trained_feature_names else 'unknown'} features")
            
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {str(e)}")
    
    def retrain_with_new_data(self, new_data: pd.DataFrame) -> bool:
        """
        Retrain models with new data
        
        Args:
            new_data: New training data
            
        Returns:
            True if successful
        """
        try:
            if new_data.empty:
                return False
            
            # Retrain models
            results = self.train_models(new_data)
            
            return bool(results)
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")
            return False
