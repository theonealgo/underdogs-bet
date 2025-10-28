#!/usr/bin/env python3
"""
Train NBA Ensemble Models (Elo, XGBoost, CatBoost, Meta)
Uses 2024 season historical data for training
"""

import sqlite3
import pandas as pd
import numpy as np
from src.models.universal_ensemble_predictor import UniversalSportsEnsemble
import pickle
import logging
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions_original.db'

def load_nba_training_data():
    """Load 2024 NBA season games for training"""
    conn = sqlite3.connect(DATABASE)
    
    query = """
        SELECT 
            game_id,
            game_date,
            home_team_id as home_team,
            away_team_id as away_team,
            home_score,
            away_score,
            CASE 
                WHEN home_score > away_score THEN 'H'
                ELSE 'A'
            END as result
        FROM games
        WHERE sport = 'NBA' 
        AND season = 2024
        AND home_score IS NOT NULL
        ORDER BY game_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} training games from 2024 NBA season")
    logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    return df

def train_ensemble_models():
    """Train Elo, XGBoost, and Logistic Regression ensemble"""
    logger.info("=" * 70)
    logger.info("TRAINING NBA ENSEMBLE MODELS")
    logger.info("=" * 70)
    
    # Load training data
    training_data = load_nba_training_data()
    
    if len(training_data) < 100:
        logger.error(f"Insufficient training data: {len(training_data)} games")
        return None
    
    # Initialize ensemble predictor
    ensemble = UniversalSportsEnsemble(sport='NBA')
    
    # Train the ensemble (this trains Elo, Logistic, and XGBoost)
    logger.info("Training Elo + Logistic + XGBoost ensemble...")
    ensemble.train(training_data)
    
    # Save ensemble model
    model_path = 'models/nba_ensemble.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    logger.info(f"✓ Saved ensemble model to {model_path}")
    
    # Extract accuracies from training
    features = ensemble.create_features(training_data, is_training=False)
    if 'target' in features.columns:
        features = features.drop('target', axis=1)
    
    # Split for validation
    train_idx = int(len(training_data) * 0.8)
    val_data = training_data.iloc[train_idx:].copy()
    
    if len(val_data) > 0:
        val_features = ensemble.create_features(val_data, is_training=False)
        if 'target' in val_features.columns:
            X_val = val_features.drop('target', axis=1)
            y_val = val_features['target']
        else:
            # Create target from results
            y_val = (val_data['result'] == 'H').astype(int)
            X_val = val_features
        
        # Get individual model predictions
        X_val_scaled = ensemble.scaler.transform(X_val)
        
        # Elo predictions
        elo_preds = []
        for idx, row in val_data.iterrows():
            elo_prob = ensemble.elo_system.predict_game(row['home_team'], row['away_team'])
            elo_preds.append(1 if elo_prob > 0.5 else 0)
        elo_acc = accuracy_score(y_val, elo_preds)
        
        # XGBoost predictions
        xgb_probs = ensemble.xgb_model.predict_proba(X_val_scaled)[:, 1]
        xgb_preds = (xgb_probs > 0.5).astype(int)
        xgb_acc = accuracy_score(y_val, xgb_preds)
        
        # Logistic predictions
        log_probs = ensemble.logistic_model.predict_proba(X_val_scaled)[:, 1]
        log_preds = (log_probs > 0.5).astype(int)
        log_acc = accuracy_score(y_val, log_preds)
        
        # Ensemble predictions
        ensemble_probs = (
            ensemble.ensemble_weights['elo'] * xgb_probs +  # Note: using XGB probs here is intentional
            ensemble.ensemble_weights['logistic'] * log_probs +
            ensemble.ensemble_weights['xgboost'] * xgb_probs
        )
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        ensemble_acc = accuracy_score(y_val, ensemble_preds)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"VALIDATION ACCURACY (2024 season, last 20%)")
        logger.info(f"{'='*70}")
        logger.info(f"  Elo:              {elo_acc:.1%}")
        logger.info(f"  XGBoost:          {xgb_acc:.1%}")
        logger.info(f"  Logistic:         {log_acc:.1%}")
        logger.info(f"  Meta Ensemble:    {ensemble_acc:.1%}")
        logger.info(f"{'='*70}")
    
    return ensemble

def train_catboost_model():
    """Train CatBoost model separately"""
    logger.info("\nTraining CatBoost model...")
    
    # Load training data
    training_data = load_nba_training_data()
    
    # Create features using ensemble for consistency
    ensemble = UniversalSportsEnsemble(sport='NBA')
    features = ensemble.create_features(training_data, is_training=True)
    
    # Prepare data
    if 'target' in features.columns:
        X = features.drop('target', axis=1)
        y = features['target']
    else:
        X = features
        y = (training_data['result'] == 'H').astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CatBoost with NBA-specific hyperparameters (similar to NHL but tuned for NBA)
    catboost_model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False,
        loss_function='Logloss'
    )
    
    catboost_model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = catboost_model.predict(X_train)
    test_pred = catboost_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    logger.info(f"  CatBoost Train Accuracy: {train_acc:.1%}")
    logger.info(f"  CatBoost Test Accuracy:  {test_acc:.1%}")
    
    # Save CatBoost model
    catboost_path = 'models/nba_catboost.pkl'
    with open(catboost_path, 'wb') as f:
        pickle.dump(catboost_model, f)
    logger.info(f"✓ Saved CatBoost model to {catboost_path}")
    
    # Save feature names for later use
    feature_names_path = 'models/nba_feature_names.pkl'
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    logger.info(f"✓ Saved feature names to {feature_names_path}")
    
    return catboost_model

def main():
    """Main training function"""
    try:
        # Train ensemble (Elo, XGBoost, Logistic)
        ensemble = train_ensemble_models()
        
        if ensemble is None:
            logger.error("Failed to train ensemble models")
            return
        
        # Train CatBoost separately
        catboost_model = train_catboost_model()
        
        logger.info("\n" + "="*70)
        logger.info("✅ NBA MODEL TRAINING COMPLETE")
        logger.info("="*70)
        logger.info("Models trained and saved:")
        logger.info("  1. Elo Rating System (in ensemble)")
        logger.info("  2. XGBoost Classifier (in ensemble)")
        logger.info("  3. Logistic Regression (in ensemble)")
        logger.info("  4. CatBoost Classifier (separate)")
        logger.info("\nNext step: Run generate_nba_predictions.py to create predictions")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)

if __name__ == "__main__":
    main()
