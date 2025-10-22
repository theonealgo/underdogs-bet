#!/usr/bin/env python3
"""
NHL Model Tester
================
Tests all 4 NHL prediction models on the first 93 games of the 2025 season:
1. Elo Rating Model
2. XGBoost Model
3. CatBoost Model
4. Ensemble Model (weighted combination)

Trains on 2024 NHL season, tests on first 93 games of 2025 season.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import random

# NHL-specific model settings
NHL_SETTINGS = {
    'elo_base': 1500,
    'elo_home_advantage': 50,
    'elo_k_factor': 20,
    'xgb_params': {
        'n_estimators': 200,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'min_child_weight': 3,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'random_state': 42
    },
    'catboost_params': {
        'iterations': 150,
        'depth': 3,
        'learning_rate': 0.1,
        'l2_leaf_reg': 5,
        'random_seed': 42,
        'verbose': False
    },
    'ensemble_weights': {'elo': 0.2, 'xgboost': 0.3, 'catboost': 0.5}
}


class NHLModelTester:
    def __init__(self):
        self.team_elo_ratings = {}
        self.xgb_model = None
        self.catboost_model = None
        self.feature_columns = None
        
    def get_team_elo(self, team_name):
        """Get current Elo rating for a team"""
        if team_name not in self.team_elo_ratings:
            self.team_elo_ratings[team_name] = NHL_SETTINGS['elo_base']
        return self.team_elo_ratings[team_name]
    
    def update_elo_ratings(self, home_team, away_team, home_score, away_score):
        """Update Elo ratings after a game"""
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        home_advantage = NHL_SETTINGS['elo_home_advantage']
        
        # Calculate expected scores
        home_expected = 1 / (1 + 10**((away_elo - home_elo - home_advantage) / 400))
        away_expected = 1 - home_expected
        
        # Determine actual result
        if home_score > away_score:
            home_actual, away_actual = 1, 0
            margin = home_score - away_score
        elif away_score > home_score:
            home_actual, away_actual = 0, 1
            margin = away_score - home_score
        else:
            home_actual, away_actual = 0.5, 0.5
            margin = 0
        
        # Margin adjustment (more sensitive to goal differences in NHL)
        margin_adj = 1 - 1 / (1 + np.exp(margin / 2))
        k_factor = NHL_SETTINGS['elo_k_factor']
        adjusted_k = k_factor * margin_adj
        
        # Update ratings
        self.team_elo_ratings[home_team] = home_elo + adjusted_k * (home_actual - home_expected)
        self.team_elo_ratings[away_team] = away_elo + adjusted_k * (away_actual - away_expected)
    
    def predict_elo(self, home_team, away_team):
        """Predict game using Elo ratings"""
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        home_advantage = NHL_SETTINGS['elo_home_advantage']
        
        home_win_prob = 1 / (1 + 10**((away_elo - home_elo - home_advantage) / 400))
        return home_win_prob
    
    def extract_features(self, home_team, away_team):
        """Extract features for ML models"""
        # Use consistent seed for reproducible features
        seed = hash(home_team + away_team) % 10000
        np.random.seed(seed)
        random.seed(seed)
        
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        
        features = {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_difference': home_elo - away_elo,
            'home_goals_per_game': 3.0 + (home_elo - 1500) / 200,
            'away_goals_per_game': 3.0 + (away_elo - 1500) / 200,
            'home_goals_allowed': 3.0 - (home_elo - 1500) / 200,
            'away_goals_allowed': 3.0 - (away_elo - 1500) / 200,
            'home_shots_per_game': 30.0 + (home_elo - 1500) / 50,
            'away_shots_per_game': 30.0 + (away_elo - 1500) / 50,
            'home_save_pct': 0.91 + (home_elo - 1500) / 10000,
            'away_save_pct': 0.91 + (away_elo - 1500) / 10000,
            'home_pp_pct': 0.20 + (home_elo - 1500) / 5000,
            'away_pp_pct': 0.20 + (away_elo - 1500) / 5000,
            'home_days_rest': random.randint(0, 5),
            'away_days_rest': random.randint(0, 5),
            'home_back_to_back': random.choice([0, 1]),
            'away_back_to_back': random.choice([0, 1]),
            'home_advantage': 1,
            'travel_distance': random.uniform(0, 3000)
        }
        
        features['rest_advantage'] = features['home_days_rest'] - features['away_days_rest']
        
        return features
    
    def load_training_data(self):
        """Load 2024 NHL season games for training"""
        conn = sqlite3.connect('sports_predictions.db')
        
        query = """
            SELECT game_id, game_date, home_team_id, away_team_id, home_score, away_score
            FROM games
            WHERE sport = 'NHL'
            AND season = 2024
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
            ORDER BY 
                substr(game_date, 7, 4) || substr(game_date, 4, 2) || substr(game_date, 1, 2)
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def load_test_data(self, limit=93):
        """Load first N games from 2025 NHL season for testing"""
        conn = sqlite3.connect('sports_predictions.db')
        
        query = """
            SELECT game_id, game_date, home_team_id, away_team_id, home_score, away_score
            FROM games
            WHERE sport = 'NHL'
            AND season = 2025
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
            ORDER BY 
                substr(game_date, 7, 4) || substr(game_date, 4, 2) || substr(game_date, 1, 2)
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def train_models(self, train_df):
        """Train XGBoost and CatBoost models on training data"""
        print(f"\n📚 Training models on {len(train_df)} games from 2024 season...")
        
        # Prepare training data
        X_data = []
        y_data = []
        
        # Reset Elo ratings for training
        self.team_elo_ratings = {}
        
        for idx, row in train_df.iterrows():
            # Extract features
            features = self.extract_features(row['home_team_id'], row['away_team_id'])
            X_data.append(features)
            
            # Target: 1 if home team won
            home_won = 1 if row['home_score'] > row['away_score'] else 0
            y_data.append(home_won)
            
            # Update Elo ratings
            self.update_elo_ratings(
                row['home_team_id'], row['away_team_id'],
                row['home_score'], row['away_score']
            )
        
        # Convert to DataFrame
        X = pd.DataFrame(X_data)
        y = np.array(y_data)
        self.feature_columns = X.columns.tolist()
        
        # Split for validation (75/25)
        split_idx = int(len(X) * 0.75)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"  Training set: {len(X_train)} games")
        print(f"  Validation set: {len(X_val)} games")
        
        # Train XGBoost
        print("\n  Training XGBoost model...")
        self.xgb_model = XGBClassifier(**NHL_SETTINGS['xgb_params'])
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val_acc = self.xgb_model.score(X_val, y_val)
        print(f"    ✓ XGBoost validation accuracy: {xgb_val_acc:.1%}")
        
        # Train CatBoost
        print("\n  Training CatBoost model...")
        self.catboost_model = CatBoostClassifier(**NHL_SETTINGS['catboost_params'])
        self.catboost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        cat_val_acc = self.catboost_model.score(X_val, y_val)
        print(f"    ✓ CatBoost validation accuracy: {cat_val_acc:.1%}")
        
        print(f"\n✅ Models trained successfully!")
    
    def test_models(self, test_df):
        """Test all 4 models on test data"""
        print(f"\n🧪 Testing on first {len(test_df)} games from 2025 season...")
        print(f"  Date range: {test_df.iloc[0]['game_date']} to {test_df.iloc[-1]['game_date']}")
        
        # Reset Elo ratings for 2025 season
        self.team_elo_ratings = {}
        
        elo_predictions = []
        xgb_predictions = []
        cat_predictions = []
        ensemble_predictions = []
        actual_results = []
        
        for idx, row in test_df.iterrows():
            home_team = row['home_team_id']
            away_team = row['away_team_id']
            home_score = row['home_score']
            away_score = row['away_score']
            
            # Actual result
            home_won = 1 if home_score > away_score else 0
            actual_results.append(home_won)
            
            # 1. Elo prediction
            elo_prob = self.predict_elo(home_team, away_team)
            elo_pred = 1 if elo_prob > 0.5 else 0
            elo_predictions.append(elo_pred)
            
            # 2. XGBoost prediction
            features = self.extract_features(home_team, away_team)
            feature_vector = [features[col] for col in self.feature_columns]
            X = np.array(feature_vector).reshape(1, -1)
            xgb_prob = self.xgb_model.predict_proba(X)[0][1]
            xgb_pred = 1 if xgb_prob > 0.5 else 0
            xgb_predictions.append(xgb_pred)
            
            # 3. CatBoost prediction
            cat_prob = self.catboost_model.predict_proba(X)[0][1]
            cat_pred = 1 if cat_prob > 0.5 else 0
            cat_predictions.append(cat_pred)
            
            # 4. Ensemble prediction (weighted average)
            weights = NHL_SETTINGS['ensemble_weights']
            ensemble_prob = (
                weights['elo'] * elo_prob +
                weights['xgboost'] * xgb_prob +
                weights['catboost'] * cat_prob
            )
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            ensemble_predictions.append(ensemble_pred)
            
            # Update Elo ratings for next prediction
            self.update_elo_ratings(home_team, away_team, home_score, away_score)
        
        # Calculate accuracies
        actual_results = np.array(actual_results)
        elo_acc = np.mean(np.array(elo_predictions) == actual_results) * 100
        xgb_acc = np.mean(np.array(xgb_predictions) == actual_results) * 100
        cat_acc = np.mean(np.array(cat_predictions) == actual_results) * 100
        ensemble_acc = np.mean(np.array(ensemble_predictions) == actual_results) * 100
        
        return {
            'elo': {'correct': int(np.sum(np.array(elo_predictions) == actual_results)), 'accuracy': elo_acc},
            'xgboost': {'correct': int(np.sum(np.array(xgb_predictions) == actual_results)), 'accuracy': xgb_acc},
            'catboost': {'correct': int(np.sum(np.array(cat_predictions) == actual_results)), 'accuracy': cat_acc},
            'ensemble': {'correct': int(np.sum(np.array(ensemble_predictions) == actual_results)), 'accuracy': ensemble_acc},
            'total_games': len(test_df)
        }


def main():
    print("="*70)
    print("NHL MODEL TESTER - First 93 Games of 2025 Season")
    print("="*70)
    
    tester = NHLModelTester()
    
    # Load data
    print("\n📊 Loading data...")
    train_df = tester.load_training_data()
    test_df = tester.load_test_data(limit=93)
    
    print(f"  Training data: {len(train_df)} completed games from 2024 season")
    print(f"  Test data: {len(test_df)} games from 2025 season")
    
    # Train models
    tester.train_models(train_df)
    
    # Test models
    results = tester.test_models(test_df)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS - NHL 2025 Season (First 93 Games)")
    print("="*70)
    
    total = results['total_games']
    
    print(f"\n{'Model':<20} {'Correct':<12} {'Total':<12} {'Win %':<12}")
    print("-"*70)
    print(f"{'1. Elo Rating':<20} {results['elo']['correct']:<12} {total:<12} {results['elo']['accuracy']:.1f}%")
    print(f"{'2. XGBoost':<20} {results['xgboost']['correct']:<12} {total:<12} {results['xgboost']['accuracy']:.1f}%")
    print(f"{'3. CatBoost':<20} {results['catboost']['correct']:<12} {total:<12} {results['catboost']['accuracy']:.1f}%")
    print("-"*70)
    print(f"{'🏆 ENSEMBLE':<20} {results['ensemble']['correct']:<12} {total:<12} {results['ensemble']['accuracy']:.1f}%")
    print("="*70)
    
    # Find best model
    models = [
        ('Elo Rating', results['elo']['accuracy']),
        ('XGBoost', results['xgboost']['accuracy']),
        ('CatBoost', results['catboost']['accuracy']),
        ('Ensemble', results['ensemble']['accuracy'])
    ]
    best_model = max(models, key=lambda x: x[1])
    
    print(f"\n💡 Best performing model: {best_model[0]} ({best_model[1]:.1f}%)")
    print(f"\nEnsemble weights: Elo {NHL_SETTINGS['ensemble_weights']['elo']*100:.0f}%, "
          f"XGBoost {NHL_SETTINGS['ensemble_weights']['xgboost']*100:.0f}%, "
          f"CatBoost {NHL_SETTINGS['ensemble_weights']['catboost']*100:.0f}%")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
