#!/usr/bin/env python3
"""
NHL Predictor - 4-Model Prediction System
==========================================
NHL-only Flask application with Elo, XGBoost, CatBoost, and Meta Ensemble models.
Uses complete 2024-2026 NHL schedule data from nhlschedules.py.
"""

from flask import Flask, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb
import pickle
import os
from datetime import datetime
import logging
from nhlschedules import get_nhl_2024_schedule, get_nhl_2025_schedule, get_nhl_2026_schedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# ELO RATING SYSTEM
# ============================================================================
class EloRatingSystem:
    def __init__(self, k_factor=22):
        self.k_factor = k_factor
        self.ratings = {}
        
    def get_rating(self, team):
        return self.ratings.get(team, 1500)
    
    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team, away_team, home_score, away_score):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        expected_home = self.expected_score(home_rating, away_rating)
        expected_away = 1 - expected_home
        
        actual_home = 1 if home_score > away_score else 0
        actual_away = 1 - actual_home
        
        new_home_rating = home_rating + self.k_factor * (actual_home - expected_home)
        new_away_rating = away_rating + self.k_factor * (actual_away - expected_away)
        
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating
    
    def predict(self, home_team, away_team):
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        return self.expected_score(home_rating, away_rating)


# ============================================================================
# NHL PREDICTOR CLASS
# ============================================================================
class NHLPredictor:
    def __init__(self):
        self.elo_system = EloRatingSystem(k_factor=22)
        self.xgb_model = None
        self.catboost_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.team_stats = {}
        
    def load_all_data(self):
        """Load complete NHL schedule data from nhlschedules.py"""
        logger.info("📥 Loading NHL data from nhlschedules.py...")
        
        schedules = []
        schedules.extend(get_nhl_2024_schedule())
        schedules.extend(get_nhl_2025_schedule())
        schedules.extend(get_nhl_2026_schedule())
        
        df = pd.DataFrame(schedules)
        
        # Parse dates
        df['date_parsed'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        
        # Filter completed games (have scores)
        completed = df[(df['home_score'].notna()) & (df['away_score'].notna())].copy()
        
        logger.info(f"✓ Loaded {len(df)} total games ({len(completed)} completed)")
        return completed
    
    def calculate_team_stats(self, games_df):
        """Calculate rolling team statistics"""
        stats = {}
        
        for team in set(list(games_df['home_team'].unique()) + list(games_df['away_team'].unique())):
            stats[team] = {
                'goals_for': [],
                'goals_against': [],
                'wins': 0,
                'losses': 0,
                'games_played': 0
            }
        
        # Sort by date
        games_df = games_df.sort_values('date_parsed')
        
        for _, game in games_df.iterrows():
            home = game['home_team']
            away = game['away_team']
            home_score = game['home_score']
            away_score = game['away_score']
            
            stats[home]['goals_for'].append(home_score)
            stats[home]['goals_against'].append(away_score)
            stats[away]['goals_for'].append(away_score)
            stats[away]['goals_against'].append(home_score)
            
            if home_score > away_score:
                stats[home]['wins'] += 1
                stats[away]['losses'] += 1
            else:
                stats[away]['wins'] += 1
                stats[home]['losses'] += 1
            
            stats[home]['games_played'] += 1
            stats[away]['games_played'] += 1
        
        self.team_stats = stats
    
    def create_features(self, home_team, away_team):
        """Create NHL-specific features for prediction"""
        features = {}
        
        # Elo ratings
        features['home_elo'] = self.elo_system.get_rating(home_team)
        features['away_elo'] = self.elo_system.get_rating(away_team)
        features['elo_diff'] = features['home_elo'] - features['away_elo']
        
        # Team stats
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        # Goals per game (last 10 games)
        home_gf = home_stats.get('goals_for', [0])
        home_ga = home_stats.get('goals_against', [0])
        away_gf = away_stats.get('goals_for', [0])
        away_ga = away_stats.get('goals_against', [0])
        
        features['home_goals_avg'] = np.mean(home_gf[-10:]) if len(home_gf) > 0 else 2.5
        features['home_goals_allowed_avg'] = np.mean(home_ga[-10:]) if len(home_ga) > 0 else 2.5
        features['away_goals_avg'] = np.mean(away_gf[-10:]) if len(away_gf) > 0 else 2.5
        features['away_goals_allowed_avg'] = np.mean(away_ga[-10:]) if len(away_ga) > 0 else 2.5
        
        # Win percentage
        home_games = home_stats.get('games_played', 1)
        away_games = away_stats.get('games_played', 1)
        features['home_win_pct'] = home_stats.get('wins', 0) / max(home_games, 1)
        features['away_win_pct'] = away_stats.get('wins', 0) / max(away_games, 1)
        
        return features
    
    def train_models(self, training_df):
        """Train all 4 models on training data"""
        logger.info(f"🏋️ Training models on {len(training_df)} games...")
        
        # Sort by date
        training_df = training_df.sort_values('date_parsed')
        
        # Calculate team stats
        self.calculate_team_stats(training_df)
        
        # Train Elo
        for _, game in training_df.iterrows():
            self.elo_system.update_ratings(
                game['home_team'], game['away_team'],
                game['home_score'], game['away_score']
            )
        
        # Prepare ML features
        X_list = []
        y_list = []
        
        for _, game in training_df.iterrows():
            features = self.create_features(game['home_team'], game['away_team'])
            X_list.append(list(features.values()))
            y_list.append(1 if game['home_score'] > game['away_score'] else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        self.feature_columns = list(self.create_features(training_df.iloc[0]['home_team'], 
                                                         training_df.iloc[0]['away_team']).keys())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=175,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.75,
            random_state=42
        )
        self.xgb_model.fit(X_scaled, y)
        
        # Train CatBoost
        self.catboost_model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
        self.catboost_model.fit(X_scaled, y)
        
        logger.info("✓ All models trained successfully")
    
    def predict(self, home_team, away_team):
        """Generate predictions from all 4 models"""
        # Elo prediction
        elo_prob = self.elo_system.predict(home_team, away_team)
        
        # ML predictions
        features = self.create_features(home_team, away_team)
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        xgb_prob = self.xgb_model.predict_proba(X_scaled)[0][1]
        catboost_prob = self.catboost_model.predict_proba(X_scaled)[0][1]
        
        # Meta ensemble: CatBoost 50%, XGBoost 30%, Elo 20%
        ensemble_prob = (catboost_prob * 0.50 + xgb_prob * 0.30 + elo_prob * 0.20)
        
        return {
            'elo': elo_prob * 100,
            'xgboost': xgb_prob * 100,
            'catboost': catboost_prob * 100,
            'ensemble': ensemble_prob * 100
        }
    
    def evaluate_models(self, testing_df):
        """Evaluate all models on test set"""
        logger.info(f"📊 Evaluating models on {len(testing_df)} test games...")
        
        results = {
            'elo': {'correct': 0, 'total': 0},
            'xgboost': {'correct': 0, 'total': 0},
            'catboost': {'correct': 0, 'total': 0},
            'ensemble': {'correct': 0, 'total': 0}
        }
        
        for _, game in testing_df.iterrows():
            predictions = self.predict(game['home_team'], game['away_team'])
            actual_winner = 'home' if game['home_score'] > game['away_score'] else 'away'
            
            for model_name in ['elo', 'xgboost', 'catboost', 'ensemble']:
                predicted_winner = 'home' if predictions[model_name] > 50 else 'away'
                results[model_name]['total'] += 1
                if predicted_winner == actual_winner:
                    results[model_name]['correct'] += 1
        
        # Calculate accuracies
        performance = {}
        for model_name in ['elo', 'xgboost', 'catboost', 'ensemble']:
            acc = (results[model_name]['correct'] / results[model_name]['total'] * 100) if results[model_name]['total'] > 0 else 0
            performance[model_name] = {
                'accuracy': round(acc, 1),
                'correct': results[model_name]['correct'],
                'total': results[model_name]['total']
            }
        
        return performance


# ============================================================================
# GLOBAL PREDICTOR INSTANCE
# ============================================================================
predictor = NHLPredictor()
performance_metrics = None

def initialize_predictor():
    """Initialize and train the predictor"""
    global predictor, performance_metrics
    
    # Load all data
    all_games = predictor.load_all_data()
    
    if len(all_games) == 0:
        logger.warning("⚠️ No completed games found")
        return
    
    # Split data: Train on 2024-Jan 2025, Test on Feb 2025-Apr 2025
    training_cutoff = pd.Timestamp('2025-02-01')
    
    training_df = all_games[all_games['date_parsed'] < training_cutoff]
    testing_df = all_games[all_games['date_parsed'] >= training_cutoff]
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"  Training: {len(training_df)} games (2024 - Jan 2025)")
    logger.info(f"  Testing: {len(testing_df)} games (Feb - Apr 2025)")
    
    if len(training_df) == 0:
        logger.warning("⚠️ No training data available")
        return
    
    # Train models
    predictor.train_models(training_df)
    
    # Evaluate on test set
    if len(testing_df) > 0:
        performance_metrics = predictor.evaluate_models(testing_df)
        
        # Get date range
        min_date = testing_df['date_parsed'].min().strftime('%d/%m/%Y')
        max_date = testing_df['date_parsed'].max().strftime('%d/%m/%Y')
        performance_metrics['date_range'] = f"{min_date} - {max_date}"
        performance_metrics['total_games'] = len(testing_df)
        
        logger.info("\n🎯 Model Performance on Test Set:")
        logger.info(f"  Elo:      {performance_metrics['elo']['accuracy']}%")
        logger.info(f"  XGBoost:  {performance_metrics['xgboost']['accuracy']}%")
        logger.info(f"  CatBoost: {performance_metrics['catboost']['accuracy']}%")
        logger.info(f"  Ensemble: {performance_metrics['ensemble']['accuracy']}%")


# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def home():
    """NHL Predictor Home Page"""
    
    template = """
<!DOCTYPE html>
<html>
<head>
    <title>🏒 NHL Predictor - 4-Model System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: #fff;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        .performance {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .performance h3 {
            font-size: 1.8em;
            margin-bottom: 15px;
            text-align: center;
        }
        .performance p {
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .perf-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .perf-card {
            background: rgba(255,255,255,0.15);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 2px solid rgba(255,255,255,0.3);
        }
        .perf-card h4 {
            font-size: 1.3em;
            margin-bottom: 10px;
            color: #fbbf24;
        }
        .perf-card .accuracy {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        .perf-card .record {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .info-box {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border-left: 4px solid #fbbf24;
        }
        .info-box h4 {
            color: #fbbf24;
            margin-bottom: 10px;
        }
        .info-box ul {
            margin-left: 20px;
        }
        .info-box li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏒 NHL Predictor</h1>
        <p class="subtitle">4-Model Prediction System: Elo • XGBoost • CatBoost • Meta Ensemble</p>
        
        {% if performance %}
        <div class="performance">
            <h3>🎯 Model Performance - Test Set ({{ performance.date_range }})</h3>
            <p><strong>Tested on {{ performance.total_games }} games from February - April 2025</strong></p>
            <p><strong>Training: 2024 to January 2025</strong></p>
            
            <div class="perf-grid">
                <div class="perf-card">
                    <h4>Elo Rating</h4>
                    <div class="accuracy">{{ performance.elo.accuracy }}%</div>
                    <div class="record">{{ performance.elo.correct }}-{{ performance.elo.total - performance.elo.correct }}</div>
                </div>
                
                <div class="perf-card">
                    <h4>XGBoost</h4>
                    <div class="accuracy">{{ performance.xgboost.accuracy }}%</div>
                    <div class="record">{{ performance.xgboost.correct }}-{{ performance.xgboost.total - performance.xgboost.correct }}</div>
                </div>
                
                <div class="perf-card">
                    <h4>CatBoost</h4>
                    <div class="accuracy">{{ performance.catboost.accuracy }}%</div>
                    <div class="record">{{ performance.catboost.correct }}-{{ performance.catboost.total - performance.catboost.correct }}</div>
                </div>
                
                <div class="perf-card" style="border: 3px solid #fbbf24;">
                    <h4>🏆 Meta Ensemble</h4>
                    <div class="accuracy">{{ performance.ensemble.accuracy }}%</div>
                    <div class="record">{{ performance.ensemble.correct }}-{{ performance.ensemble.total - performance.ensemble.correct }}</div>
                </div>
            </div>
        </div>
        {% endif %}
        
        <div class="info-box">
            <h4>📊 About This System</h4>
            <ul>
                <li><strong>Data Source:</strong> Complete NHL schedules 2024-2026 from nhlschedules.py</li>
                <li><strong>Total Games:</strong> 5,248 games across 3 seasons</li>
                <li><strong>Elo System:</strong> K-factor = 22 (optimized for NHL)</li>
                <li><strong>XGBoost:</strong> 175 estimators, max_depth=5, learning_rate=0.04</li>
                <li><strong>CatBoost:</strong> 200 iterations, depth=6, learning_rate=0.05</li>
                <li><strong>Ensemble Weights:</strong> CatBoost 50% + XGBoost 30% + Elo 20%</li>
                <li><strong>Features:</strong> Elo ratings, goals/game, goals allowed, win percentage</li>
            </ul>
        </div>
    </div>
</body>
</html>
    """
    
    return render_template_string(template, performance=performance_metrics)


if __name__ == '__main__':
    print("🏒 NHL Predictor Starting!")
    print("🎯 4-Model System: Elo + XGBoost + CatBoost + Meta Ensemble")
    print("📊 Loading complete NHL data from nhlschedules.py...")
    
    # Initialize predictor
    initialize_predictor()
    
    print("\n✓ NHL Predictor ready!")
    print("🌐 Visit http://0.0.0.0:5000 to view results\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
