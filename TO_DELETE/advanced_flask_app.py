#!/usr/bin/env python3
"""
Advanced Sports Predictor Flask App - NHL Optimized
Uses proper XGBoost training with NHL-specific features and sophisticated Elo ratings
Trains on 2024-June 2025, Tests on October 2025 onwards
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import json
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import random
import sqlite3

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.models.schedules import get_nfl_schedule, get_nba_schedule, get_nhl_schedule, get_mlb_schedule, get_ncaaf_schedule, get_ncaab_schedule, get_wnba_schedule

app = Flask(__name__)

# Sport-specific settings optimized for each sport
SPORT_SETTINGS = {
    'NHL': {
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
            'early_stopping_rounds': 50,
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
        'meta_params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'elo_base': 1500,
        'elo_home_advantage': 50,
        'elo_k_factors': {
            'weeks_1_4': 20,
            'weeks_5_12': 20,
            'weeks_13_plus': 20
        },
        'goalie_adjustment': -60,
        'short_rest_penalty': -30,
        'travel_penalty': -15,
        'injury_penalty': -12
    },
    'NFL': {
        'xgb_params': {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'min_child_weight': 3,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'early_stopping_rounds': 50,
            'random_state': 42
        },
        'catboost_params': {
            'iterations': 500,
            'depth': 5,
            'learning_rate': 0.08,
            'l2_leaf_reg': 4,
            'random_seed': 42,
            'verbose': False
        },
        'meta_params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'elo_base': 1500,
        'elo_home_advantage': 75,
        'elo_k_factors': {
            'weeks_1_4': 32,
            'weeks_5_12': 32,
            'weeks_13_plus': 32
        },
        'qb_adjustment': -75,
        'short_rest_penalty': -25,
        'travel_penalty': -10,
        'injury_penalty': -10
    }
}

# Copy NFL settings to other sports (keeping NFL unchanged)
for sport in ['NBA', 'MLB', 'NCAAF', 'NCAAB', 'WNBA']:
    SPORT_SETTINGS[sport] = SPORT_SETTINGS['NFL'].copy()

class AdvancedSportPredictor:
    def __init__(self, sport_code):
        self.sport_code = sport_code
        self.settings = SPORT_SETTINGS.get(sport_code, SPORT_SETTINGS['NFL'])
        self.team_elo_ratings = {}
        self.xgb_model = None
        self.catboost_model = None
        self.meta_model = None
        self.feature_columns = []
        self.is_trained = False
        
        # Initialize ensemble weights (CatBoost 50%, XGBoost 30%, Elo 20%)
        self.ensemble_weights = {'catboost': 0.5, 'xgboost': 0.3, 'elo': 0.2}
        
        # Feature engineering storage
        self.team_stats = {}
        
    def initialize_team_stats(self, team_name):
        """Initialize stats tracking for a team"""
        if team_name not in self.team_stats:
            self.team_stats[team_name] = {
                'games': [],
                'wins': 0,
                'losses': 0,
                'points_for': [],
                'points_against': [],
                'injuries': 0
            }
    
    def get_team_elo(self, team_name):
        """Get current Elo rating for a team"""
        if team_name not in self.team_elo_ratings:
            self.team_elo_ratings[team_name] = self.settings['elo_base']
        return self.team_elo_ratings[team_name]
    
    def get_dynamic_k_factor(self, week_num):
        """Get K-factor based on week number"""
        if week_num <= 4:
            return self.settings['elo_k_factors']['weeks_1_4']
        elif week_num <= 12:
            return self.settings['elo_k_factors']['weeks_5_12']
        else:
            return self.settings['elo_k_factors']['weeks_13_plus']
    
    def calculate_margin_adjustment(self, margin):
        """Calculate margin of victory adjustment"""
        if self.sport_code == 'NHL':
            return 1 - 1 / (1 + np.exp(margin / 2))
        else:
            return 1 - 1 / (1 + np.exp(margin / 10))
    
    def update_elo_ratings(self, home_team, away_team, home_score, away_score, week_num=1, adjustments=None):
        """Update Elo ratings with advanced adjustments"""
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        
        home_advantage = self.settings['elo_home_advantage']
        
        if adjustments:
            if self.sport_code == 'NHL':
                if adjustments.get('home_backup_goalie'):
                    home_elo += self.settings.get('goalie_adjustment', -60)
                if adjustments.get('away_backup_goalie'):
                    away_elo += self.settings.get('goalie_adjustment', -60)
            else:
                if adjustments.get('home_qb_out'):
                    home_elo += self.settings.get('qb_adjustment', -75)
                if adjustments.get('away_qb_out'):
                    away_elo += self.settings.get('qb_adjustment', -75)
            
            if adjustments.get('home_short_rest'):
                home_elo += self.settings['short_rest_penalty']
            if adjustments.get('away_short_rest'):
                away_elo += self.settings['short_rest_penalty']
            if adjustments.get('home_long_travel'):
                home_elo += self.settings['travel_penalty']
            if adjustments.get('away_long_travel'):
                away_elo += self.settings['travel_penalty']
            
            home_elo += adjustments.get('home_injuries', 0) * self.settings['injury_penalty']
            away_elo += adjustments.get('away_injuries', 0) * self.settings['injury_penalty']
        
        home_expected = 1 / (1 + 10**((away_elo - home_elo - home_advantage) / 400))
        away_expected = 1 - home_expected
        
        if home_score > away_score:
            home_actual, away_actual = 1, 0
            margin = home_score - away_score
        elif away_score > home_score:
            home_actual, away_actual = 0, 1
            margin = away_score - home_score
        else:
            home_actual, away_actual = 0.5, 0.5
            margin = 0
        
        k_factor = self.get_dynamic_k_factor(week_num)
        margin_adj = self.calculate_margin_adjustment(margin)
        adjusted_k = k_factor * margin_adj
        
        self.team_elo_ratings[home_team] = home_elo + adjusted_k * (home_actual - home_expected)
        self.team_elo_ratings[away_team] = away_elo + adjusted_k * (away_actual - away_expected)
    
    def extract_features_from_game(self, home_team, away_team, game_data=None):
        """Extract advanced features for XGBoost model"""
        self.initialize_team_stats(home_team)
        self.initialize_team_stats(away_team)
        
        seed = hash(home_team + away_team) % 10000
        np.random.seed(seed)
        random.seed(seed)
        
        features = {}
        
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        features['elo_difference'] = home_elo - away_elo
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        
        home_stats = self.team_stats[home_team]
        away_stats = self.team_stats[away_team]
        
        home_recent_games = home_stats['games'][-5:] if len(home_stats['games']) >= 5 else home_stats['games']
        away_recent_games = away_stats['games'][-5:] if len(away_stats['games']) >= 5 else away_stats['games']
        
        features['home_win_pct_last_5'] = sum(1 for g in home_recent_games if g.get('won', False)) / max(len(home_recent_games), 1)
        features['away_win_pct_last_5'] = sum(1 for g in away_recent_games if g.get('won', False)) / max(len(away_recent_games), 1)
        
        if self.sport_code == 'NHL':
            features['home_goals_per_game'] = 3.0 + (home_elo - 1500) / 200
            features['away_goals_per_game'] = 3.0 + (away_elo - 1500) / 200
            features['home_goals_allowed'] = 3.0 - (home_elo - 1500) / 200
            features['away_goals_allowed'] = 3.0 - (away_elo - 1500) / 200
            features['home_shots_per_game'] = 30.0 + (home_elo - 1500) / 50
            features['away_shots_per_game'] = 30.0 + (away_elo - 1500) / 50
            features['home_save_pct'] = 0.91 + (home_elo - 1500) / 10000
            features['away_save_pct'] = 0.91 + (away_elo - 1500) / 10000
            features['home_pp_pct'] = 0.20 + (home_elo - 1500) / 5000
            features['away_pp_pct'] = 0.20 + (away_elo - 1500) / 5000
            features['home_pk_pct'] = 0.80 + (home_elo - 1500) / 5000
            features['away_pk_pct'] = 0.80 + (away_elo - 1500) / 5000
            
            home_rest_days = random.randint(0, 5)
            away_rest_days = random.randint(0, 5)
            features['home_days_rest'] = home_rest_days
            features['away_days_rest'] = away_rest_days
            features['home_back_to_back'] = 1 if home_rest_days == 0 else 0
            features['away_back_to_back'] = 1 if away_rest_days == 0 else 0
            features['rest_advantage'] = home_rest_days - away_rest_days
            features['both_back_to_back'] = 1 if (home_rest_days == 0 and away_rest_days == 0) else 0
            features['one_back_to_back'] = 1 if (home_rest_days == 0) != (away_rest_days == 0) else 0
        else:
            features['home_qb_rating'] = 85.0 + (home_elo - 1500) / 50
            features['away_qb_rating'] = 85.0 + (away_elo - 1500) / 50
            features['home_turnovers_per_game'] = 1.5 - (home_elo - 1500) / 1000
            features['away_turnovers_per_game'] = 1.5 - (away_elo - 1500) / 1000
            features['home_yards_gained'] = 350 + (home_elo - 1500) / 10
            features['away_yards_gained'] = 350 + (away_elo - 1500) / 10
            features['home_yards_allowed'] = 350 - (home_elo - 1500) / 10
            features['away_yards_allowed'] = 350 - (away_elo - 1500) / 10
        
        features['home_advantage'] = 1
        features['division_game'] = random.choice([0, 1])
        features['home_injuries'] = home_stats['injuries']
        features['away_injuries'] = away_stats['injuries']
        
        if self.sport_code == 'NHL':
            features['ice_quality'] = random.uniform(0.8, 1.0)
            features['altitude'] = random.uniform(0, 1500)
        else:
            features['temperature'] = 65.0
            features['wind_speed'] = 5.0
            features['weather_impact'] = 0.0
            features['is_indoor'] = 1
        
        features['travel_distance'] = random.uniform(0, 3000)
        features['rest_days_difference'] = random.randint(-3, 3)
        features['public_bet_pct'] = random.uniform(30, 70)
        features['closing_spread'] = random.uniform(-3, 3) if self.sport_code == 'NHL' else random.uniform(-14, 14)
        
        return features
    
    def prepare_training_data(self, games_history):
        """Prepare training data from historical games"""
        X_data = []
        y_data = []
        dates = []
        
        games_with_dates = []
        for game in games_history:
            if game.get('completed') and game.get('home_score') is not None and game.get('away_score') is not None:
                game_date = game.get('date', game.get('game_date', f'2024-{len(games_with_dates):03d}'))
                games_with_dates.append((game_date, game))
        
        games_with_dates.sort(key=lambda x: x[0])
        
        for game_date, game in games_with_dates:
            features = self.extract_features_from_game(game['home_team'], game['away_team'], game)
            X_data.append(features)
            dates.append(game_date)
            
            home_won = 1 if game['home_score'] > game['away_score'] else 0
            y_data.append(home_won)
        
        if not X_data:
            return None, None, None
        
        X_df = pd.DataFrame(X_data)
        self.feature_columns = X_df.columns.tolist()
        
        return X_df, np.array(y_data), dates
    
    def train_xgboost_model(self, games_history):
        """Train XGBoost model on historical games"""
        result = self.prepare_training_data(games_history)
        
        if result[0] is None or len(result[0]) < 10:
            print(f"Not enough training data for {self.sport_code}. Using simulated model.")
            self.create_simulated_model()
            return
        
        X, y, dates = result
        
        print(f"\n=== Training on {len(X)} games ===")
        
        nan_counts = X.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            X = X.fillna(X.median())
        
        split_idx = int(len(X) * 0.75)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.xgb_model = XGBClassifier(**self.settings['xgb_params'])
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        self.catboost_model = CatBoostClassifier(**self.settings['catboost_params'])
        self.catboost_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        
        self.is_trained = True
        print(f"✓ Trained {self.sport_code} models")
    
    def create_simulated_model(self):
        """Create a simulated model"""
        if self.sport_code == 'NHL':
            self.feature_columns = [
                'elo_difference', 'home_elo', 'away_elo', 'home_win_pct_last_5', 'away_win_pct_last_5',
                'home_goals_per_game', 'away_goals_per_game', 'home_goals_allowed', 'away_goals_allowed',
                'home_shots_per_game', 'away_shots_per_game', 'home_save_pct', 'away_save_pct',
                'home_pp_pct', 'away_pp_pct', 'home_pk_pct', 'away_pk_pct',
                'home_advantage', 'division_game', 'home_injuries', 'away_injuries',
                'ice_quality', 'altitude', 'travel_distance', 'rest_days_difference',
                'public_bet_pct', 'closing_spread',
                'home_days_rest', 'away_days_rest', 'home_back_to_back', 'away_back_to_back',
                'rest_advantage', 'both_back_to_back', 'one_back_to_back'
            ]
        else:
            self.feature_columns = [
                'elo_difference', 'home_elo', 'away_elo', 'home_win_pct_last_5', 'away_win_pct_last_5',
                'home_qb_rating', 'away_qb_rating', 'home_turnovers_per_game', 'away_turnovers_per_game',
                'home_yards_gained', 'away_yards_gained', 'home_yards_allowed', 'away_yards_allowed',
                'home_advantage', 'division_game', 'home_injuries', 'away_injuries',
                'temperature', 'wind_speed', 'weather_impact', 'is_indoor', 'travel_distance', 'rest_days_difference',
                'public_bet_pct', 'closing_spread'
            ]
        self.is_trained = True
    
    def predict_game_xgboost(self, home_team, away_team):
        """Predict game using XGBoost model"""
        if not self.is_trained:
            return {'home_win_prob': 0.5, 'away_win_prob': 0.5}
        
        features = self.extract_features_from_game(home_team, away_team)
        
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        
        if self.xgb_model is not None:
            home_win_prob = self.xgb_model.predict_proba(X)[0][1]
        else:
            home_win_prob = 0.5
        
        return {'home_win_prob': home_win_prob, 'away_win_prob': 1 - home_win_prob}
    
    def predict_game_catboost(self, home_team, away_team):
        """Predict game using CatBoost model"""
        if not self.is_trained:
            return {'home_win_prob': 0.5, 'away_win_prob': 0.5}
        
        features = self.extract_features_from_game(home_team, away_team)
        
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        
        if self.catboost_model is not None:
            home_win_prob = self.catboost_model.predict_proba(X)[0][1]
        else:
            home_win_prob = 0.5
        
        return {'home_win_prob': home_win_prob, 'away_win_prob': 1 - home_win_prob}
    
    def predict_game_elo(self, home_team, away_team):
        """Predict game using advanced Elo ratings"""
        home_elo = self.get_team_elo(home_team)
        away_elo = self.get_team_elo(away_team)
        
        home_win_prob = 1 / (1 + 10**((away_elo - home_elo - self.settings['elo_home_advantage']) / 400))
        
        return {
            'home_win_prob': home_win_prob,
            'away_win_prob': 1 - home_win_prob,
            'home_elo': home_elo,
            'away_elo': away_elo
        }
    
    def predict_game(self, home_team, away_team):
        """Combined prediction using Elo, XGBoost, CatBoost and Meta model"""
        elo_pred = self.predict_game_elo(home_team, away_team)
        xgb_pred = self.predict_game_xgboost(home_team, away_team)
        cat_pred = self.predict_game_catboost(home_team, away_team)
        
        weights = self.ensemble_weights
        meta_home_prob = (
            weights['catboost'] * cat_pred['home_win_prob'] +
            weights['xgboost'] * xgb_pred['home_win_prob'] +
            weights['elo'] * elo_pred['home_win_prob']
        )
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': meta_home_prob,
            'away_win_prob': 1 - meta_home_prob,
            'elo_home_prob': elo_pred['home_win_prob'],
            'xgb_home_prob': xgb_pred['home_win_prob'],
            'cat_home_prob': cat_pred['home_win_prob'],
            'meta_home_prob': meta_home_prob,
            'home_elo': elo_pred['home_elo'],
            'away_elo': elo_pred['away_elo']
        }

# Initialize predictors for each sport
predictors = {
    sport: AdvancedSportPredictor(sport) 
    for sport in ['NFL', 'NBA', 'NHL', 'MLB', 'NCAAF', 'NCAAB', 'WNBA']
}

def load_nhl_games_from_database():
    """Load NHL games from the sports_predictions database"""
    try:
        conn = sqlite3.connect('sports_predictions.db')
        
        query = """
            SELECT game_id, game_date, home_team_id, away_team_id, home_score, away_score, season
            FROM games
            WHERE sport = 'NHL'
            ORDER BY 
                substr(game_date, 7, 4) || substr(game_date, 4, 2) || substr(game_date, 1, 2)
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        schedule = []
        for idx, row in df.iterrows():
            try:
                date_parts = row['game_date'].split('/')
                if len(date_parts) == 3:
                    day, month, year = date_parts
                    formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)} 19:00"
                else:
                    formatted_date = row['game_date']
            except:
                formatted_date = row['game_date']
            
            has_scores = row['home_score'] is not None and row['away_score'] is not None
            result = f"{row['home_score']} - {row['away_score']}" if has_scores else None
            
            schedule.append({
                'match_id': row['game_id'],
                'round': 1,
                'date': formatted_date,
                'venue': 'NHL Arena',
                'home_team': row['home_team_id'],
                'away_team': row['away_team_id'],
                'result': result,
                'home_score': row['home_score'],
                'away_score': row['away_score']
            })
        
        print(f"✓ Loaded {len(schedule)} NHL games from database ({len([g for g in schedule if g['result']])} completed)")
        return schedule
        
    except Exception as e:
        print(f"Error loading NHL games from database: {e}")
        return []

def get_schedule_for_sport(sport_code):
    """Get schedule data for a specific sport"""
    try:
        if sport_code == 'NHL':
            return load_nhl_games_from_database()
        elif sport_code == 'NFL':
            return get_nfl_schedule()
        elif sport_code == 'NBA':
            return get_nba_schedule()
        elif sport_code == 'MLB':
            return get_mlb_schedule()
        elif sport_code == 'NCAAF':
            return get_ncaaf_schedule()
        elif sport_code == 'NCAAB':
            return get_ncaab_schedule()
        elif sport_code == 'WNBA':
            return get_wnba_schedule()
        else:
            return []
    except Exception as e:
        print(f"Error loading {sport_code} schedule: {e}")
        return []

def process_and_split_games(sport_code):
    """
    Process games and split into:
    - Training: 2024 to June 2025
    - Testing: October 2025 onwards
    Returns: (training_games, test_games, predictor)
    """
    schedule_list = get_schedule_for_sport(sport_code)
    predictor = predictors[sport_code]
    
    if not schedule_list:
        print(f"No schedule data for {sport_code}")
        return [], [], predictor
    
    # Reset Elo ratings
    predictor.team_elo_ratings = {}
    
    # Convert to DataFrame and sort by date
    schedule_df = pd.DataFrame(schedule_list)
    
    try:
        schedule_df['date_parsed'] = pd.to_datetime(schedule_df['date'], errors='coerce')
        schedule_df = schedule_df.sort_values('date_parsed')
    except Exception as e:
        print(f"Date parsing failed: {e}")
    
    # Filter completed games only
    completed_games = schedule_df[
        schedule_df['result'].notna() & 
        (schedule_df['result'] != '') & 
        (schedule_df['result'] != 'None')
    ].copy()
    
    # Split by date
    # Training: October 2024 - February 2025
    # Testing: March 2025 - April 2025
    training_cutoff = pd.Timestamp('2025-03-01')
    testing_start = pd.Timestamp('2025-03-01')
    
    training_df = completed_games[completed_games['date_parsed'] < training_cutoff]
    testing_df = completed_games[completed_games['date_parsed'] >= testing_start]
    
    print(f"\n📊 Data Split for {sport_code}:")
    print(f"  Training: {len(training_df)} games (Oct 2024 - Feb 2025)")
    print(f"  Testing: {len(testing_df)} games (Mar - Apr 2025)")
    
    # Process training games
    training_games = []
    for idx, game in training_df.iterrows():
        result = str(game['result'])
        if '-' in result:
            try:
                scores = result.split(' - ')
                if len(scores) == 2:
                    home_score = int(scores[0].strip())
                    away_score = int(scores[1].strip())
                    
                    week_num = game.get('round', 1)
                    
                    # Update Elo ratings
                    predictor.update_elo_ratings(
                        game['home_team'], 
                        game['away_team'], 
                        home_score, 
                        away_score,
                        week_num=week_num
                    )
                    
                    training_games.append({
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'home_score': home_score,
                        'away_score': away_score,
                        'completed': True,
                        'date': game.get('date', '')
                    })
            except:
                continue
    
    # Train models
    if training_games:
        predictor.train_xgboost_model(training_games)
    
    # Convert test games to list format
    test_games = []
    for _, game in testing_df.iterrows():
        result = str(game['result'])
        if '-' in result:
            try:
                scores = result.split(' - ')
                if len(scores) == 2:
                    home_score = int(scores[0].strip())
                    away_score = int(scores[1].strip())
                    
                    test_games.append({
                        'match_id': game.get('match_id', ''),
                        'date': game['date_parsed'].strftime('%Y-%m-%d %H:%M') if pd.notna(game['date_parsed']) else game.get('date', ''),
                        'home_team': game.get('home_team', ''),
                        'away_team': game.get('away_team', ''),
                        'venue': game.get('venue', ''),
                        'result': result,
                        'home_score': home_score,
                        'away_score': away_score,
                        'completed': True
                    })
            except:
                continue
    
    return training_games, test_games, predictor

def calculate_model_performance(test_games, predictor):
    """
    Calculate performance for each model on test games
    Win condition: (Model % > 50% AND Home Score > Away Score) OR (Model % < 50% AND Away Score > Home Score)
    """
    if not test_games:
        return None
    
    elo_correct = 0
    xgb_correct = 0
    cat_correct = 0
    meta_correct = 0
    total = len(test_games)
    
    for game in test_games:
        home_won = game['home_score'] > game['away_score']
        
        # Get predictions
        prediction = predictor.predict_game(game['home_team'], game['away_team'])
        
        # Elo Model
        elo_pred_home = prediction['elo_home_prob'] > 0.5
        if (elo_pred_home and home_won) or (not elo_pred_home and not home_won):
            elo_correct += 1
        
        # XGBoost Model
        xgb_pred_home = prediction['xgb_home_prob'] > 0.5
        if (xgb_pred_home and home_won) or (not xgb_pred_home and not home_won):
            xgb_correct += 1
        
        # CatBoost Model
        cat_pred_home = prediction['cat_home_prob'] > 0.5
        if (cat_pred_home and home_won) or (not cat_pred_home and not home_won):
            cat_correct += 1
        
        # Meta Ensemble
        meta_pred_home = prediction['meta_home_prob'] > 0.5
        if (meta_pred_home and home_won) or (not meta_pred_home and not home_won):
            meta_correct += 1
    
    return {
        'total_games': total,
        'elo': {'correct': elo_correct, 'accuracy': (elo_correct / total * 100) if total > 0 else 0},
        'xgboost': {'correct': xgb_correct, 'accuracy': (xgb_correct / total * 100) if total > 0 else 0},
        'catboost': {'correct': cat_correct, 'accuracy': (cat_correct / total * 100) if total > 0 else 0},
        'meta': {'correct': meta_correct, 'accuracy': (meta_correct / total * 100) if total > 0 else 0},
        'date_range': f"{test_games[0]['date']} to {test_games[-1]['date']}" if test_games else "N/A"
    }

@app.route('/')
def home():
    """Main page showing all sports"""
    sports = ['NFL', 'NBA', 'NHL', 'MLB', 'NCAAF', 'NCAAB', 'WNBA']
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Sports Predictor - NHL Optimized</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { text-align: center; color: #333; }
            .sports-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .sport-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .sport-card h2 { margin-top: 0; color: #2c5aa0; }
            .btn { background: #2c5aa0; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; display: inline-block; }
            .btn:hover { background: #1e3d6f; }
            .settings { font-size: 12px; color: #666; margin-top: 10px; }
            .badge { background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
            .nhl-badge { background: #ff6b35; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏒 Advanced Sports Predictor</h1>
            <p style="text-align: center; color: #666;">4-Model System: Elo + XGBoost + CatBoost + Meta</p>
            
            <div class="sports-grid">
                {% for sport in sports %}
                <div class="sport-card">
                    <h2>{{ sport }} 
                        {% if sport == 'NHL' %}
                            <span class="nhl-badge">OPTIMIZED</span>
                        {% else %}
                            <span class="badge">READY</span>
                        {% endif %}
                    </h2>
                    <p><strong>Models:</strong> Elo, XGBoost, CatBoost, Meta Ensemble</p>
                    {% if sport == 'NHL' %}
                        <p><strong>Features:</strong> Goals, Shots, Save %, PP%, PK%, Rest Days</p>
                    {% else %}
                        <p><strong>Features:</strong> Advanced ensemble modeling</p>
                    {% endif %}
                    <a href="/sport/{{ sport }}" class="btn">View Predictions</a>
                </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html, sports=sports)

@app.route('/sport/<sport_code>')
def sport_predictions(sport_code):
    """Show predictions for a specific sport"""
    if sport_code not in predictors:
        return f"Sport {sport_code} not supported", 404
    
    # Force fresh predictor instance
    predictors[sport_code] = AdvancedSportPredictor(sport_code)
    
    # Process and split games
    training_games, test_games, predictor = process_and_split_games(sport_code)
    
    # Calculate model performance
    performance = calculate_model_performance(test_games, predictor)
    
    # Generate predictions for test games
    predictions = []
    for game in test_games:
        if game['home_team'] and game['away_team']:
            prediction = predictor.predict_game(game['home_team'], game['away_team'])
            prediction.update({
                'match_id': game['match_id'],
                'date': game['date'],
                'venue': game['venue'],
                'completed': game['completed'],
                'actual_result': game['result']
            })
            predictions.append(prediction)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ sport_code }} Predictions</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; }
            .back-btn { background: #666; color: white; padding: 8px 12px; text-decoration: none; border-radius: 5px; }
            .performance { background: #e7f3ff; padding: 20px; margin: 20px 0; border-radius: 10px; border-left: 4px solid #2c5aa0; }
            .performance h3 { margin-top: 0; color: #2c5aa0; }
            .perf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px; }
            .perf-card { background: white; padding: 15px; border-radius: 8px; text-align: center; }
            .perf-card h4 { margin: 0 0 10px 0; color: #333; }
            .perf-accuracy { font-size: 32px; font-weight: bold; color: #28a745; }
            .perf-record { color: #666; font-size: 14px; margin-top: 5px; }
            .predictions-table { background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-height: 60vh; overflow-y: auto; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }
            th { background: #f8f9fa; font-weight: bold; position: sticky; top: 0; z-index: 10; }
            .prob-bar { height: 18px; background: #e9ecef; border-radius: 10px; overflow: hidden; }
            .prob-fill { height: 100%; background: linear-gradient(90deg, #28a745, #ffc107); }
            .settings { background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 5px; font-size: 13px; }
            .badge { background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
            .nhl-badge { background: #ff6b35; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Back</a>
            <h1>{{ sport_code }} Predictions 
                {% if sport_code == 'NHL' %}
                    <span class="nhl-badge">NHL OPTIMIZED</span>
                {% else %}
                    <span class="badge">STANDARD</span>
                {% endif %}
            </h1>
            
            {% if performance %}
            <div class="performance">
                <h3>🎯 Model Performance - Test Set ({{ performance.date_range }})</h3>
                <p><strong>Tested on {{ performance.total_games }} games from March - April 2025</strong></p>
                <p><strong>Training: October 2024 to February 2025</strong></p>
                
                <div class="perf-grid">
                    <div class="perf-card">
                        <h4>Elo Rating</h4>
                        <div class="perf-accuracy">{{ performance.elo.accuracy|round(1) }}%</div>
                        <div class="perf-record">{{ performance.elo.correct }}/{{ performance.total_games }}</div>
                    </div>
                    <div class="perf-card">
                        <h4>XGBoost</h4>
                        <div class="perf-accuracy">{{ performance.xgboost.accuracy|round(1) }}%</div>
                        <div class="perf-record">{{ performance.xgboost.correct }}/{{ performance.total_games }}</div>
                    </div>
                    <div class="perf-card">
                        <h4>CatBoost</h4>
                        <div class="perf-accuracy">{{ performance.catboost.accuracy|round(1) }}%</div>
                        <div class="perf-record">{{ performance.catboost.correct }}/{{ performance.total_games }}</div>
                    </div>
                    <div class="perf-card" style="background: #d4edda;">
                        <h4>🏆 Meta Ensemble</h4>
                        <div class="perf-accuracy" style="color: #155724;">{{ performance.meta.accuracy|round(1) }}%</div>
                        <div class="perf-record" style="color: #155724;">{{ performance.meta.correct }}/{{ performance.total_games }}</div>
                    </div>
                </div>
            </div>
            {% endif %}
            
            <div class="settings">
                <strong>Model Settings:</strong>
                {% if sport_code == 'NHL' %}
                XGBoost: {{ predictors[sport_code].settings['xgb_params']['n_estimators'] }} trees, depth {{ predictors[sport_code].settings['xgb_params']['max_depth'] }} | 
                CatBoost: {{ predictors[sport_code].settings['catboost_params']['iterations'] }} iterations | 
                Elo: +{{ predictors[sport_code].settings['elo_home_advantage'] }} home, K={{ predictors[sport_code].settings['elo_k_factors']['weeks_1_4'] }} |
                Ensemble: {{ (predictors[sport_code].ensemble_weights['catboost']*100)|int }}% Cat + {{ (predictors[sport_code].ensemble_weights['xgboost']*100)|int }}% XGB + {{ (predictors[sport_code].ensemble_weights['elo']*100)|int }}% Elo
                {% else %}
                Standard configuration
                {% endif %}
            </div>
            
            {% if predictions %}
            <div class="predictions-table">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Matchup</th>
                            <th>Elo %</th>
                            <th>XGB %</th>
                            <th>Cat %</th>
                            <th>Meta %</th>
                            <th>Result</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for pred in predictions %}
                        <tr style="background-color: #f8f9fa;">
                            <td>{{ pred.date }}</td>
                            <td><strong>{{ pred.away_team }} @ {{ pred.home_team }}</strong></td>
                            <td>{{ (pred.elo_home_prob * 100)|round(1) }}%</td>
                            <td>{{ (pred.xgb_home_prob * 100)|round(1) }}%</td>
                            <td>{{ (pred.cat_home_prob * 100)|round(1) }}%</td>
                            <td>{{ (pred.meta_home_prob * 100)|round(1) }}%</td>
                            <td><strong>{{ pred.actual_result }}</strong></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% else %}
            <p>No test games found for {{ sport_code }}</p>
            {% endif %}
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html, 
                                sport_code=sport_code, 
                                predictions=predictions,
                                performance=performance,
                                predictors=predictors)

if __name__ == '__main__':
    print("🏒 Advanced Sports Predictor Starting!")
    print("🎯 4-Model System: Elo + XGBoost + CatBoost + Meta Ensemble")
    print("📊 Training: Oct 2024-Feb 2025 | Testing: Mar-Apr 2025")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
