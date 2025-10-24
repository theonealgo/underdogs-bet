#!/usr/bin/env python3
"""
NFL Predictions App - 2025 Season
==================================
Loads games from schedules/nfl_schedule.py
Follows the working NHL77FINAL pattern
"""

from flask import Flask, render_template_string
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add schedules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'schedules'))
from nfl_schedule import get_nfl_schedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DATABASE = 'sports_predictions_original.db'

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def parse_date(date_str):
    """Parse DD/MM/YYYY date string (handles timestamps like '05/09/2025 00:20')"""
    try:
        # Strip timestamp if present (everything after space)
        date_only = date_str.split(' ')[0] if ' ' in date_str else date_str
        return datetime.strptime(date_only, '%d/%m/%Y')
    except:
        return None

def parse_result(result_str):
    """Parse result string like '24 - 20' into (home_score, away_score)"""
    if not result_str or result_str == '':
        return None, None
    try:
        parts = result_str.split(' - ')
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
    except:
        pass
    return None, None

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_all_predictions():
    """Get ALL NFL game predictions from schedules/nfl_schedule.py
    
    Loads games from schedule file (like NHL77FINAL does)
    Enriches with predictions from database
    """
    
    # Load from schedules/nfl_schedule.py
    nfl_schedule = get_nfl_schedule()
    
    # Load predictions from database
    conn = get_db_connection()
    predictions_raw = conn.execute('''
        SELECT game_date, home_team_id, away_team_id,
               elo_home_prob, xgboost_home_prob, logistic_home_prob, win_probability as ensemble_prob
        FROM predictions
        WHERE sport = 'NFL'
    ''').fetchall()
    predictions_raw = [dict(p) for p in predictions_raw]
    conn.close()
    
    # Create lookup for predictions
    pred_lookup = {}
    for pred in predictions_raw:
        key = f"{pred['game_date']}_{pred['home_team_id']}_{pred['away_team_id']}"
        pred_lookup[key] = pred
    
    # PARSE AND SORT ALL games by actual date
    all_games_with_dates = []
    for game in nfl_schedule:
        parsed_date = parse_date(game['date'])
        if parsed_date:
            all_games_with_dates.append((parsed_date, game))
    all_games_with_dates.sort(key=lambda x: x[0])  # Sort by parsed date
    
    # Format predictions for display
    predictions = []
    for game_date, game in all_games_with_dates:
        # Parse scores from result
        home_score, away_score = parse_result(game.get('result', ''))
        
        # Get stored predictions if available
        pred_key = f"{game['date']}_{game['home_team']}_{game['away_team']}"
        pred_data = pred_lookup.get(pred_key, {})
        
        game_dict = {
            'game_date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_score': home_score,
            'away_score': away_score,
            'venue': game.get('venue', '')
        }
        
        # Use stored probabilities if available
        if pred_data.get('elo_home_prob') is not None:
            game_dict['elo_prob'] = round(pred_data['elo_home_prob'] * 100, 1)
            game_dict['xgb_prob'] = round(pred_data['xgboost_home_prob'] * 100, 1) if pred_data.get('xgboost_home_prob') else 50.0
            game_dict['cat_prob'] = round(pred_data['logistic_home_prob'] * 100, 1) if pred_data.get('logistic_home_prob') else 50.0
            game_dict['ensemble_prob'] = round(pred_data['ensemble_prob'] * 100, 1) if pred_data.get('ensemble_prob') else 50.0
        else:
            # No predictions available
            game_dict['elo_prob'] = 50.0
            game_dict['xgb_prob'] = 50.0
            game_dict['cat_prob'] = 50.0
            game_dict['ensemble_prob'] = 50.0
        
        # Determine predictions
        game_dict['elo_pick'] = game['home_team'] if game_dict['elo_prob'] > 50 else game['away_team']
        game_dict['xgb_pick'] = game['home_team'] if game_dict['xgb_prob'] > 50 else game['away_team']
        game_dict['cat_pick'] = game['home_team'] if game_dict['cat_prob'] > 50 else game['away_team']
        game_dict['ensemble_pick'] = game['home_team'] if game_dict['ensemble_prob'] > 50 else game['away_team']
        
        # Determine winner if completed
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                game_dict['winner'] = game['home_team']
                game_dict['completed'] = True
            elif away_score > home_score:
                game_dict['winner'] = game['away_team']
                game_dict['completed'] = True
            else:
                game_dict['winner'] = 'TIE'
                game_dict['completed'] = True
        else:
            game_dict['winner'] = None
            game_dict['completed'] = False
        
        predictions.append(game_dict)
    
    return predictions

def calculate_model_performance():
    """Calculate performance using games from schedules/nfl_schedule.py"""
    
    # Load all games
    all_predictions = get_all_predictions()
    
    # Filter to completed games only
    completed_games = [g for g in all_predictions if g['completed']]
    
    if len(completed_games) == 0:
        return None
    
    # Calculate accuracy for each model
    results = {
        'elo': {'correct': 0, 'total': 0},
        'xgboost': {'correct': 0, 'total': 0},
        'catboost': {'correct': 0, 'total': 0},
        'ensemble': {'correct': 0, 'total': 0}
    }
    
    dates = []
    
    for game in completed_games:
        actual_winner = game['winner']
        if actual_winner == 'TIE':
            continue  # Skip ties for accuracy calculation
        
        # Elo
        if game['elo_pick'] == actual_winner:
            results['elo']['correct'] += 1
        results['elo']['total'] += 1
        
        # XGBoost
        if game['xgb_pick'] == actual_winner:
            results['xgboost']['correct'] += 1
        results['xgboost']['total'] += 1
        
        # CatBoost
        if game['cat_pick'] == actual_winner:
            results['catboost']['correct'] += 1
        results['catboost']['total'] += 1
        
        # Ensemble
        if game['ensemble_pick'] == actual_winner:
            results['ensemble']['correct'] += 1
        results['ensemble']['total'] += 1
        
        # Track dates
        dates.append(parse_date(game['game_date']))
    
    # Calculate accuracies
    performance = {}
    for model in ['elo', 'xgboost', 'catboost', 'ensemble']:
        total = results[model]['total']
        if total > 0:
            acc = (results[model]['correct'] / total * 100)
            performance[model] = {
                'accuracy': round(acc, 1),
                'correct': results[model]['correct'],
                'total': total
            }
        else:
            performance[model] = {'accuracy': 0.0, 'correct': 0, 'total': 0}
    
    # Date range
    valid_dates = [d for d in dates if d is not None]
    if valid_dates:
        min_date = min(valid_dates).strftime('%d/%m/%Y')
        max_date = max(valid_dates).strftime('%d/%m/%Y')
        performance['date_range'] = f"{min_date} - {max_date}"
        performance['total_games'] = len(completed_games)
    
    return performance

# ============================================================================
# TEMPLATES
# ============================================================================

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NFL Predictions - 2025 Season</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #fff;
            min-height: 100vh;
        }
        .navbar {
            background: rgba(15, 23, 42, 0.95);
            padding: 15px 30px;
            border-bottom: 2px solid #059669;
        }
        .navbar-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
            text-decoration: none;
        }
        .nav-links {
            display: flex;
            gap: 20px;
        }
        .nav-links a {
            color: #94a3b8;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .nav-links a:hover, .nav-links a.active {
            background: #059669;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .page-title {
            font-size: 36px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #10b981, #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 18px;
            margin-bottom: 40px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat-label {
            color: #94a3b8;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #10b981;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-content">
            <a href="/" class="logo">🏈 NFL Predictions</a>
            <div class="nav-links">
                <a href="/" class="active">Home</a>
                <a href="/predictions">Predictions</a>
                <a href="/results">Results</a>
            </div>
        </div>
    </nav>
    <div class="container">
        <h1 class="page-title">NFL Predictions Dashboard</h1>
        <p class="subtitle">2025 Season - All Games from Schedule</p>
    </div>
</body>
</html>
"""

PREDICTIONS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NFL Predictions - 2025 Season</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #fff;
            min-height: 100vh;
        }
        .navbar {
            background: rgba(15, 23, 42, 0.95);
            padding: 15px 30px;
            border-bottom: 2px solid #059669;
        }
        .navbar-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
            text-decoration: none;
        }
        .nav-links {
            display: flex;
            gap: 20px;
        }
        .nav-links a {
            color: #94a3b8;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .nav-links a:hover, .nav-links a.active {
            background: #059669;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .page-title {
            font-size: 36px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #10b981, #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 18px;
            margin-bottom: 40px;
        }
        .predictions-table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            overflow: hidden;
        }
        .predictions-table th {
            background: rgba(5, 150, 105, 0.2);
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            color: #10b981;
        }
        .predictions-table td {
            padding: 12px 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 13px;
        }
        .matchup {
            font-weight: 500;
            color: #e2e8f0;
        }
        .score {
            color: #10b981;
            font-weight: bold;
        }
        .completed { color: #22c55e; }
        .upcoming { color: #94a3b8; }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-content">
            <a href="/" class="logo">🏈 NFL Predictions</a>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/predictions" class="active">Predictions</a>
                <a href="/results">Results</a>
            </div>
        </div>
    </nav>
    <div class="container">
        <h1 class="page-title">All NFL Predictions</h1>
        <p class="subtitle">{{ predictions|length }} games - 2025 Season</p>
        
        <table class="predictions-table">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Matchup</th>
                    <th>Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for pred in predictions %}
                <tr>
                    <td>{{ pred.game_date }}</td>
                    <td class="matchup">{{ pred.away_team }} @ {{ pred.home_team }}</td>
                    <td class="score">
                        {% if pred.completed %}
                            {{ pred.away_score }} - {{ pred.home_score }}
                        {% else %}
                            -
                        {% endif %}
                    </td>
                    <td class="{{ 'completed' if pred.completed else 'upcoming' }}">
                        {% if pred.completed %}✓ Completed{% else %}Upcoming{% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NFL Results - 2025 Season</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #fff;
            min-height: 100vh;
        }
        .navbar {
            background: rgba(15, 23, 42, 0.95);
            padding: 15px 30px;
            border-bottom: 2px solid #059669;
        }
        .navbar-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
            text-decoration: none;
        }
        .nav-links {
            display: flex;
            gap: 20px;
        }
        .nav-links a {
            color: #94a3b8;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.2s;
        }
        .nav-links a:hover, .nav-links a.active {
            background: #059669;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .page-title {
            font-size: 36px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #10b981, #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 18px;
            margin-bottom: 40px;
        }
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        .model-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .model-name {
            color: #94a3b8;
            font-size: 14px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .accuracy {
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .excellent { color: #10b981; }
        .good { color: #3b82f6; }
        .fair { color: #f59e0b; }
        .record {
            color: #94a3b8;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-content">
            <a href="/" class="logo">🏈 NFL Predictions</a>
            <div class="nav-links">
                <a href="/">Home</a>
                <a href="/predictions">Predictions</a>
                <a href="/results" class="active">Results</a>
            </div>
        </div>
    </nav>
    <div class="container">
        <h1 class="page-title">Model Performance</h1>
        <p class="subtitle">{{ performance.total_games }} completed games ({{ performance.date_range }})</p>
        
        <div class="models-grid">
            <div class="model-card">
                <div class="model-name">Elo</div>
                <div class="accuracy {{ 'excellent' if performance.elo.accuracy >= 70 else 'good' if performance.elo.accuracy >= 60 else 'fair' }}">
                    {{ performance.elo.accuracy }}%
                </div>
                <div class="record">{{ performance.elo.correct }}-{{ performance.elo.total - performance.elo.correct }}</div>
            </div>
            
            <div class="model-card">
                <div class="model-name">XGBoost</div>
                <div class="accuracy {{ 'excellent' if performance.xgboost.accuracy >= 70 else 'good' if performance.xgboost.accuracy >= 60 else 'fair' }}">
                    {{ performance.xgboost.accuracy }}%
                </div>
                <div class="record">{{ performance.xgboost.correct }}-{{ performance.xgboost.total - performance.xgboost.correct }}</div>
            </div>
            
            <div class="model-card">
                <div class="model-name">CatBoost</div>
                <div class="accuracy {{ 'excellent' if performance.catboost.accuracy >= 70 else 'good' if performance.catboost.accuracy >= 60 else 'fair' }}">
                    {{ performance.catboost.accuracy }}%
                </div>
                <div class="record">{{ performance.catboost.correct }}-{{ performance.catboost.total - performance.catboost.correct }}</div>
            </div>
            
            <div class="model-card">
                <div class="model-name">Ensemble</div>
                <div class="accuracy {{ 'excellent' if performance.ensemble.accuracy >= 70 else 'good' if performance.ensemble.accuracy >= 60 else 'fair' }}">
                    {{ performance.ensemble.accuracy }}%
                </div>
                <div class="record">{{ performance.ensemble.correct }}-{{ performance.ensemble.total - performance.ensemble.correct }}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page"""
    return render_template_string(HOME_TEMPLATE, page='home')

@app.route('/predictions')
def predictions():
    """Predictions page"""
    preds = get_all_predictions()
    return render_template_string(PREDICTIONS_TEMPLATE, page='predictions', predictions=preds)

@app.route('/results')
def results():
    """Results page"""
    performance = calculate_model_performance()
    if performance is None:
        return "No completed games yet", 404
    return render_template_string(RESULTS_TEMPLATE, page='results', performance=performance)

if __name__ == '__main__':
    print("🏈 NFL Predictions - 2025 Season")
    print("📅 Loads games from schedules/nfl_schedule.py")
    print("✓ Platform ready!")
    print("🌐 Visit http://0.0.0.0:5001\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
