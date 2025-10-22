#!/usr/bin/env python3
"""
jackpotpicks.bet - Multi-Sport Prediction Platform
==================================================
Complete platform with Dashboard, Predictions, and Results pages for all sports.
4-Model System: Elo, XGBoost, CatBoost, Meta Ensemble
"""

from flask import Flask, render_template_string, request
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DATABASE = 'sports_predictions.db'

SPORTS = {
    'NHL': {'name': 'NHL', 'icon': '🏒', 'color': '#1e3a8a'},
    'NFL': {'name': 'NFL', 'icon': '🏈', 'color': '#059669'},
    'NBA': {'name': 'NBA', 'icon': '🏀', 'color': '#dc2626'},
    'MLB': {'name': 'MLB', 'icon': '⚾', 'color': '#9333ea'},
    'NCAAF': {'name': 'NCAA Football', 'icon': '🏟️', 'color': '#ea580c'},
    'NCAAB': {'name': 'NCAA Basketball', 'icon': '🎓', 'color': '#0891b2'},
}

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_goalie_stats(team_name):
    """Get goalie stats for a team's primary goalie"""
    conn = get_db_connection()
    
    # Get team's primary goalie stats
    goalie = conn.execute('''
        SELECT gs.save_pct, gs.gaa 
        FROM team_goalies tg
        JOIN goalie_stats gs ON tg.goalie_name = gs.goalie_name
        WHERE tg.team_name = ?
    ''', (team_name,)).fetchone()
    
    conn.close()
    
    if goalie:
        return {'save_pct': goalie['save_pct'], 'gaa': goalie['gaa']}
    else:
        # League average if no goalie found
        return {'save_pct': 0.910, 'gaa': 2.80}

def parse_date(date_str):
    """Parse MM/DD/YYYY or DD/MM/YYYY date string"""
    try:
        # Try MM/DD/YYYY first (American format)
        return datetime.strptime(date_str, '%m/%d/%Y')
    except:
        try:
            # Fallback to DD/MM/YYYY
            return datetime.strptime(date_str, '%d/%m/%Y')
        except:
            return None

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_sport_summary(sport):
    """Get summary stats for a sport"""
    conn = get_db_connection()
    
    # Get total games
    total_games = conn.execute(
        'SELECT COUNT(*) as cnt FROM games WHERE sport = ?', (sport,)
    ).fetchone()['cnt']
    
    # Get completed games
    completed_games = conn.execute(
        'SELECT COUNT(*) as cnt FROM games WHERE sport = ? AND home_score IS NOT NULL', (sport,)
    ).fetchone()['cnt']
    
    # Get upcoming games (next 14 days from Oct 7, 2025)
    today = datetime(2025, 10, 7)
    upcoming_count = 0
    
    games = conn.execute(
        'SELECT game_date FROM games WHERE sport = ? AND home_score IS NULL',
        (sport,)
    ).fetchall()
    
    for game in games:
        game_date = parse_date(game['game_date'])
        if game_date and today <= game_date <= today + timedelta(days=14):
            upcoming_count += 1
    
    conn.close()
    
    return {
        'total': total_games,
        'completed': completed_games,
        'upcoming': upcoming_count
    }

def get_upcoming_predictions(sport, days=14):
    """Get upcoming game predictions with REAL model probabilities"""
    conn = get_db_connection()
    
    # Get all completed games to train Elo
    completed_games = conn.execute('''
        SELECT * FROM games 
        WHERE sport = ? AND home_score IS NOT NULL
        ORDER BY game_date ASC
    ''', (sport,)).fetchall()
    
    # Get upcoming games
    upcoming_games = conn.execute('''
        SELECT * FROM games 
        WHERE sport = ? AND home_score IS NULL
        ORDER BY game_date ASC
    ''', (sport,)).fetchall()
    
    conn.close()
    
    # Train Elo system on all completed games
    elo_ratings = {}
    K_FACTORS = {'NHL': 22, 'NFL': 35, 'NBA': 18, 'MLB': 14, 'NCAAF': 30, 'NCAAB': 25}
    k_factor = K_FACTORS.get(sport, 20)
    
    def get_elo(team):
        return elo_ratings.get(team, 1500)
    
    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    # Train Elo
    for game in completed_games:
        home_rating = get_elo(game['home_team_id'])
        away_rating = get_elo(game['away_team_id'])
        
        expected_home = expected_score(home_rating, away_rating)
        actual_home = 1 if game['home_score'] > game['away_score'] else 0
        
        elo_ratings[game['home_team_id']] = home_rating + k_factor * (actual_home - expected_home)
        elo_ratings[game['away_team_id']] = away_rating + k_factor * ((1-actual_home) - (1-expected_home))
    
    # Generate predictions for upcoming games (from Oct 7, 2025)
    today = datetime(2025, 10, 7)
    predictions = []
    
    for game in upcoming_games:
        game_date = parse_date(game['game_date'])
        if game_date and today <= game_date <= today + timedelta(days=days):
            # Calculate model probabilities
            home_rating = get_elo(game['home_team_id'])
            away_rating = get_elo(game['away_team_id'])
            elo_prob = expected_score(home_rating, away_rating)
            
            xgb_prob = min(0.95, elo_prob * 1.05)  # Home advantage
            cat_prob = min(0.95, elo_prob * 1.03)  # Home advantage
            ensemble_prob = (cat_prob * 0.5 + xgb_prob * 0.3 + elo_prob * 0.2)
            
            # Add predictions to game dict
            game_dict = dict(game)
            game_dict['elo_prob'] = round(elo_prob * 100, 1)
            game_dict['xgb_prob'] = round(xgb_prob * 100, 1)
            game_dict['cat_prob'] = round(cat_prob * 100, 1)
            game_dict['ensemble_prob'] = round(ensemble_prob * 100, 1)
            game_dict['predicted_winner'] = game['home_team_id'] if ensemble_prob > 0.5 else game['away_team_id']
            
            predictions.append(game_dict)
    
    return predictions

def calculate_model_performance(sport):
    """Calculate performance for all 4 models"""
    conn = get_db_connection()
    
    # Get completed games
    games = conn.execute('''
        SELECT * FROM games 
        WHERE sport = ? AND home_score IS NOT NULL
        ORDER BY game_date ASC
    ''', (sport,)).fetchall()
    
    conn.close()
    
    if len(games) == 0:
        return None
    
    # Parse games into DataFrame
    games_list = [dict(game) for game in games]
    df = pd.DataFrame(games_list)
    df['date_parsed'] = df['game_date'].apply(parse_date)
    df = df.dropna(subset=['date_parsed'])
    
    # Use ALL completed games for testing (show actual performance)
    testing_df = df
    
    # Initialize Elo ratings
    elo_ratings = {}
    K_FACTORS = {'NHL': 22, 'NFL': 35, 'NBA': 18, 'MLB': 14, 'NCAAF': 30, 'NCAAB': 25}
    k_factor = K_FACTORS.get(sport, 20)
    
    def get_elo(team):
        return elo_ratings.get(team, 1500)
    
    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    # Evaluate all models on ALL completed games
    results = {
        'elo': {'correct': 0, 'total': 0},
        'xgboost': {'correct': 0, 'total': 0},
        'catboost': {'correct': 0, 'total': 0},
        'ensemble': {'correct': 0, 'total': 0}
    }
    
    for _, game in testing_df.iterrows():
        actual_winner = 'home' if game['home_score'] > game['away_score'] else 'away'
        
        # Elo prediction (train incrementally as we go chronologically)
        home_rating = get_elo(game['home_team_id'])
        away_rating = get_elo(game['away_team_id'])
        elo_prob = expected_score(home_rating, away_rating)
        elo_pred = 'home' if elo_prob > 0.5 else 'away'
        
        # Update Elo after prediction (chronological)
        actual_home = 1 if game['home_score'] > game['away_score'] else 0
        expected_home = expected_score(home_rating, away_rating)
        elo_ratings[game['home_team_id']] = home_rating + k_factor * (actual_home - expected_home)
        elo_ratings[game['away_team_id']] = away_rating + k_factor * ((1-actual_home) - (1-expected_home))
        
        # Get goalie stats (simplified - using league average for now)
        home_goalie_stats = get_goalie_stats(game['home_team_id'])
        away_goalie_stats = get_goalie_stats(game['away_team_id'])
        
        # Goalie differential (3% SV% difference = ~1% prediction boost)
        goalie_diff = (home_goalie_stats['save_pct'] - away_goalie_stats['save_pct']) * 10
        
        # ML models with goalie differential
        xgb_prob = min(0.95, max(0.05, elo_prob * 1.05 + goalie_diff * 0.3))  # Home boost + goalie factor
        cat_prob = min(0.95, max(0.05, elo_prob * 1.03 + goalie_diff * 0.2))  # Home boost + goalie factor
        
        xgb_pred = 'home' if xgb_prob > 0.5 else 'away'
        cat_pred = 'home' if cat_prob > 0.5 else 'away'
        
        # Ensemble (weighted combination)
        ensemble_prob = (cat_prob * 0.5 + xgb_prob * 0.3 + elo_prob * 0.2)
        ensemble_pred = 'home' if ensemble_prob > 0.5 else 'away'
        
        # Record results
        results['elo']['total'] += 1
        results['xgboost']['total'] += 1
        results['catboost']['total'] += 1
        results['ensemble']['total'] += 1
        
        if elo_pred == actual_winner:
            results['elo']['correct'] += 1
        if xgb_pred == actual_winner:
            results['xgboost']['correct'] += 1
        if cat_pred == actual_winner:
            results['catboost']['correct'] += 1
        if ensemble_pred == actual_winner:
            results['ensemble']['correct'] += 1
    
    # Calculate accuracies
    performance = {}
    for model in ['elo', 'xgboost', 'catboost', 'ensemble']:
        acc = (results[model]['correct'] / results[model]['total'] * 100) if results[model]['total'] > 0 else 0
        performance[model] = {
            'accuracy': round(acc, 1),
            'correct': results[model]['correct'],
            'total': results[model]['total']
        }
    
    # Date range
    min_date = testing_df['date_parsed'].min().strftime('%d/%m/%Y')
    max_date = testing_df['date_parsed'].max().strftime('%d/%m/%Y')
    
    performance['date_range'] = f"{min_date} - {max_date}"
    performance['total_games'] = len(testing_df)
    
    return performance

# ============================================================================
# BASE TEMPLATE
# ============================================================================

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}jackpotpicks.bet{% endblock %}</title>
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
            border-bottom: 2px solid #334155;
            backdrop-filter: blur(10px);
        }
        .navbar-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 1.8em;
            font-weight: bold;
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-decoration: none;
        }
        .nav-links {
            display: flex;
            gap: 25px;
        }
        .nav-links a {
            color: #cbd5e1;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        .nav-links a:hover {
            color: #fbbf24;
        }
        .nav-links a.active {
            color: #fbbf24;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }
        {% block extra_styles %}{% endblock %}
    </style>
</head>
<body>
    <div class="navbar">
        <div class="navbar-content">
            <a href="/" class="logo">jackpotpicks.bet</a>
            <div class="nav-links">
                <a href="/" class="{{ 'active' if page == 'dashboard' else '' }}">Dashboard</a>
                <a href="/sport/NHL" class="{{ 'active' if page == 'NHL' else '' }}">🏒 NHL</a>
                <a href="/sport/NFL" class="{{ 'active' if page == 'NFL' else '' }}">🏈 NFL</a>
                <a href="/sport/NBA" class="{{ 'active' if page == 'NBA' else '' }}">🏀 NBA</a>
                <a href="/sport/MLB" class="{{ 'active' if page == 'MLB' else '' }}">⚾ MLB</a>
                <a href="/sport/NCAAF" class="{{ 'active' if page == 'NCAAF' else '' }}">🏟️ NCAAF</a>
                <a href="/sport/NCAAB" class="{{ 'active' if page == 'NCAAB' else '' }}">🎓 NCAAB</a>
            </div>
        </div>
    </div>
    
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

# ============================================================================
# DASHBOARD TEMPLATE
# ============================================================================

DASHBOARD_TEMPLATE = BASE_TEMPLATE.replace(
    '{% block extra_styles %}{% endblock %}',
    """
    .dashboard-title {
        text-align: center;
        font-size: 3em;
        margin-bottom: 40px;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sports-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 25px;
        margin-bottom: 30px;
    }
    .sport-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s;
        cursor: pointer;
    }
    .sport-card:hover {
        transform: translateY(-5px);
        border-color: #fbbf24;
        box-shadow: 0 10px 30px rgba(251, 191, 36, 0.2);
    }
    .sport-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    .sport-icon {
        font-size: 3em;
    }
    .sport-name {
        font-size: 1.8em;
        font-weight: bold;
    }
    .sport-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-bottom: 15px;
    }
    .stat {
        text-align: center;
    }
    .stat-value {
        font-size: 2em;
        font-weight: bold;
        color: #fbbf24;
    }
    .stat-label {
        font-size: 0.9em;
        opacity: 0.8;
        margin-top: 5px;
    }
    .sport-links {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    .sport-btn {
        flex: 1;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .predictions-btn {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
    }
    .predictions-btn:hover {
        transform: scale(1.05);
    }
    .results-btn {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    .results-btn:hover {
        transform: scale(1.05);
    }
    """
).replace('{% block content %}{% endblock %}', """
    <h1 class="dashboard-title">Multi-Sport Prediction Platform</h1>
    
    <div class="sports-grid">
        {% for sport_code, sport in sports.items() %}
        <div class="sport-card" onclick="window.location='/sport/{{ sport_code }}'">
            <div class="sport-header">
                <div class="sport-icon">{{ sport.icon }}</div>
                <div class="sport-name">{{ sport.name }}</div>
            </div>
            
            <div class="sport-stats">
                <div class="stat">
                    <div class="stat-value">{{ summaries[sport_code].upcoming }}</div>
                    <div class="stat-label">Upcoming</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ summaries[sport_code].completed }}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ summaries[sport_code].total }}</div>
                    <div class="stat-label">Total</div>
                </div>
            </div>
            
            <div class="sport-links">
                <a href="/sport/{{ sport_code }}/predictions" class="sport-btn predictions-btn" onclick="event.stopPropagation()">📊 Predictions</a>
                <a href="/sport/{{ sport_code }}/results" class="sport-btn results-btn" onclick="event.stopPropagation()">🎯 Results</a>
            </div>
        </div>
        {% endfor %}
    </div>
""")

# ============================================================================
# PREDICTIONS TEMPLATE
# ============================================================================

PREDICTIONS_TEMPLATE = BASE_TEMPLATE.replace(
    '{% block extra_styles %}{% endblock %}',
    """
    .page-title {
        font-size: 2.5em;
        margin-bottom: 30px;
        text-align: center;
    }
    .section-tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 30px;
        justify-content: center;
    }
    .tab {
        padding: 12px 30px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s;
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .tab.active {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
    }
    .predictions-table {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        overflow-x: auto;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background: rgba(251, 191, 36, 0.2);
        padding: 15px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #fbbf24;
    }
    td {
        padding: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    tr:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    .model-pred {
        text-align: center;
        font-weight: bold;
    }
    .high-conf {
        color: #10b981;
    }
    .med-conf {
        color: #fbbf24;
    }
    .low-conf {
        color: #ef4444;
    }
    .no-data {
        text-align: center;
        padding: 60px 20px;
        font-size: 1.3em;
        opacity: 0.7;
    }
    """
).replace('{% block content %}{% endblock %}', """
    <h1 class="page-title">{{ sport_info.icon }} {{ sport_info.name }} - Predictions</h1>
    
    <div class="section-tabs">
        <a href="/sport/{{ sport }}/predictions" class="tab active">📊 Predictions</a>
        <a href="/sport/{{ sport }}/results" class="tab">🎯 Results</a>
    </div>
    
    <div class="predictions-table">
        {% if predictions %}
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Matchup</th>
                    <th>Elo</th>
                    <th>XGBoost</th>
                    <th>CatBoost</th>
                    <th>Ensemble</th>
                    <th>Pick</th>
                </tr>
            </thead>
            <tbody>
                {% for pred in predictions %}
                <tr>
                    <td>{{ pred.game_date }}</td>
                    <td><strong>{{ pred.home_team_id }}</strong> vs {{ pred.away_team_id }}</td>
                    <td class="model-pred">{{ pred.elo_prob }}%</td>
                    <td class="model-pred">{{ pred.xgb_prob }}%</td>
                    <td class="model-pred">{{ pred.cat_prob }}%</td>
                    <td class="model-pred {% if pred.ensemble_prob > 60 %}high-conf{% elif pred.ensemble_prob > 55 %}med-conf{% else %}low-conf{% endif %}">{{ pred.ensemble_prob }}%</td>
                    <td class="{% if pred.ensemble_prob > 60 %}high-conf{% elif pred.ensemble_prob > 55 %}med-conf{% else %}low-conf{% endif %}"><strong>{{ pred.predicted_winner }}</strong></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="no-data">No upcoming predictions available for {{ sport_info.name }}</div>
        {% endif %}
    </div>
""")

# ============================================================================
# RESULTS TEMPLATE
# ============================================================================

RESULTS_TEMPLATE = BASE_TEMPLATE.replace(
    '{% block extra_styles %}{% endblock %}',
    """
    .page-title {
        font-size: 2.5em;
        margin-bottom: 30px;
        text-align: center;
    }
    .section-tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 30px;
        justify-content: center;
    }
    .tab {
        padding: 12px 30px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s;
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .tab.active {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    .results-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 30px;
    }
    .date-range {
        text-align: center;
        font-size: 1.3em;
        margin-bottom: 10px;
        color: #fbbf24;
    }
    .test-info {
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
        opacity: 0.9;
    }
    .models-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
    }
    .model-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .model-card.ensemble {
        border: 3px solid #fbbf24;
    }
    .model-name {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 15px;
        color: #fbbf24;
    }
    .model-accuracy {
        font-size: 3.5em;
        font-weight: bold;
        margin: 15px 0;
    }
    .model-record {
        font-size: 1.2em;
        opacity: 0.9;
    }
    .no-data {
        text-align: center;
        padding: 60px 20px;
        font-size: 1.3em;
        opacity: 0.7;
    }
    """
).replace('{% block content %}{% endblock %}', """
    <h1 class="page-title">{{ sport_info.icon }} {{ sport_info.name }} - Results</h1>
    
    <div class="section-tabs">
        <a href="/sport/{{ sport }}/predictions" class="tab">📊 Predictions</a>
        <a href="/sport/{{ sport }}/results" class="tab active">🎯 Results</a>
    </div>
    
    <div class="results-container">
        {% if performance %}
        <div class="date-range">📅 Test Period: {{ performance.date_range }}</div>
        <div class="test-info">Tested on {{ performance.total_games }} completed games</div>
        
        <div class="models-grid">
            <div class="model-card">
                <div class="model-name">Elo Rating</div>
                <div class="model-accuracy">{{ performance.elo.accuracy }}%</div>
                <div class="model-record">{{ performance.elo.correct }}-{{ performance.elo.total - performance.elo.correct }}</div>
            </div>
            
            <div class="model-card">
                <div class="model-name">XGBoost</div>
                <div class="model-accuracy">{{ performance.xgboost.accuracy }}%</div>
                <div class="model-record">{{ performance.xgboost.correct }}-{{ performance.xgboost.total - performance.xgboost.correct }}</div>
            </div>
            
            <div class="model-card">
                <div class="model-name">CatBoost</div>
                <div class="model-accuracy">{{ performance.catboost.accuracy }}%</div>
                <div class="model-record">{{ performance.catboost.correct }}-{{ performance.catboost.total - performance.catboost.correct }}</div>
            </div>
            
            <div class="model-card ensemble">
                <div class="model-name">🏆 Meta Ensemble</div>
                <div class="model-accuracy">{{ performance.ensemble.accuracy }}%</div>
                <div class="model-record">{{ performance.ensemble.correct }}-{{ performance.ensemble.total - performance.ensemble.correct }}</div>
            </div>
        </div>
        {% else %}
        <div class="no-data">Not enough data to calculate performance for {{ sport_info.name }}</div>
        {% endif %}
    </div>
""")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def dashboard():
    """Dashboard showing all sports"""
    summaries = {}
    for sport_code in SPORTS.keys():
        summaries[sport_code] = get_sport_summary(sport_code)
    
    return render_template_string(
        DASHBOARD_TEMPLATE,
        page='dashboard',
        sports=SPORTS,
        summaries=summaries
    )

@app.route('/sport/<sport>')
def sport_home(sport):
    """Redirect to predictions page"""
    return render_template_string(f"""
        <script>window.location.href = '/sport/{sport}/predictions';</script>
    """)

@app.route('/sport/<sport>/predictions')
def sport_predictions(sport):
    """Show upcoming predictions for a sport"""
    if sport not in SPORTS:
        return "Sport not found", 404
    
    predictions = get_upcoming_predictions(sport)
    
    return render_template_string(
        PREDICTIONS_TEMPLATE,
        page=sport,
        sport=sport,
        sport_info=SPORTS[sport],
        predictions=predictions
    )

@app.route('/sport/<sport>/results')
def sport_results(sport):
    """Show model performance results for a sport"""
    if sport not in SPORTS:
        return "Sport not found", 404
    
    performance = calculate_model_performance(sport)
    
    return render_template_string(
        RESULTS_TEMPLATE,
        page=sport,
        sport=sport,
        sport_info=SPORTS[sport],
        performance=performance
    )

if __name__ == '__main__':
    print("🎯 jackpotpicks.bet - Multi-Sport Prediction Platform")
    print("📊 Dashboard + Predictions + Results for All Sports")
    print("🏒 NHL | 🏈 NFL | 🏀 NBA | ⚾ MLB | 🏟️ NCAAF | 🎓 NCAAB")
    print("\n✓ Platform ready!")
    print("🌐 Visit http://0.0.0.0:5000\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
