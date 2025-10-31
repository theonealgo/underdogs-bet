#!/usr/bin/env python3
"""
jackpotpicks.bet - Multi-Sport Prediction Platform
Unified Flask app serving all sports on port 5000
Each sport is completely separate with its own data and predictions
"""
from flask import Flask, render_template_string, redirect, url_for
import sys
import os

# Add schedules to path
sys.path.insert(0, 'schedules')

app = Flask(__name__)

# ============================================================================
# LANDING PAGE
# ============================================================================

LANDING_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>jackpotpicks.bet - Sports Predictions</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            width: 100%;
        }
        
        .header {
            text-align: center;
            margin-bottom: 50px;
            color: white;
        }
        
        .header h1 {
            font-size: 3.5em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.3em;
            opacity: 0.9;
        }
        
        .sports-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .sport-card {
            background: white;
            border-radius: 16px;
            padding: 35px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-decoration: none;
            color: inherit;
        }
        
        .sport-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .sport-icon {
            font-size: 4em;
            margin-bottom: 15px;
        }
        
        .sport-name {
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 8px;
            color: #333;
        }
        
        .sport-status {
            font-size: 1em;
            color: #666;
            margin-bottom: 12px;
        }
        
        .sport-accuracy {
            font-size: 1.4em;
            font-weight: 700;
            color: #667eea;
            margin-top: 10px;
        }
        
        .active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .active .sport-name,
        .active .sport-status {
            color: white;
        }
        
        .active .sport-accuracy {
            color: #fff;
            font-size: 1.6em;
        }
        
        .coming-soon {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .coming-soon:hover {
            transform: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .footer {
            text-align: center;
            color: white;
            margin-top: 40px;
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 jackpotpicks.bet</h1>
            <p>Professional Sports Predictions Powered by Machine Learning</p>
        </div>
        
        <div class="sports-grid">
            <a href="/nhl" class="sport-card active">
                <div class="sport-icon">🏒</div>
                <div class="sport-name">NHL</div>
                <div class="sport-status">Live Now</div>
                <div class="sport-accuracy">77% Accuracy</div>
            </a>
            
            <a href="/nfl" class="sport-card active">
                <div class="sport-icon">🏈</div>
                <div class="sport-name">NFL</div>
                <div class="sport-status">Live Now</div>
                <div class="sport-accuracy">84% Accuracy</div>
            </a>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">🏀</div>
                <div class="sport-name">NBA</div>
                <div class="sport-status">Coming Soon</div>
            </div>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">⚾</div>
                <div class="sport-name">MLB</div>
                <div class="sport-status">Coming Soon</div>
            </div>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">🏀</div>
                <div class="sport-name">WNBA</div>
                <div class="sport-status">Coming Soon</div>
            </div>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">🏟️</div>
                <div class="sport-name">NCAAF</div>
                <div class="sport-status">Coming Soon</div>
            </div>
        </div>
        
        <div class="footer">
            <p>Select a sport to view predictions, results, and analysis</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def landing_page():
    """Landing page with sport selector"""
    return render_template_string(LANDING_PAGE_TEMPLATE)

# ============================================================================
# NHL ROUTES (from NHL77FINAL.py)
# ============================================================================

# Import NHL functions
from NHL77FINAL import (
    get_all_nhl_predictions,
    calculate_nhl_performance,
    NHL_PREDICTIONS_TEMPLATE,
    NHL_RESULTS_TEMPLATE
)

@app.route('/nhl')
def nhl_predictions():
    """NHL predictions page"""
    predictions = get_all_nhl_predictions()
    return render_template_string(NHL_PREDICTIONS_TEMPLATE, 
                                 page='nhl', 
                                 sport='NHL',
                                 predictions=predictions)

@app.route('/nhl/results')
def nhl_results():
    """NHL results page"""
    performance = calculate_nhl_performance()
    if performance is None:
        return "No completed games yet", 404
    return render_template_string(NHL_RESULTS_TEMPLATE, 
                                 page='nhl_results',
                                 sport='NHL',
                                 performance=performance)

# ============================================================================
# NFL ROUTES (from NFL_2025.py)
# ============================================================================

# Import NFL functions
from NFL_2025 import (
    get_all_predictions as get_all_nfl_predictions,
    calculate_model_performance as calculate_nfl_performance,
    PREDICTIONS_TEMPLATE as NFL_PREDICTIONS_TEMPLATE,
    RESULTS_TEMPLATE as NFL_RESULTS_TEMPLATE
)

@app.route('/nfl')
def nfl_predictions():
    """NFL predictions page"""
    predictions = get_all_nfl_predictions()
    return render_template_string(NFL_PREDICTIONS_TEMPLATE,
                                 page='nfl',
                                 sport='NFL', 
                                 predictions=predictions)

@app.route('/nfl/results')
def nfl_results():
    """NFL results page"""
    performance = calculate_nfl_performance()
    if performance is None:
        return "No completed games yet", 404
    return render_template_string(NFL_RESULTS_TEMPLATE,
                                 page='nfl_results',
                                 sport='NFL',
                                 performance=performance)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 jackpotpicks.bet - Multi-Sport Prediction Platform")
    print("="*60)
    print("🏒 NHL Predictions - Live (77% Accuracy)")
    print("🏈 NFL Predictions - Live (84% Accuracy)")
    print("="*60)
    print("🌐 Visit http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
