#!/usr/bin/env python3
"""
jackpotpicks.bet Dashboard
Simple dashboard that links to NHL77FINAL and NFL72FINAL apps
"""

from flask import Flask, render_template_string

app = Flask(__name__)

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>jackpotpicks.bet - Multi-Sport Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #fff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            max-width: 1200px;
            padding: 40px;
            text-align: center;
        }
        .title {
            font-size: 48px;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #10b981, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #94a3b8;
            font-size: 20px;
            margin-bottom: 60px;
        }
        .sports-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 40px;
        }
        .sport-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 40px;
            border-radius: 16px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s;
            cursor: pointer;
            text-decoration: none;
            color: white;
            display: block;
        }
        .sport-card:hover {
            transform: translateY(-5px);
            border-color: #10b981;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        }
        .sport-icon {
            font-size: 64px;
            margin-bottom: 20px;
        }
        .sport-name {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sport-accuracy {
            color: #10b981;
            font-size: 20px;
            font-weight: bold;
        }
        .sport-status {
            color: #94a3b8;
            font-size: 14px;
            margin-top: 15px;
        }
        .coming-soon {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .coming-soon:hover {
            transform: none;
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">🎯 jackpotpicks.bet</h1>
        <p class="subtitle">Multi-Sport Prediction Platform</p>
        
        <div class="sports-grid">
            <a href="http://localhost:5000" class="sport-card" target="_blank">
                <div class="sport-icon">🏒</div>
                <div class="sport-name">NHL</div>
                <div class="sport-accuracy">77% Accuracy</div>
                <div class="sport-status">✓ Running on Port 5000</div>
            </a>
            
            <a href="http://localhost:5001" class="sport-card" target="_blank">
                <div class="sport-icon">🏈</div>
                <div class="sport-name">NFL</div>
                <div class="sport-accuracy">72% Elo | 67.4% Ensemble</div>
                <div class="sport-status">✓ Running on Port 5001</div>
            </a>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">🏀</div>
                <div class="sport-name">NBA</div>
                <div class="sport-accuracy">Coming Soon</div>
                <div class="sport-status">Next to implement</div>
            </div>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">⚾</div>
                <div class="sport-name">MLB</div>
                <div class="sport-accuracy">Coming Soon</div>
                <div class="sport-status">Future implementation</div>
            </div>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">🏟️</div>
                <div class="sport-name">NCAA Football</div>
                <div class="sport-accuracy">Coming Soon</div>
                <div class="sport-status">Future implementation</div>
            </div>
            
            <div class="sport-card coming-soon">
                <div class="sport-icon">🎓</div>
                <div class="sport-name">NCAA Basketball</div>
                <div class="sport-accuracy">Coming Soon</div>
                <div class="sport-status">Future implementation</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

if __name__ == '__main__':
    print("\n🎯 jackpotpicks.bet Dashboard")
    print("=" * 50)
    print("🏒 NHL: http://localhost:5000 (77% accuracy)")
    print("🏈 NFL: http://localhost:5001 (72% Elo, 67.4% ensemble)")
    print("=" * 50)
    print("📊 Dashboard: http://0.0.0.0:3000")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=3000)
