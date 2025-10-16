"""
jackpotpicks.bet - Flask Application
Professional Sports Prediction Platform
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from src.data_storage.database import DatabaseManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'unitduel-secret-key-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id, username, email, subscription_status):
        self.id = user_id
        self.username = username
        self.email = email
        self.subscription_status = subscription_status
    
    def is_premium(self):
        return self.subscription_status == 'premium'

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('purepicks_users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, username, email, subscription_status FROM users WHERE user_id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[2], user[3])
    return None

def init_db():
    """Initialize users database"""
    conn = sqlite3.connect('purepicks_users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            subscription_status TEXT DEFAULT 'free',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stripe_customer_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_top_free_pick():
    """Get the best prediction for today as free pick"""
    db = DatabaseManager()
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    try:
        with db._get_connection() as conn:
            query = """
                SELECT 
                    p.game_id, p.sport, p.game_date,
                    g.home_team_id, g.away_team_id,
                    p.elo_home_prob, p.logistic_home_prob, p.xgboost_home_prob,
                    p.win_probability, p.predicted_winner
                FROM predictions p
                JOIN games g ON p.game_id = g.game_id
                WHERE DATE(p.game_date) >= DATE(?)
                AND DATE(p.game_date) <= DATE(?)
                AND p.xgboost_home_prob IS NOT NULL
                ORDER BY ABS(p.win_probability - 0.5) DESC
                LIMIT 1
            """
            df = pd.read_sql_query(query, conn, params=[
                today.strftime('%Y-%m-%d'),
                tomorrow.strftime('%Y-%m-%d')
            ])
            
            if not df.empty:
                pick = df.iloc[0]
                
                elo = float(pick['elo_home_prob']) if pick['elo_home_prob'] else 0.5
                logistic = float(pick['logistic_home_prob']) if pick['logistic_home_prob'] else 0.5
                xgboost = float(pick['xgboost_home_prob']) if pick['xgboost_home_prob'] else 0.5
                
                # CompositeHome = (XGB% * w1) + (Elo% * w2) + (Consensus% * w3)
                blended = (0.50 * xgboost + 0.35 * elo + 0.15 * logistic)
                
                if blended > 0.5:
                    pick_team = pick['home_team_id']
                    confidence = blended * 100
                else:
                    pick_team = pick['away_team_id']
                    confidence = (1 - blended) * 100
                
                return {
                    'sport': pick['sport'],
                    'home': pick['home_team_id'],
                    'away': pick['away_team_id'],
                    'matchup': f"{pick['away_team_id']} @ {pick['home_team_id']}",
                    'pick': pick_team,
                    'confidence': confidence,
                    'elo_home': elo * 100,
                    'logistic_home': logistic * 100,
                    'xgboost_home': xgboost * 100,
                }
    except Exception as e:
        print(f"Error getting free pick: {e}")
    
    return None

@app.route('/')
def index():
    """Landing page with free pick"""
    free_pick = get_top_free_pick()
    return render_template('index.html', free_pick=free_pick)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('purepicks_users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, username, email, password_hash, subscription_status FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            user_obj = User(user[0], user[1], user[2], user[4])
            login_user(user_obj, remember=True)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            conn = sqlite3.connect('purepicks_users.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, subscription_status) VALUES (?, ?, ?, ?)",
                (username, email, generate_password_hash(password), 'premium')
            )
            conn.commit()
            conn.close()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Dashboard with sport selection cards"""
    return render_template('dashboard.html', user=current_user)

def get_sport_predictions(sport_code, sport_name, sport_emoji):
    """Helper function to get predictions for a specific sport"""
    db = DatabaseManager()
    
    predictions = []
    
    try:
        with db._get_connection() as conn:
            # Get all predictions for this sport (dates are in DD/MM/YYYY format)
            query = """
                SELECT 
                    p.game_id, p.sport, p.game_date,
                    g.home_team_id, g.away_team_id,
                    p.elo_home_prob, p.logistic_home_prob, p.xgboost_home_prob,
                    p.win_probability
                FROM predictions p
                JOIN games g ON p.game_id = g.game_id
                WHERE p.sport = ?
                ORDER BY p.game_id
            """
            df = pd.read_sql_query(query, conn, params=[sport_code])
            
            today = datetime.now().date()
            
            for _, row in df.iterrows():
                # Parse DD/MM/YYYY date format
                game_date = None
                try:
                    game_date = datetime.strptime(row['game_date'], '%d/%m/%Y').date()
                    # Only show games from today onwards
                    if game_date < today:
                        continue
                except:
                    # If date parsing fails, include the game anyway
                    game_date = datetime.now().date()  # Default to today for sorting
                
                elo = float(row['elo_home_prob']) if row['elo_home_prob'] else 0.5
                logistic = float(row['logistic_home_prob']) if row['logistic_home_prob'] else 0.5
                xgboost = float(row['xgboost_home_prob']) if row['xgboost_home_prob'] else 0.5
                
                # CompositeHome = (XGB% * w1) + (Elo% * w2) + (Logistic% * w3)
                blended = (0.50 * xgboost + 0.35 * elo + 0.15 * logistic)
                
                if blended > 0.5:
                    pick = row['home_team_id']
                    # Show pick team's probability
                    xgb_pct = xgboost * 100
                    elo_pct = elo * 100
                    consensus_pct = blended * 100
                else:
                    pick = row['away_team_id']
                    # Show pick team's probability (flip for away team)
                    xgb_pct = (1 - xgboost) * 100
                    elo_pct = (1 - elo) * 100
                    consensus_pct = (1 - blended) * 100
                
                predictions.append({
                    'date': row['game_date'],
                    'home': row['home_team_id'],
                    'away': row['away_team_id'],
                    'pick': pick,
                    'xgboost_pct': xgb_pct,
                    'elo_pct': elo_pct,
                    'consensus_pct': consensus_pct,
                    '_sort_date': game_date  # For sorting
                })
            
            # Sort predictions chronologically
            predictions.sort(key=lambda x: x.get('_sort_date', datetime.now().date()))
            # Remove sort helper
            for pred in predictions:
                pred.pop('_sort_date', None)
                
    except Exception as e:
        print(f"Error loading {sport_name} predictions: {e}")
    
    return render_template('sport_predictions.html', 
                         predictions=predictions, 
                         sport_name=sport_name,
                         sport_emoji=sport_emoji,
                         user=current_user)

@app.route('/nfl')
def nfl_predictions():
    """NFL predictions page"""
    return get_sport_predictions('NFL', 'NFL', '🏈')

@app.route('/nba')
def nba_predictions():
    """NBA predictions page"""
    return get_sport_predictions('NBA', 'NBA', '🏀')

@app.route('/mlb')
def mlb_predictions():
    """MLB predictions page"""
    return get_sport_predictions('MLB', 'MLB', '⚾')

@app.route('/nhl')
def nhl_predictions():
    """NHL predictions page"""
    return get_sport_predictions('NHL', 'NHL', '🏒')

@app.route('/ncaaf')
def ncaaf_predictions():
    """NCAA Football predictions page"""
    return get_sport_predictions('NCAAF', 'NCAA Football', '🏈')

@app.route('/ncaab')
def ncaab_predictions():
    """NCAA Basketball predictions page"""
    return get_sport_predictions('NCAAB', 'NCAA Basketball', '🏀')

@app.route('/results')
@login_required
def results():
    """Results page showing model backtesting performance - ADMIN ONLY"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT sport, elo_correct, consensus_correct, xgboost_correct, combined_correct,
               total_games, elo_accuracy, consensus_accuracy, xgboost_accuracy, combined_accuracy
        FROM model_backtest_results
        ORDER BY sport
    ''')
    
    backtest_results = []
    for row in cursor.fetchall():
        backtest_results.append({
            'sport': row[0],
            'elo_correct': row[1],
            'consensus_correct': row[2],
            'xgboost_correct': row[3],
            'combined_correct': row[4],
            'total': row[5],
            'elo_accuracy': row[6],
            'consensus_accuracy': row[7],
            'xgboost_accuracy': row[8],
            'combined_accuracy': row[9]
        })
    
    conn.close()
    
    return render_template('results.html', backtest_results=backtest_results, user=current_user)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
