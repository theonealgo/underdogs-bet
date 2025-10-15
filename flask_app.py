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
    """Dashboard with predictions - open for testing"""
    sport_filter = request.args.get('sport', 'NFL')
    
    db = DatabaseManager()
    today = datetime.now().date()
    week_later = today + timedelta(days=7)
    
    predictions = []
    
    try:
        with db._get_connection() as conn:
            query = """
                SELECT 
                    p.game_id, p.sport, p.game_date,
                    g.home_team_id, g.away_team_id,
                    p.elo_home_prob, p.logistic_home_prob, p.xgboost_home_prob,
                    p.win_probability
                FROM predictions p
                JOIN games g ON p.game_id = g.game_id
                WHERE p.sport = ?
                AND DATE(p.game_date) >= DATE(?)
                AND DATE(p.game_date) <= DATE(?)
                ORDER BY p.game_date, p.game_id
            """
            df = pd.read_sql_query(query, conn, params=[
                sport_filter,
                today.strftime('%Y-%m-%d'),
                week_later.strftime('%Y-%m-%d')
            ])
            
            for _, row in df.iterrows():
                elo = float(row['elo_home_prob']) if row['elo_home_prob'] else 0.5
                logistic = float(row['logistic_home_prob']) if row['logistic_home_prob'] else 0.5
                xgboost = float(row['xgboost_home_prob']) if row['xgboost_home_prob'] else 0.5
                
                # CompositeHome = (XGB% * w1) + (Elo% * w2) + (Consensus% * w3)
                blended = (0.50 * xgboost + 0.35 * elo + 0.15 * logistic)
                
                if blended > 0.5:
                    pick = row['home_team_id']
                    confidence = blended * 100
                else:
                    pick = row['away_team_id']
                    confidence = (1 - blended) * 100
                
                predictions.append({
                    'date': row['game_date'],
                    'sport': row['sport'],
                    'home': row['home_team_id'],
                    'away': row['away_team_id'],
                    'pick': pick,
                    'confidence': confidence,
                    'elo_home': elo * 100,
                    'logistic_home': logistic * 100,
                    'xgboost_home': xgboost * 100
                })
    except Exception as e:
        print(f"Error loading predictions: {e}")
    
    return render_template('dashboard.html', 
                         predictions=predictions, 
                         sport_filter=sport_filter,
                         user=current_user)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
