"""
Simple Authentication System for UnitDuel.com
User will add Stripe integration later
"""

import streamlit as st
import hashlib
import sqlite3
from datetime import datetime

class AuthManager:
    def __init__(self):
        self.db_path = "purepicks_users.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize users database"""
        conn = sqlite3.connect(self.db_path)
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
    
    def _hash_password(self, password):
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def signup(self, username, email, password):
        """Create new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, subscription_status) VALUES (?, ?, ?, ?)",
                (username, email, self._hash_password(password), 'premium')
            )
            conn.commit()
            conn.close()
            return True, "Account created successfully!"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def login(self, username, password):
        """Authenticate user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, username, email, subscription_status FROM users WHERE username = ? AND password_hash = ?",
            (username, self._hash_password(password))
        )
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return True, {
                'user_id': user[0],
                'username': user[1],
                'email': user[2],
                'subscription_status': user[3]
            }
        return False, None
    
    def is_premium(self, user_data):
        """Check if user has premium access"""
        if not user_data:
            return False
        return user_data.get('subscription_status') == 'premium'
    
    def update_subscription(self, user_id, status='premium', stripe_customer_id=None):
        """Update user subscription status (for Stripe integration)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if stripe_customer_id:
            cursor.execute(
                "UPDATE users SET subscription_status = ?, stripe_customer_id = ? WHERE user_id = ?",
                (status, stripe_customer_id, user_id)
            )
        else:
            cursor.execute(
                "UPDATE users SET subscription_status = ? WHERE user_id = ?",
                (status, user_id)
            )
        conn.commit()
        conn.close()

def init_session_state():
    """Initialize session state for auth"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None

def logout():
    """Logout current user"""
    st.session_state.authenticated = False
    st.session_state.user_data = None
    st.rerun()

def require_auth():
    """Decorator-style function to require authentication"""
    init_session_state()
    if not st.session_state.authenticated:
        st.warning("⚠️ Please login to access this page")
        st.page_link("pages/02_login.py", label="Go to Login", icon="🔐")
        st.stop()
    return st.session_state.user_data

def require_premium():
    """Require premium subscription"""
    user_data = require_auth()
    auth_manager = AuthManager()
    if not auth_manager.is_premium(user_data):
        st.error("🔒 Premium Subscription Required")
        st.markdown("### Upgrade to Premium for Full Access")
        st.markdown("Get access to:")
        st.markdown("- 📊 Full 30-day predictions for all sports")
        st.markdown("- 🎯 Model breakdowns (Elo, Logistic, XGBoost)")
        st.markdown("- 📈 Advanced analytics and insights")
        
        # Placeholder for Stripe checkout button
        st.info("💳 Stripe Integration Coming Soon - User will add checkout button here")
        st.stop()
    return user_data
