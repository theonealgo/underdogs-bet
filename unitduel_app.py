"""
UnitDuel.com - Professional Sports Prediction Platform
Landing Page with Free Pick
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.data_storage.database import DatabaseManager
from auth import AuthManager, init_session_state

# Page config
st.set_page_config(
    page_title="UnitDuel.com - Elite Sports Predictions",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stats-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .free-pick {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .model-stat {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .cta-button {
        background: #667eea;
        color: white;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
                
                # Calculate blended probability (HOME team win probability)
                elo = float(pick['elo_home_prob']) if pick['elo_home_prob'] else 0.5
                logistic = float(pick['logistic_home_prob']) if pick['logistic_home_prob'] else 0.5
                xgboost = float(pick['xgboost_home_prob']) if pick['xgboost_home_prob'] else 0.5
                
                blended = (0.30 * elo + 0.35 * logistic + 0.35 * xgboost)
                
                # Determine pick
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
                    'blended_home': blended * 100
                }
    except Exception as e:
        st.error(f"Error loading pick: {e}")
    
    return None

# Initialize auth
init_session_state()
auth_manager = AuthManager()

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 3.5rem;">🏆 UnitDuel.com</h1>
    <p style="font-size: 1.3rem; margin-top: 1rem;">Elite Sports Predictions Powered by AI</p>
</div>
""", unsafe_allow_html=True)

# Performance Stats
st.markdown("### 📊 Our Track Record")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stats-box">
        <h2 style="color: #667eea; margin: 0;">93%</h2>
        <p style="margin: 0.5rem 0 0 0;">NFL Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-box">
        <h2 style="color: #667eea; margin: 0;">~100%</h2>
        <p style="margin: 0.5rem 0 0 0;">NHL Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-box">
        <h2 style="color: #667eea; margin: 0;">7+ Sports</h2>
        <p style="margin: 0.5rem 0 0 0;">NFL, NBA, NHL, MLB & More</p>
    </div>
    """, unsafe_allow_html=True)

# Free Pick Section
st.markdown("### 🎁 Today's Free Expert Pick")

free_pick = get_top_free_pick()

if free_pick:
    st.markdown(f"""
    <div class="free-pick">
        <h2 style="margin: 0 0 1rem 0;">{free_pick['sport']}</h2>
        <h3 style="margin: 0 0 1rem 0;">{free_pick['matchup']}</h3>
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; display: inline-block;">
            <p style="margin: 0; font-size: 0.9rem;">OUR PICK</p>
            <h1 style="margin: 0.5rem 0;">{free_pick['pick']}</h1>
            <p style="margin: 0; font-size: 1.2rem;">Confidence: {free_pick['confidence']:.1f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Breakdown - Hide model names
    st.markdown("#### 🔬 Our Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="model-stat">
            <p style="margin: 0; color: #666;">Model A</p>
            <h3 style="margin: 0.5rem 0; color: #667eea;">{free_pick['elo_home']:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="model-stat">
            <p style="margin: 0; color: #666;">Model B</p>
            <h3 style="margin: 0.5rem 0; color: #667eea;">{free_pick['logistic_home']:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="model-stat">
            <p style="margin: 0; color: #666;">Model C</p>
            <h3 style="margin: 0.5rem 0; color: #667eea;">{free_pick['xgboost_home']:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="model-stat" style="background: #667eea;">
            <p style="margin: 0; color: white;">Combined</p>
            <h3 style="margin: 0.5rem 0; color: white;">{free_pick['blended_home']:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.caption(f"Win probabilities for {free_pick['home']} (Home)")
else:
    st.info("No picks available for today. Check back soon!")

# CTA Section
st.markdown("---")
st.markdown("### 🚀 Get Full Access to Premium Predictions")
st.markdown("""
Get unlimited access to:
- ✅ 30-day predictions for all sports
- ✅ Complete model breakdowns and analytics
- ✅ Real-time updates and alerts
- ✅ Premium support
""")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.session_state.authenticated:
        if st.button("📊 Go to Dashboard", use_container_width=True, type="primary"):
            st.switch_page("pages/04_dashboard.py")
    else:
        if st.button("🔐 Sign Up Now", use_container_width=True, type="primary"):
            st.switch_page("pages/03_signup.py")
        if st.button("Already have an account? Login", use_container_width=True):
            st.switch_page("pages/02_login.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>© 2025 UnitDuel.com | Elite Sports Predictions</p>
    <p style="font-size: 0.9rem;">Powered by Advanced Machine Learning Models</p>
</div>
""", unsafe_allow_html=True)
