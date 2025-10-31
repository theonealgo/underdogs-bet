"""
Premium Dashboard for UnitDuel.com
30-Day Predictions View
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from auth import require_auth, logout
from src.data_storage.database import DatabaseManager

st.set_page_config(page_title="Dashboard - UnitDuel.com", page_icon="📊", layout="wide")

# Require authentication (premium check removed - all users have access)
user_data = require_auth()

# Custom CSS for clean professional look
st.markdown("""
<style>
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .game-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    .pick-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prob-bar {
        background: #f0f0f0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 style="margin: 0;">📊 Premium Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0;">Welcome back, {user_data['username']}!</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("🏠 Home"):
        st.switch_page("unitduel_app.py")
    if st.button("🚪 Logout"):
        logout()

# Sport selector
sports = ['NFL', 'NBA', 'NHL', 'MLB', 'NCAAF', 'NCAAB', 'WNBA']
selected_sport = st.selectbox("Select Sport", sports, key="sport_selector")

# Date range (30 days)
today = datetime.now().date()
end_date = today + timedelta(days=30)

st.markdown(f"### 📅 30-Day Predictions ({today.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')})")

# Fetch predictions
db = DatabaseManager()

try:
    with db._get_connection() as conn:
        query = """
            SELECT 
                p.game_id, p.game_date,
                g.home_team_id, g.away_team_id,
                p.elo_home_prob, p.logistic_home_prob, p.xgboost_home_prob
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.sport = ?
            AND DATE(p.game_date) >= DATE(?)
            AND DATE(p.game_date) <= DATE(?)
            ORDER BY p.game_date
        """
        df = pd.read_sql_query(query, conn, params=[
            selected_sport,
            today.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        ])
        
        if df.empty:
            st.info(f"No predictions available for {selected_sport} in the next 30 days")
        else:
            st.success(f"Found {len(df)} predictions for {selected_sport}")
            
            # Group by date
            df['date_only'] = pd.to_datetime(df['game_date']).dt.date
            
            for game_date in sorted(df['date_only'].unique()):
                date_games = df[df['date_only'] == game_date]
                
                # Date header
                st.markdown(f"#### 📅 {game_date.strftime('%A, %B %d, %Y')}")
                
                # Show games for this date
                for _, game in date_games.iterrows():
                    # Get probabilities
                    elo_home = float(game['elo_home_prob']) if pd.notna(game['elo_home_prob']) else 0.5
                    log_home = float(game['logistic_home_prob']) if pd.notna(game['logistic_home_prob']) else 0.5
                    xgb_home = float(game['xgboost_home_prob']) if pd.notna(game['xgboost_home_prob']) else 0.5
                    
                    # Blended probability (30% Elo + 35% Logistic + 35% XGBoost)
                    blended_home = (0.30 * elo_home + 0.35 * log_home + 0.35 * xgb_home)
                    
                    # Determine pick
                    if blended_home > 0.5:
                        pick_team = game['home_team_id']
                        pick_prob = blended_home * 100
                    else:
                        pick_team = game['away_team_id']
                        pick_prob = (1 - blended_home) * 100
                    
                    # Clean display
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {game['away_team_id']} @ {game['home_team_id']}")
                        st.markdown(f"""
                        <div class="pick-box">
                            <h3 style="margin: 0;">🎯 {pick_team}</h3>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{pick_prob:.1f}% Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### Model Analysis")
                        st.markdown(f"""
                        <div class="prob-bar">Model A: {elo_home*100:.1f}%</div>
                        <div class="prob-bar">Model B: {log_home*100:.1f}%</div>
                        <div class="prob-bar">Model C: {xgb_home*100:.1f}%</div>
                        <div class="prob-bar" style="background: #667eea; color: white; font-weight: bold;">
                            Combined: {blended_home*100:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"(Probabilities for {game['home_team_id']})")
                    
                    st.markdown("---")
                
except Exception as e:
    st.error(f"Error loading predictions: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2025 UnitDuel.com | Premium Dashboard</p>
</div>
""", unsafe_allow_html=True)
