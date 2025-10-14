"""
Premium Dashboard for PurePicks.COM
30-Day Predictions View
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from auth import require_auth, logout
from src.data_storage.database import DatabaseManager

st.set_page_config(page_title="Dashboard - PurePicks.COM", page_icon="📊", layout="wide")

# Require authentication (premium check removed - all users have access)
user_data = require_auth()

# Custom CSS for dashboard
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
        border-left: 4px solid #667eea;
    }
    .prediction-confidence {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        display: inline-block;
        margin: 0.25rem;
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
        st.switch_page("purepicks_app.py")
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
                p.elo_home_prob, p.logistic_home_prob, p.xgboost_home_prob,
                p.win_probability, p.predicted_winner
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
                    # Calculate blended probability
                    elo = float(game['elo_home_prob']) if pd.notna(game['elo_home_prob']) else 0.5
                    logistic = float(game['logistic_home_prob']) if pd.notna(game['logistic_home_prob']) else 0.5
                    xgboost = float(game['xgboost_home_prob']) if pd.notna(game['xgboost_home_prob']) else 0.5
                    
                    blended = (0.30 * elo + 0.35 * logistic + 0.35 * xgboost) * 100
                    
                    # Determine pick
                    if blended > 50:
                        pick = game['home_team_id']
                        confidence = blended
                    else:
                        pick = game['away_team_id']
                        confidence = 100 - blended
                    
                    # Display game card
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**{game['away_team_id']} @ {game['home_team_id']}**")
                        st.markdown(f"🎯 **Pick:** {pick} ({confidence:.1f}% confidence)")
                    
                    with col2:
                        st.markdown("**Model Breakdown:**")
                        st.markdown(f"""
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <span class="prediction-confidence">Elo: {elo*100:.1f}%</span>
                            <span class="prediction-confidence">Log: {logistic*100:.1f}%</span>
                            <span class="prediction-confidence">XGB: {xgboost*100:.1f}%</span>
                            <span class="prediction-confidence" style="background: #667eea; color: white;">Blended: {blended:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                
except Exception as e:
    st.error(f"Error loading predictions: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2025 PurePicks.COM | Premium Dashboard</p>
</div>
""", unsafe_allow_html=True)
