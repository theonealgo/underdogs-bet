import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)

from src.data_storage.database import DatabaseManager
from src.models.prediction_models import MLBPredictor
from src.models.nhl_predictor import NHLPredictor
from src.data_collectors.baseball_savant_scraper import BaseballSavantScraper
from src.data_collectors.result_tracker import ResultTracker
from src.models.performance_analyzer import PerformanceAnalyzer
from src.models.intelligent_retrainer import IntelligentRetrainer
from src.utils.performance_visualizer import PerformanceVisualizer
from src.utils.scheduler import DataScheduler
from src.api.prediction_api import PredictionAPI
from src.backtesting.backtester import Backtester
from src.utils.sport_data_manager import SportDataManager

# Initialize components
@st.cache_resource
def initialize_components():
    db_manager = DatabaseManager()
    predictor = MLBPredictor(db_manager=db_manager)
    nhl_predictor = NHLPredictor(db_manager=db_manager)
    api = PredictionAPI(db_manager, predictor)
    sport_data_manager = SportDataManager()
    
    result_tracker = ResultTracker(db_manager)
    performance_analyzer = PerformanceAnalyzer(db_manager)
    intelligent_retrainer = IntelligentRetrainer(db_manager, predictor)
    performance_visualizer = PerformanceVisualizer(db_manager)
    
    return (db_manager, predictor, nhl_predictor, api, sport_data_manager, result_tracker, 
            performance_analyzer, intelligent_retrainer, performance_visualizer)

# Custom CSS for dratings.com-like styling
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f5f5f5;
    }
    
    /* Header styling */
    .stApp header {
        background-color: #ffffff;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Sport navigation tabs */
    .sport-nav {
        background-color: #ffffff;
        padding: 15px 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prediction table styling */
    .pred-table {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Table headers */
    .pred-table th {
        background-color: #f8f8f8;
        padding: 12px;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Table rows */
    .pred-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #f0f0f0;
    }
    
    /* Team names */
    .team-name {
        font-weight: 600;
        color: #333;
    }
    
    /* Win percentage styling */
    .win-pct {
        font-weight: 600;
        color: #2c7a2c;
    }
    
    /* Date navigation */
    .date-nav {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 15px;
        background-color: #f8f8f8;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f8f8;
        padding: 5px;
        border-radius: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ff6b35;
        color: white;
    }
    
    /* Bet value indicators */
    .bet-value-high {
        color: #2c7a2c;
        font-weight: bold;
    }
    
    .bet-value-medium {
        color: #ff9800;
        font-weight: bold;
    }
    
    .bet-value-low {
        color: #999;
    }
    
    /* Updated timestamp */
    .updated-time {
        color: #666;
        font-size: 14px;
        font-style: italic;
        margin-bottom: 15px;
    }
    
    /* Compact dataframe styling */
    [data-testid="stDataFrame"] {
        font-size: 13px;
    }
    
    [data-testid="stDataFrame"] th {
        padding: 6px 4px !important;
        font-size: 12px !important;
    }
    
    [data-testid="stDataFrame"] td {
        padding: 6px 4px !important;
        font-size: 13px !important;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .sport-nav {
            padding: 10px;
        }
        
        [data-testid="stDataFrame"] {
            font-size: 11px;
        }
        
        [data-testid="stDataFrame"] th {
            padding: 4px 2px !important;
            font-size: 10px !important;
        }
        
        [data-testid="stDataFrame"] td {
            padding: 4px 2px !important;
            font-size: 11px !important;
        }
        
        .pred-table {
            padding: 10px;
        }
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Sports Predictions & Analysis",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    
    # Initialize components
    (db_manager, predictor, nhl_predictor, api, sport_data_manager, result_tracker, 
     performance_analyzer, intelligent_retrainer, performance_visualizer) = initialize_components()
    
    # Sport mapping (must match database sport codes and SportDataManager collector keys)
    sport_mapping = {
        "MLB ⚾": "MLB",
        "NBA 🏀": "NBA",
        "NFL 🏈": "NFL",
        "NHL 🏒": "NHL",
        "NCAA Football 🏈": "NCAAF",  # Changed from NCAA to NCAAF to match database
        "NCAA Basketball 🏀": "NCAAB",
        "WNBA 🏀": "WNBA"
    }
    
    # Top navigation header with sport pages
    st.markdown('<div class="sport-nav">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 6])
    
    with col1:
        st.markdown("### 🏆 Sports Predictions & Analysis")
    
    with col2:
        # Sport page selection - each sport is its own page
        sport_pages = ["MLB ⚾", "NBA 🏀", "NFL 🏈", "NHL 🏒", "NCAA Football 🏈", "NCAA Basketball 🏀", "WNBA 🏀"]
        
        if 'selected_sport_page' not in st.session_state:
            st.session_state.selected_sport_page = "MLB ⚾"
        
        # Create tabs for sport selection
        selected_page = st.radio(
            "Sport",
            sport_pages,
            horizontal=True,
            key="sport_page_selector",
            label_visibility="collapsed"
        )
        
        if selected_page != st.session_state.selected_sport_page:
            st.session_state.selected_sport_page = selected_page
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Extract sport code using mapping
    sport_code = sport_mapping.get(st.session_state.selected_sport_page, "MLB")
    
    # Show sport-specific page with all information
    show_sport_page(api, db_manager, sport_data_manager, result_tracker, nhl_predictor, sport_code, st.session_state.selected_sport_page)

def show_sport_page(api, db_manager, sport_data_manager, result_tracker, nhl_predictor, sport_code, sport_name):
    """Comprehensive sport page with all prediction information"""
    
    # Date navigation - use UTC-5 (EST/CDT) to match US timezones
    if 'prediction_date' not in st.session_state:
        # Default to US Eastern timezone (UTC-5) for game scheduling
        us_now = datetime.now() - timedelta(hours=5)
        st.session_state.prediction_date = us_now.date()
    
    # View mode selector
    view_mode = st.radio(
        "View Mode",
        ["7-Day View", "All Upcoming Games"],
        horizontal=True,
        key=f"view_mode_{sport_code}"
    )
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("◀ Previous Day"):
            st.session_state.prediction_date -= timedelta(days=1)
            st.rerun()
    
    with col3:
        selected_date = st.date_input(
            "Date",
            value=st.session_state.prediction_date,
            key="date_picker_main",
            label_visibility="collapsed"
        )
        if selected_date != st.session_state.prediction_date:
            st.session_state.prediction_date = selected_date
            st.rerun()
    
    with col5:
        if st.button("Next Day ▶"):
            st.session_state.prediction_date += timedelta(days=1)
            st.rerun()
    
    # Section 1: Upcoming Games
    st.markdown("## Upcoming Games")
    
    # Excel/CSV Upload Section
    with st.expander("📁 Upload Schedule (Excel/CSV)", expanded=False):
        st.warning("⚠️ Uploading a new schedule will REPLACE all existing games and predictions for this sport!")
        
        clear_old_data = st.checkbox(f"Clear old {sport_code} data before uploading", value=True, key=f"clear_old_{sport_code}")
        
        uploaded_file = st.file_uploader(
            f"Upload {sport_code} schedule (Excel or CSV)", 
            type=['xlsx', 'xls', 'csv'],
            key=f"schedule_upload_{sport_code}"
        )
        
        if uploaded_file:
            try:
                import tempfile
                import os
                
                # Clear old data if checkbox is checked
                if clear_old_data:
                    with db_manager._get_connection() as conn:
                        conn.execute("DELETE FROM predictions WHERE sport = ?", [sport_code])
                        conn.execute("DELETE FROM games WHERE sport = ?", [sport_code])
                        conn.commit()
                    st.success(f"✅ Cleared old {sport_code} data")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                st.info(f"Processing {uploaded_file.name}...")
                
                # Run universal predictor
                import subprocess
                result = subprocess.run(
                    ['python', 'universal_sports_predictor.py', sport_code, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    st.success(f"✅ Successfully loaded schedule and generated predictions!")
                    st.code(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                    st.info("Click below to see your new predictions")
                    if st.button("🔄 Refresh Now"):
                        st.rerun()
                else:
                    st.error(f"Error: {result.stderr}")
                    st.code(result.stdout)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Debug section for user to paste real game data
    with st.expander("🔧 Debug: Paste Real Game Data Here", expanded=False):
        st.write(f"If you see wrong teams or no games for **{sport_code}**, paste the actual game data from ESPN/official source below:")
        debug_data = st.text_area(
            f"Paste {sport_code} game data here",
            height=150,
            key=f"debug_{sport_code}",
            placeholder="Paste game matchups (e.g., 'Montreal @ Toronto', 'SEA @ DET')..."
        )
        if debug_data:
            st.code(debug_data)
            
            # Parse the pasted data and create games
            from src.utils.game_data_parser import parse_pasted_games
            parsed_games = parse_pasted_games(debug_data, sport_code, st.session_state.prediction_date)
            
            if parsed_games:
                st.success(f"✅ Parsed {len(parsed_games)} games from pasted data!")
                # Store parsed games in database
                try:
                    games_df = pd.DataFrame(parsed_games)
                    db_manager.store_games(games_df)
                    st.info("Games saved to database. Refresh the page to see them.")
                    st.button("🔄 Refresh", on_click=lambda: st.rerun())
                except Exception as e:
                    st.warning(f"Could not store games: {str(e)}")
            else:
                st.warning("No games found in pasted data. Make sure to use format like 'Team1 @ Team2'.")
    
    prediction_date = datetime.combine(st.session_state.prediction_date, datetime.min.time())
    
    with st.spinner("Loading predictions..."):
        try:
            if sport_code == "MLB":
                predictions = api.get_todays_predictions(date=prediction_date, sport=sport_code)
            else:
                # For non-MLB sports, use database-first approach
                date_str = st.session_state.prediction_date.strftime('%Y-%m-%d')
                
                # FIRST: Check if games exist in the database
                with db_manager._get_connection() as conn:
                    if view_mode == "All Upcoming Games":
                        # Show all future games
                        games_query = """
                            SELECT game_id, sport, home_team_id, away_team_id, game_date, status
                            FROM games
                            WHERE sport = ?
                            AND (status = 'Scheduled' OR status IS NULL OR DATE(game_date) >= DATE('now'))
                            ORDER BY game_date
                            LIMIT 100
                        """
                        games_df = pd.read_sql_query(games_query, conn, params=[sport_code])
                        
                        # Get all predictions for these games
                        pred_query = """
                            SELECT p.*, g.home_team_id, g.away_team_id
                            FROM predictions p
                            LEFT JOIN games g ON p.game_id = g.game_id
                            WHERE p.sport = ?
                            AND (g.status = 'Scheduled' OR g.status IS NULL OR DATE(p.game_date) >= DATE('now'))
                            ORDER BY p.game_date
                            LIMIT 100
                        """
                        predictions_df = pd.read_sql_query(pred_query, conn, params=[sport_code])
                    else:
                        # Show 7-day range starting from selected date
                        start_date = st.session_state.prediction_date
                        end_date = start_date + timedelta(days=7)
                        
                        # Build patterns for DD/MM/YYYY format matching
                        date_patterns = []
                        for i in range(8):  # Include start and next 7 days
                            d = start_date + timedelta(days=i)
                            date_patterns.append(f'{d.day:02d}/{d.month:02d}/{d.year}')
                        
                        # Create query with multiple date patterns
                        like_clauses = ' OR '.join(['game_date LIKE ?' for _ in date_patterns])
                        pattern_params = [f'{p}%' for p in date_patterns]
                        
                        games_query = f"""
                            SELECT game_id, sport, home_team_id, away_team_id, game_date, status
                            FROM games
                            WHERE ({like_clauses})
                            AND sport = ?
                            ORDER BY game_date
                        """
                        games_df = pd.read_sql_query(games_query, conn, params=pattern_params + [sport_code])
                        
                        # Also get predictions if they exist
                        pred_query = f"""
                            SELECT p.*, g.home_team_id, g.away_team_id
                            FROM predictions p
                            LEFT JOIN games g ON p.game_id = g.game_id
                            WHERE ({like_clauses})
                            AND p.sport = ?
                            ORDER BY p.game_date
                        """
                        predictions_df = pd.read_sql_query(pred_query, conn, params=pattern_params + [sport_code])
                
                predictions = []
                
                # If we have games in the database, use them
                if not games_df.empty:
                    # Create a mapping of predictions by game_id for quick lookup
                    pred_map = {}
                    if not predictions_df.empty:
                        for _, pred in predictions_df.iterrows():
                            pred_map[str(pred.get('game_id', ''))] = pred
                    
                    # Now iterate through ALL games and show them
                    pred_data = []
                    for _, game in games_df.iterrows():
                        game_id = str(game.get('game_id', ''))
                        home_team_id = str(game.get('home_team_id', ''))
                        away_team_id = str(game.get('away_team_id', ''))
                        game_time = str(game.get('game_date', '')).split(' ')[-1] if ' ' in str(game.get('game_date', '')) else 'TBD'
                        
                        # Check if we have a prediction for this game
                        if game_id in pred_map:
                            pred = pred_map[game_id]
                            predicted_winner = str(pred.get('predicted_winner', 'No Pick'))
                            win_prob_value = pred.get('win_probability')
                            win_prob = float(win_prob_value) if win_prob_value is not None else 0.5
                            
                            # If probabilities are equal, no clear pick
                            if abs(win_prob - 0.5) < 0.01:  # Within 1% of 50/50
                                predicted_winner = 'No Pick'
                                home_win_prob = 0.5
                            else:
                                # Determine home win probability based on who is predicted to win
                                if predicted_winner == away_team_id:
                                    home_win_prob = 1 - win_prob
                                else:
                                    home_win_prob = win_prob
                            
                            predictions.append({
                                'game_id': game_id,
                                'game_time': game_time,
                                'game_date': pred.get('game_date', prediction_date),
                                'home_team': home_team_id,
                                'away_team': away_team_id,
                                'predicted_winner': predicted_winner,
                                'home_win_probability': home_win_prob,
                                'away_win_probability': 1 - home_win_prob,
                                'predicted_total': pred.get('predicted_total'),
                                'predicted_home_score': None,
                                'predicted_away_score': None,
                                'elo_home_prob': pred.get('elo_home_prob'),
                                'logistic_home_prob': pred.get('logistic_home_prob'),
                                'xgboost_home_prob': pred.get('xgboost_home_prob')
                            })
                        else:
                            # No prediction exists for this game
                            # For NHL, generate prediction using NHL predictor
                            if sport_code == 'NHL' and nhl_predictor.is_trained:
                                # Get historical games from database for feature creation
                                with db_manager._get_connection() as conn:
                                    historical_query = """
                                        SELECT * FROM games 
                                        WHERE sport = 'NHL' 
                                        AND status = 'final'
                                        AND game_date < ?
                                        ORDER BY game_date DESC
                                        LIMIT 500
                                    """
                                    historical_games = pd.read_sql_query(historical_query, conn, params=[date_str])
                                
                                nhl_pred = nhl_predictor.predict_game(
                                    home_team_id,
                                    away_team_id,
                                    pd.to_datetime(date_str).date(),
                                    historical_games
                                )
                                
                                predicted_winner = nhl_pred['predicted_winner']
                                home_win_prob = nhl_pred['home_win_probability']
                                predicted_total = nhl_pred['predicted_total']
                                
                                predictions.append({
                                    'game_id': game_id,
                                    'game_time': game_time,
                                    'game_date': game.get('game_date', prediction_date),
                                    'home_team': home_team_id,
                                    'away_team': away_team_id,
                                    'predicted_winner': predicted_winner,
                                    'home_win_probability': home_win_prob,
                                    'away_win_probability': 1 - home_win_prob,
                                    'predicted_total': predicted_total,
                                    'predicted_home_score': None,
                                    'predicted_away_score': None,
                                    'elo_home_prob': None,
                                    'logistic_home_prob': None,
                                    'xgboost_home_prob': None
                                })
                                
                                # Store NHL prediction in database
                                pred_data.append({
                                    'sport': sport_code,
                                    'league': sport_code,
                                    'game_id': game_id,
                                    'game_date': date_str,
                                    'home_team_id': home_team_id,
                                    'away_team_id': away_team_id,
                                    'predicted_winner': predicted_winner,
                                    'win_probability': float(home_win_prob),  # Convert numpy float to Python float
                                    'predicted_total': float(predicted_total) if predicted_total is not None else None,
                                    'model_version': '1.0',
                                    'key_factors': '[]'
                                })
                            else:
                                # For other sports or if NHL model not trained, use default
                                predictions.append({
                                    'game_id': game_id,
                                    'game_time': game_time,
                                    'game_date': game.get('game_date', prediction_date),
                                    'home_team': home_team_id,
                                    'away_team': away_team_id,
                                    'predicted_winner': 'No Pick',  # No trained model
                                    'home_win_probability': 0.5,
                                    'away_win_probability': 0.5,
                                    'predicted_total': None,
                                    'predicted_home_score': None,
                                    'predicted_away_score': None,
                                    'elo_home_prob': 0.5,
                                    'logistic_home_prob': 0.5,
                                    'xgboost_home_prob': 0.5
                                })
                                
                                # Also create a prediction record in the database
                                pred_data.append({
                                    'sport': sport_code,
                                    'league': sport_code,
                                    'game_id': game_id,
                                    'game_date': date_str,
                                    'home_team_id': home_team_id,
                                    'away_team_id': away_team_id,
                                    'predicted_winner': None,  # Store as NULL when no pick
                                    'win_probability': 0.5,
                                    'predicted_total': None,
                                    'total_confidence': 0.5,
                                    'model_version': 'default_v1',
                                    'key_factors': '[]'
                                })
                    
                    # Store new predictions if any were created
                    if pred_data:
                        try:
                            db_manager.store_predictions(pd.DataFrame(pred_data))
                        except Exception as store_err:
                            logger.warning(f"Could not store predictions: {store_err}")
                
                # If no games in database, try to fetch from external API as fallback
                else:
                    todays_games = sport_data_manager.get_todays_games(sport_code, date=date_str)
                    
                    if not todays_games.empty:
                        # Store games in database
                        db_manager.store_games(todays_games)
                        
                        # Create predictions for each game
                        pred_data = []
                        for _, game in todays_games.iterrows():
                            game_id = game.get('game_id', '')
                            home_team_id = game.get('home_team_id', '')
                            away_team_id = game.get('away_team_id', '')
                            home_team_name = game.get('home_team_name', home_team_id)
                            away_team_name = game.get('away_team_name', away_team_id)
                            
                            predictions.append({
                                'game_id': game_id,
                                'game_time': game.get('game_time', 'TBD'),
                                'game_date': prediction_date,
                                'home_team': home_team_id,  # Use abbreviation
                                'away_team': away_team_id,  # Use abbreviation
                                'predicted_winner': 'No Pick',  # No trained model
                                'home_win_probability': 0.5,
                                'away_win_probability': 0.5,
                                'predicted_total': None,
                                'predicted_home_score': None,
                                'predicted_away_score': None
                            })
                            
                            pred_data.append({
                                'sport': sport_code,
                                'league': sport_code,
                                'game_id': game_id,
                                'game_date': date_str,
                                'home_team_id': home_team_id,
                                'away_team_id': away_team_id,
                                'predicted_winner': None,  # Store as NULL when no pick
                                'win_probability': 0.5,
                                'predicted_total': None,
                                'total_confidence': 0.5,
                                'model_version': 'api_v1',
                                'key_factors': '[]'
                            })
                        
                        if pred_data:
                            db_manager.store_predictions(pd.DataFrame(pred_data))
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
            predictions = []
    
    # Join odds data with predictions
    if predictions:
        try:
            from src.utils.team_resolver import TeamResolver
            team_resolver = TeamResolver()
            
            date_str = st.session_state.prediction_date.strftime('%Y-%m-%d')
            with db_manager._get_connection() as conn:
                odds_query = """
                    SELECT * FROM odds_data
                    WHERE DATE(game_date) = ?
                    AND sport = ?
                """
                odds_df = pd.read_sql_query(odds_query, conn, params=[date_str, sport_code])
            
            # Join odds data with predictions using multiple matching strategies
            for pred in predictions:
                home_team = pred.get('home_team', '')
                away_team = pred.get('away_team', '')
                
                # Strategy 1: Exact match on team IDs
                matching_odds = odds_df[
                    (odds_df['home_team'] == home_team) & (odds_df['away_team'] == away_team)
                ]
                
                # Strategy 2: If no match, try resolving odds team names to IDs and match
                if matching_odds.empty:
                    for _, odds_row in odds_df.iterrows():
                        odds_away_id = team_resolver.resolve(sport_code, str(odds_row['away_team']))
                        odds_home_id = team_resolver.resolve(sport_code, str(odds_row['home_team']))
                        
                        if (odds_away_id == away_team or odds_row['away_team'] == away_team) and \
                           (odds_home_id == home_team or odds_row['home_team'] == home_team):
                            matching_odds = pd.DataFrame([odds_row])
                            break
                
                # Strategy 3: Partial string matching as last resort
                if matching_odds.empty:
                    for _, odds_row in odds_df.iterrows():
                        odds_away = str(odds_row['away_team']).lower()
                        odds_home = str(odds_row['home_team']).lower()
                        pred_away = str(away_team).lower()
                        pred_home = str(home_team).lower()
                        
                        if (pred_away in odds_away or odds_away in pred_away) and \
                           (pred_home in odds_home or odds_home in pred_home):
                            matching_odds = pd.DataFrame([odds_row])
                            break
                
                if not matching_odds.empty:
                    odds_row = matching_odds.iloc[0]
                    pred['away_odds'] = odds_row['away_odds']
                    pred['home_odds'] = odds_row['home_odds']
                    pred['away_spread'] = odds_row['away_spread']
                    pred['away_spread_odds'] = odds_row['away_spread_odds']
                    pred['home_spread'] = odds_row['home_spread']
                    pred['home_spread_odds'] = odds_row['home_spread_odds']
                    pred['total_line'] = odds_row['total_line']
                    pred['over_odds'] = odds_row['over_odds']
                    pred['under_odds'] = odds_row['under_odds']
        except Exception as e:
            logger.warning(f"Could not join odds data: {str(e)}")
    
    show_upcoming_games_section(predictions, sport_code)
    
    # Section 2: Games In Progress
    st.markdown("## Games In Progress")
    show_games_in_progress(db_manager, sport_code, st.session_state.prediction_date)
    
    # Section 3: Completed Games
    st.markdown("## Completed Games")
    show_completed_games_section(db_manager, sport_code, st.session_state.prediction_date)
    
    # Section 4: Season Prediction Results
    st.markdown("## Season Prediction Results")
    show_season_results(db_manager, sport_code)

def show_predictions_page(api, db_manager, sport_data_manager, sport_code):
    # Date navigation - use UTC-5 (EST/CDT) to match US timezones
    if 'prediction_date' not in st.session_state:
        # Default to US Eastern timezone (UTC-5) for game scheduling
        us_now = datetime.now() - timedelta(hours=5)
        st.session_state.prediction_date = us_now.date()
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("◀ Previous Day"):
            st.session_state.prediction_date -= timedelta(days=1)
            st.rerun()
    
    with col3:
        selected_date = st.date_input(
            "Date",
            value=st.session_state.prediction_date,
            key="date_picker",
            label_visibility="collapsed"
        )
        if selected_date != st.session_state.prediction_date:
            st.session_state.prediction_date = selected_date
            st.rerun()
    
    with col5:
        if st.button("Next Day ▶"):
            st.session_state.prediction_date += timedelta(days=1)
            st.rerun()
    
    # Title and update timestamp
    st.markdown(f"## {sport_code} Predictions")
    st.markdown(f'<p class="updated-time">Updated {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>', 
                unsafe_allow_html=True)
    
    # Auto-load predictions
    prediction_date = datetime.combine(st.session_state.prediction_date, datetime.min.time())
    
    with st.spinner("Loading predictions..."):
        try:
            # Currently only MLB predictions are fully supported with ML models
            if sport_code == "MLB":
                predictions = api.get_todays_predictions(date=prediction_date)
            else:
                # For other sports, fetch games from sport data manager
                date_str = st.session_state.prediction_date.strftime('%Y-%m-%d')
                todays_games = sport_data_manager.get_todays_games(sport_code, date=date_str)
                # Convert to prediction format (basic predictions without ML model)
                predictions = []
                if not todays_games.empty:
                    for _, game in todays_games.iterrows():
                        predictions.append({
                            'game_id': game['game_id'] if 'game_id' in game else '',
                            'game_time': game['game_time'] if 'game_time' in game else 'TBD',
                            'game_date': game['game_date'] if 'game_date' in game else prediction_date,
                            'home_team': game['home_team_id'] if 'home_team_id' in game else '',  # Use abbreviation
                            'away_team': game['away_team_id'] if 'away_team_id' in game else '',  # Use abbreviation
                            'home_win_probability': 0.5,  # Neutral until ML model is trained
                            'away_win_probability': 0.5,
                            'predicted_home_score': None,
                            'predicted_away_score': None,
                            'predicted_winner': 'No Pick'  # No trained model available
                        })
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
            predictions = []
    
    # Get completed games for the same date
    try:
        with db_manager._get_connection() as conn:
            completed_query = """
                SELECT 
                    p.*,
                    g.home_score,
                    g.away_score,
                    g.status
                FROM predictions p
                LEFT JOIN games g ON p.game_id = g.game_id
                WHERE DATE(p.game_date) = DATE(?)
                AND p.sport = ?
                AND (g.status = 'completed' OR p.result_updated_at IS NOT NULL)
                ORDER BY p.game_date, p.created_at
            """
            completed_df = pd.read_sql_query(
                completed_query, 
                conn, 
                params=[st.session_state.prediction_date.strftime('%Y-%m-%d'), sport_code]
            )
    except Exception as e:
        st.warning(f"Could not load completed games: {str(e)}")
        completed_df = pd.DataFrame()
    
    # Tabs for Upcoming and Completed
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Upcoming", "✅ Completed", "📊 Season", "ℹ️ Methodology"])
    
    with tab1:
        show_upcoming_predictions(predictions, sport_code)
    
    with tab2:
        show_completed_predictions(completed_df, sport_code)
    
    with tab3:
        show_season_stats(db_manager, sport_code)
    
    with tab4:
        show_methodology(sport_code)

def show_upcoming_predictions(predictions, sport_code):
    if not predictions:
        st.info(f"No upcoming {sport_code} games found for this date. Try a different date or update data.")
        return
    
    st.markdown(f'<div class="pred-table">', unsafe_allow_html=True)
    
    # Group predictions by date
    from collections import defaultdict
    games_by_date = defaultdict(list)
    
    for pred in predictions:
        # Parse game date - handle both DD/MM/YYYY and other formats
        game_date_str = str(pred.get('game_date', ''))
        try:
            # Try DD/MM/YYYY format first
            if '/' in game_date_str:
                date_part = game_date_str.split(' ')[0]  # Remove time if present
                day, month, year = date_part.split('/')
                game_date = datetime(int(year), int(month), int(day)).date()
            else:
                # Try standard datetime parsing
                game_date = pd.to_datetime(game_date_str).date()
        except:
            game_date = datetime.now().date()  # Fallback
        
        games_by_date[game_date].append(pred)
    
    # Sort dates
    sorted_dates = sorted(games_by_date.keys())
    
    # Display games grouped by date
    for game_date in sorted_dates:
        # Display date header
        date_formatted = game_date.strftime('%A, %B %d, %Y')
        st.markdown(f"### 📅 {date_formatted}")
        
        # Display games for this date
        for pred in games_by_date[game_date]:
            try:
                away_team = pred.get('away_team', 'Away')
                home_team = pred.get('home_team', 'Home')
                win_prob_home = pred.get('home_win_probability', 0.5) * 100
                win_prob_away = pred.get('away_win_probability', 0.5) * 100
                
                # Get individual model predictions
                elo_prob = pred.get('elo_home_prob', 0.5) * 100
                log_prob = pred.get('logistic_home_prob', 0.5) * 100
                xgb_prob = pred.get('xgboost_home_prob', 0.5) * 100
                
                # Create game card
                with st.container():
                    st.markdown("---")
                    
                    # Game header
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f"### {away_team} @ {home_team}")
                    
                    # Win probabilities
                    col_away, col_home = st.columns(2)
                    with col_away:
                        st.metric("Away Win %", f"{win_prob_away:.1f}%")
                    with col_home:
                        st.metric("Home Win %", f"{win_prob_home:.1f}%")
                    
                    # Model Breakdown
                    st.markdown(f"**{home_team} Model Breakdown:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Elo", f"{elo_prob:.1f}%")
                    with col2:
                        st.metric("Logistic", f"{log_prob:.1f}%")
                    with col3:
                        st.metric("XGBoost", f"{xgb_prob:.1f}%")
                    with col4:
                        st.metric("Blended", f"{win_prob_home:.1f}%")
                    
            except Exception as e:
                st.warning(f"Error processing prediction: {str(e)}")
                continue
    
    st.markdown('</div>', unsafe_allow_html=True)

def format_american_odds(odds):
    """Format odds in American format (+150, -120, etc.)"""
    if odds is None:
        return "N/A"
    if odds > 0:
        return f"+{int(odds)}"
    return str(int(odds))

def calculate_bet_value(away_prob, home_prob, away_odds, home_odds):
    """Calculate bet value indicator"""
    if not away_odds or not home_odds:
        # Fallback to confidence-based calculation
        confidence = abs(home_prob - 0.5)
        if confidence > 0.15:
            return "🔥🔥"
        elif confidence > 0.08:
            return "🔥"
        else:
            return "→"
    
    # Calculate edge over market
    away_market_prob = american_to_prob(away_odds)
    home_market_prob = american_to_prob(home_odds)
    
    away_edge = away_prob - away_market_prob
    home_edge = home_prob - home_market_prob
    
    max_edge = max(away_edge, home_edge)
    
    if max_edge > 0.10:  # 10%+ edge
        return "🔥🔥🔥"
    elif max_edge > 0.05:  # 5-10% edge
        return "🔥🔥"
    elif max_edge > 0.02:  # 2-5% edge
        return "🔥"
    else:
        return "→"

def american_to_prob(odds):
    """Convert American odds to probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def determine_recommendation(away_prob, home_prob, away_odds, home_odds):
    """Determine recommended bet based on edge"""
    if not away_odds or not home_odds:
        # Just pick the higher probability
        if away_prob > home_prob + 0.05:
            return f"Away"
        elif home_prob > away_prob + 0.05:
            return f"Home"
        else:
            return "Pass"
    
    away_edge = away_prob - american_to_prob(away_odds)
    home_edge = home_prob - american_to_prob(home_odds)
    
    if away_edge > 0.02 and away_edge > home_edge:
        return f"Away ({format_american_odds(away_odds)})"
    elif home_edge > 0.02 and home_edge > away_edge:
        return f"Home ({format_american_odds(home_odds)})"
    else:
        return "Pass"

def show_completed_predictions(completed_df, sport_code):
    if completed_df.empty:
        st.info(f"No completed {sport_code} games found for this date.")
        return
    
    st.markdown(f'<div class="pred-table">', unsafe_allow_html=True)
    st.markdown(f"### Completed Games")
    
    # Create display DataFrame
    display_data = []
    for _, row in completed_df.iterrows():
        try:
            actual_winner = row['home_team'] if row.get('home_score', 0) > row.get('away_score', 0) else row['away_team']
            predicted_winner = row.get('predicted_winner', 'N/A')
            correct = "✅" if actual_winner == predicted_winner else "❌"
            
            home_win_prob = row.get('home_win_probability', 0.5) * 100
            away_win_prob = row.get('away_win_probability', 0.5) * 100
            
            display_data.append({
                'Time': row.get('game_time', 'TBD'),
                'Away Team': row.get('away_team', ''),
                'Away Win %': f"{away_win_prob:.1f}%",
                'Home Team': row.get('home_team', ''),
                'Home Win %': f"{home_win_prob:.1f}%",
                'Final Score': f"{row.get('away_score', 0)} - {row.get('home_score', 0)}",
                'Predicted': predicted_winner,
                'Actual': actual_winner,
                'Result': correct
            })
        except Exception as e:
            continue
    
    if display_data:
        df = pd.DataFrame(display_data)
        st.dataframe(
            df,
            width='stretch',
            hide_index=True
        )
        
        # Calculate and display accuracy
        correct_count = len([d for d in display_data if d['Result'] == "✅"])
        total_count = len(display_data)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        st.markdown(f"**Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)**")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_season_stats(db_manager, sport_code):
    st.markdown(f'<div class="pred-table">', unsafe_allow_html=True)
    st.markdown("### Season Prediction Results")
    
    try:
        with db_manager._get_connection() as conn:
            season_query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN win_prediction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                    ROUND(AVG(CASE WHEN win_prediction_correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy,
                    AVG(total_absolute_error) as avg_total_error,
                    AVG(win_probability) as avg_confidence
                FROM predictions
                WHERE sport = ?
                AND result_updated_at IS NOT NULL
                AND game_date >= DATE('now', '-90 days')
            """
            
            season_stats = pd.read_sql_query(season_query, conn, params=[sport_code])
            
            if not season_stats.empty and season_stats['total_predictions'].iloc[0] > 0:
                total_preds = int(season_stats['total_predictions'].iloc[0])
                correct_preds = int(season_stats['correct_predictions'].iloc[0])
                accuracy = season_stats['accuracy'].iloc[0]
                
                # Calculate ROI (assuming $100 bets at -110 odds for all predictions)
                # Win: +$90.91 (bet $110 to win $100)
                # Loss: -$110
                units_won = correct_preds * 0.9091  # Win $90.91 per correct prediction
                units_lost = (total_preds - correct_preds) * 1.0  # Lose $100 per incorrect
                net_profit = units_won - units_lost
                roi = (net_profit / total_preds) * 100 if total_preds > 0 else 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Predictions", total_preds)
                
                with col2:
                    st.metric("Correct", correct_preds)
                
                with col3:
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                with col4:
                    roi_delta = "positive" if roi > 0 else "negative"
                    st.metric("ROI", f"{roi:.1f}%", delta_color="normal")
                
                with col5:
                    st.metric("Net Profit", f"${net_profit:.2f}", delta_color="normal" if net_profit > 0 else "inverse")
                
                # Weekly performance chart
                weekly_query = """
                    SELECT 
                        strftime('%Y-%W', game_date) as week,
                        COUNT(*) as predictions,
                        ROUND(AVG(CASE WHEN win_prediction_correct = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as accuracy
                    FROM predictions
                    WHERE sport = ?
                    AND result_updated_at IS NOT NULL
                    AND game_date >= DATE('now', '-90 days')
                    GROUP BY strftime('%Y-%W', game_date)
                    ORDER BY week
                """
                
                weekly_df = pd.read_sql_query(weekly_query, conn, params=[sport_code])
                
                if not weekly_df.empty:
                    fig = px.line(
                        weekly_df,
                        x='week',
                        y='accuracy',
                        title='Weekly Prediction Accuracy',
                        markers=True
                    )
                    fig.update_layout(
                        xaxis_title="Week",
                        yaxis_title="Accuracy (%)",
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                st.info("No season data available yet. Predictions will appear here once games are completed.")
                
    except Exception as e:
        st.error(f"Error loading season stats: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_methodology(sport_code):
    st.markdown(f'<div class="pred-table">', unsafe_allow_html=True)
    st.markdown("### Prediction Methodology")
    
    st.markdown(f"""
    Our {sport_code} prediction system uses advanced machine learning algorithms to forecast game outcomes:
    
    #### Data Sources
    - **Historical Game Data**: Comprehensive game results and statistics
    - **Team Performance Metrics**: Advanced statistics including offensive and defensive ratings
    - **Player Statistics**: Individual player performance data
    - **Betting Odds**: Market odds from multiple sportsbooks
    - **Real-time Updates**: Live data feeds for the most current information
    
    #### Machine Learning Models
    - **XGBoost Classifier**: For win/loss predictions
    - **XGBoost Regressor**: For total score predictions
    - **Feature Engineering**: Over 50+ features including:
        - Team strength ratings
        - Recent form (rolling averages)
        - Head-to-head history
        - Home/away splits
        - Rest days and scheduling factors
    
    #### Model Performance
    - Models are continuously retrained with new data
    - Automated performance monitoring and improvement
    - Cross-validation ensures robust predictions
    
    #### Bet Value Indicators
    - 🔥🔥 High Value: Strong confidence, significant edge over market
    - 🔥 Medium Value: Moderate confidence, some edge over market
    - → Low Value: Close game, minimal edge
    
    #### Important Notes
    - All predictions are for informational and entertainment purposes only
    - Past performance does not guarantee future results
    - Always gamble responsibly
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance_page(predictor, db_manager, sport_code):
    st.header(f"⚡ {sport_code} Model Performance Metrics")
    
    try:
        metrics = predictor.get_model_metrics()
        
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Winner Prediction Model")
                win_metrics = metrics.get('classification', {})
                st.metric("Accuracy", f"{win_metrics.get('accuracy', 0):.3f}")
                st.metric("Precision", f"{win_metrics.get('precision', 0):.3f}")
                st.metric("Recall", f"{win_metrics.get('recall', 0):.3f}")
                st.metric("F1 Score", f"{win_metrics.get('f1_score', 0):.3f}")
            
            with col2:
                st.subheader("Totals Prediction Model")
                total_metrics = metrics.get('regression', {})
                st.metric("MAE", f"{total_metrics.get('mae', 0):.3f}")
                st.metric("RMSE", f"{total_metrics.get('rmse', 0):.3f}")
                st.metric("R² Score", f"{total_metrics.get('r2_score', 0):.3f}")
                st.metric("MAPE", f"{total_metrics.get('mape', 0):.1%}")
            
            if 'feature_importance' in metrics:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame(metrics['feature_importance'])
                fig = px.bar(
                    importance_df.head(10), 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No model metrics available. Train the model first.")
            
        if st.button("Retrain Models"):
            with st.spinner("Training models..."):
                try:
                    training_data = db_manager.get_training_data()
                    if not training_data.empty:
                        predictor.train_models(training_data)
                        st.success("Models retrained successfully!")
                        st.rerun()
                    else:
                        st.warning("No training data available")
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    
    except Exception as e:
        st.error(f"Error loading model performance: {str(e)}")

def show_data_pipeline_page(db_manager, sport_code):
    st.header(f"🔧 {sport_code} Data Pipeline Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Sources")
        
        try:
            latest_savant = db_manager.get_latest_data_timestamp('statcast')
            if latest_savant:
                st.success(f"✅ Baseball Savant: {latest_savant}")
            else:
                st.warning("⚠️ Baseball Savant: No data")
        except:
            st.error("❌ Baseball Savant: Connection error")
        
        try:
            latest_odds = db_manager.get_latest_data_timestamp('odds')
            if latest_odds:
                st.success(f"✅ Odds Data: {latest_odds}")
            else:
                st.warning("⚠️ Odds Data: No data")
        except:
            st.error("❌ Odds Data: Connection error")
    
    with col2:
        st.subheader("Database Status")
        try:
            stats = db_manager.get_database_stats()
            st.metric("Total Games", stats.get('total_games', 0))
            st.metric("Statcast Records", stats.get('statcast_records', 0))
            st.metric("Odds Records", stats.get('odds_records', 0))
            st.metric("Database Size", f"{stats.get('db_size_mb', 0):.1f} MB")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
    
    st.subheader("Manual Data Collection")
    
    if st.button("🔄 Run Full Data Update"):
        with st.spinner("Running full data update..."):
            try:
                scheduler = DataScheduler(db_manager)
                result = scheduler.run_daily_update()
                
                if result['success']:
                    st.success("Full data update completed successfully!")
                    for message in result['messages']:
                        st.info(message)
                else:
                    st.error("Data update failed")
                    for error in result['errors']:
                        st.error(error)
                        
            except Exception as e:
                st.error(f"Error running data update: {str(e)}")

def show_backtesting_page(db_manager, predictor, sport_code):
    st.header(f"📈 {sport_code} Model Backtesting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Backtest Start", value=datetime.now() - timedelta(days=90))
    with col2:
        end_date = st.date_input("Backtest End", value=datetime.now() - timedelta(days=30))
    with col3:
        min_confidence = st.slider("Minimum Confidence", 0.5, 0.9, 0.6)
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                backtester = Backtester(db_manager, predictor)
                results = backtester.run_backtest(
                    datetime.combine(start_date, datetime.min.time()), 
                    datetime.combine(end_date, datetime.min.time()), 
                    min_confidence
                )
                
                if results:
                    st.subheader("Backtest Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", results['total_predictions'])
                    with col2:
                        st.metric("Winner Accuracy", f"{results['winner_accuracy']:.1%}")
                    with col3:
                        st.metric("Totals MAE", f"{results['totals_mae']:.2f}")
                    with col4:
                        st.metric("ROI", f"{results['roi']:.1%}")
                    
                    if 'daily_performance' in results:
                        daily_perf = pd.DataFrame(results['daily_performance'])
                        fig = px.line(
                            daily_perf, 
                            x='date', 
                            y='cumulative_accuracy',
                            title="Accuracy Over Time"
                        )
                        st.plotly_chart(fig, width='stretch')
                    
                    if st.checkbox("Show Detailed Results"):
                        if 'predictions' in results:
                            pred_df = pd.DataFrame(results['predictions'])
                            st.dataframe(pred_df)
                else:
                    st.warning("No backtest results available")
                    
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

def show_upcoming_games_section(predictions, sport_code):
    """Show upcoming games section for sport page"""
    if not predictions:
        st.info(f"No upcoming {sport_code} games found for this date.")
        return
    
    # Reuse the existing display logic
    show_upcoming_predictions(predictions, sport_code)

def show_games_in_progress(db_manager, sport_code, game_date):
    """Show games currently in progress"""
    try:
        with db_manager._get_connection() as conn:
            query = """
                SELECT g.*, p.win_probability as orig_win_pct,
                       p.predicted_total
                FROM games g
                LEFT JOIN predictions p ON g.game_id = p.game_id
                WHERE DATE(g.game_date) = DATE(?)
                AND g.sport = ?
                AND g.status = 'in_progress'
                ORDER BY g.game_date
            """
            in_progress_df = pd.read_sql_query(
                query, 
                conn, 
                params=[game_date.strftime('%Y-%m-%d'), sport_code]
            )
        
        if in_progress_df.empty:
            st.info(f"No {sport_code} games currently in progress.")
            return
        
        # Display in-progress games
        for _, game in in_progress_df.iterrows():
            away_score = game.get('away_score')
            home_score = game.get('home_score')
            
            # Only show score if we have it
            if away_score is not None and home_score is not None:
                st.write(f"**{game['away_team_id']} {away_score} @ {game['home_team_id']} {home_score}** - In Progress")
            else:
                st.write(f"**{game['away_team_id']} @ {game['home_team_id']}** - In Progress")
                
    except Exception as e:
        st.warning(f"Could not load in-progress games: {str(e)}")

def show_completed_games_section(db_manager, sport_code, game_date):
    """Show completed games section for sport page"""
    try:
        with db_manager._get_connection() as conn:
            completed_query = """
                SELECT p.*, g.home_score as actual_home_score, 
                       g.away_score as actual_away_score, g.status
                FROM predictions p
                LEFT JOIN games g ON p.game_id = g.game_id
                WHERE DATE(p.game_date) = DATE(?)
                AND p.sport = ?
                AND (g.status = 'completed' OR p.result_updated_at IS NOT NULL)
                ORDER BY p.game_date DESC, p.created_at DESC
                LIMIT 20
            """
            completed_df = pd.read_sql_query(
                completed_query, 
                conn, 
                params=[game_date.strftime('%Y-%m-%d'), sport_code]
            )
    except Exception as e:
        st.warning(f"Could not load completed games: {str(e)}")
        completed_df = pd.DataFrame()
    
    if completed_df.empty:
        st.info(f"No completed {sport_code} games found for this date.")
        return
    
    # Reuse existing display logic
    show_completed_predictions(completed_df, sport_code)

def show_season_results(db_manager, sport_code):
    """Show season prediction results"""
    st.markdown("### Time Period")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        time_period = st.selectbox(
            "Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Season"],
            label_visibility="collapsed"
        )
    
    # Calculate date range based on selection
    end_date = datetime.now()
    if time_period == "Last 24 Hours":
        start_date = end_date - timedelta(hours=24)
    elif time_period == "Last 7 Days":
        start_date = end_date - timedelta(days=7)
    elif time_period == "Last 30 Days":
        start_date = end_date - timedelta(days=30)
    else:  # Season
        start_date = datetime(end_date.year, 1, 1)
    
    # Reuse existing season stats function
    show_season_stats(db_manager, sport_code)

if __name__ == "__main__":
    main()
