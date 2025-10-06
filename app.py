import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_storage.database import DatabaseManager
from src.models.prediction_models import MLBPredictor
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
    api = PredictionAPI(db_manager, predictor)
    sport_data_manager = SportDataManager()
    
    result_tracker = ResultTracker(db_manager)
    performance_analyzer = PerformanceAnalyzer(db_manager)
    intelligent_retrainer = IntelligentRetrainer(db_manager, predictor)
    performance_visualizer = PerformanceVisualizer(db_manager)
    
    return (db_manager, predictor, api, sport_data_manager, result_tracker, 
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
    (db_manager, predictor, api, sport_data_manager, result_tracker, 
     performance_analyzer, intelligent_retrainer, performance_visualizer) = initialize_components()
    
    # Sport mapping (must match SportDataManager collector keys)
    sport_mapping = {
        "MLB ⚾": "MLB",
        "NBA 🏀": "NBA",
        "NFL 🏈": "NFL",
        "NHL 🏒": "NHL",
        "NCAA Football 🏈": "NCAA",  # Maps to NCAA Football collector
        "NCAA Basketball 🏀": "NCAAB",
        "WNBA 🏀": "WNBA"
    }
    
    # Top navigation header
    st.markdown('<div class="sport-nav">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    
    with col1:
        st.markdown("### 🏆 SPORTS PREDICTIONS")
    
    with col2:
        # Sport selection dropdown
        sports = ["MLB ⚾", "NBA 🏀", "NFL 🏈", "NHL 🏒", "NCAA Football 🏈", "NCAA Basketball 🏀", "WNBA 🏀"]
        
        if 'selected_sport' not in st.session_state:
            st.session_state.selected_sport = "MLB ⚾"
        
        selected_sport = st.selectbox(
            "Select Sport",
            sports,
            index=sports.index(st.session_state.selected_sport) if st.session_state.selected_sport in sports else 0,
            key="sport_dropdown"
        )
        
        if selected_sport != st.session_state.selected_sport:
            st.session_state.selected_sport = selected_sport
            st.rerun()
    
    with col3:
        # Additional pages dropdown
        page = st.selectbox(
            "Page",
            ["Predictions", "Model Performance", "Backtesting", "Data Pipeline"],
            key="page_dropdown"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Extract sport code using mapping
    sport_code = sport_mapping.get(st.session_state.selected_sport, "MLB")
    
    # Route to appropriate page
    if page == "Predictions":
        show_predictions_page(api, db_manager, sport_data_manager, sport_code)
    elif page == "Model Performance":
        show_model_performance_page(predictor, db_manager, sport_code)
    elif page == "Backtesting":
        show_backtesting_page(db_manager, predictor, sport_code)
    elif page == "Data Pipeline":
        show_data_pipeline_page(db_manager, sport_code)

def show_predictions_page(api, db_manager, sport_data_manager, sport_code):
    # Date navigation
    if 'prediction_date' not in st.session_state:
        st.session_state.prediction_date = datetime.now().date()
    
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
                            'game_id': game.get('game_id', ''),
                            'game_time': game.get('game_time', 'TBD'),
                            'game_date': game.get('game_date', prediction_date),
                            'home_team': game.get('home_team', ''),
                            'away_team': game.get('away_team', ''),
                            'home_win_probability': 0.5,  # Neutral until ML model is trained
                            'away_win_probability': 0.5,
                            'predicted_home_score': None,
                            'predicted_away_score': None,
                            'predicted_winner': 'TBD - ML model training in progress'
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
    st.markdown(f"### Upcoming Games for {st.session_state.prediction_date.strftime('%B %d, %Y')}")
    
    # Create DataFrame for display
    display_data = []
    for pred in predictions:
        try:
            win_prob_home = pred.get('home_win_probability', 0.5) * 100
            win_prob_away = pred.get('away_win_probability', 0.5) * 100
            
            home_score = pred.get('predicted_home_score', 0)
            away_score = pred.get('predicted_away_score', 0)
            total_runs = home_score + away_score if home_score and away_score else 0
            
            # Get odds data if available
            away_odds = pred.get('away_odds')
            home_odds = pred.get('home_odds')
            away_spread = pred.get('away_spread')
            away_spread_odds = pred.get('away_spread_odds')
            home_spread = pred.get('home_spread')
            home_spread_odds = pred.get('home_spread_odds')
            total_line = pred.get('total_line')
            over_odds = pred.get('over_odds')
            under_odds = pred.get('under_odds')
            
            # Format moneyline odds
            away_ml = format_american_odds(away_odds) if away_odds else "N/A"
            home_ml = format_american_odds(home_odds) if home_odds else "N/A"
            
            # Format spread
            if away_spread is not None and away_spread_odds:
                away_spread_display = f"{away_spread:+.1f} ({format_american_odds(away_spread_odds)})"
            else:
                away_spread_display = "N/A"
            
            if home_spread is not None and home_spread_odds:
                home_spread_display = f"{home_spread:+.1f} ({format_american_odds(home_spread_odds)})"
            else:
                home_spread_display = "N/A"
            
            # Format totals
            if total_line and over_odds and under_odds:
                totals_display = f"O{total_line} ({format_american_odds(over_odds)}) / U{total_line} ({format_american_odds(under_odds)})"
            else:
                totals_display = "N/A"
            
            # Calculate bet value based on odds comparison
            bet_value = calculate_bet_value(win_prob_away/100, win_prob_home/100, away_odds, home_odds)
            
            # Determine recommended bet
            recommended = determine_recommendation(win_prob_away/100, win_prob_home/100, away_odds, home_odds)
            
            display_data.append({
                'Time': pred.get('game_time', 'TBD'),
                'Away Team': pred.get('away_team', ''),
                'Away Win %': f"{win_prob_away:.1f}%",
                'Away ML': away_ml,
                'Away Spread': away_spread_display,
                'Home Team': pred.get('home_team', ''),
                'Home Win %': f"{win_prob_home:.1f}%",
                'Home ML': home_ml,
                'Home Spread': home_spread_display,
                'Predicted Score': f"{away_score:.0f}-{home_score:.0f}" if home_score and away_score else "N/A",
                'Total': f"{total_runs:.1f}" if total_runs else "N/A",
                'O/U': totals_display,
                'Bet Value': bet_value,
                'Pick': recommended
            })
        except Exception as e:
            st.warning(f"Error processing prediction: {str(e)}")
            continue
    
    if display_data:
        df = pd.DataFrame(display_data)
        
        # Style the dataframe - compact mobile-friendly view
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("⏰", width="small"),
                "Away Team": st.column_config.TextColumn("Away", width="small"),
                "Away Win %": st.column_config.TextColumn("W%", width="small"),
                "Away ML": st.column_config.TextColumn("ML", width="small"),
                "Away Spread": st.column_config.TextColumn("Sprd", width="small"),
                "Home Team": st.column_config.TextColumn("Home", width="small"),
                "Home Win %": st.column_config.TextColumn("W%", width="small"),
                "Home ML": st.column_config.TextColumn("ML", width="small"),
                "Home Spread": st.column_config.TextColumn("Sprd", width="small"),
                "Predicted Score": st.column_config.TextColumn("Score", width="small"),
                "Total": st.column_config.TextColumn("Tot", width="small"),
                "O/U": st.column_config.TextColumn("O/U", width="small"),
                "Bet Value": st.column_config.TextColumn("Val", width="small"),
                "Pick": st.column_config.TextColumn("Pick", width="small")
            }
        )
    
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
                    AVG(total_runs_error) as avg_total_error
                FROM predictions
                WHERE sport = ?
                AND result_updated_at IS NOT NULL
                AND game_date >= DATE('now', '-90 days')
            """
            
            season_stats = pd.read_sql_query(season_query, conn, params=[sport_code])
            
            if not season_stats.empty and season_stats['total_predictions'].iloc[0] > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", int(season_stats['total_predictions'].iloc[0]))
                
                with col2:
                    st.metric("Correct", int(season_stats['correct_predictions'].iloc[0]))
                
                with col3:
                    st.metric("Accuracy", f"{season_stats['accuracy'].iloc[0]:.1f}%")
                
                with col4:
                    avg_error = season_stats['avg_total_error'].iloc[0]
                    if pd.notna(avg_error):
                        st.metric("Avg Total Error", f"{avg_error:.2f}")
                
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
                    st.plotly_chart(fig, use_container_width=True)
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
                st.plotly_chart(fig, use_container_width=True)
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
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if st.checkbox("Show Detailed Results"):
                        if 'predictions' in results:
                            pred_df = pd.DataFrame(results['predictions'])
                            st.dataframe(pred_df)
                else:
                    st.warning("No backtest results available")
                    
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

if __name__ == "__main__":
    main()
