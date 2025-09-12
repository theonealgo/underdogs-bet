import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_storage.database import DatabaseManager
from models.prediction_models import MLBPredictor
from data_collectors.baseball_savant_scraper import BaseballSavantScraper
from utils.scheduler import DataScheduler
from api.prediction_api import PredictionAPI
from backtesting.backtester import Backtester

# Initialize components
@st.cache_resource
def initialize_components():
    db_manager = DatabaseManager()
    predictor = MLBPredictor()
    api = PredictionAPI(db_manager, predictor)
    return db_manager, predictor, api

def main():
    st.set_page_config(
        page_title="MLB Prediction System",
        page_icon="⚾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚾ MLB Game Prediction System")
    st.markdown("Real-time MLB game winner and totals predictions using machine learning")
    
    # Initialize components
    db_manager, predictor, api = initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Today's Predictions", "Historical Data", "Model Performance", "Data Pipeline", "Backtesting"]
    )
    
    if page == "Today's Predictions":
        show_predictions_page(api, db_manager)
    elif page == "Historical Data":
        show_historical_data_page(db_manager)
    elif page == "Model Performance":
        show_model_performance_page(predictor, db_manager)
    elif page == "Data Pipeline":
        show_data_pipeline_page(db_manager)
    elif page == "Backtesting":
        show_backtesting_page(db_manager, predictor)

def show_predictions_page(api, db_manager):
    st.header("Today's Game Predictions")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Data Controls")
        if st.button("🔄 Update Data", help="Fetch latest data from sources"):
            with st.spinner("Updating data..."):
                try:
                    # Import MLB schedule collector
                    from data_collectors.mlb_schedule_collector import MLBScheduleCollector
                    
                    # Update today's schedule first
                    schedule_collector = MLBScheduleCollector()
                    todays_games = schedule_collector.get_todays_games()
                    if not todays_games.empty:
                        db_manager.store_games(todays_games)
                        st.success(f"Found {len(todays_games)} games for today!")
                    else:
                        st.warning("No games scheduled for today")
                    
                    # Update Baseball Savant data
                    savant_scraper = BaseballSavantScraper()
                    savant_data = savant_scraper.get_recent_games(days=7)
                    if not savant_data.empty:
                        db_manager.store_statcast_data(savant_data)
                        st.success("Baseball Savant data updated!")
                    
                    # Note: OddsShark dependency removed - no longer using odds data
                        
                except Exception as e:
                    st.error(f"Error updating data: {str(e)}")
        
        if st.button("🤖 Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    predictions = api.get_todays_predictions()
                    if predictions:
                        st.success(f"Generated {len(predictions)} predictions!")
                        st.session_state['predictions'] = predictions
                    else:
                        st.warning("No games found for today")
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
    
    with col1:
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            
            for prediction in predictions:
                with st.expander(f"🏟️ {prediction['away_team']} @ {prediction['home_team']}", expanded=True):
                    col1_inner, col2_inner, col3_inner = st.columns(3)
                    
                    with col1_inner:
                        st.metric(
                            "Winner Prediction",
                            prediction['predicted_winner'],
                            f"{prediction['win_probability']:.1%} confidence"
                        )
                    
                    with col2_inner:
                        st.metric(
                            "Total Runs",
                            f"{prediction['predicted_total']:.1f}",
                            f"{prediction['total_confidence']:.1%} confidence"
                        )
                    
                    with col3_inner:
                        st.metric(
                            "Game Time",
                            prediction.get('game_time', 'TBD'),
                            f"Model Score: {prediction.get('model_score', 0):.2f}"
                        )
                    
                    # Show key factors
                    if 'key_factors' in prediction:
                        st.write("**Key Factors:**")
                        for factor in prediction['key_factors'][:3]:
                            st.write(f"• {factor}")
        else:
            st.info("Click 'Generate Predictions' to see today's game predictions")

def show_historical_data_page(db_manager):
    st.header("Historical Data Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    if st.button("Load Historical Data"):
        try:
            # Get historical game data
            historical_data = db_manager.get_historical_games(start_date, end_date)
            
            if not historical_data.empty:
                st.subheader(f"Games from {start_date} to {end_date}")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Games", len(historical_data))
                with col2:
                    avg_runs = historical_data['total_runs'].mean() if 'total_runs' in historical_data.columns else 0
                    st.metric("Avg Total Runs", f"{avg_runs:.1f}")
                with col3:
                    home_wins = historical_data['home_win'].sum() if 'home_win' in historical_data.columns else 0
                    home_win_pct = home_wins / len(historical_data) if len(historical_data) > 0 else 0
                    st.metric("Home Win %", f"{home_win_pct:.1%}")
                with col4:
                    unique_teams = historical_data[['home_team', 'away_team']].stack().nunique() if not historical_data.empty else 0
                    st.metric("Teams", unique_teams)
                
                # Data visualization
                if 'total_runs' in historical_data.columns:
                    fig = px.histogram(
                        historical_data, 
                        x='total_runs',
                        title="Distribution of Total Runs",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data
                st.subheader("Raw Data")
                st.dataframe(historical_data.head(100))
            else:
                st.warning("No historical data found for the selected date range")
                
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")

def show_model_performance_page(predictor, db_manager):
    st.header("Model Performance Metrics")
    
    try:
        # Get model metrics
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
            
            # Feature importance
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
                    # Get training data
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

def show_data_pipeline_page(db_manager):
    st.header("Data Pipeline Status")
    
    # Pipeline status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Sources")
        
        # Check Baseball Savant data
        try:
            latest_savant = db_manager.get_latest_data_timestamp('statcast')
            if latest_savant:
                st.success(f"✅ Baseball Savant: {latest_savant}")
            else:
                st.warning("⚠️ Baseball Savant: No data")
        except:
            st.error("❌ Baseball Savant: Connection error")
        
        # Check OddsShark data
        try:
            latest_odds = db_manager.get_latest_data_timestamp('odds')
            if latest_odds:
                st.success(f"✅ OddsShark: {latest_odds}")
            else:
                st.warning("⚠️ OddsShark: No data")
        except:
            st.error("❌ OddsShark: Connection error")
    
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
    
    # Manual data collection
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

def show_backtesting_page(db_manager, predictor):
    st.header("Model Backtesting")
    
    # Backtesting parameters
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
                results = backtester.run_backtest(start_date, end_date, min_confidence)
                
                if results:
                    st.subheader("Backtest Results")
                    
                    # Overall metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", results['total_predictions'])
                    with col2:
                        st.metric("Winner Accuracy", f"{results['winner_accuracy']:.1%}")
                    with col3:
                        st.metric("Totals MAE", f"{results['totals_mae']:.2f}")
                    with col4:
                        st.metric("ROI", f"{results['roi']:.1%}")
                    
                    # Performance over time
                    if 'daily_performance' in results:
                        daily_perf = pd.DataFrame(results['daily_performance'])
                        fig = px.line(
                            daily_perf, 
                            x='date', 
                            y='cumulative_accuracy',
                            title="Accuracy Over Time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
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
