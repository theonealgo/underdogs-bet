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

# Initialize components
@st.cache_resource
def initialize_components():
    db_manager = DatabaseManager()
    predictor = MLBPredictor()
    api = PredictionAPI(db_manager, predictor)
    
    # Initialize learning system components
    result_tracker = ResultTracker(db_manager)
    performance_analyzer = PerformanceAnalyzer(db_manager)
    intelligent_retrainer = IntelligentRetrainer(db_manager, predictor)
    performance_visualizer = PerformanceVisualizer(db_manager)
    
    return (db_manager, predictor, api, result_tracker, 
            performance_analyzer, intelligent_retrainer, performance_visualizer)

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
    (db_manager, predictor, api, result_tracker, 
     performance_analyzer, intelligent_retrainer, performance_visualizer) = initialize_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Today's Predictions", "Historical Data", "Model Performance", 
         "Learning System", "Result Tracking", "Data Pipeline", "Backtesting"]
    )
    
    if page == "Today's Predictions":
        show_predictions_page(api, db_manager)
    elif page == "Historical Data":
        show_historical_data_page(db_manager)
    elif page == "Model Performance":
        show_model_performance_page(predictor, db_manager)
    elif page == "Learning System":
        show_learning_system_page(performance_analyzer, intelligent_retrainer, performance_visualizer)
    elif page == "Result Tracking":
        show_result_tracking_page(result_tracker, performance_visualizer, db_manager)
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
                    from src.data_collectors.mlb_schedule_collector import MLBScheduleCollector
                    
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
                results = backtester.run_backtest(datetime.combine(start_date, datetime.min.time()), 
                                                  datetime.combine(end_date, datetime.min.time()), 
                                                  min_confidence)
                
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

def show_learning_system_page(performance_analyzer, intelligent_retrainer, performance_visualizer):
    st.header("🧠 Automated Learning System")
    st.markdown("Monitor how the model learns from its mistakes and improves over time")
    
    # Learning system status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Current Performance")
        try:
            summary = performance_visualizer.get_performance_summary_stats(days=7)
            
            if 'error' not in summary:
                accuracy = summary.get('average_accuracy')
                if accuracy:
                    st.metric("7-Day Accuracy", f"{accuracy:.1%}", 
                             delta=f"{summary.get('accuracy_trend', 0):.2%}" if summary.get('accuracy_trend') else None)
                    st.metric("Total Predictions", summary.get('total_predictions', 0))
                    st.metric("Status", summary.get('accuracy_status', 'unknown').title())
                else:
                    st.warning("No recent performance data available")
            else:
                st.error("Error loading performance data")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("🔄 Retraining Status")
        try:
            retraining_status = intelligent_retrainer.get_retraining_status()
            
            if 'error' not in retraining_status:
                if retraining_status.get('retraining_in_progress'):
                    st.warning("🔄 Retraining in progress...")
                else:
                    last_retrain = retraining_status.get('last_retrain_date')
                    if last_retrain:
                        days_since = retraining_status.get('days_since_retrain', 0)
                        st.metric("Last Retrain", f"{days_since} days ago")
                    else:
                        st.info("No retraining history")
                
                evaluation = retraining_status.get('current_evaluation', {})
                if evaluation.get('retraining_needed'):
                    st.warning(f"⚠️ Retraining needed: {evaluation.get('priority', 'unknown')} priority")
                else:
                    st.success("✅ Performance stable")
            else:
                st.error("Error loading retraining status")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with col3:
        st.subheader("⚡ Actions")
        
        if st.button("🔍 Analyze Performance", help="Run detailed error analysis"):
            with st.spinner("Analyzing performance..."):
                try:
                    analysis = performance_analyzer.analyze_prediction_errors(days_back=14)
                    if 'error' not in analysis:
                        st.session_state['analysis_results'] = analysis
                        st.success("Analysis completed!")
                    else:
                        st.error(f"Analysis failed: {analysis.get('error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.button("🔄 Check Retraining Need", help="Evaluate if retraining is needed"):
            with st.spinner("Evaluating retraining need..."):
                try:
                    evaluation = intelligent_retrainer.evaluate_retraining_need()
                    if 'error' not in evaluation:
                        st.session_state['retraining_evaluation'] = evaluation
                        if evaluation.get('retraining_needed'):
                            st.warning(f"Retraining recommended: {evaluation.get('priority')} priority")
                        else:
                            st.success("No retraining needed currently")
                    else:
                        st.error(f"Evaluation failed: {evaluation.get('error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.button("🚀 Force Retrain", help="Force model retraining"):
            with st.spinner("Starting forced retraining..."):
                try:
                    result = intelligent_retrainer.force_retraining("Manual trigger from UI")
                    if result.get('success'):
                        st.success("Retraining completed successfully!")
                        improvement = result.get('improvements', {}).get('accuracy_change')
                        if improvement:
                            st.metric("Accuracy Change", f"{improvement:+.2%}")
                    else:
                        st.error(f"Retraining failed: {result.get('error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Performance visualizations
    st.subheader("📊 Performance Trends")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Accuracy Trends", "Learning Progress", "Error Analysis", "Team Performance"])
    
    with tab1:
        try:
            fig = performance_visualizer.create_accuracy_trend_chart(days=30)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating accuracy chart: {str(e)}")
    
    with tab2:
        try:
            fig = performance_visualizer.create_learning_progress_chart()
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating learning progress chart: {str(e)}")
    
    with tab3:
        try:
            fig = performance_visualizer.create_error_analysis_chart(days=14)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating error analysis chart: {str(e)}")
    
    with tab4:
        try:
            fig = performance_visualizer.create_team_performance_chart(days=30)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating team performance chart: {str(e)}")
    
    # Show detailed analysis results if available
    if 'analysis_results' in st.session_state:
        st.subheader("🔍 Latest Analysis Results")
        analysis = st.session_state['analysis_results']
        
        if 'actionable_insights' in analysis:
            insights = analysis['actionable_insights']
            if insights:
                st.write("**Actionable Insights:**")
                for insight in insights:
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(insight.get('priority'), "🔵")
                    st.write(f"{priority_color} **{insight.get('issue')}**: {insight.get('description')}")
                    st.write(f"   📋 Action: {insight.get('action')}")
            else:
                st.success("✅ No significant issues found")
    
    # Show retraining evaluation if available
    if 'retraining_evaluation' in st.session_state:
        st.subheader("🔄 Retraining Evaluation")
        evaluation = st.session_state['retraining_evaluation']
        
        if evaluation.get('triggers'):
            st.write("**Detected Triggers:**")
            for trigger in evaluation['triggers']:
                severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(trigger.get('severity'), "🔵")
                st.write(f"{severity_color} {trigger.get('description')}")
        
        if evaluation.get('recommendations'):
            st.write("**Recommendations:**")
            for rec in evaluation['recommendations']:
                st.write(f"• {rec}")

def show_result_tracking_page(result_tracker, performance_visualizer, db_manager):
    st.header("📊 Result Tracking & Accuracy Monitoring")
    st.markdown("Track how predictions compare to actual game results")
    
    # Result tracking controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Data Controls")
        
        if st.button("🔄 Fetch Latest Results", help="Get actual game results from MLB API"):
            with st.spinner("Fetching game results..."):
                try:
                    results = result_tracker.fetch_and_update_results()
                    if results.get('success'):
                        st.success(f"Updated {results.get('predictions_updated', 0)} predictions with actual results")
                        st.info(f"Processed {results.get('games_processed', 0)} completed games")
                        if results.get('insights_generated'):
                            st.info(f"Generated {results['insights_generated']} new insights")
                    else:
                        st.error(f"Failed to fetch results: {results.get('error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        days_back = st.selectbox("Days to analyze", [7, 14, 30, 60], index=1)
        
        if st.button("📈 Refresh Analysis"):
            st.rerun()
    
    with col1:
        # Recent accuracy metrics
        try:
            recent_accuracy = result_tracker.get_recent_accuracy(days=days_back)
            
            if 'error' not in recent_accuracy and not recent_accuracy.get('no_data'):
                st.subheader(f"📊 Last {days_back} Days Performance")
                
                col1_inner, col2_inner, col3_inner, col4_inner = st.columns(4)
                
                with col1_inner:
                    accuracy = recent_accuracy.get('average_accuracy')
                    st.metric("Accuracy", f"{accuracy:.1%}" if accuracy else "N/A")
                
                with col2_inner:
                    st.metric("Total Predictions", recent_accuracy.get('total_predictions', 0))
                
                with col3_inner:
                    st.metric("Correct Predictions", recent_accuracy.get('total_correct', 0))
                
                with col4_inner:
                    mae = recent_accuracy.get('average_mae')
                    st.metric("Average MAE", f"{mae:.2f}" if mae else "N/A")
            else:
                st.warning("No recent accuracy data available")
                
        except Exception as e:
            st.error(f"Error loading recent accuracy: {str(e)}")
    
    # Visualization tabs
    st.subheader("📈 Performance Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Confidence Analysis", "Recent Trends", "Team Breakdown"])
    
    with tab1:
        try:
            fig = performance_visualizer.create_prediction_confidence_analysis(days=days_back)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Understanding Confidence Analysis:**
            - **Calibration Plot**: Shows how well prediction confidence matches actual accuracy
            - **Confidence Distribution**: Shows the range of confidence levels in predictions
            - **Accuracy by Confidence**: Higher confidence predictions should be more accurate
            """)
        except Exception as e:
            st.error(f"Error creating confidence analysis: {str(e)}")
    
    with tab2:
        try:
            fig = performance_visualizer.create_accuracy_trend_chart(days=days_back)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating trends chart: {str(e)}")
    
    with tab3:
        try:
            fig = performance_visualizer.create_team_performance_chart(days=days_back)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Team Performance Insights:**
            - Teams on the left are harder to predict accurately
            - Teams with high error rates may need special attention
            - Look for systematic biases in predictions
            """)
        except Exception as e:
            st.error(f"Error creating team performance chart: {str(e)}")
    
    # Recent predictions with results
    st.subheader("🎯 Recent Predictions vs Results")
    
    try:
        with db_manager.get_connection() as conn:
            query = """
                SELECT 
                    game_date,
                    home_team_id,
                    away_team_id,
                    predicted_winner,
                    actual_winner,
                    win_probability,
                    win_prediction_correct,
                    predicted_total,
                    actual_total,
                    total_absolute_error
                FROM predictions 
                WHERE sport = 'MLB' 
                AND result_updated_at IS NOT NULL
                AND game_date >= DATE('now', '-{} days')
                ORDER BY game_date DESC, game_id
                LIMIT 50
            """.format(days_back)
            
            df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            # Add some formatting
            df['win_correct'] = df['win_prediction_correct'].replace({1: '✅', 0: '❌'})
            df['confidence'] = df['win_probability'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            df['total_error'] = df['total_absolute_error'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
            
            # Display formatted table
            display_df = df[['game_date', 'home_team_id', 'away_team_id', 'win_correct', 
                           'confidence', 'predicted_total', 'actual_total', 'total_error']]
            display_df.columns = ['Date', 'Home', 'Away', 'Win ✓', 'Confidence', 'Pred Total', 'Actual Total', 'Error']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Summary stats
            correct_pct = df['win_prediction_correct'].mean() if len(df) > 0 else 0
            avg_error = df['total_absolute_error'].mean() if len(df) > 0 else 0
            
            st.info(f"**Summary**: {correct_pct:.1%} win accuracy, {avg_error:.2f} average total error across {len(df)} games")
        else:
            st.warning("No recent predictions with results found")
            
    except Exception as e:
        st.error(f"Error loading recent predictions: {str(e)}")

if __name__ == "__main__":
    main()
