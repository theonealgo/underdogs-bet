import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import sqlite3

from src.data_storage.database import DatabaseManager


class PerformanceVisualizer:
    """
    Create comprehensive visualizations for model performance monitoring and learning trends
    """
    
    def __init__(self, db_manager: DatabaseManager, sport: str = 'MLB'):
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.sport = sport
        
        # Color scheme for consistent visualizations
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'accuracy': '#2ca02c',
            'error': '#d62728',
            'prediction': '#1f77b4',
            'actual': '#ff7f0e'
        }
    
    def create_accuracy_trend_chart(self, days: int = 30) -> go.Figure:
        """
        Create accuracy trend visualization over time
        
        Args:
            days: Number of days to show in the trend
            
        Returns:
            Plotly figure showing accuracy trends
        """
        try:
            # Get accuracy data from database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT 
                        date_period,
                        accuracy_rate,
                        total_predictions,
                        total_mae,
                        confidence_calibration
                    FROM prediction_accuracy
                    WHERE sport = ? 
                    AND date_period >= DATE(?)
                    AND date_period <= DATE(?)
                    ORDER BY date_period
                """
                
                df = pd.read_sql_query(query, conn, params=[self.sport, start_date.date(), end_date.date()])
            
            if df.empty:
                # Return empty chart with message
                fig = go.Figure()
                fig.add_annotation(
                    text="No accuracy data available for the selected period",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(title="Model Accuracy Trends - No Data Available")
                return fig
            
            # Convert date_period to datetime
            df['date_period'] = pd.to_datetime(df['date_period'])
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Prediction Accuracy Over Time', 'Total Predictions Volume'),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            # Main accuracy line
            fig.add_trace(
                go.Scatter(
                    x=df['date_period'],
                    y=df['accuracy_rate'],
                    mode='lines+markers',
                    name='Accuracy Rate',
                    line=dict(color=self.colors['accuracy'], width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                '<b>Accuracy:</b> %{y:.1%}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add accuracy threshold line
            fig.add_shape(
                type="line",
                x0=0, x1=1, y0=0.52, y1=0.52,
                xref="x domain", yref="y",
                line=dict(dash="dash", color=self.colors['warning']),
                row=1, col=1
            )
            fig.add_annotation(
                x=0.02, y=0.52, text="Minimum Target (52%)", 
                xref="x domain", yref="y", row=1, col=1,
                showarrow=False, font=dict(size=10)
            )
            
            # Add excellent performance line
            fig.add_shape(
                type="line",
                x0=0, x1=1, y0=0.55, y1=0.55,
                xref="x domain", yref="y",
                line=dict(dash="dot", color=self.colors['success']),
                row=1, col=1
            )
            fig.add_annotation(
                x=0.02, y=0.55, text="Excellent (55%)", 
                xref="x domain", yref="y", row=1, col=1,
                showarrow=False, font=dict(size=10)
            )
            
            # Volume bar chart
            fig.add_trace(
                go.Bar(
                    x=df['date_period'],
                    y=df['total_predictions'],
                    name='Daily Predictions',
                    marker_color=self.colors['primary'],
                    opacity=0.7,
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                '<b>Predictions:</b> %{y}<br>' +
                                '<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Calculate trend line
            if len(df) >= 5:
                # Simple linear trend
                x_numeric = np.arange(len(df))
                z = np.polyfit(x_numeric, df['accuracy_rate'], 1)
                trend_line = np.poly1d(z)(x_numeric)
                
                fig.add_trace(
                    go.Scatter(
                        x=df['date_period'],
                        y=trend_line,
                        mode='lines',
                        name='Trend',
                        line=dict(color=self.colors['danger'], width=2, dash='dash'),
                        opacity=0.8,
                        hovertemplate='<b>Trend:</b> %{y:.1%}<br><extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"Model Performance Trends (Last {days} Days)",
                showlegend=True,
                height=600,
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            # Update axes
            fig.update_yaxes(title_text="Accuracy Rate", tickformat='.1%', row=1, col=1)
            fig.update_yaxes(title_text="Prediction Count", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating accuracy trend chart: {str(e)}")
            # Return error chart
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color='red')
            )
            fig.update_layout(title="Error in Accuracy Trend Chart")
            return fig
    
    def create_error_analysis_chart(self, days: int = 14) -> go.Figure:
        """
        Create error analysis visualization showing different types of prediction errors
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Plotly figure showing error analysis
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT 
                        win_probability,
                        win_prediction_correct,
                        total_prediction_error,
                        total_absolute_error,
                        home_team_id,
                        away_team_id,
                        game_date
                    FROM predictions
                    WHERE sport = ? 
                    AND game_date >= DATE(?)
                    AND game_date <= DATE(?)
                    AND result_updated_at IS NOT NULL
                """
                
                df = pd.read_sql_query(query, conn, params=[self.sport, start_date.date(), end_date.date()])
            
            if df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No prediction data available for error analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(title="Error Analysis - No Data Available")
                return fig
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Confidence vs Accuracy Calibration',
                    'Total Prediction Errors Distribution',
                    'Win Prediction Accuracy by Confidence',
                    'Error Patterns Over Time'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Confidence calibration plot
            win_df = df[df['win_prediction_correct'].notna()].copy()
            if not win_df.empty:
                # Create confidence bins
                confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                win_df['confidence_bin'] = pd.cut(win_df['win_probability'], bins=confidence_bins)
                
                calibration_data = win_df.groupby('confidence_bin').agg({
                    'win_probability': 'mean',
                    'win_prediction_correct': 'mean'
                }).reset_index()
                
                # Perfect calibration line
                fig.add_trace(
                    go.Scatter(
                        x=[0.5, 1.0], y=[0.5, 1.0],
                        mode='lines',
                        name='Perfect Calibration',
                        line=dict(color='gray', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Actual calibration
                fig.add_trace(
                    go.Scatter(
                        x=calibration_data['win_probability'],
                        y=calibration_data['win_prediction_correct'],
                        mode='lines+markers',
                        name='Model Calibration',
                        line=dict(color=self.colors['primary'], width=3),
                        marker=dict(size=10),
                        hovertemplate='<b>Predicted:</b> %{x:.1%}<br>' +
                                    '<b>Actual:</b> %{y:.1%}<br>' +
                                    '<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 2. Total prediction errors histogram
            total_errors_df = df[df['total_prediction_error'].notna()]
            if not total_errors_df.empty:
                fig.add_trace(
                    go.Histogram(
                        x=total_errors_df['total_prediction_error'],
                        nbinsx=20,
                        name='Error Distribution',
                        marker_color=self.colors['warning'],
                        opacity=0.7,
                        hovertemplate='<b>Error Range:</b> %{x}<br>' +
                                    '<b>Count:</b> %{y}<br>' +
                                    '<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # 3. Win accuracy by confidence level
            if not win_df.empty:
                confidence_accuracy = win_df.groupby('confidence_bin').agg({
                    'win_prediction_correct': ['count', 'sum']
                }).reset_index()
                
                confidence_accuracy.columns = ['confidence_bin', 'total', 'correct']
                confidence_accuracy['accuracy'] = confidence_accuracy['correct'] / confidence_accuracy['total']
                confidence_accuracy['confidence_mid'] = confidence_accuracy['confidence_bin'].apply(
                    lambda x: x.mid if hasattr(x, 'mid') else 0.75
                )
                
                fig.add_trace(
                    go.Bar(
                        x=confidence_accuracy['confidence_mid'],
                        y=confidence_accuracy['accuracy'],
                        name='Win Accuracy',
                        marker_color=self.colors['success'],
                        hovertemplate='<b>Confidence:</b> %{x:.1%}<br>' +
                                    '<b>Accuracy:</b> %{y:.1%}<br>' +
                                    '<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # 4. Error patterns over time
            df['game_date'] = pd.to_datetime(df['game_date'])
            daily_errors = df.groupby(df['game_date'].dt.date).agg({
                'total_absolute_error': 'mean',
                'win_prediction_correct': 'mean'
            }).reset_index()
            
            if not daily_errors.empty:
                fig.add_trace(
                    go.Scatter(
                        x=daily_errors['game_date'],
                        y=daily_errors['total_absolute_error'],
                        mode='lines+markers',
                        name='Daily MAE',
                        line=dict(color=self.colors['error']),
                        yaxis='y1',
                        hovertemplate='<b>Date:</b> %{x}<br>' +
                                    '<b>MAE:</b> %{y:.2f}<br>' +
                                    '<extra></extra>'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"Prediction Error Analysis (Last {days} Days)",
                showlegend=True,
                height=700,
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Predicted Probability", row=1, col=1)
            fig.update_yaxes(title_text="Actual Accuracy", row=1, col=1)
            fig.update_xaxes(title_text="Prediction Error (runs)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_xaxes(title_text="Confidence Level", row=2, col=1)
            fig.update_yaxes(title_text="Win Accuracy", row=2, col=1, tickformat='.1%')
            fig.update_xaxes(title_text="Date", row=2, col=2)
            fig.update_yaxes(title_text="Mean Absolute Error", row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating error analysis chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating error analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color='red')
            )
            fig.update_layout(title="Error in Error Analysis Chart")
            return fig
    
    def create_team_performance_chart(self, days: int = 30) -> go.Figure:
        """
        Create team-specific performance analysis chart
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Plotly figure showing team performance
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT 
                        home_team_id,
                        away_team_id,
                        win_prediction_correct,
                        total_absolute_error,
                        predicted_winner,
                        actual_winner
                    FROM predictions
                    WHERE sport = ? 
                    AND game_date >= DATE(?)
                    AND game_date <= DATE(?)
                    AND result_updated_at IS NOT NULL
                """
                
                df = pd.read_sql_query(query, conn, params=[self.sport, start_date.date(), end_date.date()])
            
            if df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No team performance data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(title="Team Performance Analysis - No Data Available")
                return fig
            
            # Aggregate team statistics
            team_stats = []
            
            for _, row in df.iterrows():
                for team in [row['home_team_id'], row['away_team_id']]:
                    # Convert values to scalars to avoid Series boolean issues
                    win_correct_val = row['win_prediction_correct']
                    total_error_val = row['total_absolute_error']
                    
                    # Handle NaN values safely for scalar values
                    try:
                        if win_correct_val is None or str(win_correct_val).lower() in ['nan', 'none']:
                            win_val = None
                        else:
                            win_val = win_correct_val
                    except Exception:
                        win_val = win_correct_val
                        
                    try:
                        if total_error_val is None or str(total_error_val).lower() in ['nan', 'none']:
                            error_val = None
                        else:
                            error_val = total_error_val
                    except Exception:
                        error_val = total_error_val
                    
                    team_stats.append({
                        'team': team,
                        'win_correct': win_val,
                        'total_error': error_val
                    })
            
            team_df = pd.DataFrame(team_stats)
            
            # Calculate team metrics
            team_metrics = team_df.groupby('team').agg({
                'win_correct': ['count', 'sum', 'mean'],
                'total_error': 'mean'
            }).round(3)
            
            team_metrics.columns = ['games', 'wins_correct', 'win_accuracy', 'avg_total_error']
            team_metrics = team_metrics.reset_index()
            
            # Filter teams with at least 3 games
            team_metrics = team_metrics[team_metrics['games'] >= 3]
            
            if team_metrics.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="Insufficient team data for analysis (need 3+ games per team)",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(title="Team Performance Analysis - Insufficient Data")
                return fig
            
            # Sort by win accuracy  
            team_metrics = team_metrics.iloc[team_metrics['win_accuracy'].argsort()]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Win Prediction Accuracy by Team', 'Total Prediction Error by Team'),
                horizontal_spacing=0.1
            )
            
            # Win accuracy bar chart
            colors = [self.colors['success'] if acc >= 0.5 else self.colors['danger'] 
                     for acc in team_metrics['win_accuracy']]
            
            fig.add_trace(
                go.Bar(
                    x=team_metrics['win_accuracy'],
                    y=team_metrics['team'],
                    orientation='h',
                    name='Win Accuracy',
                    marker_color=colors,
                    hovertemplate='<b>Team:</b> %{y}<br>' +
                                '<b>Accuracy:</b> %{x:.1%}<br>' +
                                '<b>Games:</b> %{customdata}<br>' +
                                '<extra></extra>',
                    customdata=team_metrics['games']
                ),
                row=1, col=1
            )
            
            # Total error bar chart
            avg_error_series = team_metrics['avg_total_error']
            has_non_na_errors = bool((~avg_error_series.isna()).any())
            if has_non_na_errors:
                error_colors = [self.colors['success'] if err <= 1.0 else self.colors['warning'] if err <= 1.5 else self.colors['danger'] 
                               for err in team_metrics['avg_total_error'].fillna(0)]
                
                fig.add_trace(
                    go.Bar(
                        x=team_metrics['avg_total_error'],
                        y=team_metrics['team'],
                        orientation='h',
                        name='Avg Total Error',
                        marker_color=error_colors,
                        hovertemplate='<b>Team:</b> %{y}<br>' +
                                    '<b>Avg Error:</b> %{x:.2f} runs<br>' +
                                    '<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # Add reference lines
            fig.add_shape(
                type="line",
                x0=0.5, x1=0.5, y0=0, y1=1,
                xref="x", yref="y domain",
                line=dict(dash="dash", color="gray"),
                row=1, col=1
            )
            fig.add_annotation(
                x=0.5, y=0.95, text="50%", 
                xref="x", yref="y domain", row=1, col=1,
                showarrow=False, font=dict(size=10)
            )
            
            avg_error_check = team_metrics['avg_total_error']
            has_non_na_check = bool((~avg_error_check.isna()).any())
            if has_non_na_check:
                avg_error = team_metrics['avg_total_error'].mean()
                fig.add_shape(
                    type="line",
                    x0=avg_error, x1=avg_error, y0=0, y1=1,
                    xref="x", yref="y domain",
                    line=dict(dash="dash", color="gray"),
                    row=1, col=2
                )
                fig.add_annotation(
                    x=avg_error, y=0.95, text=f"Avg: {avg_error:.2f}", 
                    xref="x", yref="y domain", row=1, col=2,
                    showarrow=False, font=dict(size=10)
                )
            
            # Update layout
            fig.update_layout(
                title=f"Team Performance Analysis (Last {days} Days)",
                showlegend=False,
                height=max(400, len(team_metrics) * 25 + 100),
                margin=dict(t=100, b=50, l=100, r=50)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Win Prediction Accuracy", tickformat='.1%', row=1, col=1)
            fig.update_xaxes(title_text="Average Total Error (runs)", row=1, col=2)
            fig.update_yaxes(title_text="Team", row=1, col=1)
            fig.update_yaxes(title_text="Team", row=1, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating team performance chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating team performance chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color='red')
            )
            fig.update_layout(title="Error in Team Performance Chart")
            return fig
    
    def create_learning_progress_chart(self) -> go.Figure:
        """
        Create learning progress visualization showing model improvements over time
        
        Returns:
            Plotly figure showing learning progress
        """
        try:
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Get retraining events and performance before/after
                retrain_query = """
                    SELECT date_recorded, metric_value, metric_name
                    FROM model_metrics
                    WHERE sport = ? 
                    AND model_type = 'xgboost'
                    AND metric_name IN ('retraining_completed', 'accuracy_improvement')
                    ORDER BY date_recorded
                """
                
                retrain_df = pd.read_sql_query(retrain_query, conn)
                
                # Get accuracy trends
                accuracy_query = """
                    SELECT date_period, accuracy_rate
                    FROM prediction_accuracy
                    WHERE sport = ?
                    ORDER BY date_period
                """
                
                accuracy_df = pd.read_sql_query(accuracy_query, conn)
            
            if accuracy_df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No learning progress data available yet",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(title="Learning Progress - No Data Available")
                return fig
            
            # Convert dates
            accuracy_df['date_period'] = pd.to_datetime(accuracy_df['date_period'])
            
            # Create the main figure
            fig = go.Figure()
            
            # Add accuracy trend line
            fig.add_trace(
                go.Scatter(
                    x=accuracy_df['date_period'],
                    y=accuracy_df['accuracy_rate'],
                    mode='lines+markers',
                    name='Model Accuracy',
                    line=dict(color=self.colors['primary'], width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>Date:</b> %{x}<br>' +
                                '<b>Accuracy:</b> %{y:.1%}<br>' +
                                '<extra></extra>'
                )
            )
            
            # Add retraining markers if any
            if not retrain_df.empty:
                retrain_completed = retrain_df[retrain_df['metric_name'] == 'retraining_completed']
                if not retrain_completed.empty:
                    retrain_dates = pd.to_datetime(retrain_completed['date_recorded'])
                    
                    for retrain_date in retrain_dates:
                        # Find accuracy around retraining date
                        nearby_accuracy = accuracy_df[
                            (accuracy_df['date_period'] >= retrain_date - timedelta(days=1)) &
                            (accuracy_df['date_period'] <= retrain_date + timedelta(days=1))
                        ]
                        
                        if len(nearby_accuracy) > 0:
                            acc_series = nearby_accuracy['accuracy_rate']
                            if isinstance(acc_series, pd.Series) and not acc_series.empty:
                                acc_value = acc_series.iloc[0]
                            elif hasattr(acc_series, '__len__') and len(acc_series) > 0:
                                try:
                                    # Handle numpy array or pandas Series access safely
                                    if isinstance(acc_series, pd.Series) and not acc_series.empty:
                                        acc_value = float(acc_series.iloc[0])
                                    elif hasattr(acc_series, '__getitem__') and len(acc_series) > 0:
                                        first_val = acc_series[0]
                                        # Safer type checking and conversion
                                        try:
                                            if first_val is not None:
                                                # Check if it's a valid numeric value
                                                if isinstance(first_val, (int, float, np.number)):
                                                    # Simple numeric validation without pandas isna
                                                    try:
                                                        numeric_val = float(first_val)
                                                        if np.isfinite(numeric_val):
                                                            acc_value = numeric_val
                                                        else:
                                                            acc_value = 0.5
                                                    except (ValueError, TypeError):
                                                        acc_value = 0.5
                                                else:
                                                    acc_value = 0.5
                                            else:
                                                acc_value = 0.5
                                        except (TypeError, ValueError):
                                            acc_value = 0.5
                                        else:
                                            acc_value = 0.5
                                    else:
                                        acc_value = 0.5
                                except (TypeError, ValueError, IndexError, AttributeError):
                                    acc_value = 0.5
                            else:
                                acc_value = 0.5
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[retrain_date],
                                    y=[acc_value],
                                    mode='markers',
                                    name='Model Retrained',
                                    marker=dict(
                                        symbol='star',
                                        size=15,
                                        color=self.colors['warning'],
                                        line=dict(width=2, color='white')
                                    ),
                                    hovertemplate='<b>Retraining Event</b><br>' +
                                                '<b>Date:</b> %{x}<br>' +
                                                '<b>Accuracy:</b> %{y:.1%}<br>' +
                                                '<extra></extra>',
                                    showlegend=True
                                )
                            )
            
            # Add target accuracy lines
            fig.add_hline(
                y=0.52, line_dash="dash", line_color=self.colors['warning'],
                annotation_text="Minimum Target (52%)"
            )
            
            fig.add_hline(
                y=0.55, line_dash="dot", line_color=self.colors['success'],
                annotation_text="Excellence Target (55%)"
            )
            
            # Calculate overall learning trend
            if len(accuracy_df) >= 5:
                x_numeric = np.arange(len(accuracy_df))
                z = np.polyfit(x_numeric, accuracy_df['accuracy_rate'], 1)
                trend_slope = z[0]
                
                # Add trend annotation
                trend_direction = "improving" if trend_slope > 0 else "declining" if trend_slope < 0 else "stable"
                trend_color = self.colors['success'] if trend_slope > 0 else self.colors['danger'] if trend_slope < 0 else self.colors['primary']
                
                fig.add_annotation(
                    text=f"Overall Trend: {trend_direction.title()} ({trend_slope:.3f}/day)",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, xanchor='left', yanchor='top',
                    showarrow=False,
                    bgcolor=trend_color,
                    bordercolor="white",
                    borderwidth=1,
                    font=dict(color="white", size=12)
                )
            
            # Update layout
            fig.update_layout(
                title="Model Learning Progress Over Time",
                xaxis_title="Date",
                yaxis_title="Prediction Accuracy",
                showlegend=True,
                height=500,
                margin=dict(t=80, b=50, l=50, r=50),
                yaxis=dict(tickformat='.1%', range=[0.45, 0.65])
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating learning progress chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating learning progress chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color='red')
            )
            fig.update_layout(title="Error in Learning Progress Chart")
            return fig
    
    def create_prediction_confidence_analysis(self, days: int = 14) -> go.Figure:
        """
        Create detailed confidence analysis showing calibration and reliability
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Plotly figure showing confidence analysis
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                query = """
                    SELECT 
                        win_probability,
                        win_prediction_correct,
                        total_confidence,
                        total_absolute_error
                    FROM predictions
                    WHERE sport = ? 
                    AND game_date >= DATE(?)
                    AND game_date <= DATE(?)
                    AND result_updated_at IS NOT NULL
                    AND win_probability IS NOT NULL
                """
                
                df = pd.read_sql_query(query, conn, params=[self.sport, start_date.date(), end_date.date()])
            
            if df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No confidence data available for analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(title="Confidence Analysis - No Data Available")
                return fig
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Confidence Calibration Plot',
                    'Confidence Distribution',
                    'Accuracy by Confidence Bins',
                    'Confidence vs Total Error'
                )
            )
            
            # 1. Confidence calibration plot
            df_clean = df.dropna(subset=['win_probability', 'win_prediction_correct'])
            
            if not df_clean.empty:
                # Create confidence bins for calibration
                bins = np.arange(0.5, 1.05, 0.05)
                df_clean['conf_bin'] = pd.cut(df_clean['win_probability'], bins=bins)
                
                calibration = df_clean.groupby('conf_bin').agg({
                    'win_probability': 'mean',
                    'win_prediction_correct': ['mean', 'count']
                }).reset_index()
                
                calibration.columns = ['conf_bin', 'avg_confidence', 'actual_accuracy', 'count']
                calibration = calibration[calibration['count'] >= 3]  # At least 3 predictions per bin
                
                if not calibration.empty:
                    # Perfect calibration line
                    fig.add_trace(
                        go.Scatter(
                            x=[0.5, 1.0], y=[0.5, 1.0],
                            mode='lines',
                            name='Perfect Calibration',
                            line=dict(color='gray', dash='dash', width=2),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Actual calibration
                    fig.add_trace(
                        go.Scatter(
                            x=calibration['avg_confidence'],
                            y=calibration['actual_accuracy'],
                            mode='lines+markers',
                            name='Model Calibration',
                            line=dict(color=self.colors['primary'], width=3),
                            marker=dict(size=8),
                            hovertemplate='<b>Predicted:</b> %{x:.1%}<br>' +
                                        '<b>Actual:</b> %{y:.1%}<br>' +
                                        '<b>Count:</b> %{customdata}<br>' +
                                        '<extra></extra>',
                            customdata=calibration['count']
                        ),
                        row=1, col=1
                    )
            
            # 2. Confidence distribution histogram
            fig.add_trace(
                go.Histogram(
                    x=df['win_probability'],
                    nbinsx=20,
                    name='Confidence Distribution',
                    marker_color=self.colors['primary'],
                    opacity=0.7,
                    hovertemplate='<b>Confidence:</b> %{x:.2f}<br>' +
                                '<b>Count:</b> %{y}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. Accuracy by confidence bins
            if not df_clean.empty:
                # Broader bins for accuracy analysis
                confidence_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
                accuracy_by_range = []
                
                for low, high in confidence_ranges:
                    range_data = df_clean[
                        (df_clean['win_probability'] >= low) & 
                        (df_clean['win_probability'] < high)
                    ]
                    
                    if len(range_data) > 0:
                        accuracy_by_range.append({
                            'range': f"{low:.1f}-{high:.1f}",
                            'mid': (low + high) / 2,
                            'accuracy': range_data['win_prediction_correct'].mean(),
                            'count': len(range_data)
                        })
                
                if accuracy_by_range:
                    range_df = pd.DataFrame(accuracy_by_range)
                    
                    fig.add_trace(
                        go.Bar(
                            x=range_df['range'],
                            y=range_df['accuracy'],
                            name='Accuracy by Range',
                            marker_color=self.colors['success'],
                            hovertemplate='<b>Range:</b> %{x}<br>' +
                                        '<b>Accuracy:</b> %{y:.1%}<br>' +
                                        '<b>Count:</b> %{customdata}<br>' +
                                        '<extra></extra>',
                            customdata=range_df['count']
                        ),
                        row=2, col=1
                    )
            
            # 4. Confidence vs Total Error (if total confidence available)
            total_conf_df = df.dropna(subset=['total_confidence', 'total_absolute_error'])
            if not total_conf_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=total_conf_df['total_confidence'],
                        y=total_conf_df['total_absolute_error'],
                        mode='markers',
                        name='Total Confidence vs Error',
                        marker=dict(
                            color=self.colors['warning'],
                            size=6,
                            opacity=0.6
                        ),
                        hovertemplate='<b>Confidence:</b> %{x:.2f}<br>' +
                                    '<b>Error:</b> %{y:.2f} runs<br>' +
                                    '<extra></extra>'
                    ),
                    row=2, col=2
                )
                
                # Add trend line for confidence vs error
                if len(total_conf_df) >= 10:
                    conf_vals = np.array(total_conf_df['total_confidence'].values, dtype=np.float64)
                    error_vals = np.array(total_conf_df['total_absolute_error'].values, dtype=np.float64)
                    z = np.polyfit(conf_vals, error_vals, 1)
                    sorted_conf_series = total_conf_df.sort_values('total_confidence')['total_confidence']
                    sorted_conf_vals = np.array(sorted_conf_series.values, dtype=np.float64)
                    trend_line = np.poly1d(z)(sorted_conf_vals)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_conf_vals,
                            y=trend_line,
                            mode='lines',
                            name='Error Trend',
                            line=dict(color=self.colors['danger'], width=2),
                            showlegend=False
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title=f"Prediction Confidence Analysis (Last {days} Days)",
                showlegend=True,
                height=700,
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Predicted Probability", row=1, col=1)
            fig.update_yaxes(title_text="Actual Accuracy", row=1, col=1)
            fig.update_xaxes(title_text="Win Probability", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_xaxes(title_text="Confidence Range", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1, tickformat='.1%')
            fig.update_xaxes(title_text="Total Confidence", row=2, col=2)
            fig.update_yaxes(title_text="Total Error (runs)", row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating confidence analysis chart: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating confidence analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color='red')
            )
            fig.update_layout(title="Error in Confidence Analysis Chart")
            return fig
    
    def get_performance_summary_stats(self, days: int = 7) -> Dict:
        """
        Get summary statistics for performance dashboard
        
        Args:
            days: Number of days to summarize
            
        Returns:
            Dictionary with key performance metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.db_manager.db_path) as conn:
                # Get recent accuracy metrics
                accuracy_query = """
                    SELECT 
                        AVG(accuracy_rate) as avg_accuracy,
                        SUM(total_predictions) as total_predictions,
                        SUM(correct_predictions) as total_correct,
                        AVG(total_mae) as avg_mae,
                        AVG(confidence_calibration) as avg_calibration_error
                    FROM prediction_accuracy
                    WHERE sport = ? 
                    AND date_period >= DATE(?)
                    AND date_period <= DATE(?)
                """
                
                cursor = conn.execute(accuracy_query, (start_date.date(), end_date.date()))
                result = cursor.fetchone()
                
                # Get trend information
                trend_query = """
                    SELECT accuracy_trend, improvement_score, retraining_triggered
                    FROM performance_trends
                    WHERE sport = ? AND model_type = 'xgboost'
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                
                trend_cursor = conn.execute(trend_query)
                trend_result = trend_cursor.fetchone()
                
                # Get last retraining date
                retrain_query = """
                    SELECT MAX(date_recorded) as last_retrain
                    FROM model_metrics
                    WHERE sport = ? AND metric_name = 'retraining_completed'
                """
                
                retrain_cursor = conn.execute(retrain_query)
                retrain_result = retrain_cursor.fetchone()
            
            summary = {
                'period_days': days,
                'timestamp': datetime.now().isoformat()
            }
            
            if result:
                avg_acc, total_pred, total_correct, avg_mae, avg_cal_error = result
                
                summary.update({
                    'average_accuracy': avg_acc,
                    'total_predictions': total_pred or 0,
                    'total_correct': total_correct or 0,
                    'average_mae': avg_mae,
                    'confidence_calibration_error': avg_cal_error,
                    'accuracy_status': self._get_accuracy_status(avg_acc),
                    'data_coverage': 'sufficient' if total_pred and total_pred >= 10 else 'limited'
                })
            else:
                summary.update({
                    'average_accuracy': None,
                    'total_predictions': 0,
                    'accuracy_status': 'no_data',
                    'data_coverage': 'none'
                })
            
            if trend_result:
                trend, improvement, retraining = trend_result
                summary.update({
                    'accuracy_trend': trend,
                    'improvement_score': improvement,
                    'retraining_needed': bool(retraining),
                    'trend_direction': 'improving' if trend and trend > 0 else 'declining' if trend and trend < 0 else 'stable'
                })
            
            if retrain_result and retrain_result[0]:
                last_retrain = datetime.fromisoformat(retrain_result[0])
                days_since_retrain = (datetime.now() - last_retrain).days
                summary.update({
                    'last_retrain_date': retrain_result[0],
                    'days_since_retrain': days_since_retrain
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary stats: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_accuracy_status(self, accuracy: Optional[float]) -> str:
        """Get accuracy status classification"""
        if accuracy is None:
            return 'unknown'
        elif accuracy >= 0.55:
            return 'excellent'
        elif accuracy >= 0.52:
            return 'good'
        elif accuracy >= 0.50:
            return 'fair'
        else:
            return 'poor'