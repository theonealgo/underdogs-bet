# MLB Prediction System

## Overview

This is a comprehensive MLB game prediction system that uses machine learning to predict game winners and totals (over/under). The system combines real-time data collection from Baseball Savant and OddsShark with XGBoost models to generate daily predictions. It features automated data pipelines, backtesting capabilities, and a Streamlit web interface for visualization and interaction.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Multi-page application with navigation sidebar
- **Interactive Dashboards**: Real-time predictions, historical data visualization, model performance metrics
- **Plotly Visualizations**: Dynamic charts and graphs for data presentation
- **Caching Strategy**: Uses `@st.cache_resource` for component initialization to improve performance

### Backend Architecture
- **Modular Design**: Clean separation of concerns across data collection, storage, modeling, and API layers
- **Object-Oriented Structure**: Each major component (DatabaseManager, MLBPredictor, Scrapers) is encapsulated in dedicated classes
- **Logging System**: Comprehensive logging with rotation and multiple handlers for debugging and monitoring
- **Scheduler Framework**: Automated data collection and model retraining using the `schedule` library

### Data Storage Solutions
- **SQLite Database**: Local database for storing statcast data, odds data, predictions, and model performance metrics
- **Table Structure**: Separate tables for different data types (statcast_data, odds_data, predictions, etc.)
- **Data Persistence**: Models and scalers saved using pickle for consistent predictions

### Machine Learning Pipeline
- **Feature Engineering**: Comprehensive feature creation including team performance, pitching/batting stats, situational factors, and rolling statistics
- **XGBoost Models**: Separate models for winner prediction (binary classification) and totals prediction (regression)
- **Cross-Validation**: Built-in model evaluation and hyperparameter tuning
- **Backtesting Framework**: Historical performance evaluation with configurable date ranges and confidence thresholds

### Data Collection Strategy
- **Baseball Savant Integration**: Uses pybaseball library for Statcast data, team statistics, and schedules
- **Web Scraping**: Custom scraper for OddsShark betting trends and market data
- **Rate Limiting**: Built-in delays and retry logic to respect API limits
- **Error Handling**: Graceful degradation when data sources are unavailable

### Automation Framework
- **Scheduled Tasks**: Daily data updates, prediction generation, and weekly model retraining
- **Threading Support**: Non-blocking scheduler execution
- **Configurable Timing**: Multiple daily update cycles to capture fresh data

## External Dependencies

### Core ML/Data Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Feature preprocessing, model evaluation, and utilities
- **xgboost**: Primary machine learning algorithm for both classification and regression

### Data Collection
- **pybaseball**: Official Baseball Savant/Statcast data access
- **requests**: HTTP client for web scraping
- **beautifulsoup4**: HTML parsing for odds data
- **trafilatura**: Web content extraction

### Web Interface
- **streamlit**: Web application framework
- **plotly**: Interactive data visualization

### Utilities
- **schedule**: Task scheduling and automation
- **sqlite3**: Database connectivity (built-in Python)
- **logging**: System monitoring and debugging (built-in Python)

### Data Sources
- **Baseball Savant/Statcast**: Primary source for MLB game data, player statistics, and advanced metrics
- **OddsShark**: Betting trends, market data, and odds information
- **Alternative APIs**: System designed to accommodate SportsData.io, Sportradar, or The Odds API as future integrations

The system is designed with modularity in mind, allowing for easy integration of additional sports leagues (NBA, NFL, NHL) and data sources while maintaining the same architectural patterns.