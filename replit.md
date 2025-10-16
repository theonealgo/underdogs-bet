# jackpotpicks.bet - Multi-Sport Prediction System

## Overview
jackpotpicks.bet is a comprehensive multi-sport game prediction platform (NFL, NBA, NHL, MLB, NCAA Football) that uses machine learning to predict game winners and totals (over/under). It features a dual-source data collection strategy (API for playoffs, Excel for regular season), automated data pipelines, backtesting capabilities, and a Flask web application for visualization and interaction. The platform aims to provide accurate game predictions and insights across various sports.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Flask Web Application**: Professional web app with Jinja2 templates, responsive design, and consistent navigation.
- **Routes**:
    - Landing page (/) for 14-day MLB predictions.
    - Authentication: /login, /signup, /logout.
    - Dashboard (/dashboard) for sport selection.
    - Sport-specific pages: /nfl, /nba, /mlb, /nhl, /ncaaf, /ncaab.
    - Results (/results) for model backtesting performance (admin only).
- **UI/UX**: Clean, modern UI with gradient accents, mobile-friendly layout, and compact table displays for predictions.

### Backend Architecture
- **Modular Design**: Separation of concerns across data collection, storage, modeling, and API layers.
- **Object-Oriented Structure**: Encapsulated components (DatabaseManager, MLBPredictor, Scrapers).
- **Logging System**: Comprehensive logging for debugging and monitoring.
- **Scheduler Framework**: `schedule` library for automated data collection and model retraining.

### Data Storage Solutions
- **SQLite Database**: Local database for statcast data, odds data, predictions, and model performance metrics.
- **Data Persistence**: Models and scalers saved using pickle.

### Machine Learning Pipeline
- **Feature Engineering**: Comprehensive feature creation including team performance, pitching/batting stats, situational factors, and rolling statistics.
- **XGBoost Models**: Separate models for winner prediction (classification) and totals prediction (regression).
- **Ensemble Model**: Composite prediction using weighted combination of XGBoost (50%), Elo (35%), and Logistic Regression (15%).
- **Cross-Validation**: Built-in model evaluation and hyperparameter tuning.
- **Backtesting Framework**: Historical performance evaluation with configurable date ranges and confidence thresholds, validating against actual game results.

### Data Collection Strategy
- **Dual-Source System**: Automatic switching between Excel files (user-provided in `schedules/`) for regular season and official league APIs for playoffs.
- **API Integration**: Utilizes MLB Stats API, NHL API, NBA, NFL, WNBA, NCAA collectors with rate limiting and error handling.
- **Excel Schedule Support**: Flexible column and date format detection for user-provided schedules.

### Automation Framework
- **Scheduled Tasks**: Daily data updates, prediction generation, and weekly model retraining.
- **Threading Support**: Non-blocking scheduler execution.
- **Configurable Timing**: Multiple daily update cycles.

## External Dependencies

### Core ML/Data Libraries
- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computing.
- **scikit-learn**: Feature preprocessing, model evaluation.
- **xgboost**: Primary machine learning algorithm.

### Data Collection
- **pybaseball**: Baseball Savant/Statcast data.
- **openpyxl**: Excel file reading.
- **requests**: HTTP client for API calls.
- **beautifulsoup4**: HTML parsing for odds data.
- **trafilatura**: Web content extraction.

### Web Interface
- **flask**: Web application framework.
- **flask-login**: User session management and authentication.
- **flask-wtf**: Form handling and CSRF protection.
- **werkzeug**: Password hashing.

### Utilities
- **schedule**: Task scheduling and automation.
- **sqlite3**: Database connectivity.
- **logging**: System monitoring and debugging.

### Data Sources
- **MLB Stats API**: Official MLB game data.
- **NHL API**: Official NHL game data.
- **NBA, NFL, WNBA, NCAA APIs**: Sport-specific data collectors.
- **Baseball Savant/Statcast**: Advanced MLB metrics.
- **Excel Schedules**: User-provided regular season schedules in `schedules/` directory.