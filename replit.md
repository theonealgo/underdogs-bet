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
- **Sport-Specific Elo K-Factors**: Optimized for each sport (NFL: 35, NBA: 18, NHL: 22, MLB: 14, NCAAF: 30) based on season length and game variance.
- **Sport-Specific XGBoost Hyperparameters**: Custom-tuned for each sport:
  - NFL: n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=1, reg_alpha=0.1
  - NBA: n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.7 (82-game season optimization)
  - NHL: n_estimators=175, max_depth=5, learning_rate=0.04, subsample=0.75 (randomness handling)
  - MLB: n_estimators=250, max_depth=3, learning_rate=0.02, subsample=0.6 (162-game season)
  - NCAAF: n_estimators=160, max_depth=6, learning_rate=0.06, subsample=0.85 (12-game season)
- **Comprehensive Feature Engineering** (with chronological filtering to prevent target leakage):
  - **Rolling Window Features**: Sport-specific windows (NFL: 3/5-game, NBA/NHL: 3/5/10-game, MLB: 5/10/15-game, NCAAF: 3-game)
  - **Lag Variables**: Previous 2 games stats for immediate momentum capture
  - **Fatigue Metrics**: Rest days, back-to-back indicators, rest advantage differential
  - **Matchup Features**: Offensive vs defensive ratings, turnover differentials, EPA matchups (NFL), pitching vs batting (MLB)
  - **NFL-Specific**: ~25 features (yards, EPA rolling/lag, turnover margin, rest, matchup)
  - **NBA-Specific**: ~20 features (scoring rolling/lag, back-to-back, matchup)
  - **NHL-Specific**: ~20 features (goals rolling/lag, back-to-back, matchup)
  - **MLB-Specific**: ~18 features (ERA/OPS/runs rolling/lag, pitching/batting matchup)
  - **NCAAF-Specific**: ~15 features (points/yards rolling/lag, rest, matchup)
- **Chronological Filtering**: Each game prediction uses ONLY data from games before that date (team_stats[date < game_date]) to prevent target leakage
- **Ensemble Model**: Composite prediction using weighted combination of XGBoost (50%), Elo (35%), and Logistic Regression (15%).
- **Cross-Validation**: Built-in model evaluation and hyperparameter tuning.
- **Backtesting Framework**: Historical performance evaluation with configurable date ranges and confidence thresholds, validating against actual game results.
- **League-Average Fallback**: For sports without historical data, generates balanced 50/50 predictions with 5% home field advantage using balanced dummy training data (5 wins, 5 losses per team). Deterministic and reproducible.

### Data Collection Strategy
- **Dual-Source System**: Automatic switching between Excel files (user-provided in `schedules/`) for regular season and official league APIs for playoffs.
- **API Integration**: Utilizes MLB Stats API, NHL API, NBA, NFL, WNBA, NCAA collectors with rate limiting and error handling.
- **Excel Schedule Support**: Multi-format date parsing supporting ISO (YYYY-MM-DD HH:MM), legacy DD/MM/YYYY, NBA text format ("Tue, Oct 21, 2025"), and NCAAF formats.
- **Date Format Consistency**: All dates stored as DD/MM/YYYY strings in database to prevent automatic pandas date conversion and month/day confusion.
- **Active Sports**: NFL (207 predictions), NBA (1204 predictions), NHL (1264 predictions), MLB (5 predictions), NCAAF (29 predictions). NCAAB and WNBA schedules pending data.

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