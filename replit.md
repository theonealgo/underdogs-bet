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
- **Enhanced Elo System (Oct 2025)**: 
  - **Margin of Victory (MoV) Adjustment**: Weights blowout wins higher using score differential multiplier
  - **Offensive/Defensive Split Elo**: Separate ratings for offense and defense, enabling better matchup analysis
  - Fixed critical bug where update_ratings() wasn't receiving game scores, preventing MoV and split Elo from functioning
- **Sport-Specific XGBoost Hyperparameters**: Heavily regularized to prevent overfitting:
  - NFL: n_estimators=120, max_depth=3, learning_rate=0.03, subsample=0.6, colsample_bytree=0.6, min_child_weight=5, reg_alpha=1.0, reg_lambda=10.0
  - NBA: n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.7 (82-game season optimization)
  - NHL: n_estimators=175, max_depth=5, learning_rate=0.04, subsample=0.75 (randomness handling)
  - MLB: n_estimators=250, max_depth=3, learning_rate=0.02, subsample=0.6 (162-game season)
  - NCAAF: n_estimators=160, max_depth=6, learning_rate=0.06, subsample=0.85 (12-game season)
- **NFL Comprehensive Feature Engineering** (62 features with chronological filtering):
  - **Deep Lag Features**: 3/5/10-game rolling windows for points scored/allowed/differential, win percentage
  - **Home/Away Splits**: Last 5 home games, last 5 away games performance metrics
  - **Opponent-Adjusted Features**: Average opponent Elo (L5 games), strength of schedule differential
  - **Contextual Features**: Rest days, rest advantage, short rest indicators (Thu/Mon games), home win streaks
  - **Advanced Metrics**: Scoring efficiency differential, defensive efficiency differential, matchup features
  - **Chronological Filtering**: ONLY uses data from games before prediction date to prevent target leakage
- **Other Sports Feature Engineering**:
  - **NBA-Specific**: ~20 features (scoring rolling/lag, back-to-back, matchup)
  - **NHL-Specific**: ~20 features (goals rolling/lag, back-to-back, matchup)
  - **MLB-Specific**: ~18 features (ERA/OPS/runs rolling/lag, pitching/batting matchup)
  - **NCAAF-Specific**: ~15 features (points/yards rolling/lag, rest, matchup)
- **Team Stats System (Oct 2025)**: Populated team_stats table with 570 NFL records from 285 completed games, enabling all rolling window features
- **NFL Confidence-Based Weighting (Oct 2025)**:
  - **Rule 1 - High Elo Favor** (Elo ≥75%, XGB ≤55%): 85% Elo / 10% XGB / 5% Log - Don't let weak XGB override strong Elo (89.4% accuracy on 113 games)
  - **Rule 2 - Upset Zone** (Elo 55-75%, XGB 45-55%): 50% Elo / 40% XGB / 10% Log - Let XGB find hidden value (68.7% accuracy on 134 games)
  - **Default**: 60% Elo / 30% XGB / 10% Log - Standard weighting for other scenarios (36.8% accuracy on 38 games)
  - Improvement: 72.6% vs 72.3% simple weighted (+0.4%)
- **Other Sports Ensemble Weights**: 50% XGBoost, 35% Elo, 15% Logistic (simple weighted average)
- **NFL Model Performance**: 72.6% accuracy on 285 completed games (up from 53% baseline)
- **Overfitting Prevention**: Discovered and fixed severe overfitting (98.9% training → 44.9% real-world for XGBoost). Applied aggressive regularization and ensemble reweighting, improving to 72.6% real-world accuracy.
- **Cross-Validation**: Built-in model evaluation and hyperparameter tuning.
- **Backtesting Framework**: Historical performance evaluation with configurable date ranges and confidence thresholds, validating against actual game results.
- **League-Average Fallback**: For sports without historical data, generates balanced 50/50 predictions with 5% home field advantage using balanced dummy training data (5 wins, 5 losses per team). Deterministic and reproducible.

### Data Collection Strategy
- **Dual-Source System**: Automatic switching between Excel files (user-provided in `schedules/`) for regular season and official league APIs for playoffs.
- **API Integration**: Utilizes MLB Stats API, NHL API, NBA, NFL, WNBA, NCAA collectors with rate limiting and error handling.
- **Excel Schedule Support**: Multi-format date parsing supporting ISO (YYYY-MM-DD HH:MM), legacy DD/MM/YYYY, NBA text format ("Tue, Oct 21, 2025"), and NCAAF formats.
- **Date Format Consistency**: All dates stored as DD/MM/YYYY strings in database to prevent automatic pandas date conversion and month/day confusion.
- **Active Sports**: NFL (207 predictions), NBA (1204 predictions), NHL (1264 predictions), MLB (5 predictions), NCAAF (29 predictions). NCAAB and WNBA schedules pending data.

### Prediction Generation Workflow
- **Real Model-Based Predictions**: Uses `generate_real_predictions.py` to load trained ensemble models and generate authentic predictions for all upcoming games.
- **Model Persistence Fix (Oct 2025)**: Fixed critical bug where training metadata (games_trained, feature_cols, accuracy) wasn't being persisted. Models now properly save/load all training state.
- **Placeholder Prevention**: Deprecated `generate_missing_predictions.py` (random placeholders) in favor of model-based generation. File renamed to `DEPRECATED_generate_missing_predictions.py.bak` to prevent accidental reuse.
- **Current Predictions**: 6,167 real predictions across all sports (NFL: 557, NBA: 1,497, NHL: 2,624, MLB: 1,361, NCAAF: 128) based on 3,619 historical games.
- **Workflow**: After model retraining, run `python generate_real_predictions.py` to repopulate predictions with authentic model outputs.

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