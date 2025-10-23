# jackpotpicks.bet - Multi-Sport Prediction System

## Overview
jackpotpicks.bet is a multi-sport game prediction platform for NFL, NBA, NHL, MLB, and NCAA Football. It utilizes machine learning to forecast game winners and totals (over/under), employing a dual-source data collection strategy (API for playoffs, Excel for regular season), automated data pipelines, and backtesting capabilities. The platform aims to provide accurate predictions and insights via a Flask web application. The business vision is to deliver a cutting-edge prediction system with high market potential in sports analytics, striving for professional-level accuracy.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The platform features a professional Flask web application with Jinja2 templates, offering a responsive and mobile-friendly design with consistent navigation. The UI/UX emphasizes a clean, modern aesthetic with gradient accents and compact table displays for predictions. Key routes include a landing page, authentication flows, a sport-selection dashboard, sport-specific prediction pages, and an admin-only results page for model backtesting.

### Backend Architecture
The backend is built with a modular, object-oriented design, separating concerns across data collection, storage, modeling, and API layers. It includes a comprehensive logging system for monitoring and debugging, and utilizes the `schedule` library for automated data collection and model retraining. SQLite serves as the local database for storing statcast data, odds, predictions, and model performance metrics, with models and scalers persisted using pickle.

### Machine Learning Pipeline
The core of the system is its machine learning pipeline, featuring sport-specific Elo K-factors optimized for each sport and enhanced with Margin of Victory (MoV) and Offensive/Defensive Split Elo adjustments. XGBoost models are used with heavily regularized, sport-specific hyperparameters to prevent overfitting. Feature engineering is extensive, particularly for NFL, including deep lag features, home/away splits, opponent-adjusted metrics, contextual features (rest days, short rest), advanced metrics, weather data, betting market meta, and situational meta. Other sports also have tailored feature sets, such as NHL goalie differentials. The system employs confidence-based weighting for NFL predictions and dynamic ensemble weighting for other sports, combining XGBoost, Elo, and Logistic Regression models. Overfitting prevention is a priority, addressed through aggressive regularization and ensemble reweighting. The pipeline includes cross-validation for model evaluation and hyperparameter tuning, and a robust backtesting framework for historical performance assessment. A league-average fallback mechanism ensures predictions for sports without extensive historical data.

### Data Collection Strategy
A dual-source system automatically switches between user-provided Excel files for the regular season and official league APIs for playoffs. API integrations include MLB Stats API, NHL API, NBA, NFL, WNBA, and NCAA collectors, with rate limiting and error handling. Excel schedule support handles multiple date formats, ensuring consistency.

### Schedule Architecture (Oct 23, 2025)
**Modular Sport-Specific Schedules** - Each sport maintains its own schedule configuration:
- **Location**: `schedules/` directory contains sport-specific Python files
- **NFL Schedule**: `schedules/nfl_schedule.py` - Complete 2025 season (272 games, Sept 5, 2025 - Jan 5, 2026)
- **NHL Schedule**: `nhlschedules.py` - Complete 2025-26 season (1,312 games, Oct 7, 2025 - Apr 16, 2026)
- **Format**: Each schedule file exports `get_<sport>_schedule()` function returning list of game dictionaries
- **Structure**: Games include match_id, round/date, venue, home_team, away_team, result (if completed)
- **Loading Process**: Run `python load_schedules.py ALL` to import all schedules into database
  - **NFL Only**: `python load_schedules.py NFL`
  - **NHL Only**: `python load_schedules.py NHL`
  - **Clears old data**: Automatically removes previous season data before importing
  - **Yearly Update**: Run once per season to load new schedules
- **Season Start Dates** (hardcoded in nhl_predictor.py):
  - NFL: September 5, 2025
  - NHL: October 7, 2025
  - NBA: October 22, 2025
  - MLB: March 27, 2025
  - NCAAF: August 30, 2025
  - NCAAB: November 4, 2025
- **Benefits**: Isolated schedule management per sport while maintaining unified multi-sport platform
- **Future**: Additional sports (NBA, MLB, NCAAF, NCAAB) will follow same pattern

### Prediction Generation Workflow
The system uses `generate_real_predictions.py` to create authentic predictions from trained ensemble models for all upcoming games. A critical fix ensures training metadata (games_trained, feature_cols, accuracy) is persisted with models. Placeholder generation has been deprecated in favor of model-based outputs.

### Automation Framework
Automated tasks include daily data updates, prediction generation, and weekly model retraining, managed by a configurable, non-blocking scheduler with threading support.

### NHL Goalie Stats Integration (Oct 22, 2025)
**Current Implementation (V3 with nhl-api-py)** - Professional library integration with dual-source data collection:
- **Library**: `nhl-api-py` - Official Python client for NHL APIs (replaces custom integration)
- **Data Collection**: Dual-source approach for complete coverage
  - **Stats API**: 25 goalies with actual 2025-26 season stats (SV%, GAA, wins, losses)
  - **Roster API**: 50 additional goalies from team rosters (league avg stats for those without games)
  - **Total**: 75 goalies with 32/32 team coverage (up from 31/32)
- **Team Mapping**: Automated via `map_team_goalies_v2.py`
  - Selects goalie with most games played from each team's roster
  - Full name format (first + last) for better data integrity
  - **Known Limitation**: Some goalies have 0 GP due to trades/roster moves (use league avg stats)
- **Feature**: Goalie differential (save % difference) weighted 0.3 for XGBoost, 0.2 for CatBoost
- **Accuracy Results on 76 games (Oct 7-18, 2025)**:
  - Elo: 52.6% (unchanged - doesn't use goalie feature)
  - XGBoost: 44.7% (vs 43.4% baseline without goalies)
  - CatBoost: 48.7% (vs 43.4% baseline without goalies)
  - Meta: 47.4% (vs 43.4% baseline without goalies)
- **Automation**: Auto-initialization on app startup if tables don't exist
- **Manual Refresh**: `python fetch_nhl_goalies_v3.py` then `python map_team_goalies_v2.py`
- **Manual Re-init**: Drop tables via `sqlite3 sports_predictions.db "DROP TABLE goalie_stats; DROP TABLE team_goalies;"` then restart
- **Failure Mode**: Graceful fallback to league-average stats (91.0% SV%, 2.80 GAA)
- **Future Enhancement**: Consider storing player IDs to handle mid-season trades and name collisions

## External Dependencies

### Core ML/Data Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML utilities
- **xgboost**: Machine learning algorithm

### Data Collection
- **nhl-api-py**: Official NHL data client (stats, rosters, schedules)
- **pybaseball**: Baseball data
- **openpyxl**: Excel file reading
- **requests**: HTTP client
- **beautifulsoup4**: HTML parsing
- **trafilatura**: Web content extraction

### Web Interface
- **flask**: Web framework
- **flask-login**: User authentication
- **flask-wtf**: Form handling
- **werkzeug**: Password hashing

### Utilities
- **schedule**: Task scheduling
- **sqlite3**: Database connectivity
- **logging**: System logging

### Data Sources
- **MLB Stats API**: Official MLB data
- **NHL API (via nhl-api-py)**: Official NHL stats, rosters, schedules
- **NBA, NFL, WNBA, NCAA APIs**: Sport-specific data
- **The Odds API**: Betting odds data
- **Baseball Savant/Statcast**: Advanced MLB metrics
- **Excel Schedules**: User-provided schedules