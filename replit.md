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

### Prediction Generation Workflow
The system uses `generate_real_predictions.py` to create authentic predictions from trained ensemble models for all upcoming games. A critical fix ensures training metadata (games_trained, feature_cols, accuracy) is persisted with models. Placeholder generation has been deprecated in favor of model-based outputs.

### Automation Framework
Automated tasks include daily data updates, prediction generation, and weekly model retraining, managed by a configurable, non-blocking scheduler with threading support.

### NHL Goalie Stats Integration (Oct 22, 2025)
**V1 Implementation (Production)** - The main NHL predictor now includes real goalie stats:
- **API Source**: NHL Official API (free tier) - fetches goalie save %, GAA, wins, losses
- **Data**: 63 goalies from 2025-26 season
- **Team Mappings**: 31/32 NHL teams mapped to primary starting goalies (last names as keys)
- **Feature**: Goalie differential (save % difference) weighted 0.3 for XGBoost, 0.2 for CatBoost
- **Accuracy Impact on 76 games (Oct 7-18, 2025)**: CatBoost +4.0% (43.4%→47.4%), Meta +5.3% (43.4%→48.7%), XGBoost 0% (47.4%→47.4%)
- **Best Model**: Elo at 52.6% (no goalie feature) - all models near coin-flip accuracy
- **Automation**: Auto-initialization on app startup if tables don't exist (requires `python nhl_predictor.py` direct execution)
- **Manual Refresh**: `python fetch_nhl_goalies.py` then `python map_team_goalies.py`
- **Manual Re-init**: Drop tables via `sqlite3 sports_predictions.db "DROP TABLE goalie_stats; DROP TABLE team_goalies;"` then restart
- **Failure Mode**: Graceful fallback to league-average stats (91.0% SV%, 2.80 GAA)
- **Data Integrity**: Fixed incorrect mappings (Colorado Avalanche → Georgiev not Wedgewood)

### NHL V2 - API Enhanced Version (Testing - DEPRECATED)
NHL V2 was an enhanced testing version that integrated real-time API data from the NHL Official API and The Odds API. This approach has been superseded by the V1 goalie stats integration above, which is simpler and production-ready.

## External Dependencies

### Core ML/Data Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML utilities
- **xgboost**: Machine learning algorithm

### Data Collection
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
- **NHL API**: Official NHL data
- **NBA, NFL, WNBA, NCAA APIs**: Sport-specific data
- **The Odds API**: Betting odds data
- **Baseball Savant/Statcast**: Advanced MLB metrics
- **Excel Schedules**: User-provided schedules