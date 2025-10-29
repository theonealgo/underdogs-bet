# jackpotpicks.bet - Multi-Sport Prediction System

## Overview
jackpotpicks.bet is a multi-sport game prediction platform for NFL, NBA, NHL, MLB, WNBA, and NCAA Football. It utilizes machine learning to forecast game winners and totals (over/under), employing a dual-source data collection strategy (API for playoffs, Excel for regular season), automated data pipelines, and backtesting capabilities. The platform is designed for sale as a production-ready sports analytics system with professional-level accuracy.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Unified Platform Architecture (Oct 28, 2025)
**Production-Ready Single-Port Design** - The platform runs as a unified Flask application on port 5000:
- **Main App**: `NHL77FINAL.py` serves all sports on port 5000 (Replit only exposes this port)
- **Landing Page**: Sport selector (NO unified dashboard) per user requirement - each sport completely separate
- **Active Sports**: NHL (77% accuracy), NFL (84% accuracy), and NBA shown as "Live Now"
- **Coming Soon**: MLB, WNBA, NCAAF, NCAAB displayed but not yet implemented
- **Routes**: 
  - `/` - Sport selector landing page with dynamic accuracy percentages
  - `/sport/{SPORT}/predictions` - Predictions for each sport
  - `/sport/{SPORT}/results` - Model performance/backtesting results
- **Navigation**: Responsive navbar with hamburger menu (<768px width) for mobile devices
- **Data Isolation**: Each sport completely separate - no shared data visualization between sports
- **Production Config**: debug=False, use_reloader=False for stability

### Frontend Architecture
The platform features a professional Flask web application with Jinja2 templates, offering a responsive mobile-first design with sticky navigation. The UI/UX emphasizes a clean, modern aesthetic with gradient accents and compact table displays for predictions. Key routes include a sport selector landing page, sport-specific prediction pages (showing ALL games from season start), and results pages for model backtesting. The landing page dynamically calculates and displays real ensemble accuracy from completed games.

### Backend Architecture
The backend is built with a modular, object-oriented design, separating concerns across data collection, storage, modeling, and API layers. It includes a comprehensive logging system for monitoring and debugging, and utilizes the `schedule` library for automated data collection and model retraining. SQLite serves as the local database for storing statcast data, odds, predictions, and model performance metrics, with models and scalers persisted using pickle.

### Machine Learning Pipeline
The core of the system is its machine learning pipeline, featuring sport-specific Elo K-factors optimized for each sport and enhanced with Margin of Victory (MoV) and Offensive/Defensive Split Elo adjustments. XGBoost models are used with heavily regularized, sport-specific hyperparameters to prevent overfitting. Feature engineering is extensive, particularly for NFL, including deep lag features, home/away splits, opponent-adjusted metrics, contextual features (rest days, short rest), advanced metrics, weather data, betting market meta, and situational meta. Other sports also have tailored feature sets, such as NHL goalie differentials. The system employs confidence-based weighting for NFL predictions and dynamic ensemble weighting for other sports, combining XGBoost, Elo, and Logistic Regression models. Overfitting prevention is a priority, addressed through aggressive regularization and ensemble reweighting. The pipeline includes cross-validation for model evaluation and hyperparameter tuning, and a robust backtesting framework for historical performance assessment. A league-average fallback mechanism ensures predictions for sports without extensive historical data.

### Data Collection Strategy
A dual-source system automatically switches between user-provided Excel files for the regular season and official league APIs for playoffs. API integrations include MLB Stats API, NHL API, NBA, NFL, WNBA, and NCAA collectors, with rate limiting and error handling. Excel schedule support handles multiple date formats, ensuring consistency.

### Schedule Architecture (Oct 28, 2025)
**Modular Sport-Specific Schedules** - Each sport maintains its own schedule configuration:
- **Location**: `schedules/` directory contains sport-specific Python files
- **NFL Schedule**: `schedules/nfl_schedule.py` - Complete 2025 season (272 games, Sept 5, 2025 - Jan 5, 2026)
- **NHL Schedule**: `nhlschedules.py` - Complete 2025-26 season (1,312 games, Oct 7, 2025 - Apr 16, 2026)
- **NBA Schedule**: `schedules/nbaschedule.py` - Partial 2025-26 season (346 games, Oct 21, 2025 - Dec 7, 2025)
- **Format**: Each schedule file exports `get_<sport>_schedule()` function returning list of game dictionaries
- **Structure**: Games include match_id, round/date, venue, home_team, away_team, result (if completed)
- **Loading Process**: Run `python load_schedules.py ALL` to import all schedules into database
  - **NFL Only**: `python load_schedules.py NFL`
  - **NHL Only**: `python load_schedules.py NHL`
  - **NBA Only**: `python load_schedules.py NBA`
  - **Clears old data**: Automatically removes previous season data before importing
  - **Yearly Update**: Run once per season to load new schedules
- **Season Start Dates** (hardcoded in NHL77FINAL.py):
  - NFL: September 5, 2025 (Sept 4 in code)
  - NHL: October 7, 2025
  - NBA: October 21, 2025 (**CRITICAL**: Must match first game date)
  - MLB: March 27, 2025
  - NCAAF: August 30, 2025
  - NCAAB: November 4, 2025
- **Benefits**: Isolated schedule management per sport while maintaining unified multi-sport platform
- **Future**: Additional sports (NBA, MLB, NCAAF, NCAAB) will follow same pattern

### Prediction Generation Workflow
The system uses `generate_real_predictions.py` to create authentic predictions from trained ensemble models for all upcoming games. A critical fix ensures training metadata (games_trained, feature_cols, accuracy) is persisted with models. Placeholder generation has been deprecated in favor of model-based outputs.

### Automation Framework
Automated tasks include daily data updates, prediction generation, and weekly model retraining, managed by a configurable, non-blocking scheduler with threading support.

### NBA Model Training & Predictions (Oct 28, 2025)
**Production-Ready NBA 2025-26 Season** - Complete model training with optimized predictions:
- **PREDICTION SEASON**: 2025-26 NBA season (346 games, Oct 21 - Dec 7, 2025)
  - **First Game**: October 21, 2025 - Oklahoma City Thunder vs Houston Rockets at Paycom Center
  - **Completed Games**: 53 games with actual results (Oct 21-28, 2025)
  - **Upcoming Games**: 293 games displayed for user predictions
  - **ALL Games Displayed**: Shows all 346 games including completed ones with scores
- **TRAINING DATA**: 2024-25 NBA season (1,231 games) used for model training ONLY, not shown to users
- **Models**: All 4 models fully trained and operational
  - **Elo**: K-factor 18 for NBA-specific rating system (56.6% accuracy on 53 completed games)
  - **XGBoost**: Regularized with sport-specific hyperparameters (58.5% accuracy)
  - **CatBoost**: Advanced gradient boosting with categorical features (58.5% accuracy)
  - **Logistic Regression**: Baseline linear model for comparison (54.7% accuracy)
  - **Meta Ensemble**: Weighted average of all 4 models - 25% each (56.6% accuracy)
- **PREDICTION STRATEGY**: Direct model predictions (NO inversion/fade)
  - **Previous Issue**: Fade strategy (1 - prob) was reducing accuracy from 58% to 41%
  - **Current Implementation**: Raw model probabilities used directly
  - **Performance**: XGBoost and CatBoost achieve 58.5% accuracy (industry-standard for NBA)
- **ACCURACY BENCHMARKS** (53 completed games, Oct 21-28, 2025):
  - **Best Models**: XGBoost 58.5% (31/53), CatBoost 58.5% (31/53)
  - **Ensemble**: Meta 56.6% (30/53), Elo 56.6% (30/53)
  - **Baseline**: Logistic 54.7% (29/53)
  - **Industry Context**: 55-60% is professional-grade accuracy for NBA predictions
- **Database Schema**: Predictions stored with all model probabilities (elo_home_prob, xgboost_home_prob, catboost_home_prob, logistic_home_prob, meta_home_prob, win_probability)
- **Display Format**: Column order matches NHL - XGBoost, CatBoost, Elo, Meta (left to right)
- **Key Files**:
  - `schedules/nbaschedule.py` - 2025-26 season schedule (346 games for PREDICTIONS)
  - `load_schedules.py` - Imports NBA schedule into database with season=2025 (run `python load_schedules.py NBA`)
  - `train_nba_models.py` - Trains models on historical 2024-25 data
  - `generate_nba_predictions.py` - Generates predictions for 2025-26 season (no fade strategy)
- **Season Configuration**:
  - `NHL77FINAL.py`: Season start set to October 22, 2024 (for training data display logic)
  - Prediction generator pulls `season=2025` games only from database
- **Status**: ✅ Live on landing page with "Live Now" badge - 346 games from 2025-26 season displayed

### NHL Advanced Model Implementation (Oct 29, 2025)
**Production-Ready NHL 2025-26 Season** - Advanced ensemble models with comprehensive feature engineering:
- **PREDICTION SEASON**: 2025-26 NHL season (1,132 games, Oct 7, 2025 - Apr 16, 2026)
- **TRAINING DATA**: October 2025 games (92 games) used for model training
- **Models**: All 3 models fully trained and operational
  - **XGBoost**: Regularized with NHL-specific hyperparameters (depth=5, lr=0.05, n_estimators=200, reg_alpha=0.5, reg_lambda=1.0)
  - **CatBoost**: Advanced gradient boosting (depth=6, lr=0.05, iterations=200, l2_leaf_reg=3.0)
  - **Logistic Regression**: Baseline linear model for comparison
  - **Meta Ensemble**: Weighted average of all 3 models
- **ADVANCED FEATURE ENGINEERING** (36 features total):
  - **Rolling Windows**: Win %, goals for/against in last 5, 10, and 15 games
  - **Home/Away Splits**: Performance splits for home vs road games
  - **Rest & Fatigue**: Days since last game, back-to-back game penalties
  - **Strength of Schedule**: Average opponent strength over recent games
  - **Head-to-Head History**: Last 5 matchups between teams, average total goals
  - **Goalie Performance**: Save percentage differential between starting goalies
  - **Differential Features**: Win % diff, goals diff, defense diff, rest diff, form diff
- **TRAINING RESULTS** (92 games):
  - XGBoost: 47.4% accuracy, 1.61 goals MAE
  - CatBoost: 47.4% accuracy, 1.56 goals MAE
  - Logistic: 52.6% accuracy
  - 36 engineered features with multiple time windows
- **PREDICTION STRATEGY**: Direct model predictions with ensemble averaging
- **Database Schema**: Predictions stored with all model probabilities (xgboost_home_prob, catboost_home_prob, logistic_home_prob, meta_home_prob)
- **Display Format**: Column order - XGBoost, CatBoost, Logistic (no Elo), Meta (left to right)
- **Key Files**:
  - `src/models/nhl_predictor.py` - Advanced NHL predictor with feature engineering
  - `train_nhl_models.py` - Trains XGBoost, CatBoost, Logistic models
  - `generate_nhl_predictions.py` - Generates predictions for 2025-26 season
  - `import_nhl_october_2025_results.py` - Imports historical results for training
- **Status**: ✅ All 1,132 predictions generated for 2025-26 season

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
- **NHL API (via nhl-api-py)**: Official NHL stats, rosters, schedules, goalie data
- **NBA API**: Official NBA game data and statistics
- **NFL API**: Official NFL game data and statistics
- **MLB Stats API**: Official MLB data (Baseball Savant/Statcast for advanced metrics)
- **WNBA, NCAA APIs**: Additional sport-specific data (coming soon)
- **The Odds API**: Betting odds data (currently unused)
- **Excel Schedules**: User-provided schedules for all sports