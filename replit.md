# Multi-Sport Prediction System

## Overview

This is a comprehensive multi-sport game prediction system (MLB, NBA, NFL, NHL, NCAA Football, NCAA Basketball, WNBA) that uses machine learning to predict game winners and totals (over/under). The system features a dual-source data collection strategy: API data for playoff games and Excel files for regular season schedules. It includes automated data pipelines, backtesting capabilities, and a Streamlit web interface for visualization and interaction.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Flask Web Application**: Professional web app with Jinja2 templates
- **Routes**: Landing page (/), Login (/login), Signup (/signup), Dashboard (/dashboard), Logout (/logout)
- **Authentication**: Flask-Login session-based authentication with secure password hashing
- **Responsive Design**: Clean, modern UI with gradient accents and mobile-friendly layout
- **Template Inheritance**: Base template with consistent navigation and styling

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
- **Dual-Source System**: Automatic switching between Excel schedules and API data
  - **Regular Season**: Reads from Excel files in `schedules/` directory (user-provided)
  - **Playoffs**: Uses official league APIs (MLB Stats API, NHL API, etc.)
  - **Auto-Fallback**: If Excel not available, automatically uses API
- **API Integration**: 
  - MLB Stats API for baseball games with playoff detection (gameType field)
  - NHL API for hockey games with date filtering and playoff detection
  - NBA, NFL, WNBA, NCAA collectors with standardized interfaces
- **Excel Schedule Support**: 
  - Flexible column detection (Date/Away/Home with various naming)
  - Multiple date format support (YYYY-MM-DD, MM/DD/YYYY, etc.)
  - Season-specific file naming (MLB.xlsx, MLB_2025.xlsx)
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
- **openpyxl**: Excel file reading for schedule imports
- **requests**: HTTP client for API calls
- **beautifulsoup4**: HTML parsing for odds data
- **trafilatura**: Web content extraction

### Web Interface
- **flask**: Web application framework
- **flask-login**: User session management and authentication
- **flask-wtf**: Form handling and CSRF protection
- **werkzeug**: Password hashing and security utilities

### Utilities
- **schedule**: Task scheduling and automation
- **sqlite3**: Database connectivity (built-in Python)
- **logging**: System monitoring and debugging (built-in Python)

### Data Sources
- **MLB Stats API**: Official MLB game schedules, scores, and playoff detection
- **NHL API**: Official NHL game schedules with playoff indicators
- **NBA, NFL, WNBA, NCAA APIs**: Sport-specific data collectors
- **Baseball Savant/Statcast**: Advanced MLB metrics and player statistics
- **Excel Schedules**: User-provided regular season schedules in `schedules/` directory
  - Place Excel files named `MLB.xlsx`, `NBA.xlsx`, etc.
  - See `schedules/README.md` for format details

### Recent Updates (October 2025)

**Flask Conversion (Oct 15, 2025)**
- **Complete migration from Streamlit to Flask** for better control and customization
- Implemented Flask-Login authentication system with session management
- Created professional HTML templates with Jinja2 (base, index, login, signup, dashboard)
- Maintained all original functionality: landing page with free pick, authentication, dashboard with sport filtering
- Dashboard shows 7-day predictions with three-model ensemble breakdown
- **Ensemble Model**: CompositeHome = (XGBoost × 50%) + (Elo × 35%) + (Consensus × 15%)
  - XGBoost given highest weight for better accuracy
  - Elo Rating provides stable baseline
  - Consensus model offers additional validation
- Login removed for testing phase - open dashboard access
- Clean, modern UI with gradient design and responsive layout
- All users automatically set to premium subscription (Stripe integration pending)
- Workflow updated to run Flask on port 5000

**Schedule Import System (Oct 15, 2025)**
- Created `import_schedules.py` to import games and predictions from `models/schedules.py`
- Successfully imported 2,857 games across all sports (NFL: 272, NBA: 1,237, NHL: 1,312, MLB: 36, partial NCAAF)
- Generated 1,471 predictions (NFL: 207, NHL: 1,264) using ensemble models
- Fixed database corruption where MLB showed NFL team names
- Predictions now properly stored with Elo, Logistic, and XGBoost probabilities

**Universal Ensemble Prediction System (Oct 13, 2025)**
- Created universal ensemble predictor using **Elo Ratings + GLMNet + XGBoost**
- Blended ensemble with optimized weights: 30% Elo + 35% GLMNet + 35% XGBoost
- Works with **all sports**: MLB, NFL, NBA, NHL, NCAA Football, NCAA Basketball, WNBA
- CSV-based approach for easy data import (auto-detects columns)
- Handles multiple result formats: scores ("24-20"), H/A, Home/Away
- Model persistence: Trained models saved to `models/{sport}_ensemble.pkl`
- NFL testing: 78% XGBoost accuracy, 61% GLMNet accuracy on 64-game training set
- Database integration: Automatically stores games and predictions
- Usage: `python universal_sports_predictor.py <SPORT> <csv_file>`

**Dual-Source Scheduling System**
- Fixed NHL collector to properly filter games by date (was returning full week instead of single day)
- Fixed MLB collector to detect playoff games using gameType field
- Added `is_playoff` flag to all games for playoff identification
- Created `ExcelScheduleReader` utility for parsing user-provided schedule files
- Updated `SportDataManager` to automatically choose between Excel and API sources
- Added `schedules/` directory with README for user-provided Excel schedules

**NHL Prediction System (Oct 2025)**
- Created NHL predictor with XGBoost models for winner and totals prediction
- Trained models on 268 historical NHL games (Oct-Nov 2024)
- Implemented feature engineering using team stats (win%, goals for/against, recent form)
- Fixed data corruption issue: Convert numpy floats to Python floats before database storage to prevent BLOB corruption
- Updated UI to display "Total: X.X" for NHL games (shows total goals) vs "X-X" for MLB games (shows individual scores)
- Fixed display logic to handle zero scores correctly (uses `is not None` checks instead of truthy checks)
- NHL predictions now show realistic probabilities (54-94% range) instead of flat 50% defaults

**7-Day Prediction View (Oct 13, 2025)**
- Changed default view from single day to **7-day range** to handle Excel schedule date offset issues
- Excel schedules may have timezone/entry date offsets (e.g., game labeled "tomorrow" actually plays "today")
- New view shows all games across 7-day window with clear date headers (e.g., "📅 Sunday, October 13, 2025")
- Predictions grouped and sorted by date for easy navigation
- Handles both DD/MM/YYYY (database format) and YYYY-MM-DD date formats seamlessly
- Alternative "All Upcoming Games" view still available for viewing complete season schedule
- Database column naming fixed: Uses `logistic_home_prob` (not `glmnet_home_prob`) to match ensemble system

The system is designed with modularity in mind, allowing for easy integration of additional sports leagues and data sources while maintaining the same architectural patterns.