# Multi-Sport Prediction System

## Overview

This is a comprehensive multi-sport game prediction system (MLB, NBA, NFL, NHL, NCAA Football, NCAA Basketball, WNBA) that uses machine learning to predict game winners and totals (over/under). The system features a dual-source data collection strategy: API data for playoff games and Excel files for regular season schedules. It includes automated data pipelines, backtesting capabilities, and a Streamlit web interface for visualization and interaction.

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
- **streamlit**: Web application framework
- **plotly**: Interactive data visualization

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

**Dual-Source Scheduling System**
- Fixed NHL collector to properly filter games by date (was returning full week instead of single day)
- Fixed MLB collector to detect playoff games using gameType field
- Added `is_playoff` flag to all games for playoff identification
- Created `ExcelScheduleReader` utility for parsing user-provided schedule files
- Updated `SportDataManager` to automatically choose between Excel and API sources
- Added `schedules/` directory with README for user-provided Excel schedules

The system is designed with modularity in mind, allowing for easy integration of additional sports leagues and data sources while maintaining the same architectural patterns.