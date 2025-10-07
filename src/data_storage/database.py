import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np

class DatabaseManager:
    """
    Manages SQLite database for multi-sport prediction system
    """
    
    def __init__(self, db_path: str = "sports_predictions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize multi-sport database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create normalized teams table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS teams (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        league TEXT NOT NULL,
                        team_id TEXT NOT NULL,
                        team_name TEXT NOT NULL,
                        abbreviation TEXT,
                        season_active INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(sport, league, team_id, season_active)
                    )
                """)
                
                # Create normalized games table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS games (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        league TEXT NOT NULL,
                        game_id TEXT NOT NULL,
                        season INTEGER NOT NULL,
                        game_date DATE NOT NULL,
                        home_team_id TEXT NOT NULL,
                        away_team_id TEXT NOT NULL,
                        status TEXT DEFAULT 'scheduled',
                        home_score INTEGER,
                        away_score INTEGER,
                        source_keys TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(sport, league, game_id)
                    )
                """)
                
                # Create statcast data table (baseball-specific with sport column)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS statcast_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT DEFAULT 'MLB',
                        game_pk TEXT,
                        game_date DATE,
                        home_team TEXT,
                        away_team TEXT,
                        home_score INTEGER,
                        away_score INTEGER,
                        total_runs INTEGER,
                        home_win INTEGER,
                        pitch_type TEXT,
                        release_speed REAL,
                        launch_speed REAL,
                        launch_angle REAL,
                        hit_distance_sc REAL,
                        release_spin_rate REAL,
                        plate_x REAL,
                        plate_z REAL,
                        hard_hit INTEGER,
                        barrel INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create odds data table for real betting lines
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS odds_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_date DATE NOT NULL,
                        sport TEXT NOT NULL,
                        away_team TEXT NOT NULL,
                        home_team TEXT NOT NULL,
                        away_odds INTEGER,
                        home_odds INTEGER,
                        away_implied_prob REAL,
                        home_implied_prob REAL,
                        away_spread REAL,
                        away_spread_odds INTEGER,
                        home_spread REAL,
                        home_spread_odds INTEGER,
                        total_line REAL,
                        over_odds INTEGER,
                        under_odds INTEGER,
                        bookmaker_count INTEGER,
                        collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(game_date, sport, away_team, home_team)
                    )
                """)
                
                # Create sport-aware predictions table with result tracking - no filler data
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        league TEXT NOT NULL,
                        game_id TEXT NOT NULL,
                        game_date DATE NOT NULL,
                        home_team_id TEXT NOT NULL,
                        away_team_id TEXT NOT NULL,
                        predicted_winner TEXT,
                        win_probability REAL,
                        predicted_total REAL,
                        total_confidence REAL,
                        model_version TEXT,
                        key_factors TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        actual_winner INTEGER,
                        actual_home_score INTEGER,
                        actual_away_score INTEGER,
                        actual_total REAL,
                        win_prediction_correct INTEGER,
                        total_prediction_error REAL,
                        total_absolute_error REAL,
                        result_updated_at TIMESTAMP,
                        UNIQUE(sport, game_date, home_team_id, away_team_id)
                    )
                """)
                
                # Create sport-aware team stats table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS team_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        league TEXT NOT NULL,
                        team_id TEXT NOT NULL,
                        season INTEGER NOT NULL,
                        date DATE,
                        games_played INTEGER,
                        wins INTEGER,
                        losses INTEGER,
                        metrics TEXT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(sport, league, team_id, season, date)
                    )
                """)
                
                # Create sport-aware model metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        model_version TEXT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create team metrics table for advanced baseball statistics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS team_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        league TEXT NOT NULL,
                        team_id TEXT NOT NULL,
                        season INTEGER NOT NULL,
                        date DATE NOT NULL,
                        -- Core Pythagorean Stats
                        runs_scored INTEGER,
                        runs_allowed INTEGER,
                        run_differential INTEGER,
                        games_played INTEGER,
                        wins INTEGER,
                        losses INTEGER,
                        -- Top Correlated Pitching Stats (11 of top 19)
                        era REAL,
                        fip REAL,
                        lob_percent REAL,
                        war_pitching REAL,
                        whip REAL,
                        h_per_9 REAL,
                        batting_avg_against REAL,
                        saves INTEGER,
                        bb_per_k_pitchers REAL,
                        wrc_plus_pitchers REAL,
                        iso_pitchers REAL,
                        -- Top Hitting Stats
                        war_hitting REAL,
                        obp REAL,
                        slg REAL,
                        wrc_plus REAL,
                        iso REAL,
                        woba REAL,
                        ops REAL,
                        -- Pythagorean Calculations
                        pythag_wins REAL,
                        pythag_win_pct REAL,
                        pythag_exponent REAL DEFAULT 2.0,
                        -- Rolling Windows (14 and 30 days)
                        runs_scored_14 REAL,
                        runs_allowed_14 REAL,
                        run_diff_14 REAL,
                        era_14 REAL,
                        fip_14 REAL,
                        pythag_win_pct_14 REAL,
                        runs_scored_30 REAL,
                        runs_allowed_30 REAL,
                        run_diff_30 REAL,
                        era_30 REAL,
                        fip_30 REAL,
                        pythag_win_pct_30 REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(sport, league, team_id, date)
                    )
                """)
                
                # Create prediction accuracy tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_accuracy (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        model_version TEXT,
                        date_period DATE NOT NULL,
                        total_predictions INTEGER NOT NULL,
                        correct_predictions INTEGER NOT NULL,
                        accuracy_rate REAL NOT NULL,
                        avg_win_confidence REAL,
                        total_mae REAL,
                        total_rmse REAL,
                        confidence_calibration REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(sport, model_type, model_version, date_period)
                    )
                """)
                
                # Create model performance trends table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_trends (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        trend_period TEXT NOT NULL, -- 'daily', 'weekly', 'monthly'
                        period_start DATE NOT NULL,
                        period_end DATE NOT NULL,
                        accuracy_trend REAL,
                        mae_trend REAL,
                        volume_trend REAL,
                        improvement_score REAL,
                        retraining_triggered INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create learning insights table for storing what the model learned
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_insights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sport TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        insight_type TEXT NOT NULL, -- 'error_pattern', 'feature_importance', 'retraining_trigger'
                        insight_category TEXT, -- 'team_bias', 'total_overestimate', etc.
                        insight_description TEXT NOT NULL,
                        confidence_score REAL,
                        action_taken TEXT,
                        improvement_measure REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_teams_sport_league ON teams(sport, league)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_games_sport_date ON games(sport, game_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_statcast_game_date ON statcast_data(game_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_statcast_game_pk ON statcast_data(game_pk)")
                # Add missing columns to existing predictions table for learning system
                self._migrate_predictions_table(conn)
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_sport_date ON predictions(sport, game_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_result_tracking ON predictions(game_id, result_updated_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_team_stats_sport_team ON team_stats(sport, team_id, season)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_sport ON model_metrics(sport, model_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_accuracy_sport_date ON prediction_accuracy(sport, model_type, date_period)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_trends_sport ON performance_trends(sport, model_type, period_start)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_insights_type ON learning_insights(sport, model_type, insight_type)")
                
                conn.commit()
                self.logger.info("Multi-sport database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _migrate_predictions_table(self, conn):
        """Add missing columns to existing predictions table for learning system"""
        try:
            # Check if result tracking columns exist
            cursor = conn.execute("PRAGMA table_info(predictions)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            # Define new columns needed for learning system
            new_columns = {
                'actual_winner': 'INTEGER',
                'actual_home_score': 'INTEGER', 
                'actual_away_score': 'INTEGER',
                'actual_total': 'REAL',
                'win_prediction_correct': 'INTEGER',
                'total_prediction_error': 'REAL',
                'total_absolute_error': 'REAL',
                'result_updated_at': 'TIMESTAMP'
            }
            
            # Add missing columns
            columns_added = 0
            for column_name, column_type in new_columns.items():
                if column_name not in existing_columns:
                    try:
                        conn.execute(f"ALTER TABLE predictions ADD COLUMN {column_name} {column_type}")
                        columns_added += 1
                        self.logger.info(f"Added column {column_name} to predictions table")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" not in str(e):
                            self.logger.warning(f"Could not add column {column_name}: {str(e)}")
            
            if columns_added > 0:
                self.logger.info(f"Migration completed: Added {columns_added} learning system columns to predictions table")
            else:
                self.logger.debug("Predictions table already has all learning system columns")
                
        except Exception as e:
            self.logger.error(f"Error migrating predictions table: {str(e)}")
            # Don't raise - migration failure shouldn't prevent database initialization
    
    def store_statcast_data(self, data: pd.DataFrame) -> bool:
        """
        Store Statcast data in database
        
        Args:
            data: DataFrame with Statcast data
            
        Returns:
            True if successful
        """
        try:
            if data.empty:
                self.logger.warning("No Statcast data to store")
                return False
            
            # Prepare data for insertion
            columns_mapping = {
                'game_pk': 'game_pk',
                'game_date': 'game_date',
                'home_team': 'home_team',
                'away_team': 'away_team',
                'home_score': 'home_score',
                'away_score': 'away_score',
                'total_runs': 'total_runs',
                'home_win': 'home_win',
                'pitch_type': 'pitch_type',
                'release_speed': 'release_speed',
                'launch_speed': 'launch_speed',
                'launch_angle': 'launch_angle',
                'hit_distance_sc': 'hit_distance_sc',
                'release_spin_rate': 'release_spin_rate',
                'plate_x': 'plate_x',
                'plate_z': 'plate_z',
                'hard_hit': 'hard_hit',
                'barrel': 'barrel'
            }
            
            # Select and rename columns
            insert_data = pd.DataFrame()
            for db_col, df_col in columns_mapping.items():
                if df_col in data.columns:
                    insert_data[db_col] = data[df_col]
                else:
                    insert_data[db_col] = None
            
            # Add sport column (defaults to MLB for baseball data)
            insert_data['sport'] = 'MLB'
            
            # Convert data types
            if 'game_date' in insert_data.columns:
                insert_data['game_date'] = pd.to_datetime(insert_data['game_date']).dt.date
            
            # Handle NaN values
            insert_data = insert_data.replace({np.nan: None})
            
            with sqlite3.connect(self.db_path) as conn:
                # Insert data, replacing duplicates
                insert_data.to_sql('statcast_data', conn, if_exists='append', index=False)
                
                rows_inserted = len(insert_data)
                self.logger.info(f"Stored {rows_inserted} Statcast records")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing Statcast data: {str(e)}")
            return False
    
    def store_teams(self, teams_df: pd.DataFrame) -> bool:
        """
        Store team information in database
        
        Args:
            teams_df: DataFrame with team information (sport, league, team_id, team_name, etc.)
            
        Returns:
            True if successful
        """
        try:
            if teams_df.empty:
                self.logger.warning("No team data to store")
                return False
            
            # Ensure required columns exist
            required_cols = ['sport', 'league', 'team_id', 'team_name']
            missing_cols = [col for col in required_cols if col not in teams_df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Add current season if not present
            if 'season_active' not in teams_df.columns:
                teams_df['season_active'] = datetime.now().year
            
            # Handle NaN values
            teams_df = teams_df.replace({np.nan: None})
            
            with sqlite3.connect(self.db_path) as conn:
                # Use ON CONFLICT DO UPDATE to preserve created_at and update changed fields
                cursor = conn.cursor()
                
                for _, row in teams_df.iterrows():
                    cursor.execute("""
                        INSERT INTO teams 
                        (sport, league, team_id, team_name, abbreviation, season_active)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(sport, league, team_id, season_active) DO UPDATE SET
                            team_name = excluded.team_name,
                            abbreviation = excluded.abbreviation
                    """, (
                        row.get('sport'),
                        row.get('league'), 
                        row.get('team_id'),
                        row.get('team_name'),
                        row.get('abbreviation'),
                        row.get('season_active')
                    ))
                
                conn.commit()
                rows_inserted = len(teams_df)
                self.logger.info(f"Processed {rows_inserted} team records")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing team data: {str(e)}")
            return False
    
    def store_games(self, games_df: pd.DataFrame) -> bool:
        """
        Store game information in database
        
        Args:
            games_df: DataFrame with game information
            
        Returns:
            True if successful
        """
        try:
            if games_df.empty:
                self.logger.warning("No game data to store")
                return False
            
            # Ensure required columns exist
            required_cols = ['sport', 'league', 'game_id', 'game_date', 'home_team_id', 'away_team_id']
            missing_cols = [col for col in required_cols if col not in games_df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Add current season if not present
            if 'season' not in games_df.columns:
                games_df['season'] = datetime.now().year
            
            # Convert source_keys to JSON string if it's a dict
            if 'source_keys' in games_df.columns:
                games_df['source_keys'] = games_df['source_keys'].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
            
            # Handle NaN values
            games_df = games_df.replace({np.nan: None})
            
            with sqlite3.connect(self.db_path) as conn:
                # Use ON CONFLICT DO UPDATE to preserve created_at and update changed fields
                cursor = conn.cursor()
                
                for _, row in games_df.iterrows():
                    cursor.execute("""
                        INSERT INTO games 
                        (sport, league, game_id, season, game_date, home_team_id, away_team_id, 
                         status, home_score, away_score, source_keys)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(sport, league, game_id) DO UPDATE SET
                            status = excluded.status,
                            home_score = excluded.home_score,
                            away_score = excluded.away_score,
                            source_keys = excluded.source_keys,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        row.get('sport'),
                        row.get('league'),
                        row.get('game_id'),
                        row.get('season'),
                        row.get('game_date'),
                        row.get('home_team_id'),
                        row.get('away_team_id'),
                        row.get('status', 'scheduled'),
                        row.get('home_score'),
                        row.get('away_score'),
                        row.get('source_keys')
                    ))
                
                conn.commit()
                rows_inserted = len(games_df)
                self.logger.info(f"Processed {rows_inserted} game records")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing game data: {str(e)}")
            return False
    
    def store_team_stats(self, team_stats_df: pd.DataFrame) -> bool:
        """
        Store team statistics in database
        
        Args:
            team_stats_df: DataFrame with team statistics
            
        Returns:
            True if successful
        """
        try:
            if team_stats_df.empty:
                self.logger.warning("No team stats data to store")
                return False
            
            # Ensure required columns exist
            required_cols = ['sport', 'league', 'team_id', 'season']
            missing_cols = [col for col in required_cols if col not in team_stats_df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Convert metrics dict to JSON string
            if 'metrics' in team_stats_df.columns:
                team_stats_df['metrics'] = team_stats_df['metrics'].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
            
            # Handle NaN values
            team_stats_df = team_stats_df.replace({np.nan: None})
            
            with sqlite3.connect(self.db_path) as conn:
                # Use ON CONFLICT DO UPDATE to preserve creation time and update relevant fields
                cursor = conn.cursor()
                
                for _, row in team_stats_df.iterrows():
                    cursor.execute("""
                        INSERT INTO team_stats 
                        (sport, league, team_id, season, date, games_played, wins, losses, metrics)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(sport, league, team_id, season, date) DO UPDATE SET
                            games_played = excluded.games_played,
                            wins = excluded.wins,
                            losses = excluded.losses,
                            metrics = excluded.metrics,
                            last_updated = CURRENT_TIMESTAMP
                    """, (
                        row.get('sport'),
                        row.get('league'),
                        row.get('team_id'),
                        row.get('season'),
                        row.get('date'),
                        row.get('games_played'),
                        row.get('wins'),
                        row.get('losses'),
                        row.get('metrics')
                    ))
                
                conn.commit()
                rows_inserted = len(team_stats_df)
                self.logger.info(f"Processed {rows_inserted} team stats records")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing team stats data: {str(e)}")
            return False
    
    def save_odds(self, odds_df: pd.DataFrame, sport: str) -> bool:
        """Save odds data to database including moneyline, spreads, and totals"""
        if odds_df.empty:
            return True
            
        try:
            with self._get_connection() as conn:
                for _, row in odds_df.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO odds_data 
                        (game_date, sport, away_team, home_team, away_odds, home_odds, 
                         away_implied_prob, home_implied_prob, away_spread, away_spread_odds,
                         home_spread, home_spread_odds, total_line, over_odds, under_odds,
                         bookmaker_count, collected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['game_date'], sport, row['away_team'], row['home_team'],
                        row['away_odds'], row['home_odds'], row['away_implied_prob'],
                        row['home_implied_prob'], 
                        row.get('away_spread'), row.get('away_spread_odds'),
                        row.get('home_spread'), row.get('home_spread_odds'),
                        row.get('total_line'), row.get('over_odds'), row.get('under_odds'),
                        row['bookmaker_count'], row['collected_at']
                    ))
                conn.commit()
                self.logger.info(f"Saved {len(odds_df)} odds records for {sport}")
                return True
        except Exception as e:
            self.logger.error(f"Error saving odds: {str(e)}")
            return False
    
    def store_odds_data(self, odds_df: pd.DataFrame, sport: str = 'MLB') -> bool:
        """Alias for save_odds to maintain compatibility"""
        return self.save_odds(odds_df, sport)
    
    def get_odds_for_game(self, game_date, sport: str, away_team: str, home_team: str) -> Optional[Dict]:
        """Get odds for a specific game"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT away_odds, home_odds, away_implied_prob, home_implied_prob
                    FROM odds_data 
                    WHERE game_date = ? AND sport = ? AND away_team = ? AND home_team = ?
                    ORDER BY collected_at DESC LIMIT 1
                """, (game_date, sport, away_team, home_team))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'away_odds': row[0],
                        'home_odds': row[1], 
                        'away_implied_prob': row[2],
                        'home_implied_prob': row[3]
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting odds: {str(e)}")
            return None
    
    def store_predictions(self, predictions: List[Dict]) -> bool:
        """
        Store model predictions in database (sport-aware)
        
        Args:
            predictions: List of prediction dictionaries with sport-aware schema
            
        Returns:
            True if successful
        """
        try:
            if not predictions:
                return False
            
            predictions_df = pd.DataFrame(predictions)
            
            # Ensure required columns for new sport-aware schema
            # predicted_total and total_confidence are optional for winner-only models
            required_columns = [
                'sport', 'league', 'game_id', 'game_date', 'home_team_id', 'away_team_id',
                'predicted_winner', 'win_probability'
            ]
            
            missing_cols = [col for col in required_columns if col not in predictions_df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Convert key_factors to JSON string if it's a dict/list
            if 'key_factors' in predictions_df.columns:
                predictions_df['key_factors'] = predictions_df['key_factors'].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )
            
            # Handle date conversion
            if 'game_date' in predictions_df.columns:
                predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date']).dt.date
            
            predictions_df = predictions_df.replace({np.nan: None})
            
            with sqlite3.connect(self.db_path) as conn:
                # Use INSERT OR IGNORE for predictions to avoid duplicate errors
                cursor = conn.cursor()
                
                for _, row in predictions_df.iterrows():
                    cursor.execute("""
                        INSERT OR IGNORE INTO predictions 
                        (sport, league, game_id, game_date, home_team_id, away_team_id,
                         predicted_winner, win_probability, predicted_total, total_confidence,
                         model_version, key_factors)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('sport'),
                        row.get('league'),
                        row.get('game_id'),
                        row.get('game_date'),
                        row.get('home_team_id'),
                        row.get('away_team_id'),
                        row.get('predicted_winner'),
                        row.get('win_probability'),
                        row.get('predicted_total'),
                        row.get('total_confidence'),
                        row.get('model_version'),
                        row.get('key_factors')
                    ))
                
                conn.commit()
                self.logger.info(f"Stored {len(predictions_df)} predictions")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing predictions: {str(e)}")
            return False
    
    def get_historical_games(self, start_date: datetime, end_date: datetime, 
                           sport: str = None, league: str = None) -> pd.DataFrame:
        """
        Get historical game data for analysis (sport-aware)
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            sport: Optional sport filter (e.g., 'MLB', 'NBA')
            league: Optional league filter (e.g., 'MLB', 'NBA', 'NCAA')
            
        Returns:
            DataFrame with historical game data
        """
        try:
            # Build query with optional sport/league filters
            where_conditions = ["g.game_date BETWEEN ? AND ?"]
            params = [start_date.date(), end_date.date()]
            
            if sport:
                where_conditions.append("g.sport = ?")
                params.append(sport)
            
            if league:
                where_conditions.append("g.league = ?") 
                params.append(league)
            
            query = f"""
                SELECT 
                    g.sport,
                    g.league,
                    g.game_id,
                    g.game_date,
                    g.season,
                    ht.team_name as home_team,
                    at.team_name as away_team,
                    g.home_team_id,
                    g.away_team_id,
                    g.home_score,
                    g.away_score,
                    g.status,
                    g.source_keys
                FROM games g
                LEFT JOIN teams ht ON g.home_team_id = ht.team_id AND g.sport = ht.sport AND g.league = ht.league AND g.season = ht.season_active
                LEFT JOIN teams at ON g.away_team_id = at.team_id AND g.sport = at.sport AND g.league = at.league AND g.season = at.season_active
                WHERE {' AND '.join(where_conditions)}
                    AND g.status = 'final'
                    AND g.home_score IS NOT NULL
                    AND g.away_score IS NOT NULL
                ORDER BY g.game_date DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
            self.logger.info(f"Retrieved {len(df)} historical games for {sport or 'all sports'}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical games: {str(e)}")
            return pd.DataFrame()
    
    def get_training_data(self, days_back: int = 90) -> pd.DataFrame:
        """
        Get training data for ML models
        
        Args:
            days_back: Number of days back to fetch data
            
        Returns:
            DataFrame with training features and targets
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get aggregated game data with team stats
            query = """
                WITH game_aggregates AS (
                    SELECT 
                        game_pk,
                        game_date,
                        home_team,
                        away_team,
                        MAX(home_score) as home_score,
                        MAX(away_score) as away_score,
                        MAX(total_runs) as total_runs,
                        MAX(home_win) as home_win,
                        COUNT(*) as total_pitches,
                        AVG(release_speed) as avg_velocity,
                        AVG(CASE WHEN launch_speed IS NOT NULL THEN launch_speed END) as avg_exit_velocity,
                        SUM(CASE WHEN hard_hit = 1 THEN 1 ELSE 0 END) as hard_hits,
                        SUM(CASE WHEN barrel = 1 THEN 1 ELSE 0 END) as barrels
                    FROM statcast_data
                    WHERE game_date BETWEEN ? AND ?
                        AND game_pk IS NOT NULL
                    GROUP BY game_pk, game_date, home_team, away_team
                )
                SELECT *
                FROM game_aggregates
                WHERE total_runs IS NOT NULL
                    AND home_win IS NOT NULL
                ORDER BY game_date
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[start_date.date(), end_date.date()])
            
            self.logger.info(f"Retrieved {len(df)} training records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {str(e)}")
            return pd.DataFrame()
    
    def get_odds_for_game(self, home_team: str, away_team: str, game_date: datetime) -> pd.DataFrame:
        """
        Get betting odds for a specific game including moneyline, spreads, and totals
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation  
            game_date: Date of the game
            
        Returns:
            DataFrame with odds data
        """
        try:
            query = """
                SELECT home_odds, away_odds, home_implied_prob, away_implied_prob,
                       away_spread, away_spread_odds, home_spread, home_spread_odds,
                       total_line, over_odds, under_odds
                FROM odds_data
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                ORDER BY collected_at DESC
                LIMIT 1
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[home_team, away_team, game_date.date()])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting odds data: {str(e)}")
            return pd.DataFrame()
    
    def get_head_to_head_games(self, home_team: str, away_team: str, limit: int = 10) -> pd.DataFrame:
        """
        Get head-to-head games between two teams
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            limit: Maximum number of recent games to return
            
        Returns:
            DataFrame with historical matchup data
        """
        try:
            query = """
                SELECT game_date, home_team, away_team, home_score, away_score, home_win
                FROM statcast_data
                WHERE ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
                    AND home_score IS NOT NULL AND away_score IS NOT NULL
                GROUP BY game_pk
                ORDER BY game_date DESC
                LIMIT ?
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[home_team, away_team, away_team, home_team, limit])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting head-to-head games: {str(e)}")
            return pd.DataFrame()
    
    def store_team_metrics(self, metrics_data: pd.DataFrame) -> bool:
        """
        Store team metrics data with Pythagorean calculations
        
        Args:
            metrics_data: DataFrame with team metrics
            
        Returns:
            True if successful
        """
        try:
            if metrics_data.empty:
                self.logger.warning("No team metrics data to store")
                return False
            
            # Ensure required columns exist
            required_columns = ['sport', 'league', 'team_id', 'season', 'date']
            missing_columns = [col for col in required_columns if col not in metrics_data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns in team metrics: {missing_columns}")
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                # Use INSERT OR REPLACE to handle updates
                metrics_data.to_sql('team_metrics', conn, if_exists='append', index=False, method='multi')
                
            self.logger.info(f"Stored {len(metrics_data)} team metrics records")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing team metrics: {str(e)}")
            return False
    
    def get_team_metrics(self, team_id: str, date: datetime, days_back: int = 1) -> pd.DataFrame:
        """
        Get team metrics for a specific team and date range
        
        Args:
            team_id: Team abbreviation
            date: Reference date  
            days_back: Days to look back from date
            
        Returns:
            DataFrame with team metrics
        """
        try:
            start_date = date - timedelta(days=days_back)
            
            query = """
                SELECT *
                FROM team_metrics
                WHERE team_id = ? AND date BETWEEN ? AND ?
                ORDER BY date DESC
                LIMIT 1
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[team_id, start_date.date(), date.date()])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting team metrics: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_data_timestamp(self, table: str) -> Optional[str]:
        """
        Get the latest data timestamp for a table
        
        Args:
            table: Table name ('statcast' or 'odds')
            
        Returns:
            Latest timestamp string or None
        """
        try:
            table_mapping = {
                'statcast': 'statcast_data'
            }
            
            actual_table = table_mapping.get(table, table)
            
            query = f"SELECT MAX(created_at) FROM {actual_table}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query)
                result = cursor.fetchone()[0]
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting latest timestamp for {table}: {str(e)}")
            return None
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with database stats
        """
        try:
            stats = {}
            
            with sqlite3.connect(self.db_path) as conn:
                # Count records in each table
                tables = ['statcast_data', 'predictions', 'team_stats', 'teams', 'games', 'model_metrics']
                
                for table in tables:
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        stats[f"{table.replace('_data', '')}_records"] = count
                    except:
                        stats[f"{table.replace('_data', '')}_records"] = 0
                
                # Get unique games count
                try:
                    cursor = conn.execute("SELECT COUNT(DISTINCT game_pk) FROM statcast_data WHERE game_pk IS NOT NULL")
                    stats['total_games'] = cursor.fetchone()[0]
                except:
                    stats['total_games'] = 0
            
            # Get database file size
            try:
                if os.path.exists(self.db_path):
                    size_bytes = os.path.getsize(self.db_path)
                    stats['db_size_mb'] = size_bytes / (1024 * 1024)
                else:
                    stats['db_size_mb'] = 0
            except:
                stats['db_size_mb'] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def get_team_recent_performance(self, team: str, days: int = 30) -> pd.DataFrame:
        """
        Get recent performance data for a team
        
        Args:
            team: Team abbreviation
            days: Number of days back to look
            
        Returns:
            DataFrame with recent team performance
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT DISTINCT
                    game_date,
                    CASE WHEN home_team = ? THEN 'home' ELSE 'away' END as venue,
                    CASE WHEN home_team = ? THEN away_team ELSE home_team END as opponent,
                    CASE 
                        WHEN home_team = ? AND home_win = 1 THEN 1
                        WHEN away_team = ? AND home_win = 0 THEN 1
                        ELSE 0
                    END as team_won,
                    total_runs,
                    CASE WHEN home_team = ? THEN home_score ELSE away_score END as team_score,
                    CASE WHEN home_team = ? THEN away_score ELSE home_score END as opponent_score
                FROM statcast_data
                WHERE (home_team = ? OR away_team = ?)
                    AND game_date >= ?
                    AND total_runs IS NOT NULL
                ORDER BY game_date DESC
            """
            
            params = [team] * 8 + [cutoff_date.date()]
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting team performance for {team}: {str(e)}")
            return pd.DataFrame()
    
    def store_model_metrics(self, metrics: Dict, sport: str = 'MLB') -> bool:
        """
        Store model performance metrics
        
        Args:
            metrics: Dictionary with model metrics
            sport: Sport name (required by schema)
            
        Returns:
            True if successful
        """
        try:
            records = []
            
            for model_type, model_metrics in metrics.items():
                if isinstance(model_metrics, dict):
                    for metric_name, metric_value in model_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            records.append({
                                'sport': sport,
                                'model_type': model_type,
                                'metric_name': metric_name,
                                'metric_value': float(metric_value)
                            })
            
            if records:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for record in records:
                        cursor.execute("""
                            INSERT INTO model_metrics 
                            (sport, model_type, metric_name, metric_value)
                            VALUES (?, ?, ?, ?)
                        """, (
                            record['sport'],
                            record['model_type'],
                            record['metric_name'],
                            record['metric_value']
                        ))
                    
                    conn.commit()
                
                self.logger.info(f"Stored {len(records)} model metrics for {sport}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing model metrics: {str(e)}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """
        Clean up old data to manage database size
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            True if successful
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                # Delete old statcast data
                conn.execute("DELETE FROM statcast_data WHERE game_date < ?", [cutoff_date.date()])
                
                # Delete old model metrics
                conn.execute("DELETE FROM model_metrics WHERE date_recorded < ?", [cutoff_date])
                
                # Delete old predictions
                conn.execute("DELETE FROM predictions WHERE game_date < ?", [cutoff_date.date()])
                
                conn.commit()
                
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
            return False
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
