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
    Manages SQLite database for MLB prediction system
    """
    
    def __init__(self, db_path: str = "mlb_predictions.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create statcast data table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS statcast_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                
                # Create odds data table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS odds_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_date DATE,
                        away_team TEXT,
                        home_team TEXT,
                        away_moneyline TEXT,
                        home_moneyline TEXT,
                        spread REAL,
                        total REAL,
                        source_url TEXT,
                        trend_text TEXT,
                        bet_type TEXT,
                        record TEXT,
                        extracted_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create predictions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_date DATE,
                        away_team TEXT,
                        home_team TEXT,
                        predicted_winner TEXT,
                        win_probability REAL,
                        predicted_total REAL,
                        total_confidence REAL,
                        model_version TEXT,
                        key_factors TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create team stats table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS team_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        team TEXT,
                        year INTEGER,
                        games_played INTEGER,
                        wins INTEGER,
                        losses INTEGER,
                        batting_avg REAL,
                        era REAL,
                        home_runs INTEGER,
                        runs_scored INTEGER,
                        runs_allowed INTEGER,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create model metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_type TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_statcast_game_date ON statcast_data(game_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_statcast_game_pk ON statcast_data(game_pk)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_game_date ON odds_data(game_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_game_date ON predictions(game_date)")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
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
    
    def store_odds_data(self, data: List[Dict]) -> bool:
        """
        Store odds data in database
        
        Args:
            data: List of odds dictionaries
            
        Returns:
            True if successful
        """
        try:
            if not data:
                self.logger.warning("No odds data to store")
                return False
            
            # Convert to DataFrame for easier handling
            odds_df = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_columns = [
                'away_team', 'home_team', 'away_moneyline', 'home_moneyline',
                'spread', 'total', 'source_url', 'trend_text', 'bet_type', 'record'
            ]
            
            for col in required_columns:
                if col not in odds_df.columns:
                    odds_df[col] = None
            
            # Add game_date if not present
            if 'game_date' not in odds_df.columns:
                odds_df['game_date'] = datetime.now().date()
            
            # Handle extracted_at timestamp
            if 'extracted_at' in odds_df.columns:
                odds_df['extracted_at'] = pd.to_datetime(odds_df['extracted_at'])
            else:
                odds_df['extracted_at'] = datetime.now()
            
            # Clean and convert data types
            if 'spread' in odds_df.columns:
                odds_df['spread'] = pd.to_numeric(odds_df['spread'], errors='coerce')
            if 'total' in odds_df.columns:
                odds_df['total'] = pd.to_numeric(odds_df['total'], errors='coerce')
            
            # Replace NaN with None for SQLite
            odds_df = odds_df.replace({np.nan: None})
            
            with sqlite3.connect(self.db_path) as conn:
                odds_df.to_sql('odds_data', conn, if_exists='append', index=False)
                
                rows_inserted = len(odds_df)
                self.logger.info(f"Stored {rows_inserted} odds records")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing odds data: {str(e)}")
            return False
    
    def store_predictions(self, predictions: List[Dict]) -> bool:
        """
        Store model predictions in database
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            True if successful
        """
        try:
            if not predictions:
                return False
            
            predictions_df = pd.DataFrame(predictions)
            
            # Ensure required columns
            required_columns = [
                'game_date', 'away_team', 'home_team', 'predicted_winner',
                'win_probability', 'predicted_total', 'total_confidence', 'model_version'
            ]
            
            for col in required_columns:
                if col not in predictions_df.columns:
                    predictions_df[col] = None
            
            # Convert key_factors to JSON string if it's a list
            if 'key_factors' in predictions_df.columns:
                predictions_df['key_factors'] = predictions_df['key_factors'].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )
            
            # Handle date conversion
            if 'game_date' in predictions_df.columns:
                predictions_df['game_date'] = pd.to_datetime(predictions_df['game_date']).dt.date
            
            predictions_df = predictions_df.replace({np.nan: None})
            
            with sqlite3.connect(self.db_path) as conn:
                predictions_df.to_sql('predictions', conn, if_exists='append', index=False)
                
                self.logger.info(f"Stored {len(predictions_df)} predictions")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing predictions: {str(e)}")
            return False
    
    def get_historical_games(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical game data for analysis
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with historical game data
        """
        try:
            query = """
                SELECT DISTINCT
                    game_pk,
                    game_date,
                    home_team,
                    away_team,
                    home_score,
                    away_score,
                    total_runs,
                    home_win
                FROM statcast_data
                WHERE game_date BETWEEN ? AND ?
                    AND game_pk IS NOT NULL
                    AND total_runs IS NOT NULL
                ORDER BY game_date DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[start_date.date(), end_date.date()])
                
            self.logger.info(f"Retrieved {len(df)} historical games")
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
                'statcast': 'statcast_data',
                'odds': 'odds_data'
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
                tables = ['statcast_data', 'odds_data', 'predictions', 'team_stats']
                
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
    
    def store_model_metrics(self, metrics: Dict) -> bool:
        """
        Store model performance metrics
        
        Args:
            metrics: Dictionary with model metrics
            
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
                                'model_type': model_type,
                                'metric_name': metric_name,
                                'metric_value': float(metric_value)
                            })
            
            if records:
                metrics_df = pd.DataFrame(records)
                
                with sqlite3.connect(self.db_path) as conn:
                    metrics_df.to_sql('model_metrics', conn, if_exists='append', index=False)
                
                self.logger.info(f"Stored {len(records)} model metrics")
                
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
                
                # Delete old odds data
                conn.execute("DELETE FROM odds_data WHERE game_date < ?", [cutoff_date.date()])
                
                # Delete old predictions
                conn.execute("DELETE FROM predictions WHERE game_date < ?", [cutoff_date.date()])
                
                conn.commit()
                
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
            return False
