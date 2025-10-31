"""
Excel schedule reader for importing game schedules from Excel files.
"""

import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class ExcelScheduleReader:
    """
    Read game schedules from Excel files.
    
    Expected Excel format:
    - Date column: Game date
    - Away Team column: Away team name/abbreviation
    - Home Team column: Home team name/abbreviation
    - Optional: Time, Location, etc.
    """
    
    def __init__(self, excel_file_path: str):
        """
        Initialize with path to Excel schedule file.
        
        Args:
            excel_file_path: Path to Excel file with schedule
        """
        self.excel_file_path = excel_file_path
        self.logger = logging.getLogger(__name__)
    
    def read_schedule(self, sport: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Read schedule from Excel file.
        
        Args:
            sport: Sport code (MLB, NHL, NBA, etc.)
            sheet_name: Excel sheet name (defaults to first sheet)
        
        Returns:
            DataFrame with standardized game schedule
        """
        try:
            if not os.path.exists(self.excel_file_path):
                self.logger.error(f"Excel file not found: {self.excel_file_path}")
                return pd.DataFrame()
            
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(self.excel_file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(self.excel_file_path)
            
            self.logger.info(f"Read {len(df)} rows from Excel file")
            
            # Convert to standardized format
            games = self._standardize_schedule(df, sport)
            
            return games
            
        except Exception as e:
            self.logger.error(f"Error reading Excel schedule: {str(e)}")
            return pd.DataFrame()
    
    def _standardize_schedule(self, df: pd.DataFrame, sport: str) -> pd.DataFrame:
        """
        Convert Excel data to standardized game format.
        
        Tries to automatically detect column names.
        """
        # Common column name variations
        date_cols = ['Date', 'date', 'Game Date', 'game_date', 'DATE']
        away_cols = ['Away', 'away', 'Away Team', 'away_team', 'Visitor', 'visitor', 'AWAY']
        home_cols = ['Home', 'home', 'Home Team', 'home_team', 'HOME']
        time_cols = ['Time', 'time', 'Game Time', 'game_time', 'TIME']
        
        # Find matching columns
        date_col = self._find_column(df, date_cols)
        away_col = self._find_column(df, away_cols)
        home_col = self._find_column(df, home_cols)
        time_col = self._find_column(df, time_cols)
        
        if not date_col or not away_col or not home_col:
            self.logger.error(f"Could not find required columns. Found: {df.columns.tolist()}")
            return pd.DataFrame()
        
        games = []
        for idx, row in df.iterrows():
            try:
                # Parse date
                game_date = self._parse_date(row[date_col])
                if not game_date:
                    continue
                
                away_team = str(row[away_col]).strip()
                home_team = str(row[home_col]).strip()
                
                if not away_team or not home_team or away_team == 'nan' or home_team == 'nan':
                    continue
                
                game_id = f"{sport}_{game_date.strftime('%Y%m%d')}_{away_team}_{home_team}"
                
                game_time = 'TBD'
                if time_col and time_col in row:
                    game_time = str(row[time_col])
                
                games.append({
                    'sport': sport,
                    'league': sport,
                    'game_id': game_id,
                    'game_date': game_date,
                    'home_team_id': home_team,
                    'away_team_id': away_team,
                    'home_team_name': home_team,
                    'away_team_name': away_team,
                    'season': game_date.year,
                    'status': 'scheduled',
                    'game_time': game_time,
                    'source_keys': '{"source": "excel_import"}'
                })
            except Exception as e:
                self.logger.warning(f"Error parsing row {idx}: {str(e)}")
                continue
        
        self.logger.info(f"Parsed {len(games)} games from Excel")
        return pd.DataFrame(games)
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column that matches one of the possible names."""
        for col in df.columns:
            if col in possible_names:
                return col
        return None
    
    def _parse_date(self, date_value) -> Optional[date]:
        """Parse date from various formats."""
        try:
            if pd.isna(date_value):
                return None
            
            # If already a datetime/date object
            if isinstance(date_value, (datetime, date)):
                if isinstance(date_value, datetime):
                    return date_value.date()
                return date_value
            
            # Try parsing string
            date_str = str(date_value).strip()
            
            # Try common date formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%m/%d/%y',
                '%d/%m/%Y',
                '%Y%m%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            
            # Let pandas try
            return pd.to_datetime(date_value).date()
            
        except Exception as e:
            self.logger.warning(f"Could not parse date: {date_value}")
            return None
