from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

class BaseDataCollector(ABC):
    """
    Abstract base class for sport-specific data collectors.
    
    All sport data collectors must implement these methods to ensure
    consistent data access across different sports.
    """
    
    def __init__(self, sport: str, league: str):
        """
        Initialize the data collector.
        
        Args:
            sport: Sport name (e.g., 'MLB', 'NBA', 'NFL', 'NHL', 'CFB', 'CBB')
            league: League name (e.g., 'MLB', 'NBA', 'NFL', 'NHL', 'NCAA')
        """
        self.sport = sport
        self.league = league
    
    @abstractmethod
    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get game schedule for a date range.
        
        Args:
            start_date: Start date for schedule
            end_date: End date for schedule
            
        Returns:
            DataFrame with columns:
            - sport: Sport name (e.g., 'MLB', 'NBA')
            - league: League name (e.g., 'MLB', 'NBA', 'NCAA')
            - game_id: Unique game identifier
            - game_date: Date of the game
            - home_team_id: Home team identifier
            - home_team_name: Home team name
            - away_team_id: Away team identifier  
            - away_team_name: Away team name
            - season: Season year
            - status: Game status (scheduled, in_progress, final, etc.)
            - source_keys: Optional JSON dict with source-specific IDs
        """
        pass
    
    @abstractmethod
    def get_games(self, game_date: date) -> pd.DataFrame:
        """
        Get games for a specific date.
        
        Args:
            game_date: Date to get games for
            
        Returns:
            DataFrame with same structure as get_schedule() plus:
            - home_score: Home team final score (if game completed)
            - away_score: Away team final score (if game completed)
            Note: Must include sport, league, and optional source_keys columns
        """
        pass
    
    @abstractmethod
    def get_team_stats(self, season: int, rolling_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get team statistics for a season.
        
        Args:
            season: Season year
            rolling_days: If provided, get rolling stats for last N days
            
        Returns:
            DataFrame with columns:
            - sport: Sport name (e.g., 'MLB', 'NBA')
            - league: League name (e.g., 'MLB', 'NBA', 'NCAA')
            - team_id: Team identifier
            - team_name: Team name
            - season: Season year
            - date: Date of stats (for rolling stats)
            - games_played: Number of games played
            - wins: Number of wins
            - losses: Number of losses
            - [sport-specific stats columns]
        """
        pass
    
    @abstractmethod
    def get_recent_form(self, team_id: str, games: int = 10) -> Dict:
        """
        Get recent form/performance for a team.
        
        Args:
            team_id: Team identifier
            games: Number of recent games to analyze
            
        Returns:
            Dictionary with recent performance metrics
            Note: Returns Dict for flexibility, but should include sport/league context
        """
        pass
    
    @abstractmethod
    def get_head_to_head(self, team1_id: str, team2_id: str, 
                        seasons: int = 3) -> pd.DataFrame:
        """
        Get head-to-head matchup history between two teams.
        
        Args:
            team1_id: First team identifier
            team2_id: Second team identifier
            seasons: Number of seasons to look back
            
        Returns:
            DataFrame with historical matchup data
        """
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that collected data meets requirements.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
    
    def get_supported_stats(self) -> List[str]:
        """
        Get list of statistics this collector supports.
        
        Returns:
            List of supported statistic names
        """
        return []
    
    def get_data_source(self) -> str:
        """
        Get the name of the data source this collector uses.
        
        Returns:
            Data source name
        """
        return "Unknown"