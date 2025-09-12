import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging


class MLBScheduleCollector:
    """
    Collector for MLB game schedules using the official MLB Stats API.
    Fetches today's games and upcoming games to populate the database.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
    def get_todays_games(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Get today's MLB games from the MLB Stats API.
        
        Args:
            date: Date in YYYY-MM-DD format. If None, uses today.
            
        Returns:
            DataFrame with today's games
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,  # MLB
                'date': date,
                'hydrate': 'team,linescore'
            }
            
            self.logger.info(f"Fetching MLB schedule for {date}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            if 'dates' in data and data['dates']:
                for date_entry in data['dates']:
                    if 'games' in date_entry:
                        for game in date_entry['games']:
                            game_info = self._parse_game(game)
                            if game_info:
                                games.append(game_info)
            
            if not games:
                self.logger.warning(f"No games found for {date}")
                return pd.DataFrame()
            
            df = pd.DataFrame(games)
            self.logger.info(f"Found {len(df)} games for {date}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching today's games: {str(e)}")
            return pd.DataFrame()
    
    def get_games_for_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get games for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with games in the date range
        """
        try:
            url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,  # MLB
                'startDate': start_date,
                'endDate': end_date,
                'hydrate': 'team,linescore'
            }
            
            self.logger.info(f"Fetching MLB schedule from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            if 'dates' in data and data['dates']:
                for date_entry in data['dates']:
                    if 'games' in date_entry:
                        for game in date_entry['games']:
                            game_info = self._parse_game(game)
                            if game_info:
                                games.append(game_info)
            
            if not games:
                self.logger.warning(f"No games found from {start_date} to {end_date}")
                return pd.DataFrame()
            
            df = pd.DataFrame(games)
            self.logger.info(f"Found {len(df)} games from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching games for date range: {str(e)}")
            return pd.DataFrame()
    
    def _parse_game(self, game: Dict) -> Optional[Dict]:
        """
        Parse a single game from the MLB API response.
        
        Args:
            game: Game data from MLB API
            
        Returns:
            Parsed game info or None if parsing fails
        """
        try:
            # Basic game info
            game_pk = game.get('gamePk')
            if not game_pk:
                return None
            
            # Game date and time
            game_date = game.get('gameDate', '')
            if game_date:
                game_datetime = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                game_date = game_datetime.strftime('%Y-%m-%d')
            else:
                game_date = datetime.now().strftime('%Y-%m-%d')
            
            # Teams
            teams = game.get('teams', {})
            home_team = teams.get('home', {}).get('team', {})
            away_team = teams.get('away', {}).get('team', {})
            
            home_team_id = home_team.get('abbreviation', home_team.get('name', ''))
            away_team_id = away_team.get('abbreviation', away_team.get('name', ''))
            
            if not home_team_id or not away_team_id:
                return None
            
            # Game status
            status = game.get('status', {})
            status_code = status.get('statusCode', 'S')
            detailed_state = status.get('detailedState', 'Scheduled')
            
            # Scores (if game is final)
            home_score = None
            away_score = None
            
            if status_code in ['F', 'O']:  # Final or Other (completed)
                home_score = teams.get('home', {}).get('score')
                away_score = teams.get('away', {}).get('score')
            
            return {
                'game_id': str(game_pk),
                'sport': 'MLB',
                'league': 'MLB',
                'game_date': game_date,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_name': home_team.get('name', home_team_id),
                'away_team_name': away_team.get('name', away_team_id),
                'status': detailed_state,
                'status_code': status_code,
                'home_score': home_score,
                'away_score': away_score,
                'venue': game.get('venue', {}).get('name', ''),
                'game_time': game.get('gameDate', ''),
                'inning': game.get('linescore', {}).get('currentInning', None),
                'inning_state': game.get('linescore', {}).get('inningState', '')
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing game: {str(e)}")
            return None
    
    def get_team_abbreviations(self) -> Dict[str, str]:
        """
        Get mapping of team IDs to abbreviations.
        
        Returns:
            Dictionary mapping team IDs to abbreviations
        """
        try:
            url = f"{self.base_url}/teams"
            params = {'sportId': 1}  # MLB
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            teams = {}
            
            if 'teams' in data:
                for team in data['teams']:
                    team_id = team.get('id')
                    abbreviation = team.get('abbreviation')
                    if team_id and abbreviation:
                        teams[str(team_id)] = abbreviation
            
            return teams
            
        except Exception as e:
            self.logger.error(f"Error fetching team abbreviations: {str(e)}")
            return {}