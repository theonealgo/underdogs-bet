"""
WNBA data collector for schedules, scores, and team statistics.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class WNBADataCollector:
    """
    WNBA data collector using ESPN API and other public data sources.
    
    Provides access to WNBA game schedules, scores, team statistics,
    and historical data through various public APIs.
    """
    
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_todays_games(self) -> pd.DataFrame:
        """Get today's WNBA games."""
        try:
            today = datetime.now().strftime('%Y%m%d')
            url = f"{self.base_url}/scoreboard?dates={today}"
            
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            games = []
            if 'events' in data:
                for event in data['events']:
                    game_info = self._parse_game_data(event)
                    if game_info:
                        games.append(game_info)
            
            return pd.DataFrame(games)
            
        except Exception as e:
            logger.error(f"Error fetching WNBA games: {e}")
            return pd.DataFrame()
    
    def get_schedule(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get WNBA schedule for date range."""
        try:
            games = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
            while current_date <= end_date_obj:
                date_str = current_date.strftime('%Y%m%d')
                url = f"{self.base_url}/scoreboard?dates={date_str}"
                
                response = self.session.get(url)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'events' in data:
                        for event in data['events']:
                            game_info = self._parse_game_data(event)
                            if game_info:
                                games.append(game_info)
                
                current_date += timedelta(days=1)
            
            return pd.DataFrame(games)
            
        except Exception as e:
            logger.error(f"Error fetching WNBA schedule: {e}")
            return pd.DataFrame()
    
    def get_team_stats(self, season: Optional[int] = None) -> pd.DataFrame:
        """Get WNBA team statistics."""
        try:
            if season is None:
                season = datetime.now().year
            
            url = f"{self.base_url}/teams"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            teams = []
            if 'sports' in data and len(data['sports']) > 0:
                if 'leagues' in data['sports'][0] and len(data['sports'][0]['leagues']) > 0:
                    if 'teams' in data['sports'][0]['leagues'][0]:
                        for team_data in data['sports'][0]['leagues'][0]['teams']:
                            team_info = self._parse_team_data(team_data['team'])
                            if team_info:
                                teams.append(team_info)
            
            return pd.DataFrame(teams)
            
        except Exception as e:
            logger.error(f"Error fetching WNBA team stats: {e}")
            return pd.DataFrame()
    
    def _parse_game_data(self, event: Dict) -> Optional[Dict]:
        """Parse individual game data from ESPN API response."""
        try:
            game_info = {
                'sport': 'WNBA',
                'league': 'WNBA',
                'game_id': event.get('id'),
                'game_date': event.get('date'),
                'season': event.get('season', {}).get('year'),
                'status': event.get('status', {}).get('type', {}).get('name', 'Scheduled')
            }
            
            # Parse teams
            if 'competitions' in event and len(event['competitions']) > 0:
                competition = event['competitions'][0]
                
                if 'competitors' in competition:
                    for competitor in competition['competitors']:
                        team = competitor.get('team', {})
                        home_away = competitor.get('homeAway')
                        
                        if home_away == 'home':
                            game_info['home_team_id'] = team.get('id')
                            game_info['home_team'] = team.get('displayName')
                            game_info['home_abbreviation'] = team.get('abbreviation')
                            if 'score' in competitor:
                                game_info['home_score'] = competitor['score']
                        else:
                            game_info['away_team_id'] = team.get('id')
                            game_info['away_team'] = team.get('displayName')
                            game_info['away_abbreviation'] = team.get('abbreviation')
                            if 'score' in competitor:
                                game_info['away_score'] = competitor['score']
            
            return game_info
            
        except Exception as e:
            logger.error(f"Error parsing WNBA game data: {e}")
            return None
    
    def _parse_team_data(self, team: Dict) -> Optional[Dict]:
        """Parse team data from ESPN API response."""
        try:
            return {
                'team_id': team.get('id'),
                'team_name': team.get('displayName'),
                'abbreviation': team.get('abbreviation'),
                'location': team.get('location'),
                'color': team.get('color'),
                'alternate_color': team.get('alternateColor'),
                'sport': 'WNBA',
                'league': 'WNBA'
            }
            
        except Exception as e:
            logger.error(f"Error parsing WNBA team data: {e}")
            return None
    
    def standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize WNBA data to match database schema."""
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_columns = ['sport', 'league', 'game_id', 'home_team_id', 'away_team_id']
        for col in required_columns:
            if col not in data.columns:
                data[col] = None
        
        # Convert date format if needed
        if 'game_date' in data.columns:
            data['game_date'] = pd.to_datetime(data['game_date'], errors='coerce')
        
        return data