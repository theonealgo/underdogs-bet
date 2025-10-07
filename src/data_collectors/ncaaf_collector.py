import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from src.interfaces.base_collector import BaseDataCollector


class NCAAFDataCollector(BaseDataCollector):
    """
    NCAA Football data collector using College Football Data API.
    
    Provides access to college football game schedules, scores, team statistics,
    and historical data through the collegefootballdata.com API.
    """
    
    def __init__(self):
        super().__init__(sport='NCAAF', league='NCAA')
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.api_key = os.getenv('CFBD_API_KEY')
        self.base_url = 'https://api.collegefootballdata.com'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
            'Accept': 'application/json',
            'User-Agent': 'Multi-Sport-Prediction-System/1.0'
        }
        
        # Cache for teams and current season
        self._teams_cache = None
        self._team_id_map = {}
        self._current_season = self._get_current_ncaa_season()
        
        # Cache for season games to avoid excessive API calls
        self._season_games_cache = {}
        self._cache_timestamps = {}
        
        # Initialize team data
        self._initialize_teams()
        
        if not self.api_key:
            self.logger.warning("CFBD_API_KEY not found. Some functionality may be limited.")
    
    def _get_current_ncaa_season(self) -> int:
        """Determine current NCAA football season year"""
        now = datetime.now()
        year = now.year
        
        # NCAA football season starts in late August/early September 
        # and ends in January (bowl games and playoffs)
        # August-December belongs to that year's season
        # January-July belongs to the previous year's season
        if now.month >= 8:  # Aug, Sep, Oct, Nov, Dec
            return year
        else:  # Jan-Jul (bowl games from previous season, off-season)
            return year - 1
    
    def _date_to_ncaa_season(self, target_date: date) -> int:
        """Convert a date to its corresponding NCAA football season year"""
        if target_date.month >= 8:  # August-December
            return target_date.year
        else:  # January-July
            return target_date.year - 1
    
    def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[List]:
        """Make a request to the CFBD API"""
        if not self.api_key:
            self.logger.error("CFBD API key is required for data access")
            return None
        
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                self.logger.warning("API rate limit exceeded")
                return None
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Error making API request to {endpoint}: {str(e)}")
            return None
    
    def _initialize_teams(self):
        """Initialize NCAA football teams data"""
        try:
            # Get FBS teams (Division 1 FBS - top level college football)
            teams_data = self._make_api_request('/teams/fbs')
            
            if teams_data:
                self._teams_cache = teams_data
                
                # Create mappings for easier lookups - use both school name and abbreviation as keys
                for team in teams_data:
                    team_abbrev = team.get('abbreviation', '')
                    team_name = team.get('school', '')
                    
                    if team_abbrev and team_name:
                        team_info = {
                            'abbreviation': team_abbrev,
                            'full_name': team_name,
                            'mascot': team.get('mascot', ''),
                            'conference': team.get('conference', ''),
                            'division': team.get('division', ''),
                            'color': team.get('color', ''),
                            'alt_color': team.get('alt_color', ''),
                            'logos': team.get('logos', [])
                        }
                        
                        # Map both abbreviation and full name to the same info
                        self._team_id_map[team_abbrev] = team_info
                        self._team_id_map[team_name] = team_info
                
                self.logger.info(f"Initialized {len(self._teams_cache)} NCAA FBS teams")
            else:
                # Fallback: create basic team mapping
                self._initialize_basic_teams()
                
        except Exception as e:
            self.logger.warning(f"Error initializing NCAA teams via API: {str(e)}")
            self._initialize_basic_teams()
    
    def _initialize_basic_teams(self):
        """Initialize basic NCAA football team mappings as fallback"""
        # Sample of major FBS teams (Power 5 conferences mainly)
        basic_teams = [
            # SEC
            {'abbrev': 'ALA', 'name': 'Alabama', 'conference': 'SEC'},
            {'abbrev': 'UGA', 'name': 'Georgia', 'conference': 'SEC'},
            {'abbrev': 'LSU', 'name': 'LSU', 'conference': 'SEC'},
            {'abbrev': 'FLA', 'name': 'Florida', 'conference': 'SEC'},
            {'abbrev': 'AUB', 'name': 'Auburn', 'conference': 'SEC'},
            {'abbrev': 'TAMU', 'name': 'Texas A&M', 'conference': 'SEC'},
            {'abbrev': 'TENN', 'name': 'Tennessee', 'conference': 'SEC'},
            {'abbrev': 'ARK', 'name': 'Arkansas', 'conference': 'SEC'},
            {'abbrev': 'UK', 'name': 'Kentucky', 'conference': 'SEC'},
            {'abbrev': 'MISS', 'name': 'Ole Miss', 'conference': 'SEC'},
            {'abbrev': 'MSST', 'name': 'Mississippi State', 'conference': 'SEC'},  # Changed from MSU to avoid collision
            {'abbrev': 'MIZ', 'name': 'Missouri', 'conference': 'SEC'},
            {'abbrev': 'SC', 'name': 'South Carolina', 'conference': 'SEC'},
            {'abbrev': 'VAN', 'name': 'Vanderbilt', 'conference': 'SEC'},
            
            # Big Ten
            {'abbrev': 'OSU', 'name': 'Ohio State', 'conference': 'Big Ten'},
            {'abbrev': 'MICH', 'name': 'Michigan', 'conference': 'Big Ten'},
            {'abbrev': 'PSU', 'name': 'Penn State', 'conference': 'Big Ten'},
            {'abbrev': 'WIS', 'name': 'Wisconsin', 'conference': 'Big Ten'},
            {'abbrev': 'MSU', 'name': 'Michigan State', 'conference': 'Big Ten'},
            {'abbrev': 'IOWA', 'name': 'Iowa', 'conference': 'Big Ten'},
            {'abbrev': 'MIN', 'name': 'Minnesota', 'conference': 'Big Ten'},
            {'abbrev': 'NEB', 'name': 'Nebraska', 'conference': 'Big Ten'},
            {'abbrev': 'ILL', 'name': 'Illinois', 'conference': 'Big Ten'},
            {'abbrev': 'IND', 'name': 'Indiana', 'conference': 'Big Ten'},
            {'abbrev': 'PUR', 'name': 'Purdue', 'conference': 'Big Ten'},
            {'abbrev': 'NW', 'name': 'Northwestern', 'conference': 'Big Ten'},
            {'abbrev': 'RUT', 'name': 'Rutgers', 'conference': 'Big Ten'},
            {'abbrev': 'MD', 'name': 'Maryland', 'conference': 'Big Ten'},
            
            # ACC
            {'abbrev': 'CLEM', 'name': 'Clemson', 'conference': 'ACC'},
            {'abbrev': 'FSU', 'name': 'Florida State', 'conference': 'ACC'},
            {'abbrev': 'MIA', 'name': 'Miami', 'conference': 'ACC'},
            {'abbrev': 'UNC', 'name': 'North Carolina', 'conference': 'ACC'},
            {'abbrev': 'NCST', 'name': 'NC State', 'conference': 'ACC'},
            {'abbrev': 'DUKE', 'name': 'Duke', 'conference': 'ACC'},
            {'abbrev': 'WAKE', 'name': 'Wake Forest', 'conference': 'ACC'},
            {'abbrev': 'VT', 'name': 'Virginia Tech', 'conference': 'ACC'},
            {'abbrev': 'UVA', 'name': 'Virginia', 'conference': 'ACC'},
            {'abbrev': 'GT', 'name': 'Georgia Tech', 'conference': 'ACC'},
            {'abbrev': 'LOU', 'name': 'Louisville', 'conference': 'ACC'},
            {'abbrev': 'PITT', 'name': 'Pittsburgh', 'conference': 'ACC'},
            {'abbrev': 'SYR', 'name': 'Syracuse', 'conference': 'ACC'},
            {'abbrev': 'BC', 'name': 'Boston College', 'conference': 'ACC'},
            
            # Big 12
            {'abbrev': 'TEX', 'name': 'Texas', 'conference': 'Big 12'},
            {'abbrev': 'OU', 'name': 'Oklahoma', 'conference': 'Big 12'},
            {'abbrev': 'OKST', 'name': 'Oklahoma State', 'conference': 'Big 12'},
            {'abbrev': 'TCU', 'name': 'TCU', 'conference': 'Big 12'},
            {'abbrev': 'BAY', 'name': 'Baylor', 'conference': 'Big 12'},
            {'abbrev': 'TTU', 'name': 'Texas Tech', 'conference': 'Big 12'},
            {'abbrev': 'KU', 'name': 'Kansas', 'conference': 'Big 12'},
            {'abbrev': 'KSU', 'name': 'Kansas State', 'conference': 'Big 12'},
            {'abbrev': 'ISU', 'name': 'Iowa State', 'conference': 'Big 12'},
            {'abbrev': 'WVU', 'name': 'West Virginia', 'conference': 'Big 12'},
            
            # Pac-12
            {'abbrev': 'USC', 'name': 'USC', 'conference': 'Pac-12'},
            {'abbrev': 'UCLA', 'name': 'UCLA', 'conference': 'Pac-12'},
            {'abbrev': 'ORE', 'name': 'Oregon', 'conference': 'Pac-12'},
            {'abbrev': 'ORST', 'name': 'Oregon State', 'conference': 'Pac-12'},
            {'abbrev': 'WASH', 'name': 'Washington', 'conference': 'Pac-12'},
            {'abbrev': 'WSU', 'name': 'Washington State', 'conference': 'Pac-12'},
            {'abbrev': 'STAN', 'name': 'Stanford', 'conference': 'Pac-12'},
            {'abbrev': 'CAL', 'name': 'California', 'conference': 'Pac-12'},
            {'abbrev': 'ASU', 'name': 'Arizona State', 'conference': 'Pac-12'},
            {'abbrev': 'ARIZ', 'name': 'Arizona', 'conference': 'Pac-12'},
            {'abbrev': 'UTAH', 'name': 'Utah', 'conference': 'Pac-12'},
            {'abbrev': 'COL', 'name': 'Colorado', 'conference': 'Pac-12'}
        ]
        
        self._teams_cache = basic_teams
        for team in basic_teams:
            team_abbrev = team['abbrev']
            team_name = team['name']
            
            team_info = {
                'abbreviation': team_abbrev,
                'full_name': team_name,
                'conference': team['conference'],
                'mascot': '',
                'division': '',
                'color': '',
                'alt_color': '',
                'logos': []
            }
            
            # Map both abbreviation and full name to the same info
            self._team_id_map[team_abbrev] = team_info
            self._team_id_map[team_name] = team_info
        
        self.logger.info(f"Initialized {len(basic_teams)} NCAA FBS teams (basic fallback)")
    
    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get NCAA football game schedule for a date range.
        
        Args:
            start_date: Start date for schedule
            end_date: End date for schedule
            
        Returns:
            DataFrame with standardized schedule columns
        """
        try:
            self.logger.info(f"Getting NCAA football schedule from {start_date} to {end_date}")
            
            # Determine which NCAA seasons to query based on date range
            seasons = set()
            current_date = start_date
            while current_date <= end_date:
                seasons.add(self._date_to_ncaa_season(current_date))
                current_date += timedelta(days=30)  # Sample dates across the range
            
            seasons = sorted(list(seasons))
            schedule_data = []
            
            for season in seasons:
                try:
                    # Get games for this season
                    season_games = self._make_api_request('/games', {'year': season})
                    
                    if season_games:
                        for game in season_games:
                            game_date_str = game.get('startDate')
                            if game_date_str:
                                # Parse the date
                                try:
                                    game_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                                    game_date = game_datetime.date()
                                    
                                    # Check if game is within our date range
                                    if start_date <= game_date <= end_date:
                                        schedule_data.append(self._parse_schedule_game(game, season))
                                        
                                except (ValueError, TypeError) as e:
                                    self.logger.warning(f"Error parsing game date {game_date_str}: {str(e)}")
                                    continue
                
                except Exception as e:
                    self.logger.warning(f"Error getting NCAA schedule for season {season}: {str(e)}")
                    continue
            
            if not schedule_data:
                self.logger.warning(f"No NCAA football games found for date range {start_date} to {end_date}")
                return self._create_empty_schedule_df()
            
            result_df = pd.DataFrame(schedule_data)
            self.logger.info(f"Retrieved {len(result_df)} NCAA football games")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA football schedule: {str(e)}")
            return self._create_empty_schedule_df()
    
    def _get_team_id_from_name(self, team_name: str) -> str:
        """Convert CFBD team name to standardized team ID"""
        if not team_name:
            return ''
        
        # First check if it's already a known abbreviation or name
        if team_name in self._team_id_map:
            return self._team_id_map[team_name].get('abbreviation', team_name)
        
        # Try to find by case-insensitive match
        for key, team_info in self._team_id_map.items():
            if (team_info.get('full_name', '').lower() == team_name.lower() or
                team_info.get('abbreviation', '').lower() == team_name.lower()):
                return team_info.get('abbreviation', team_name)
        
        # Create a fallback abbreviation
        return team_name.upper().replace(' ', '')[:4]
    
    def _parse_schedule_game(self, game: Dict, season: int) -> Dict:
        """Parse a single game from NCAA schedule data"""
        try:
            # Determine game status
            status = 'scheduled'
            if game.get('completed', False):
                status = 'final'
            elif game.get('startDate'):
                try:
                    start_date_str = game.get('startDate')
                    if start_date_str:
                        game_datetime = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                        if game_datetime < datetime.now():
                            status = 'final'
                except:
                    pass
            
            # Get team information - normalize team names to consistent IDs
            home_team_name = str(game.get('homeTeam', ''))
            away_team_name = str(game.get('awayTeam', ''))
            
            # Convert to standardized team IDs
            home_team_id = self._get_team_id_from_name(home_team_name)
            away_team_id = self._get_team_id_from_name(away_team_name)
            
            # Get display names (prefer full names from our mapping)
            home_display_name = self._team_id_map.get(home_team_id, {}).get('full_name', home_team_name)
            away_display_name = self._team_id_map.get(away_team_id, {}).get('full_name', away_team_name)
            
            # Parse game date
            game_date = None
            start_date_str = game.get('startDate')
            if start_date_str:
                try:
                    # Handle different date formats from CFBD API
                    if 'T' in start_date_str:
                        if start_date_str.endswith('Z'):
                            game_datetime = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                        else:
                            game_datetime = datetime.fromisoformat(start_date_str)
                    else:
                        # Simple date format
                        game_datetime = datetime.strptime(start_date_str, '%Y-%m-%d')
                    game_date = game_datetime.date()
                except Exception as date_e:
                    self.logger.warning(f"Could not parse date {start_date_str}: {date_e}")
                    pass
            
            return {
                'sport': self.sport,
                'league': self.league,
                'game_id': str(game.get('id', '')),
                'game_date': game_date,
                'home_team_id': home_team_id,
                'home_team_name': home_display_name,
                'away_team_id': away_team_id,
                'away_team_name': away_display_name,
                'season': season,
                'status': status,
                'source_keys': json.dumps({
                    'cfbd_game_id': game.get('id'),
                    'week': game.get('week'),
                    'season_type': game.get('seasonType', 'regular'),
                    'conference_game': game.get('conferenceGame', False),
                    'neutral_site': game.get('neutralSite', False),
                    'original_home_team': home_team_name,
                    'original_away_team': away_team_name
                })
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing schedule game: {str(e)}")
            return {}
    
    def _get_cached_season_games(self, season: int) -> Optional[List]:
        """Get season games from cache or API with caching"""
        current_time = datetime.now()
        cache_key = f"season_{season}"
        
        # Check if we have cached data and it's not too old (cache for 1 hour)
        if (cache_key in self._season_games_cache and 
            cache_key in self._cache_timestamps and
            (current_time - self._cache_timestamps[cache_key]).seconds < 3600):
            return self._season_games_cache[cache_key]
        
        # Fetch fresh data from API
        season_games = self._make_api_request('/games', {'year': season})
        
        if season_games:
            self._season_games_cache[cache_key] = season_games
            self._cache_timestamps[cache_key] = current_time
            self.logger.info(f"Cached {len(season_games)} games for season {season}")
        
        return season_games
    
    def get_todays_games(self, game_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get today's NCAA football games (or for a specific date).
        
        Args:
            game_date: Optional date to get games for. If None, uses current UTC date - 5 hours (US time)
        
        Returns:
            DataFrame with NCAA football games
        """
        if game_date is None:
            # Use US timezone (UTC - 5 hours) to match game scheduling
            us_now = datetime.now() - timedelta(hours=5)
            game_date = us_now.date()
        elif isinstance(game_date, str):
            game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
        
        return self.get_games(game_date)
    
    def get_games(self, game_date: date) -> pd.DataFrame:
        """
        Get NCAA football games for a specific date with scores.
        
        Args:
            game_date: Date to get games for
            
        Returns:
            DataFrame with games including scores
        """
        try:
            self.logger.info(f"Getting NCAA football games for {game_date}")
            
            # Get games from schedule first
            games_df = self.get_schedule(game_date, game_date)
            
            if games_df.empty:
                return self._create_empty_games_df()
            
            # Get the season for this date to cache season data
            game_season = self._date_to_ncaa_season(game_date)
            season_games = self._get_cached_season_games(game_season)
            
            # Enhance with scores for completed games
            enhanced_games = []
            
            for _, game in games_df.iterrows():
                game_dict = game.to_dict()
                
                # Add score columns with initial None values
                game_dict['home_score'] = None
                game_dict['away_score'] = None
                
                # If game is final or past date, try to get scores from cached season data
                if game['status'] == 'final' or game['game_date'] < date.today():
                    try:
                        if season_games:
                            # Find this specific game in cached season data
                            for api_game in season_games:
                                if str(api_game.get('id', '')) == game['game_id']:
                                    # Extract scores if available
                                    if api_game.get('homePoints') is not None:
                                        game_dict['home_score'] = int(api_game['homePoints'])
                                    if api_game.get('awayPoints') is not None:
                                        game_dict['away_score'] = int(api_game['awayPoints'])
                                    
                                    # Update status if completed
                                    if api_game.get('completed', False):
                                        game_dict['status'] = 'final'
                                    
                                    break
                    
                    except Exception as e:
                        self.logger.warning(f"Could not get scores for game {game['game_id']}: {str(e)}")
                
                enhanced_games.append(game_dict)
            
            result_df = pd.DataFrame(enhanced_games)
            self.logger.info(f"Retrieved {len(result_df)} NCAA football games with scores")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA football games for {game_date}: {str(e)}")
            return self._create_empty_games_df()
    
    def get_team_stats(self, season: int, rolling_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get NCAA football team statistics for a season.
        
        Args:
            season: Season year (e.g., 2024 for 2024 season)
            rolling_days: If provided, get rolling stats for last N days
            
        Returns:
            DataFrame with team statistics
        """
        try:
            self.logger.info(f"Getting NCAA football team stats for season {season}")
            
            if rolling_days:
                return self._get_rolling_team_stats(season, rolling_days)
            else:
                return self._get_season_team_stats(season)
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA football team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def _get_season_team_stats(self, season: int) -> pd.DataFrame:
        """Get full season team statistics"""
        try:
            # Get team season stats from API
            team_stats = self._make_api_request('/stats/season', {'year': season})
            
            if not team_stats:
                self.logger.warning(f"No team stats available for season {season}")
                return self._create_empty_stats_df()
            
            stats_list = []
            
            # Process each team's stats
            for team_stat in team_stats:
                try:
                    team_name = team_stat.get('team', '')
                    team_abbrev = None
                    
                    # Find team abbreviation
                    for abbrev, team_info in self._team_id_map.items():
                        if team_info['full_name'].lower() == team_name.lower():
                            team_abbrev = abbrev
                            break
                    
                    if not team_abbrev:
                        team_abbrev = team_name.upper()[:4]  # Fallback
                    
                    # Get conference for this team
                    conference = team_stat.get('conference', '')
                    
                    # Extract offensive stats
                    games_played = team_stat.get('games', 0)
                    total_yards = team_stat.get('totalYards', 0)
                    passing_yards = team_stat.get('passingYards', 0)
                    rushing_yards = team_stat.get('rushingYards', 0)
                    
                    # Calculate per-game averages
                    total_yards_per_game = total_yards / max(games_played, 1)
                    passing_yards_per_game = passing_yards / max(games_played, 1)
                    rushing_yards_per_game = rushing_yards / max(games_played, 1)
                    
                    # Get wins and losses (if available)
                    wins = team_stat.get('wins', 0)
                    losses = team_stat.get('losses', 0)
                    
                    # Calculate additional stats
                    points_per_game = team_stat.get('pointsPerGame', 0.0)
                    turnovers = team_stat.get('turnovers', 0)
                    
                    stats_list.append({
                        'sport': self.sport,
                        'league': self.league,
                        'team_id': team_abbrev,
                        'team_name': team_name,
                        'season': season,
                        'date': None,  # Full season stats
                        'games_played': games_played,
                        'wins': wins,
                        'losses': losses,
                        # NCAA-specific stats
                        'total_yards_per_game': float(total_yards_per_game),
                        'passing_yards_per_game': float(passing_yards_per_game),
                        'rushing_yards_per_game': float(rushing_yards_per_game),
                        'points_per_game': float(points_per_game),
                        'turnovers': turnovers,
                        'conference': conference,
                        'total_yards': total_yards,
                        'passing_yards': passing_yards,
                        'rushing_yards': rushing_yards
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing stats for team {team_stat.get('team', 'unknown')}: {str(e)}")
                    continue
            
            return pd.DataFrame(stats_list)
            
        except Exception as e:
            self.logger.error(f"Error getting season team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def _get_rolling_team_stats(self, season: int, rolling_days: int) -> pd.DataFrame:
        """Get rolling team statistics for recent games"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=rolling_days)
            
            # Get recent games
            games_df = self.get_schedule(start_date, end_date)
            
            if games_df.empty:
                return self._create_empty_stats_df()
            
            # Calculate rolling stats from recent games
            stats_list = []
            
            for team_abbrev in self._team_id_map.keys():
                team_games = games_df[
                    (games_df['home_team_id'] == team_abbrev) | 
                    (games_df['away_team_id'] == team_abbrev)
                ]
                
                if len(team_games) > 0:
                    team_name = self._team_id_map[team_abbrev]['full_name']
                    
                    # Calculate basic rolling stats from completed games
                    wins = 0
                    losses = 0
                    total_points = 0
                    total_points_allowed = 0
                    completed_games = 0
                    
                    for _, game in team_games.iterrows():
                        if game['status'] == 'final':
                            is_home = game['home_team_id'] == team_abbrev
                            
                            # Get actual game scores
                            game_date_value = game['game_date']
                            if isinstance(game_date_value, date):
                                enhanced_game_df = self.get_games(game_date_value)
                                game_with_scores = enhanced_game_df[enhanced_game_df['game_id'] == game['game_id']]
                            else:
                                continue
                            
                            if not game_with_scores.empty:
                                game_row = game_with_scores.iloc[0]
                                
                                if game_row.get('home_score') is not None and game_row.get('away_score') is not None:
                                    team_score = game_row['home_score'] if is_home else game_row['away_score']
                                    opp_score = game_row['away_score'] if is_home else game_row['home_score']
                                    
                                    if team_score > opp_score:
                                        wins += 1
                                    else:
                                        losses += 1
                                    
                                    total_points += team_score
                                    total_points_allowed += opp_score
                                    completed_games += 1
                    
                    # Calculate averages
                    avg_points_scored = total_points / max(completed_games, 1)
                    avg_points_allowed = total_points_allowed / max(completed_games, 1)
                    
                    # Estimate yards from points (rough approximation)
                    # College football averages roughly 400-500 yards per game
                    estimated_total_yards = avg_points_scored * 12  # ~12 yards per point
                    estimated_passing_yards = estimated_total_yards * 0.6  # 60% passing
                    estimated_rushing_yards = estimated_total_yards * 0.4  # 40% rushing
                    
                    stats_list.append({
                        'sport': self.sport,
                        'league': self.league,
                        'team_id': team_abbrev,
                        'team_name': team_name,
                        'season': season,
                        'date': end_date,
                        'games_played': completed_games,
                        'wins': wins,
                        'losses': losses,
                        # Estimated NCAA stats based on scoring
                        'total_yards_per_game': float(estimated_total_yards),
                        'passing_yards_per_game': float(estimated_passing_yards),
                        'rushing_yards_per_game': float(estimated_rushing_yards),
                        'points_per_game': float(avg_points_scored),
                        'turnovers': 0,  # Would need detailed game data
                        'conference': self._team_id_map[team_abbrev]['conference'],
                        'total_yards': int(estimated_total_yards * completed_games),
                        'passing_yards': int(estimated_passing_yards * completed_games),
                        'rushing_yards': int(estimated_rushing_yards * completed_games)
                    })
            
            return pd.DataFrame(stats_list)
            
        except Exception as e:
            self.logger.error(f"Error getting rolling team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def get_recent_form(self, team_id: str, games: int = 10) -> Dict:
        """
        Get recent form/performance for an NCAA football team.
        
        Args:
            team_id: NCAA team identifier
            games: Number of recent games to analyze
            
        Returns:
            Dictionary with recent performance metrics
        """
        try:
            self.logger.info(f"Getting recent form for NCAA football team {team_id}")
            
            # Get recent games from the last 90 days (college football season is shorter)
            end_date = date.today()
            start_date = end_date - timedelta(days=90)
            
            games_df = self.get_schedule(start_date, end_date)
            
            # Filter games for this team
            team_games = games_df[
                (games_df['home_team_id'] == team_id) | 
                (games_df['away_team_id'] == team_id)
            ].head(games)  # Get most recent games
            
            if team_games.empty:
                return {
                    'sport': self.sport, 
                    'league': self.league, 
                    'team_id': team_id, 
                    'error': 'No recent games found'
                }
            
            # Calculate form metrics
            wins = 0
            losses = 0
            total_points = 0
            total_points_allowed = 0
            completed_games = 0
            
            for _, game in team_games.iterrows():
                if game['status'] == 'final':
                    is_home = game['home_team_id'] == team_id
                    
                    # Get actual game scores
                    game_date_value = game['game_date']
                    if isinstance(game_date_value, date):
                        enhanced_game_df = self.get_games(game_date_value)
                        game_with_scores = enhanced_game_df[enhanced_game_df['game_id'] == game['game_id']]
                    else:
                        continue
                    
                    if not game_with_scores.empty:
                        game_row = game_with_scores.iloc[0]
                        
                        if game_row.get('home_score') is not None and game_row.get('away_score') is not None:
                            team_score = game_row['home_score'] if is_home else game_row['away_score']
                            opp_score = game_row['away_score'] if is_home else game_row['home_score']
                            
                            if team_score > opp_score:
                                wins += 1
                            else:
                                losses += 1
                            
                            total_points += team_score
                            total_points_allowed += opp_score
                            completed_games += 1
            
            form_dict = {
                'sport': self.sport,
                'league': self.league,
                'team_id': team_id,
                'games_analyzed': completed_games,
                'wins': wins,
                'losses': losses,
                'win_percentage': wins / max(completed_games, 1),
                'avg_points_scored': total_points / max(completed_games, 1),
                'avg_points_allowed': total_points_allowed / max(completed_games, 1),
                'point_differential': (total_points - total_points_allowed) / max(completed_games, 1),
                'last_5_record': f"{wins}-{losses}" if completed_games <= 5 else f"{wins}-{losses}",
                'form_trend': 'improving' if wins > losses else 'declining' if losses > wins else 'stable'
            }
            
            return form_dict
            
        except Exception as e:
            self.logger.error(f"Error getting recent form for team {team_id}: {str(e)}")
            return {'sport': self.sport, 'league': self.league, 'team_id': team_id, 'error': str(e)}
    
    def get_head_to_head(self, team1_id: str, team2_id: str, seasons: int = 5) -> pd.DataFrame:
        """
        Get head-to-head matchup history between two NCAA football teams.
        
        Args:
            team1_id: First team identifier
            team2_id: Second team identifier
            seasons: Number of seasons to look back
            
        Returns:
            DataFrame with historical matchup data
        """
        try:
            self.logger.info(f"Getting NCAA football head-to-head: {team1_id} vs {team2_id}")
            
            matchups = []
            
            # Get seasons to analyze
            seasons_to_check = [self._current_season - i for i in range(seasons)]
            
            for season in seasons_to_check:
                try:
                    # Get games for this season
                    season_games = self._make_api_request('/games', {'year': season})
                    
                    if season_games:
                        # Filter for matchups between the two teams
                        for game in season_games:
                            home_team = game.get('home_team', '')
                            away_team = game.get('away_team', '')
                            
                            # Check if this is a matchup between our two teams
                            if ((home_team == team1_id and away_team == team2_id) or
                                (home_team == team2_id and away_team == team1_id)):
                                
                                if game.get('completed', False) and game.get('home_points') is not None:
                                    # Parse game data
                                    game_date = None
                                    if game.get('start_date'):
                                        try:
                                            game_datetime = datetime.fromisoformat(game.get('start_date').replace('Z', '+00:00'))
                                            game_date = game_datetime.date()
                                        except:
                                            pass
                                    
                                    # Determine which team is which
                                    if home_team == team1_id:
                                        team1_score = int(game['home_points'])
                                        team2_score = int(game['away_points'])
                                        location = 'home'
                                    else:
                                        team1_score = int(game['away_points'])
                                        team2_score = int(game['home_points'])
                                        location = 'away'
                                    
                                    winner_id = team1_id if team1_score > team2_score else team2_id
                                    
                                    matchups.append({
                                        'sport': self.sport,
                                        'league': self.league,
                                        'game_id': str(game['id']),
                                        'game_date': game_date,
                                        'season': season,
                                        'team1_id': team1_id,
                                        'team1_score': team1_score,
                                        'team2_id': team2_id,
                                        'team2_score': team2_score,
                                        'winner_id': winner_id,
                                        'location': location
                                    })
                
                except Exception as e:
                    self.logger.warning(f"Error getting season {season} matchups: {str(e)}")
                    continue
            
            if not matchups:
                self.logger.warning(f"No head-to-head data found for teams {team1_id} vs {team2_id}")
                return pd.DataFrame()
            
            return pd.DataFrame(matchups)
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA football head-to-head data: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that collected NCAA football data meets requirements.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns based on data type
        required_base_columns = ['sport', 'league']
        missing_base = [col for col in required_base_columns if col not in df.columns]
        
        if missing_base:
            issues.append(f"Missing required base columns: {missing_base}")
        
        # Check sport/league values
        if 'sport' in df.columns and not all(df['sport'] == 'NCAAF'):
            issues.append("Sport column should contain only 'NCAAF' values")
        
        if 'league' in df.columns and not all(df['league'] == 'NCAA'):
            issues.append("League column should contain only 'NCAA' values")
        
        # Check for schedule/games data
        if 'game_id' in df.columns:
            schedule_columns = ['game_date', 'home_team_id', 'away_team_id', 'season']
            missing_schedule = [col for col in schedule_columns if col not in df.columns]
            
            if missing_schedule:
                issues.append(f"Missing schedule columns: {missing_schedule}")
            
            # Check for duplicate game IDs
            if 'game_id' in df.columns and df['game_id'].duplicated().any():
                issues.append("Duplicate game IDs found")
        
        # Check for team stats data
        if 'team_id' in df.columns:
            stats_columns = ['team_name', 'games_played', 'wins', 'losses']
            missing_stats = [col for col in stats_columns if col not in df.columns]
            
            if missing_stats:
                issues.append(f"Missing stats columns: {missing_stats}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_supported_stats(self) -> List[str]:
        """Get list of NCAA football statistics this collector supports"""
        return [
            'total_yards_per_game', 'passing_yards_per_game', 'rushing_yards_per_game',
            'points_per_game', 'turnovers', 'total_yards', 'passing_yards', 'rushing_yards',
            'wins', 'losses', 'win_percentage', 'conference'
        ]
    
    def get_data_source(self) -> str:
        """Get the name of the data source this collector uses"""
        return "College Football Data API (collegefootballdata.com)"
    
    def _create_empty_schedule_df(self) -> pd.DataFrame:
        """Create empty DataFrame with schedule columns"""
        return pd.DataFrame({
            'sport': pd.Series([], dtype='object'),
            'league': pd.Series([], dtype='object'),
            'game_id': pd.Series([], dtype='object'),
            'game_date': pd.Series([], dtype='object'),
            'home_team_id': pd.Series([], dtype='object'),
            'home_team_name': pd.Series([], dtype='object'),
            'away_team_id': pd.Series([], dtype='object'),
            'away_team_name': pd.Series([], dtype='object'),
            'season': pd.Series([], dtype='int64'),
            'status': pd.Series([], dtype='object'),
            'source_keys': pd.Series([], dtype='object')
        })
    
    def _create_empty_games_df(self) -> pd.DataFrame:
        """Create empty DataFrame with games columns"""
        return pd.DataFrame({
            'sport': pd.Series([], dtype='object'),
            'league': pd.Series([], dtype='object'),
            'game_id': pd.Series([], dtype='object'),
            'game_date': pd.Series([], dtype='object'),
            'home_team_id': pd.Series([], dtype='object'),
            'home_team_name': pd.Series([], dtype='object'),
            'away_team_id': pd.Series([], dtype='object'),
            'away_team_name': pd.Series([], dtype='object'),
            'season': pd.Series([], dtype='int64'),
            'status': pd.Series([], dtype='object'),
            'home_score': pd.Series([], dtype='float64'),
            'away_score': pd.Series([], dtype='float64'),
            'source_keys': pd.Series([], dtype='object')
        })
    
    def _create_empty_stats_df(self) -> pd.DataFrame:
        """Create empty DataFrame with stats columns"""
        return pd.DataFrame({
            'sport': pd.Series([], dtype='object'),
            'league': pd.Series([], dtype='object'),
            'team_id': pd.Series([], dtype='object'),
            'team_name': pd.Series([], dtype='object'),
            'season': pd.Series([], dtype='int64'),
            'date': pd.Series([], dtype='object'),
            'games_played': pd.Series([], dtype='int64'),
            'wins': pd.Series([], dtype='int64'),
            'losses': pd.Series([], dtype='int64'),
            'total_yards_per_game': pd.Series([], dtype='float64'),
            'passing_yards_per_game': pd.Series([], dtype='float64'),
            'rushing_yards_per_game': pd.Series([], dtype='float64'),
            'points_per_game': pd.Series([], dtype='float64'),
            'turnovers': pd.Series([], dtype='int64'),
            'conference': pd.Series([], dtype='object'),
            'total_yards': pd.Series([], dtype='int64'),
            'passing_yards': pd.Series([], dtype='int64'),
            'rushing_yards': pd.Series([], dtype='int64')
        })