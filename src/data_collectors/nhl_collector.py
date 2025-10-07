import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import requests

# Try to import nhl_api, fallback to None if not available
try:
    from nhl_api import NhlApi
except ImportError:
    try:
        from nhl_api_py import NhlApi
    except ImportError:
        NhlApi = None

from src.interfaces.base_collector import BaseDataCollector


class NHLDataCollector(BaseDataCollector):
    """
    NHL data collector using nhl-api-py library and NHL official APIs.
    
    Provides access to NHL game schedules, scores, team statistics,
    and historical data through the official NHL.com APIs.
    """
    
    def __init__(self):
        super().__init__(sport='NHL', league='NHL')
        self.logger = logging.getLogger(__name__)
        
        # Initialize NHL API client (if available)
        self.nhl_api = NhlApi() if NhlApi is not None else None
        
        # Base URLs for direct API calls
        self.api_web_base = "https://api-web.nhle.com/v1/"
        self.api_stats_base = "https://api.nhle.com/stats/rest/"
        
        # Cache for NHL teams
        self._teams_cache = None
        self._team_id_map = {}
        
        # Initialize team data
        self._initialize_teams()
    
    def _initialize_teams(self):
        """Initialize NHL teams data and create ID mappings"""
        try:
            # Get teams using the API wrapper (if available)
            teams_data = None
            if self.nhl_api is not None:
                teams_data = self.nhl_api.stats.team_stats_leaders(season="20242025", game_type=2)
            else:
                self.logger.info("NHL API wrapper not available, using fallback team initialization")
            
            if teams_data and 'data' in teams_data:
                self._teams_cache = teams_data['data']
                
                # Create mappings for easier lookups
                for team in self._teams_cache:
                    team_id = str(team.get('teamId', ''))
                    team_abbrev = team.get('teamAbbrevs', {}).get('default', '')
                    team_name = team.get('teamFullName', '')
                    
                    if team_id:
                        self._team_id_map[team_id] = {
                            'abbreviation': team_abbrev,
                            'full_name': team_name,
                            'logo': team.get('teamLogo', '')
                        }
                
                self.logger.info(f"Initialized {len(self._teams_cache)} NHL teams")
            else:
                # Fallback: create basic team mapping
                self._initialize_basic_teams()
                
        except Exception as e:
            self.logger.warning(f"Error initializing NHL teams via API: {str(e)}")
            self._initialize_basic_teams()
    
    def _initialize_basic_teams(self):
        """Initialize basic NHL team mappings as fallback"""
        # Basic NHL team data (32 teams for 2024-25 season)
        basic_teams = [
            {'id': 1, 'abbrev': 'NJD', 'name': 'New Jersey Devils'},
            {'id': 2, 'abbrev': 'NYI', 'name': 'New York Islanders'},
            {'id': 3, 'abbrev': 'NYR', 'name': 'New York Rangers'},
            {'id': 4, 'abbrev': 'PHI', 'name': 'Philadelphia Flyers'},
            {'id': 5, 'abbrev': 'PIT', 'name': 'Pittsburgh Penguins'},
            {'id': 6, 'abbrev': 'BOS', 'name': 'Boston Bruins'},
            {'id': 7, 'abbrev': 'BUF', 'name': 'Buffalo Sabres'},
            {'id': 8, 'abbrev': 'MTL', 'name': 'Montreal Canadiens'},
            {'id': 9, 'abbrev': 'OTT', 'name': 'Ottawa Senators'},
            {'id': 10, 'abbrev': 'TOR', 'name': 'Toronto Maple Leafs'},
            {'id': 12, 'abbrev': 'CAR', 'name': 'Carolina Hurricanes'},
            {'id': 13, 'abbrev': 'FLA', 'name': 'Florida Panthers'},
            {'id': 14, 'abbrev': 'TBL', 'name': 'Tampa Bay Lightning'},
            {'id': 15, 'abbrev': 'WSH', 'name': 'Washington Capitals'},
            {'id': 16, 'abbrev': 'CHI', 'name': 'Chicago Blackhawks'},
            {'id': 17, 'abbrev': 'DET', 'name': 'Detroit Red Wings'},
            {'id': 18, 'abbrev': 'NSH', 'name': 'Nashville Predators'},
            {'id': 19, 'abbrev': 'STL', 'name': 'St. Louis Blues'},
            {'id': 20, 'abbrev': 'CGY', 'name': 'Calgary Flames'},
            {'id': 21, 'abbrev': 'COL', 'name': 'Colorado Avalanche'},
            {'id': 22, 'abbrev': 'EDM', 'name': 'Edmonton Oilers'},
            {'id': 23, 'abbrev': 'VAN', 'name': 'Vancouver Canucks'},
            {'id': 24, 'abbrev': 'ANA', 'name': 'Anaheim Ducks'},
            {'id': 25, 'abbrev': 'DAL', 'name': 'Dallas Stars'},
            {'id': 26, 'abbrev': 'LAK', 'name': 'Los Angeles Kings'},
            {'id': 27, 'abbrev': 'SJS', 'name': 'San Jose Sharks'},
            {'id': 28, 'abbrev': 'CBJ', 'name': 'Columbus Blue Jackets'},
            {'id': 29, 'abbrev': 'MIN', 'name': 'Minnesota Wild'},
            {'id': 30, 'abbrev': 'WPG', 'name': 'Winnipeg Jets'},
            {'id': 54, 'abbrev': 'VGK', 'name': 'Vegas Golden Knights'},
            {'id': 55, 'abbrev': 'SEA', 'name': 'Seattle Kraken'},
            {'id': 59, 'abbrev': 'UTA', 'name': 'Utah Hockey Club'}
        ]
        
        self._teams_cache = basic_teams
        for team in basic_teams:
            team_id = str(team['id'])
            self._team_id_map[team_id] = {
                'abbreviation': team['abbrev'],
                'full_name': team['name'],
                'logo': ''
            }
        
        self.logger.info(f"Initialized {len(basic_teams)} NHL teams (basic fallback)")
    
    def _normalize_season(self, season_input) -> int:
        """
        Normalize NHL season format to proper season year.
        
        NHL seasons are typically formatted as 20242025 for 2024-25 season.
        
        Args:
            season_input: Season identifier (could be string or int)
            
        Returns:
            Normalized season year as integer (start year of season)
        """
        try:
            season_str = str(season_input)
            
            # Handle NHL's 8-digit season format like 20242025 -> 2024
            if len(season_str) == 8 and season_str.startswith('20'):
                return int(season_str[:4])
            
            # Handle regular 4-digit years
            if len(season_str) == 4:
                return int(season_str)
            
            # Fallback to current year
            current_year = datetime.now().year
            # NHL season runs Oct-June, so adjust for season start
            if datetime.now().month >= 10:
                return current_year
            else:
                return current_year - 1
            
        except (ValueError, TypeError):
            return datetime.now().year
    
    def _get_nhl_season_string(self, year: int) -> str:
        """Convert year to NHL season string format (e.g., 2024 -> '20242025')"""
        return f"{year}{year + 1}"
    
    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get NHL game schedule for a date range.
        
        Args:
            start_date: Start date for schedule
            end_date: End date for schedule
            
        Returns:
            DataFrame with standardized schedule columns
        """
        try:
            self.logger.info(f"Getting NHL schedule from {start_date} to {end_date}")
            
            schedule_data = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    # Get daily schedule using NHL API
                    date_str = current_date.strftime('%Y-%m-%d')
                    
                    # Try using the API wrapper first (if available)
                    try:
                        daily_games = None
                        if self.nhl_api is not None:
                            daily_games = self.nhl_api.schedule.get_schedule(start_date=date_str, end_date=date_str)
                        else:
                            raise Exception("NHL API wrapper not available")
                        
                        if daily_games and 'gameWeek' in daily_games:
                            for week in daily_games['gameWeek']:
                                # Handle different API response structures
                                if isinstance(week.get('games'), list):
                                    # Direct games list
                                    for game in week.get('games', []):
                                        schedule_data.append(self._parse_schedule_game(game, current_date))
                                else:
                                    # Nested structure with date groups
                                    for game_date in week.get('games', []):
                                        if isinstance(game_date, dict) and 'games' in game_date:
                                            for game in game_date.get('games', []):
                                                schedule_data.append(self._parse_schedule_game(game, current_date))
                                        else:
                                            # Direct game object
                                            schedule_data.append(self._parse_schedule_game(game_date, current_date))
                    
                    except Exception as api_error:
                        self.logger.warning(f"API wrapper failed for {date_str}, trying direct API: {str(api_error)}")
                        
                        # Fallback to direct API call
                        url = f"{self.api_web_base}schedule/{date_str}"
                        response = requests.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            daily_data = response.json()
                            
                            if 'gameWeek' in daily_data:
                                for week in daily_data['gameWeek']:
                                    # Handle different API response structures
                                    if isinstance(week.get('games'), list):
                                        # Direct games list
                                        for game in week.get('games', []):
                                            schedule_data.append(self._parse_schedule_game(game, current_date))
                                    else:
                                        # Nested structure with date groups
                                        for day in week.get('games', []):
                                            if isinstance(day, dict) and 'games' in day:
                                                for game in day.get('games', []):
                                                    schedule_data.append(self._parse_schedule_game(game, current_date))
                                            else:
                                                # Direct game object
                                                schedule_data.append(self._parse_schedule_game(day, current_date))
                
                except Exception as e:
                    self.logger.warning(f"Error getting games for {current_date}: {str(e)}")
                
                current_date += timedelta(days=1)
            
            if not schedule_data:
                self.logger.warning(f"No NHL games found for date range {start_date} to {end_date}")
                return self._create_empty_schedule_df()
            
            result_df = pd.DataFrame(schedule_data)
            self.logger.info(f"Retrieved {len(result_df)} NHL games")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NHL schedule: {str(e)}")
            return self._create_empty_schedule_df()
    
    def _parse_schedule_game(self, game: Dict, game_date: date) -> Dict:
        """Parse a single game from NHL schedule API"""
        try:
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
            # Determine game status with improved mapping
            game_state = game.get('gameState', '').upper()
            status = 'scheduled'
            if game_state in ['FINAL', 'OFF', 'OVER']:
                status = 'final'
            elif game_state in ['LIVE', 'CRIT', 'PROG']:
                status = 'in_progress'
            elif game_state in ['PRE', 'PREVIEW']:
                status = 'scheduled'
            elif game_state in ['PPD', 'SUSP']:
                status = 'postponed'
            
            # Get season year from game date
            season_year = self._get_season_year_from_date(game_date)
            
            # Extract team names properly from NHL API structure
            home_place = home_team.get('placeName', {}).get('default', '')
            home_common = home_team.get('commonName', {}).get('default', '')
            away_place = away_team.get('placeName', {}).get('default', '')
            away_common = away_team.get('commonName', {}).get('default', '')
            
            # Build full team names (e.g., "Buffalo Sabres")
            home_team_name = f"{home_place} {home_common}".strip() if home_place and home_common else home_team.get('abbrev', '')
            away_team_name = f"{away_place} {away_common}".strip() if away_place and away_common else away_team.get('abbrev', '')

            # Extract scores if available (for final games)
            home_score = None
            away_score = None
            if status == 'final':
                home_score = home_team.get('score')
                away_score = away_team.get('score')

            return {
                'sport': self.sport,
                'league': self.league,
                'game_id': str(game.get('id', '')),
                'game_date': game_date,
                'home_team_id': str(home_team.get('id', '')),
                'home_team_name': home_team_name,
                'away_team_id': str(away_team.get('id', '')),
                'away_team_name': away_team_name,
                'season': season_year,
                'status': status,
                'home_score': home_score,
                'away_score': away_score,
                'source_keys': json.dumps({
                    'nhl_game_id': str(game.get('id', '')),
                    'game_state': game.get('gameState', ''),
                    'season_string': self._get_nhl_season_string(season_year)
                })
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing schedule game: {str(e)}")
            return {}
    
    def _get_season_year_from_date(self, game_date: date) -> int:
        """Determine NHL season year from game date"""
        year = game_date.year
        
        # NHL season starts in October and ends in June
        # October-December belongs to the season starting that year
        # January-June belongs to the season that started the previous year
        if game_date.month >= 10:  # Oct, Nov, Dec
            return year
        else:  # Jan-Sep (mainly Jan-June for regular season)
            return year - 1
    
    def _detect_overtime_loss(self, game_id: str, team_score: int, opp_score: int) -> bool:
        """
        Detect if a loss was in overtime/shootout (worth 1 point in NHL).
        
        Args:
            game_id: NHL game identifier
            team_score: Team's final score
            opp_score: Opponent's final score
            
        Returns:
            True if this was an overtime/shootout loss, False otherwise
        """
        # Only check for OT loss if team actually lost
        if team_score >= opp_score:
            return False
        
        try:
            # Try to get game details to determine if it went to overtime
            url = f"{self.api_web_base}gamecenter/{game_id}/boxscore"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                boxscore = response.json()
                
                # Check for overtime indicators in the game data
                game_outcome = boxscore.get('gameOutcome', {})
                last_period_type = game_outcome.get('lastPeriodType', '')
                
                # NHL period types: REG (regulation), OT (overtime), SO (shootout)
                if last_period_type in ['OT', 'SO']:
                    return True
                
                # Alternative check: look at scoring by period
                # If there are more than 3 periods of play, it went to OT
                summary = boxscore.get('summary', {})
                scoring = summary.get('scoring', [])
                
                if scoring:
                    # Check if any goals were scored in period 4+ (overtime periods)
                    for period_scoring in scoring:
                        period_descriptor = period_scoring.get('periodDescriptor', {})
                        period_type = period_descriptor.get('periodType', '')
                        
                        if period_type in ['OT', 'SO']:
                            return True
                
                # Additional check: period summary information
                periods = boxscore.get('periodByPeriod', [])
                for period in periods:
                    period_type = period.get('periodType', '')
                    if period_type in ['OT', 'SO']:
                        return True
                        
        except Exception as e:
            self.logger.warning(f"Could not determine OT status for game {game_id}: {str(e)}")
        
        # Default: assume regulation loss if we can't determine otherwise
        return False
    
    def get_todays_games(self, game_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get today's NHL games (or for a specific date).
        
        Args:
            game_date: Optional date to get games for. If None, uses current UTC date - 5 hours (US time)
        
        Returns:
            DataFrame with NHL games
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
        Get NHL games for a specific date with scores.
        
        Args:
            game_date: Date to get games for
            
        Returns:
            DataFrame with games including scores
        """
        try:
            self.logger.info(f"Getting NHL games for {game_date}")
            
            # Get games from schedule first
            games_df = self.get_schedule(game_date, game_date)
            
            if games_df.empty:
                return self._create_empty_games_df()
            
            # Enhance with scores for completed games
            enhanced_games = []
            
            for _, game in games_df.iterrows():
                game_dict = game.to_dict()
                
                # Preserve existing scores from schedule if available
                existing_home_score = game.get('home_score')
                existing_away_score = game.get('away_score')
                
                # If game is final, try to get more detailed scores from boxscore
                if game['status'] == 'final':
                    try:
                        game_id = game['game_id']
                        
                        # Try to get boxscore for enhanced/detailed scores
                        url = f"{self.api_web_base}gamecenter/{game_id}/boxscore"
                        response = requests.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            boxscore = response.json()
                            
                            if 'homeTeam' in boxscore and 'awayTeam' in boxscore:
                                # Only overwrite if boxscore has valid scores
                                boxscore_home = boxscore['homeTeam'].get('score')
                                boxscore_away = boxscore['awayTeam'].get('score')
                                
                                if boxscore_home is not None and boxscore_away is not None:
                                    game_dict['home_score'] = boxscore_home
                                    game_dict['away_score'] = boxscore_away
                                else:
                                    # Keep existing schedule scores
                                    game_dict['home_score'] = existing_home_score
                                    game_dict['away_score'] = existing_away_score
                            else:
                                # Keep existing schedule scores
                                game_dict['home_score'] = existing_home_score
                                game_dict['away_score'] = existing_away_score
                        else:
                            # Keep existing schedule scores if boxscore API fails
                            game_dict['home_score'] = existing_home_score
                            game_dict['away_score'] = existing_away_score
                    
                    except Exception as e:
                        self.logger.warning(f"Could not get boxscore for game {game_id}, preserving schedule scores: {str(e)}")
                        # Keep existing schedule scores
                        game_dict['home_score'] = existing_home_score
                        game_dict['away_score'] = existing_away_score
                else:
                    # For non-final games, preserve existing scores (usually None)
                    game_dict['home_score'] = existing_home_score
                    game_dict['away_score'] = existing_away_score
                
                enhanced_games.append(game_dict)
            
            result_df = pd.DataFrame(enhanced_games)
            self.logger.info(f"Retrieved {len(result_df)} NHL games with scores")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NHL games for {game_date}: {str(e)}")
            return self._create_empty_games_df()
    
    def get_team_stats(self, season: int, rolling_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get NHL team statistics for a season.
        
        Args:
            season: Season year (e.g., 2024 for 2024-25 season)
            rolling_days: If provided, get rolling stats for last N days
            
        Returns:
            DataFrame with team statistics
        """
        try:
            self.logger.info(f"Getting NHL team stats for season {season}")
            
            season_str = self._get_nhl_season_string(season)
            
            if rolling_days:
                return self._get_rolling_team_stats(season, rolling_days)
            else:
                return self._get_season_team_stats(season_str)
            
        except Exception as e:
            self.logger.error(f"Error getting NHL team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def _get_season_team_stats(self, season_str: str) -> pd.DataFrame:
        """Get full season team statistics"""
        try:
            stats_list = []
            
            # Get team stats using the stats API
            url = f"{self.api_stats_base}en/team/summary"
            params = {
                'sort': 'points',
                'cayenneExp': f'seasonId={season_str} and gameTypeId=2'  # Regular season
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    for team_data in data['data']:
                        team_id = str(team_data.get('teamId', ''))
                        team_name = team_data.get('teamFullName', '')
                        
                        stats_list.append({
                            'sport': self.sport,
                            'league': self.league,
                            'team_id': team_id,
                            'team_name': team_name,
                            'season': int(season_str[:4]),
                            'date': None,  # Full season stats
                            'games_played': int(team_data.get('gamesPlayed', 0)),
                            'wins': int(team_data.get('wins', 0)),
                            'losses': int(team_data.get('losses', 0)),
                            # NHL-specific stats
                            'goals_per_game': float(team_data.get('goalsForPerGame', 0)),
                            'goals_against_per_game': float(team_data.get('goalsAgainstPerGame', 0)),
                            'power_play_pct': float(team_data.get('powerPlayPct', 0)),
                            'penalty_kill_pct': float(team_data.get('penaltyKillPct', 0)),
                            'shots_per_game': float(team_data.get('shotsForPerGame', 0)),
                            'shots_against_per_game': float(team_data.get('shotsAgainstPerGame', 0)),
                            'save_pct': float(team_data.get('savePct', 0)),
                            'points': int(team_data.get('points', 0)),
                            'points_pct': float(team_data.get('pointPct', 0)),
                            'overtime_losses': int(team_data.get('otLosses', 0))
                        })
            
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
            
            for team_id in set(list(games_df['home_team_id']) + list(games_df['away_team_id'])):
                if not team_id:
                    continue
                
                team_games = games_df[
                    (games_df['home_team_id'] == team_id) | 
                    (games_df['away_team_id'] == team_id)
                ]
                
                if len(team_games) > 0:
                    # Get team name
                    team_name = ''
                    if team_id in self._team_id_map:
                        team_name = self._team_id_map[team_id]['full_name']
                    else:
                        sample_game = team_games.iloc[0]
                        team_name = sample_game['home_team_name'] if sample_game['home_team_id'] == team_id else sample_game['away_team_name']
                    
                    # Calculate rolling stats with overtime loss detection
                    wins = 0
                    losses = 0
                    overtime_losses = 0
                    
                    for _, game in team_games.iterrows():
                        if game['status'] == 'final':
                            is_home = game['home_team_id'] == team_id
                            
                            if game.get('home_score') is not None and game.get('away_score') is not None:
                                team_score = game['home_score'] if is_home else game['away_score']
                                opp_score = game['away_score'] if is_home else game['home_score']
                                
                                if team_score > opp_score:
                                    wins += 1
                                else:
                                    # Check if this was an overtime/shootout loss (worth 1 point)
                                    if self._detect_overtime_loss(str(game['game_id']), int(team_score), int(opp_score)):
                                        overtime_losses += 1
                                    else:
                                        losses += 1
                    
                    stats_list.append({
                        'sport': self.sport,
                        'league': self.league,
                        'team_id': team_id,
                        'team_name': team_name,
                        'season': season,
                        'date': end_date,
                        'games_played': wins + losses + overtime_losses,
                        'wins': wins,
                        'losses': losses,
                        # Placeholder NHL stats - would need detailed game data
                        'goals_per_game': 0.0,
                        'goals_against_per_game': 0.0,
                        'power_play_pct': 0.0,
                        'penalty_kill_pct': 0.0,
                        'shots_per_game': 0.0,
                        'shots_against_per_game': 0.0,
                        'save_pct': 0.0,
                        'points': wins * 2 + overtime_losses,  # NHL points: 2 for wins + 1 for OT/SO losses
                        'points_pct': (wins * 2 + overtime_losses) / ((wins + losses + overtime_losses) * 2) if (wins + losses + overtime_losses) > 0 else 0.0,
                        'overtime_losses': overtime_losses
                    })
            
            return pd.DataFrame(stats_list)
            
        except Exception as e:
            self.logger.error(f"Error getting rolling team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def get_recent_form(self, team_id: str, games: int = 10) -> Dict:
        """
        Get recent form/performance for an NHL team.
        
        Args:
            team_id: NHL team identifier
            games: Number of recent games to analyze
            
        Returns:
            Dictionary with recent performance metrics
        """
        try:
            self.logger.info(f"Getting recent form for NHL team {team_id}")
            
            # Get recent games from the last 30 days
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            games_df = self.get_schedule(start_date, end_date)
            
            # Filter games for this team
            team_games = games_df[
                (games_df['home_team_id'] == team_id) | 
                (games_df['away_team_id'] == team_id)
            ].sort_values('game_date', ascending=False).head(games)
            
            if team_games.empty:
                return {
                    'sport': self.sport, 
                    'league': self.league, 
                    'team_id': team_id, 
                    'error': 'No recent games found'
                }
            
            # Calculate form metrics with overtime loss detection
            wins = 0
            losses = 0
            overtime_losses = 0
            
            for _, game in team_games.iterrows():
                if game['status'] == 'final':
                    is_home = game['home_team_id'] == team_id
                    
                    if game.get('home_score') is not None and game.get('away_score') is not None:
                        team_score = game['home_score'] if is_home else game['away_score']
                        opp_score = game['away_score'] if is_home else game['home_score']
                        
                        if team_score > opp_score:
                            wins += 1
                        else:
                            # Check if this was an overtime/shootout loss (worth 1 point)
                            if self._detect_overtime_loss(str(game['game_id']), int(team_score), int(opp_score)):
                                overtime_losses += 1
                            else:
                                losses += 1
            
            total_games = wins + losses + overtime_losses
            form_dict = {
                'sport': self.sport,
                'league': self.league,
                'team_id': team_id,
                'games_analyzed': total_games,
                'wins': wins,
                'losses': losses,
                'overtime_losses': overtime_losses,  # Now properly detected from game data
                'points': wins * 2 + overtime_losses,  # NHL points: 2 for wins + 1 for OT/SO losses
                'win_percentage': wins / total_games if total_games > 0 else 0,
                'points_percentage': (wins * 2 + overtime_losses) / (total_games * 2) if total_games > 0 else 0,
                'last_5_record': f"{wins}-{losses}-{overtime_losses}" if total_games <= 5 else f"{wins}-{losses}-{overtime_losses}",
                'form_trend': 'improving' if (wins * 2 + overtime_losses) > (losses * 2) else 'declining' if (wins * 2 + overtime_losses) < (losses * 2) else 'stable'
            }
            
            return form_dict
            
        except Exception as e:
            self.logger.error(f"Error getting recent form for team {team_id}: {str(e)}")
            return {'sport': self.sport, 'league': self.league, 'team_id': team_id, 'error': str(e)}
    
    def get_head_to_head(self, team1_id: str, team2_id: str, seasons: int = 3) -> pd.DataFrame:
        """
        Get head-to-head matchup history between two NHL teams.
        
        Args:
            team1_id: First team identifier
            team2_id: Second team identifier
            seasons: Number of seasons to look back
            
        Returns:
            DataFrame with historical matchup data
        """
        try:
            self.logger.info(f"Getting NHL head-to-head: {team1_id} vs {team2_id}")
            
            matchups = []
            current_year = datetime.now().year
            
            # Get current season year (accounting for NHL season structure)
            if datetime.now().month >= 10:
                current_season_year = current_year
            else:
                current_season_year = current_year - 1
            
            for i in range(seasons):
                season_year = current_season_year - i
                
                try:
                    # Get games for the entire season and filter for matchups
                    season_start = date(season_year, 10, 1)
                    season_end = date(season_year + 1, 6, 30)
                    
                    # Get schedule for the season (in chunks to avoid timeout)
                    all_games = []
                    
                    # Split season into quarters to avoid large API calls
                    quarter_dates = [
                        (season_start, date(season_year, 12, 31)),
                        (date(season_year + 1, 1, 1), date(season_year + 1, 3, 31)),
                        (date(season_year + 1, 4, 1), season_end)
                    ]
                    
                    for start_dt, end_dt in quarter_dates:
                        try:
                            quarter_games = self.get_schedule(start_dt, end_dt)
                            if not quarter_games.empty:
                                all_games.append(quarter_games)
                        except Exception as e:
                            self.logger.warning(f"Error getting quarter schedule: {str(e)}")
                            continue
                    
                    if all_games:
                        season_games = pd.concat(all_games, ignore_index=True)
                        
                        # Filter for matchups between the two teams
                        team_matchups = season_games[
                            ((season_games['home_team_id'] == team1_id) & (season_games['away_team_id'] == team2_id)) |
                            ((season_games['home_team_id'] == team2_id) & (season_games['away_team_id'] == team1_id))
                        ]
                        
                        for _, game in team_matchups.iterrows():
                            if game['status'] == 'final' and game.get('home_score') is not None:
                                # Determine which team is which
                                if game['home_team_id'] == team1_id:
                                    team1_score = game['home_score']
                                    team2_score = game['away_score']
                                    location = 'home'
                                else:
                                    team1_score = game['away_score']
                                    team2_score = game['home_score']
                                    location = 'away'
                                
                                winner_id = team1_id if team1_score > team2_score else team2_id
                                
                                matchups.append({
                                    'sport': self.sport,
                                    'league': self.league,
                                    'game_id': str(game['game_id']),
                                    'game_date': game['game_date'],
                                    'season': season_year,
                                    'team1_id': team1_id,
                                    'team1_score': int(team1_score),
                                    'team2_id': team2_id,
                                    'team2_score': int(team2_score),
                                    'winner_id': winner_id,
                                    'location': location
                                })
                
                except Exception as e:
                    self.logger.warning(f"Error getting season {season_year} matchups: {str(e)}")
                    continue
            
            if not matchups:
                self.logger.warning(f"No head-to-head data found for teams {team1_id} vs {team2_id}")
                return pd.DataFrame()
            
            return pd.DataFrame(matchups)
            
        except Exception as e:
            self.logger.error(f"Error getting NHL head-to-head data: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that collected NHL data meets requirements.
        
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
        if 'sport' in df.columns and not all(df['sport'] == 'NHL'):
            issues.append("Sport column should contain only 'NHL' values")
        
        if 'league' in df.columns and not all(df['league'] == 'NHL'):
            issues.append("League column should contain only 'NHL' values")
        
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
        """Get list of NHL statistics this collector supports"""
        return [
            'goals_per_game', 'goals_against_per_game', 'power_play_pct', 
            'penalty_kill_pct', 'shots_per_game', 'shots_against_per_game',
            'save_pct', 'points', 'points_pct', 'overtime_losses',
            'wins', 'losses', 'win_percentage'
        ]
    
    def get_data_source(self) -> str:
        """Get the name of the data source this collector uses"""
        return "NHL.com (via nhl-api-py and official NHL APIs)"
    
    def _create_empty_schedule_df(self) -> pd.DataFrame:
        """Create empty DataFrame with schedule columns"""
        return pd.DataFrame(columns=[
            'sport', 'league', 'game_id', 'game_date', 
            'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
            'season', 'status', 'source_keys'
        ])
    
    def _create_empty_games_df(self) -> pd.DataFrame:
        """Create empty DataFrame with games columns"""
        return pd.DataFrame(columns=[
            'sport', 'league', 'game_id', 'game_date',
            'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
            'season', 'status', 'home_score', 'away_score', 'source_keys'
        ])
    
    def _create_empty_stats_df(self) -> pd.DataFrame:
        """Create empty DataFrame with stats columns"""
        return pd.DataFrame(columns=[
            'sport', 'league', 'team_id', 'team_name', 'season', 'date',
            'games_played', 'wins', 'losses', 'goals_per_game',
            'goals_against_per_game', 'power_play_pct', 'penalty_kill_pct',
            'shots_per_game', 'shots_against_per_game', 'save_pct',
            'points', 'points_pct', 'overtime_losses'
        ])