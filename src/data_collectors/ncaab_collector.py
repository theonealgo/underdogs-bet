import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import warnings

# Suppress pandas and urllib warnings from sportsipy
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sportsipy.ncaab.teams import Teams
    from sportsipy.ncaab.schedule import Schedule
    from sportsipy.ncaab.boxscore import Boxscore
    SPORTSIPY_AVAILABLE = True
except ImportError:
    # Create dummy classes for type safety when sportsipy is not available
    Teams = None  # type: ignore
    Schedule = None  # type: ignore
    Boxscore = None  # type: ignore
    SPORTSIPY_AVAILABLE = False

from interfaces.base_collector import BaseDataCollector


class NCAABDataCollector(BaseDataCollector):
    """
    NCAA Basketball data collector using Sportsipy library.
    
    Provides access to college basketball game schedules, scores, team statistics,
    and historical data through the sportsreference.com (sportsipy) API.
    """
    
    def __init__(self):
        super().__init__(sport='NCAAB', league='NCAA')
        self.logger = logging.getLogger(__name__)
        
        if not SPORTSIPY_AVAILABLE:
            self.logger.error("Sportsipy library not available. Please install: pip install sportsipy")
            raise ImportError("Sportsipy library is required for NCAA Basketball data collection")
        
        # Cache for teams and current season
        self._teams_cache = None
        self._team_id_map = {}
        self._current_season = self._get_current_ncaab_season()
        
        # Initialize team data
        self._initialize_teams()
    
    def _get_current_ncaab_season(self) -> int:
        """Determine current NCAA basketball season year"""
        now = datetime.now()
        year = now.year
        
        # NCAA basketball season starts in November and ends in April (March Madness)
        # November-April belongs to the ending year's season (e.g., 2024-25 season ends in 2025)
        # May-October belongs to the upcoming season 
        if now.month >= 11:  # Nov, Dec
            return year + 1  # 2024-25 season
        elif now.month <= 4:  # Jan, Feb, Mar, Apr
            return year      # 2024-25 season (ends in 2025)
        else:  # May-Oct (off-season)
            return year + 1  # Next season starts in November
    
    def _date_to_ncaab_season(self, target_date: date) -> int:
        """Convert a date to its corresponding NCAA basketball season year"""
        if target_date.month >= 11:  # November-December
            return target_date.year + 1
        elif target_date.month <= 4:  # January-April
            return target_date.year
        else:  # May-October (off-season)
            return target_date.year + 1
    
    def _initialize_teams(self):
        """Initialize NCAA basketball teams data"""
        try:
            if not SPORTSIPY_AVAILABLE or Teams is None:
                raise ImportError("Sportsipy not available")
            
            # Get current season teams from sportsipy
            self.logger.info(f"Loading NCAA basketball teams for season {self._current_season}")
            teams = Teams(year=self._current_season)
            
            teams_list = []
            for team in teams:
                try:
                    # Create team data dictionary
                    team_data = {
                        'abbreviation': team.abbreviation,
                        'name': team.name,
                        'conference': getattr(team, 'conference', ''),
                        'wins': getattr(team, 'wins', 0),
                        'losses': getattr(team, 'losses', 0),
                        'win_percentage': getattr(team, 'win_percentage', 0.0),
                        'srs': getattr(team, 'simple_rating_system', 0.0),
                        'sos': getattr(team, 'strength_of_schedule', 0.0)
                    }
                    teams_list.append(team_data)
                    
                    # Add to team mapping
                    team_abbrev = team.abbreviation
                    self._team_id_map[team_abbrev] = {
                        'abbreviation': team_abbrev,
                        'full_name': team.name,
                        'conference': getattr(team, 'conference', ''),
                        'season': self._current_season
                    }
                    
                    # Also map by name for flexibility
                    self._team_id_map[team.name] = self._team_id_map[team_abbrev]
                    
                except Exception as e:
                    self.logger.warning(f"Error processing team {getattr(team, 'name', 'unknown')}: {str(e)}")
                    continue
            
            self._teams_cache = teams_list
            unique_teams = len([k for k in self._team_id_map.keys() if isinstance(self._team_id_map[k].get('abbreviation'), str)])
            self.logger.info(f"Initialized {unique_teams} NCAA basketball teams")
            
        except Exception as e:
            self.logger.warning(f"Error initializing NCAA basketball teams: {str(e)}")
            self._initialize_basic_teams()
    
    def _initialize_basic_teams(self):
        """Initialize basic NCAA basketball team mappings as fallback"""
        # Major Division I teams (Power 6 conferences + strong mid-majors)
        basic_teams = [
            # ACC
            {'abbrev': 'DUKE', 'name': 'Duke', 'conference': 'ACC'},
            {'abbrev': 'UNC', 'name': 'North Carolina', 'conference': 'ACC'},
            {'abbrev': 'VT', 'name': 'Virginia Tech', 'conference': 'ACC'},
            {'abbrev': 'UVA', 'name': 'Virginia', 'conference': 'ACC'},
            {'abbrev': 'WAKE', 'name': 'Wake Forest', 'conference': 'ACC'},
            {'abbrev': 'NCST', 'name': 'NC State', 'conference': 'ACC'},
            {'abbrev': 'FSU', 'name': 'Florida State', 'conference': 'ACC'},
            {'abbrev': 'CLEM', 'name': 'Clemson', 'conference': 'ACC'},
            {'abbrev': 'GT', 'name': 'Georgia Tech', 'conference': 'ACC'},
            {'abbrev': 'LOU', 'name': 'Louisville', 'conference': 'ACC'},
            {'abbrev': 'PITT', 'name': 'Pittsburgh', 'conference': 'ACC'},
            {'abbrev': 'SYR', 'name': 'Syracuse', 'conference': 'ACC'},
            {'abbrev': 'BC', 'name': 'Boston College', 'conference': 'ACC'},
            {'abbrev': 'MIA', 'name': 'Miami', 'conference': 'ACC'},
            {'abbrev': 'ND', 'name': 'Notre Dame', 'conference': 'ACC'},
            
            # SEC
            {'abbrev': 'UK', 'name': 'Kentucky', 'conference': 'SEC'},
            {'abbrev': 'UF', 'name': 'Florida', 'conference': 'SEC'},
            {'abbrev': 'UGA', 'name': 'Georgia', 'conference': 'SEC'},
            {'abbrev': 'UT', 'name': 'Tennessee', 'conference': 'SEC'},
            {'abbrev': 'AUB', 'name': 'Auburn', 'conference': 'SEC'},
            {'abbrev': 'ALA', 'name': 'Alabama', 'conference': 'SEC'},
            {'abbrev': 'ARK', 'name': 'Arkansas', 'conference': 'SEC'},
            {'abbrev': 'LSU', 'name': 'LSU', 'conference': 'SEC'},
            {'abbrev': 'MISS', 'name': 'Ole Miss', 'conference': 'SEC'},
            {'abbrev': 'MSU', 'name': 'Mississippi State', 'conference': 'SEC'},
            {'abbrev': 'SC', 'name': 'South Carolina', 'conference': 'SEC'},
            {'abbrev': 'VAN', 'name': 'Vanderbilt', 'conference': 'SEC'},
            {'abbrev': 'MO', 'name': 'Missouri', 'conference': 'SEC'},
            {'abbrev': 'TAMU', 'name': 'Texas A&M', 'conference': 'SEC'},
            
            # Big Ten
            {'abbrev': 'MSU', 'name': 'Michigan State', 'conference': 'Big Ten'},
            {'abbrev': 'UM', 'name': 'Michigan', 'conference': 'Big Ten'},
            {'abbrev': 'OSU', 'name': 'Ohio State', 'conference': 'Big Ten'},
            {'abbrev': 'PU', 'name': 'Purdue', 'conference': 'Big Ten'},
            {'abbrev': 'IU', 'name': 'Indiana', 'conference': 'Big Ten'},
            {'abbrev': 'WIS', 'name': 'Wisconsin', 'conference': 'Big Ten'},
            {'abbrev': 'IOWA', 'name': 'Iowa', 'conference': 'Big Ten'},
            {'abbrev': 'ILL', 'name': 'Illinois', 'conference': 'Big Ten'},
            {'abbrev': 'NW', 'name': 'Northwestern', 'conference': 'Big Ten'},
            {'abbrev': 'MIN', 'name': 'Minnesota', 'conference': 'Big Ten'},
            {'abbrev': 'NEB', 'name': 'Nebraska', 'conference': 'Big Ten'},
            {'abbrev': 'PSU', 'name': 'Penn State', 'conference': 'Big Ten'},
            {'abbrev': 'RUT', 'name': 'Rutgers', 'conference': 'Big Ten'},
            {'abbrev': 'MD', 'name': 'Maryland', 'conference': 'Big Ten'},
            
            # Big 12
            {'abbrev': 'KU', 'name': 'Kansas', 'conference': 'Big 12'},
            {'abbrev': 'KSU', 'name': 'Kansas State', 'conference': 'Big 12'},
            {'abbrev': 'BAY', 'name': 'Baylor', 'conference': 'Big 12'},
            {'abbrev': 'TEX', 'name': 'Texas', 'conference': 'Big 12'},
            {'abbrev': 'TTU', 'name': 'Texas Tech', 'conference': 'Big 12'},
            {'abbrev': 'TCU', 'name': 'TCU', 'conference': 'Big 12'},
            {'abbrev': 'OU', 'name': 'Oklahoma', 'conference': 'Big 12'},
            {'abbrev': 'OSU', 'name': 'Oklahoma State', 'conference': 'Big 12'},
            {'abbrev': 'WVU', 'name': 'West Virginia', 'conference': 'Big 12'},
            {'abbrev': 'ISU', 'name': 'Iowa State', 'conference': 'Big 12'},
            
            # Big East
            {'abbrev': 'NOVA', 'name': 'Villanova', 'conference': 'Big East'},
            {'abbrev': 'UConn', 'name': 'UConn', 'conference': 'Big East'},
            {'abbrev': 'CREI', 'name': 'Creighton', 'conference': 'Big East'},
            {'abbrev': 'MARQ', 'name': 'Marquette', 'conference': 'Big East'},
            {'abbrev': 'PROV', 'name': 'Providence', 'conference': 'Big East'},
            {'abbrev': 'SHAL', 'name': 'Seton Hall', 'conference': 'Big East'},
            {'abbrev': 'STJN', 'name': 'St. Johns', 'conference': 'Big East'},
            {'abbrev': 'GTWN', 'name': 'Georgetown', 'conference': 'Big East'},
            {'abbrev': 'BUTL', 'name': 'Butler', 'conference': 'Big East'},
            {'abbrev': 'XAVI', 'name': 'Xavier', 'conference': 'Big East'},
            
            # Pac-12
            {'abbrev': 'UCLA', 'name': 'UCLA', 'conference': 'Pac-12'},
            {'abbrev': 'USC', 'name': 'USC', 'conference': 'Pac-12'},
            {'abbrev': 'ARIZ', 'name': 'Arizona', 'conference': 'Pac-12'},
            {'abbrev': 'ASU', 'name': 'Arizona State', 'conference': 'Pac-12'},
            {'abbrev': 'ORE', 'name': 'Oregon', 'conference': 'Pac-12'},
            {'abbrev': 'ORST', 'name': 'Oregon State', 'conference': 'Pac-12'},
            {'abbrev': 'WASH', 'name': 'Washington', 'conference': 'Pac-12'},
            {'abbrev': 'WSU', 'name': 'Washington State', 'conference': 'Pac-12'},
            {'abbrev': 'STAN', 'name': 'Stanford', 'conference': 'Pac-12'},
            {'abbrev': 'CAL', 'name': 'California', 'conference': 'Pac-12'},
            {'abbrev': 'COLO', 'name': 'Colorado', 'conference': 'Pac-12'},
            {'abbrev': 'UTAH', 'name': 'Utah', 'conference': 'Pac-12'}
        ]
        
        self._teams_cache = basic_teams
        for team in basic_teams:
            team_abbrev = team['abbrev']
            self._team_id_map[team_abbrev] = {
                'abbreviation': team_abbrev,
                'full_name': team['name'],
                'conference': team['conference'],
                'season': self._current_season
            }
            # Also map by name
            self._team_id_map[team['name']] = self._team_id_map[team_abbrev]
        
        self.logger.info(f"Initialized {len(basic_teams)} NCAA basketball teams (basic fallback)")
    
    def _get_team_id_from_name(self, name: str) -> str:
        """Convert team name to standardized team ID"""
        if not name:
            return ''
        
        # Try exact match first
        if name in self._team_id_map:
            return self._team_id_map[name].get('abbreviation', name)
        
        # Try case-insensitive match
        for team_key, team_info in self._team_id_map.items():
            if isinstance(team_key, str) and team_key.lower() == name.lower():
                return team_info.get('abbreviation', name)
            
            # Check full name match
            full_name = team_info.get('full_name', '')
            if full_name.lower() == name.lower():
                return team_info.get('abbreviation', name)
        
        # Fallback: return uppercased abbreviated form
        return name.upper()[:4] if len(name) > 4 else name.upper()
    
    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get NCAA basketball game schedule for a date range.
        
        Args:
            start_date: Start date for schedule
            end_date: End date for schedule
            
        Returns:
            DataFrame with standardized schedule columns
        """
        try:
            self.logger.info(f"Getting NCAA basketball schedule from {start_date} to {end_date}")
            
            # Determine which NCAA basketball seasons to query
            seasons = set()
            current_date = start_date
            while current_date <= end_date:
                seasons.add(self._date_to_ncaab_season(current_date))
                current_date += timedelta(days=30)  # Sample dates across the range
            
            seasons = sorted(list(seasons))
            schedule_data = []
            
            for season in seasons:
                try:
                    if not SPORTSIPY_AVAILABLE or Teams is None:
                        self.logger.warning(f"Sportsipy not available for season {season}")
                        continue
                        
                    # Get games for all teams in this season
                    teams = Teams(year=season)
                    
                    for team in teams:
                        try:
                            if Schedule is None:
                                continue
                            team_schedule = Schedule(team.abbreviation, year=season)
                            
                            for game in team_schedule:
                                # Parse game date
                                game_date = None
                                if hasattr(game, 'date') and game.date:
                                    try:
                                        if isinstance(game.date, date):
                                            game_date = game.date
                                        else:
                                            game_date = datetime.strptime(str(game.date), '%Y-%m-%d').date()
                                    except:
                                        continue
                                
                                # Check if game is within our date range
                                if game_date and start_date <= game_date <= end_date:
                                    # Avoid duplicate games (each game appears in both teams' schedules)
                                    game_id = self._generate_game_id(game, season, game_date)
                                    
                                    # Check if we already added this game
                                    if any(g.get('game_id') == game_id for g in schedule_data):
                                        continue
                                    
                                    schedule_data.append(self._parse_schedule_game(game, team, season, game_date))
                        
                        except Exception as e:
                            # Individual team schedule failures shouldn't stop the whole process
                            self.logger.debug(f"Error getting schedule for team {getattr(team, 'abbreviation', 'unknown')}: {str(e)}")
                            continue
                    
                except Exception as e:
                    self.logger.warning(f"Error getting NCAA basketball data for season {season}: {str(e)}")
                    continue
            
            if not schedule_data:
                self.logger.warning(f"No NCAA basketball games found for date range {start_date} to {end_date}")
                return self._create_empty_schedule_df()
            
            result_df = pd.DataFrame(schedule_data)
            
            # Remove duplicates based on game_id
            result_df = result_df.drop_duplicates(subset=['game_id']).reset_index(drop=True)
            
            self.logger.info(f"Retrieved {len(result_df)} NCAA basketball games")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA basketball schedule: {str(e)}")
            return self._create_empty_schedule_df()
    
    def _generate_game_id(self, game, season: int, game_date: date) -> str:
        """Generate a unique game ID from game information"""
        try:
            # Get team abbreviations
            home_team = getattr(game, 'opponent_abbr', '') or getattr(game, 'opponent_name', '')
            away_team = getattr(game, 'team_abbr', '') or ''
            
            # Determine home vs away based on location
            if hasattr(game, 'location'):
                if game.location == '@':
                    # Away game for the team we're looking at
                    away_team_id = away_team
                    home_team_id = home_team
                else:
                    # Home game for the team we're looking at
                    away_team_id = home_team  
                    home_team_id = away_team
            else:
                # Default assumption
                away_team_id = home_team
                home_team_id = away_team
            
            # Create consistent game ID (alphabetical order to ensure uniqueness)
            teams = sorted([home_team_id, away_team_id])
            game_id = f"{season}_{game_date}_{teams[0]}_{teams[1]}"
            return game_id
            
        except Exception:
            # Fallback to a basic ID
            return f"{season}_{game_date}_{str(hash(str(game)))}"
    
    def _parse_schedule_game(self, game, team, season: int, game_date: date) -> Dict:
        """Parse a single game from NCAA basketball schedule data"""
        try:
            # Determine game status
            status = 'scheduled'
            if hasattr(game, 'boxscore_index') and game.boxscore_index:
                status = 'final'
            elif game_date < date.today():
                status = 'final'  # Past games are likely final
            
            # Get opponent and location information
            opponent_name = getattr(game, 'opponent_name', '')
            opponent_abbr = getattr(game, 'opponent_abbr', '') or self._get_team_id_from_name(opponent_name)
            team_abbr = team.abbreviation
            team_name = team.name
            location = getattr(game, 'location', '')
            
            # Determine home vs away
            if location == '@':
                # Away game for the team we're looking at
                home_team_id = opponent_abbr
                home_team_name = opponent_name
                away_team_id = team_abbr
                away_team_name = team_name
            else:
                # Home game for the team we're looking at
                home_team_id = team_abbr
                home_team_name = team_name
                away_team_id = opponent_abbr
                away_team_name = opponent_name
            
            game_id = self._generate_game_id(game, season, game_date)
            
            return {
                'sport': self.sport,
                'league': self.league,
                'game_id': game_id,
                'game_date': game_date,
                'home_team_id': home_team_id,
                'home_team_name': home_team_name,
                'away_team_id': away_team_id,
                'away_team_name': away_team_name,
                'season': season,
                'status': status,
                'source_keys': json.dumps({
                    'boxscore_index': getattr(game, 'boxscore_index', ''),
                    'location': location,
                    'neutral_site': location == 'N',
                    'conference_game': getattr(game, 'conference_game', False)
                })
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing schedule game: {str(e)}")
            return {}
    
    def get_games(self, game_date: date) -> pd.DataFrame:
        """
        Get NCAA basketball games for a specific date with scores.
        
        Args:
            game_date: Date to get games for
            
        Returns:
            DataFrame with games including scores
        """
        try:
            self.logger.info(f"Getting NCAA basketball games for {game_date}")
            
            # Get games from schedule first
            games_df = self.get_schedule(game_date, game_date)
            
            if games_df.empty:
                return self._create_empty_games_df()
            
            # Enhance with scores for completed games
            enhanced_games = []
            
            for _, game in games_df.iterrows():
                game_dict = game.to_dict()
                
                # Add score columns with initial None values
                game_dict['home_score'] = None
                game_dict['away_score'] = None
                
                # If game is final or past date, try to get scores
                if game['status'] == 'final' or game['game_date'] < date.today():
                    try:
                        # Try to get boxscore data
                        source_keys_str = game.get('source_keys')
                        if source_keys_str is None:
                            source_keys_str = '{}'
                        source_keys = json.loads(source_keys_str)
                        boxscore_index = source_keys.get('boxscore_index')
                        
                        if boxscore_index and Boxscore is not None:
                            try:
                                boxscore = Boxscore(boxscore_index)
                                
                                if boxscore.home_points is not None and boxscore.away_points is not None:
                                    game_dict['home_score'] = int(boxscore.home_points)
                                    game_dict['away_score'] = int(boxscore.away_points)
                                    game_dict['status'] = 'final'
                                    
                            except Exception as boxscore_e:
                                self.logger.debug(f"Could not get boxscore for {boxscore_index}: {str(boxscore_e)}")
                    
                    except Exception as e:
                        self.logger.warning(f"Could not get scores for game {game['game_id']}: {str(e)}")
                
                enhanced_games.append(game_dict)
            
            result_df = pd.DataFrame(enhanced_games)
            self.logger.info(f"Retrieved {len(result_df)} NCAA basketball games with scores")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA basketball games for {game_date}: {str(e)}")
            return self._create_empty_games_df()
    
    def get_team_stats(self, season: int, rolling_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get NCAA basketball team statistics for a season.
        
        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
            rolling_days: If provided, get rolling stats for last N days
            
        Returns:
            DataFrame with team statistics
        """
        try:
            self.logger.info(f"Getting NCAA basketball team stats for season {season}")
            
            if rolling_days:
                return self._get_rolling_team_stats(season, rolling_days)
            else:
                return self._get_season_team_stats(season)
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA basketball team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def _get_season_team_stats(self, season: int) -> pd.DataFrame:
        """Get full season team statistics"""
        try:
            if not SPORTSIPY_AVAILABLE or Teams is None:
                self.logger.warning(f"Sportsipy not available for season {season}")
                return self._create_empty_stats_df()
                
            teams = Teams(year=season)
            stats_list = []
            
            for team in teams:
                try:
                    team_abbrev = team.abbreviation
                    team_name = team.name
                    conference = getattr(team, 'conference', '')
                    
                    # Get basic stats
                    games_played = getattr(team, 'games_played', 0)
                    wins = getattr(team, 'wins', 0)
                    losses = getattr(team, 'losses', 0)
                    win_percentage = getattr(team, 'win_percentage', 0.0)
                    
                    # Get advanced stats
                    points_per_game = getattr(team, 'points_per_game', 0.0)
                    points_against_per_game = getattr(team, 'opp_points_per_game', 0.0)
                    field_goal_percentage = getattr(team, 'field_goal_percentage', 0.0)
                    three_point_percentage = getattr(team, 'three_point_percentage', 0.0)
                    free_throw_percentage = getattr(team, 'free_throw_percentage', 0.0)
                    rebounds_per_game = getattr(team, 'total_rebounds_per_game', 0.0)
                    assists_per_game = getattr(team, 'assists_per_game', 0.0)
                    turnovers_per_game = getattr(team, 'turnovers_per_game', 0.0)
                    steals_per_game = getattr(team, 'steals_per_game', 0.0)
                    blocks_per_game = getattr(team, 'blocks_per_game', 0.0)
                    
                    # Advanced metrics
                    simple_rating_system = getattr(team, 'simple_rating_system', 0.0)
                    strength_of_schedule = getattr(team, 'strength_of_schedule', 0.0)
                    
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
                        'win_percentage': float(win_percentage) if win_percentage else 0.0,
                        # NCAA Basketball specific stats
                        'points_per_game': float(points_per_game) if points_per_game else 0.0,
                        'points_against_per_game': float(points_against_per_game) if points_against_per_game else 0.0,
                        'point_differential': float(points_per_game - points_against_per_game) if points_per_game and points_against_per_game else 0.0,
                        'field_goal_percentage': float(field_goal_percentage) if field_goal_percentage else 0.0,
                        'three_point_percentage': float(three_point_percentage) if three_point_percentage else 0.0,
                        'free_throw_percentage': float(free_throw_percentage) if free_throw_percentage else 0.0,
                        'rebounds_per_game': float(rebounds_per_game) if rebounds_per_game else 0.0,
                        'assists_per_game': float(assists_per_game) if assists_per_game else 0.0,
                        'turnovers_per_game': float(turnovers_per_game) if turnovers_per_game else 0.0,
                        'steals_per_game': float(steals_per_game) if steals_per_game else 0.0,
                        'blocks_per_game': float(blocks_per_game) if blocks_per_game else 0.0,
                        'simple_rating_system': float(simple_rating_system) if simple_rating_system else 0.0,
                        'strength_of_schedule': float(strength_of_schedule) if strength_of_schedule else 0.0,
                        'conference': conference
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing stats for team {getattr(team, 'name', 'unknown')}: {str(e)}")
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
            
            # Get all unique teams from recent games
            all_teams = set(games_df['home_team_id'].tolist() + games_df['away_team_id'].tolist())
            
            for team_id in all_teams:
                team_games = games_df[
                    (games_df['home_team_id'] == team_id) | 
                    (games_df['away_team_id'] == team_id)
                ]
                
                if len(team_games) > 0:
                    team_name = self._team_id_map.get(team_id, {}).get('full_name', team_id)
                    
                    # Calculate basic rolling stats from completed games
                    wins = 0
                    losses = 0
                    total_points = 0
                    total_points_allowed = 0
                    completed_games = 0
                    
                    for _, game in team_games.iterrows():
                        if game['status'] == 'final':
                            is_home = game['home_team_id'] == team_id
                            
                            # Get game with scores - convert game_date to proper date type
                            game_date_val = game['game_date']
                            if isinstance(game_date_val, date):
                                enhanced_game_df = self.get_games(game_date_val)
                            elif hasattr(game_date_val, 'date'):
                                try:
                                    date_method = getattr(game_date_val, 'date', None)
                                    if callable(date_method):
                                        enhanced_game_df = self.get_games(date_method())
                                    else:
                                        continue
                                except (AttributeError, TypeError):
                                    continue
                            elif isinstance(game_date_val, str):
                                try:
                                    parsed_date = datetime.strptime(game_date_val, '%Y-%m-%d').date()
                                    enhanced_game_df = self.get_games(parsed_date)
                                except (ValueError, TypeError):
                                    continue
                            else:
                                continue
                            
                            game_with_scores = enhanced_game_df[enhanced_game_df['game_id'] == game['game_id']]
                            
                            if not game_with_scores.empty:
                                game_row = game_with_scores.iloc[0]
                                
                                home_score = game_row.get('home_score')
                                away_score = game_row.get('away_score')
                                
                                if home_score is not None and away_score is not None:
                                    team_score = home_score if is_home else away_score
                                    opp_score = away_score if is_home else home_score
                                    
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
                    
                    stats_list.append({
                        'sport': self.sport,
                        'league': self.league,
                        'team_id': team_id,
                        'team_name': team_name,
                        'season': season,
                        'date': end_date,
                        'games_played': completed_games,
                        'wins': wins,
                        'losses': losses,
                        'win_percentage': wins / max(completed_games, 1),
                        # Estimated NCAA Basketball stats based on scoring
                        'points_per_game': float(avg_points_scored),
                        'points_against_per_game': float(avg_points_allowed),
                        'point_differential': float(avg_points_scored - avg_points_allowed),
                        'field_goal_percentage': 0.0,  # Would need detailed game data
                        'three_point_percentage': 0.0,
                        'free_throw_percentage': 0.0,
                        'rebounds_per_game': 0.0,
                        'assists_per_game': 0.0,
                        'turnovers_per_game': 0.0,
                        'steals_per_game': 0.0,
                        'blocks_per_game': 0.0,
                        'simple_rating_system': 0.0,
                        'strength_of_schedule': 0.0,
                        'conference': self._team_id_map.get(team_id, {}).get('conference', '')
                    })
            
            return pd.DataFrame(stats_list)
            
        except Exception as e:
            self.logger.error(f"Error getting rolling team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def get_recent_form(self, team_id: str, games: int = 10) -> Dict:
        """
        Get recent form/performance for an NCAA basketball team.
        
        Args:
            team_id: NCAA team identifier
            games: Number of recent games to analyze
            
        Returns:
            Dictionary with recent performance metrics
        """
        try:
            self.logger.info(f"Getting recent form for NCAA basketball team {team_id}")
            
            # Get recent games from the last 60 days (basketball season is dense)
            end_date = date.today()
            start_date = end_date - timedelta(days=60)
            
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
                    
                    # Get actual game scores - convert game_date to proper date type
                    game_date_val = game['game_date']
                    if isinstance(game_date_val, date):
                        enhanced_game_df = self.get_games(game_date_val)
                    elif hasattr(game_date_val, 'date'):
                        try:
                            date_method = getattr(game_date_val, 'date', None)
                            if callable(date_method):
                                enhanced_game_df = self.get_games(date_method())
                            else:
                                continue
                        except (AttributeError, TypeError):
                            continue
                    elif isinstance(game_date_val, str):
                        try:
                            parsed_date = datetime.strptime(game_date_val, '%Y-%m-%d').date()
                            enhanced_game_df = self.get_games(parsed_date)
                        except (ValueError, TypeError):
                            continue
                    else:
                        continue
                    
                    game_with_scores = enhanced_game_df[enhanced_game_df['game_id'] == game['game_id']]
                    
                    if not game_with_scores.empty:
                        game_row = game_with_scores.iloc[0]
                        
                        home_score = game_row.get('home_score')
                        away_score = game_row.get('away_score')
                        
                        if home_score is not None and away_score is not None:
                            team_score = home_score if is_home else away_score
                            opp_score = away_score if is_home else home_score
                            
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
                'last_10_record': f"{wins}-{losses}" if completed_games <= 10 else f"{wins}-{losses}",
                'form_trend': 'improving' if wins > losses else 'declining' if losses > wins else 'stable'
            }
            
            return form_dict
            
        except Exception as e:
            self.logger.error(f"Error getting recent form for team {team_id}: {str(e)}")
            return {'sport': self.sport, 'league': self.league, 'team_id': team_id, 'error': str(e)}
    
    def get_head_to_head(self, team1_id: str, team2_id: str, seasons: int = 5) -> pd.DataFrame:
        """
        Get head-to-head matchup history between two NCAA basketball teams.
        
        Args:
            team1_id: First team identifier
            team2_id: Second team identifier
            seasons: Number of seasons to look back
            
        Returns:
            DataFrame with historical matchup data
        """
        try:
            self.logger.info(f"Getting NCAA basketball head-to-head: {team1_id} vs {team2_id}")
            
            matchups = []
            
            # Get seasons to analyze
            seasons_to_check = [self._current_season - i for i in range(seasons)]
            
            for season in seasons_to_check:
                try:
                    # Get schedule for both teams in this season
                    for team_id in [team1_id, team2_id]:
                        try:
                            if not SPORTSIPY_AVAILABLE or Schedule is None:
                                continue
                            team_schedule = Schedule(team_id, year=season)
                            
                            for game in team_schedule:
                                # Check if this is a matchup against the other team
                                opponent_abbr = getattr(game, 'opponent_abbr', '')
                                opponent_name = getattr(game, 'opponent_name', '')
                                other_team_id = team2_id if team_id == team1_id else team1_id
                                
                                if (opponent_abbr == other_team_id or 
                                    self._get_team_id_from_name(opponent_name) == other_team_id):
                                    
                                    # Get game date
                                    game_date = None
                                    if hasattr(game, 'date') and game.date:
                                        try:
                                            if isinstance(game.date, date):
                                                game_date = game.date
                                            else:
                                                game_date = datetime.strptime(str(game.date), '%Y-%m-%d').date()
                                        except:
                                            continue
                                    
                                    if game_date:
                                        # Get game scores if available
                                        boxscore_index = getattr(game, 'boxscore_index', '')
                                        
                                        if boxscore_index and Boxscore is not None:
                                            try:
                                                boxscore = Boxscore(boxscore_index)
                                                
                                                if (boxscore.home_points is not None and 
                                                    boxscore.away_points is not None):
                                                    
                                                    # Determine which team was home/away
                                                    location = getattr(game, 'location', '')
                                                    if location == '@':
                                                        # Away game for current team
                                                        away_team_id = team_id
                                                        home_team_id = other_team_id
                                                        away_score = boxscore.away_points
                                                        home_score = boxscore.home_points
                                                    else:
                                                        # Home game for current team
                                                        home_team_id = team_id
                                                        away_team_id = other_team_id
                                                        home_score = boxscore.home_points
                                                        away_score = boxscore.away_points
                                                    
                                                    winner_id = home_team_id if home_score > away_score else away_team_id
                                                    
                                                    # Standardize to team1 vs team2 format
                                                    if team_id == team1_id:
                                                        team1_score = home_score if home_team_id == team1_id else away_score
                                                        team2_score = away_score if home_team_id == team1_id else home_score
                                                        location_ref = 'home' if home_team_id == team1_id else 'away'
                                                    else:
                                                        team1_score = away_score if home_team_id == team2_id else home_score
                                                        team2_score = home_score if home_team_id == team2_id else away_score
                                                        location_ref = 'away' if home_team_id == team2_id else 'home'
                                                    
                                                    # Avoid duplicates
                                                    game_id = f"{season}_{game_date}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
                                                    
                                                    if not any(m.get('game_id') == game_id for m in matchups):
                                                        matchups.append({
                                                            'sport': self.sport,
                                                            'league': self.league,
                                                            'game_id': game_id,
                                                            'game_date': game_date,
                                                            'season': season,
                                                            'team1_id': team1_id,
                                                            'team1_score': int(team1_score),
                                                            'team2_id': team2_id,
                                                            'team2_score': int(team2_score),
                                                            'winner_id': winner_id,
                                                            'location': location_ref
                                                        })
                                            
                                            except Exception as boxscore_e:
                                                self.logger.debug(f"Could not get boxscore for {boxscore_index}: {str(boxscore_e)}")
                                                continue
                        
                        except Exception as team_e:
                            self.logger.debug(f"Error getting schedule for team {team_id} in season {season}: {str(team_e)}")
                            continue
                
                except Exception as e:
                    self.logger.warning(f"Error getting season {season} matchups: {str(e)}")
                    continue
            
            if not matchups:
                self.logger.warning(f"No head-to-head data found for teams {team1_id} vs {team2_id}")
                return pd.DataFrame()
            
            return pd.DataFrame(matchups)
            
        except Exception as e:
            self.logger.error(f"Error getting NCAA basketball head-to-head data: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that collected NCAA basketball data meets requirements.
        
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
        if 'sport' in df.columns and not all(df['sport'] == 'NCAAB'):
            issues.append("Sport column should contain only 'NCAAB' values")
        
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
        """Get list of NCAA basketball statistics this collector supports"""
        return [
            'points_per_game', 'points_against_per_game', 'point_differential',
            'field_goal_percentage', 'three_point_percentage', 'free_throw_percentage',
            'rebounds_per_game', 'assists_per_game', 'turnovers_per_game',
            'steals_per_game', 'blocks_per_game', 'simple_rating_system',
            'strength_of_schedule', 'wins', 'losses', 'win_percentage', 'conference'
        ]
    
    def get_data_source(self) -> str:
        """Get the name of the data source this collector uses"""
        return "Sportsipy (sportsreference.com)"
    
    def _create_empty_schedule_df(self) -> pd.DataFrame:
        """Create empty DataFrame with schedule columns"""
        return pd.DataFrame({
            'sport': pd.Series(dtype='object'),
            'league': pd.Series(dtype='object'),
            'game_id': pd.Series(dtype='object'),
            'game_date': pd.Series(dtype='object'),
            'home_team_id': pd.Series(dtype='object'),
            'home_team_name': pd.Series(dtype='object'),
            'away_team_id': pd.Series(dtype='object'),
            'away_team_name': pd.Series(dtype='object'),
            'season': pd.Series(dtype='int64'),
            'status': pd.Series(dtype='object'),
            'source_keys': pd.Series(dtype='object')
        })
    
    def _create_empty_games_df(self) -> pd.DataFrame:
        """Create empty DataFrame with games columns"""
        return pd.DataFrame({
            'sport': pd.Series(dtype='object'),
            'league': pd.Series(dtype='object'),
            'game_id': pd.Series(dtype='object'),
            'game_date': pd.Series(dtype='object'),
            'home_team_id': pd.Series(dtype='object'),
            'home_team_name': pd.Series(dtype='object'),
            'away_team_id': pd.Series(dtype='object'),
            'away_team_name': pd.Series(dtype='object'),
            'season': pd.Series(dtype='int64'),
            'status': pd.Series(dtype='object'),
            'home_score': pd.Series(dtype='Int64'),
            'away_score': pd.Series(dtype='Int64'),
            'source_keys': pd.Series(dtype='object')
        })
    
    def _create_empty_stats_df(self) -> pd.DataFrame:
        """Create empty DataFrame with stats columns"""
        return pd.DataFrame({
            'sport': pd.Series(dtype='object'),
            'league': pd.Series(dtype='object'),
            'team_id': pd.Series(dtype='object'),
            'team_name': pd.Series(dtype='object'),
            'season': pd.Series(dtype='int64'),
            'date': pd.Series(dtype='object'),
            'games_played': pd.Series(dtype='int64'),
            'wins': pd.Series(dtype='int64'),
            'losses': pd.Series(dtype='int64'),
            'win_percentage': pd.Series(dtype='float64'),
            'points_per_game': pd.Series(dtype='float64'),
            'points_against_per_game': pd.Series(dtype='float64'),
            'point_differential': pd.Series(dtype='float64'),
            'field_goal_percentage': pd.Series(dtype='float64'),
            'three_point_percentage': pd.Series(dtype='float64'),
            'free_throw_percentage': pd.Series(dtype='float64'),
            'rebounds_per_game': pd.Series(dtype='float64'),
            'assists_per_game': pd.Series(dtype='float64'),
            'turnovers_per_game': pd.Series(dtype='float64'),
            'steals_per_game': pd.Series(dtype='float64'),
            'blocks_per_game': pd.Series(dtype='float64'),
            'simple_rating_system': pd.Series(dtype='float64'),
            'strength_of_schedule': pd.Series(dtype='float64'),
            'conference': pd.Series(dtype='object')
        })