import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import warnings

# Suppress pandas warnings from nfl_data_py
warnings.filterwarnings('ignore', category=FutureWarning)

import nfl_data_py as nfl

from src.interfaces.base_collector import BaseDataCollector


class NFLDataCollector(BaseDataCollector):
    """
    NFL data collector using nfl_data_py library.
    
    Provides access to NFL game schedules, scores, team statistics,
    and historical data through the nflverse data ecosystem.
    """
    
    def __init__(self):
        super().__init__(sport='NFL', league='NFL')
        self.logger = logging.getLogger(__name__)
        
        # Cache for NFL teams and current season
        self._teams_cache = None
        self._team_id_map = {}
        self._current_season = self._get_current_nfl_season()
        
        # Initialize team data
        self._initialize_teams()
    
    def _get_current_nfl_season(self) -> int:
        """Determine current NFL season year"""
        now = datetime.now()
        return self._date_to_nfl_season(now.date())
    
    def _date_to_nfl_season(self, game_date: date) -> int:
        """
        Convert a date to its corresponding NFL season year.
        
        NFL season logic:
        - September-December: Current year's season
        - January-February: Previous year's season (playoffs/Super Bowl)
        - March-August: Off-season, assign to upcoming season
        
        Args:
            game_date: Date to convert to NFL season
            
        Returns:
            NFL season year
        """
        year = game_date.year
        
        if game_date.month >= 9:  # Sep, Oct, Nov, Dec - current season
            return year
        elif game_date.month <= 2:  # Jan-Feb - previous year's season (playoffs/Super Bowl)
            return year - 1
        else:  # Mar-Aug - off-season, assign to upcoming season
            return year
    
    def _initialize_teams(self):
        """Initialize NFL teams data and create ID mappings"""
        try:
            # Get team descriptions from nfl_data_py
            teams_df = nfl.import_team_desc()
            
            if not teams_df.empty:
                self._teams_cache = teams_df.to_dict('records')
                
                # Create mappings for easier lookups
                for team in self._teams_cache:
                    team_abbrev = team.get('team_abbr', '')
                    team_name = team.get('team_name', '')
                    
                    self._team_id_map[team_abbrev] = {
                        'abbreviation': team_abbrev,
                        'full_name': f"{team.get('team_name', '')}",
                        'conference': team.get('team_conf', ''),
                        'division': team.get('team_division', ''),
                        'logo_espn': team.get('team_logo_espn', ''),
                        'color': team.get('team_color', ''),
                        'color2': team.get('team_color2', '')
                    }
                
                self.logger.info(f"Initialized {len(self._teams_cache)} NFL teams")
            else:
                # Fallback: create basic team mapping
                self._initialize_basic_teams()
                
        except Exception as e:
            self.logger.warning(f"Error initializing NFL teams via nfl_data_py: {str(e)}")
            self._initialize_basic_teams()
    
    def _initialize_basic_teams(self):
        """Initialize basic NFL team mappings as fallback"""
        # Basic NFL team data (32 teams for 2024 season)
        basic_teams = [
            # AFC East
            {'abbrev': 'BUF', 'name': 'Buffalo Bills'},
            {'abbrev': 'MIA', 'name': 'Miami Dolphins'},
            {'abbrev': 'NE', 'name': 'New England Patriots'},
            {'abbrev': 'NYJ', 'name': 'New York Jets'},
            # AFC North
            {'abbrev': 'BAL', 'name': 'Baltimore Ravens'},
            {'abbrev': 'CIN', 'name': 'Cincinnati Bengals'},
            {'abbrev': 'CLE', 'name': 'Cleveland Browns'},
            {'abbrev': 'PIT', 'name': 'Pittsburgh Steelers'},
            # AFC South
            {'abbrev': 'HOU', 'name': 'Houston Texans'},
            {'abbrev': 'IND', 'name': 'Indianapolis Colts'},
            {'abbrev': 'JAX', 'name': 'Jacksonville Jaguars'},
            {'abbrev': 'TEN', 'name': 'Tennessee Titans'},
            # AFC West
            {'abbrev': 'DEN', 'name': 'Denver Broncos'},
            {'abbrev': 'KC', 'name': 'Kansas City Chiefs'},
            {'abbrev': 'LV', 'name': 'Las Vegas Raiders'},
            {'abbrev': 'LAC', 'name': 'Los Angeles Chargers'},
            # NFC East
            {'abbrev': 'DAL', 'name': 'Dallas Cowboys'},
            {'abbrev': 'NYG', 'name': 'New York Giants'},
            {'abbrev': 'PHI', 'name': 'Philadelphia Eagles'},
            {'abbrev': 'WAS', 'name': 'Washington Commanders'},
            # NFC North
            {'abbrev': 'CHI', 'name': 'Chicago Bears'},
            {'abbrev': 'DET', 'name': 'Detroit Lions'},
            {'abbrev': 'GB', 'name': 'Green Bay Packers'},
            {'abbrev': 'MIN', 'name': 'Minnesota Vikings'},
            # NFC South
            {'abbrev': 'ATL', 'name': 'Atlanta Falcons'},
            {'abbrev': 'CAR', 'name': 'Carolina Panthers'},
            {'abbrev': 'NO', 'name': 'New Orleans Saints'},
            {'abbrev': 'TB', 'name': 'Tampa Bay Buccaneers'},
            # NFC West
            {'abbrev': 'ARI', 'name': 'Arizona Cardinals'},
            {'abbrev': 'LAR', 'name': 'Los Angeles Rams'},
            {'abbrev': 'SF', 'name': 'San Francisco 49ers'},
            {'abbrev': 'SEA', 'name': 'Seattle Seahawks'}
        ]
        
        self._teams_cache = basic_teams
        for team in basic_teams:
            team_abbrev = team['abbrev']
            self._team_id_map[team_abbrev] = {
                'abbreviation': team_abbrev,
                'full_name': team['name'],
                'conference': '',
                'division': '',
                'logo_espn': '',
                'color': '',
                'color2': ''
            }
        
        self.logger.info(f"Initialized {len(basic_teams)} NFL teams (basic fallback)")
    
    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get NFL game schedule for a date range.
        
        Args:
            start_date: Start date for schedule
            end_date: End date for schedule
            
        Returns:
            DataFrame with standardized schedule columns
        """
        try:
            self.logger.info(f"Getting NFL schedule from {start_date} to {end_date}")
            
            # Determine which NFL seasons to query based on date range
            seasons = set()
            
            # Add seasons that could contain games in our date range
            start_season = self._date_to_nfl_season(start_date)
            end_season = self._date_to_nfl_season(end_date)
            
            # Include all seasons from start to end
            for season in range(start_season, end_season + 1):
                seasons.add(season)
            
            # Ensure we include current season if it overlaps
            seasons.add(self._current_season)
            seasons = sorted(list(seasons))
            
            schedule_data = []
            
            for season in seasons:
                try:
                    # Get schedule for this season
                    season_schedule = nfl.import_schedules([season])
                    
                    if not season_schedule.empty:
                        # Filter games within date range
                        season_schedule = season_schedule.copy()
                        # Convert to datetime and extract date with proper None handling
                        gameday_series = pd.to_datetime(season_schedule['gameday'], errors='coerce')
                        season_schedule['gameday'] = [d.date() if pd.notna(d) else None for d in gameday_series]
                        
                        # Filter with proper None handling
                        valid_dates_mask = pd.notna(season_schedule['gameday'])
                        filtered_games = season_schedule[
                            valid_dates_mask & 
                            (season_schedule['gameday'] >= start_date) & 
                            (season_schedule['gameday'] <= end_date)
                        ].copy()
                        
                        if not filtered_games.empty:  # type: ignore
                            for _, game in filtered_games.iterrows():  # type: ignore
                                schedule_data.append(self._parse_schedule_game(game, season))
                
                except Exception as e:
                    self.logger.warning(f"Error getting NFL schedule for season {season}: {str(e)}")
                    continue
            
            if not schedule_data:
                self.logger.warning(f"No NFL games found for date range {start_date} to {end_date}")
                return self._create_empty_schedule_df()
            
            result_df = pd.DataFrame(schedule_data)
            self.logger.info(f"Retrieved {len(result_df)} NFL games")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NFL schedule: {str(e)}")
            return self._create_empty_schedule_df()
    
    def _parse_schedule_game(self, game: pd.Series, season: int) -> Dict:
        """Parse a single game from NFL schedule data"""
        try:
            # Determine game status
            status = 'scheduled'
            if not pd.isna(game.get('result')):  # type: ignore
                status = 'final'
            else:
                # Only set as 'final' if we have actual results, not just past dates
                game_date = game.get('gameday')
                if game_date and isinstance(game_date, date) and game_date < date.today():
                    # Check if we have scores to confirm it's actually final
                    if not pd.isna(game.get('home_score')) and not pd.isna(game.get('away_score')):  # type: ignore
                        status = 'final'
            
            # Get team information
            home_team = str(game.get('home_team', ''))
            away_team = str(game.get('away_team', ''))
            
            home_team_name = self._team_id_map.get(home_team, {}).get('full_name', home_team)
            away_team_name = self._team_id_map.get(away_team, {}).get('full_name', away_team)
            
            return {
                'sport': self.sport,
                'league': self.league,
                'game_id': str(game.get('game_id', '')),
                'game_date': game.get('gameday'),
                'home_team_id': home_team,
                'home_team_name': home_team_name,
                'away_team_id': away_team,
                'away_team_name': away_team_name,
                'season': season,
                'status': status,
                'source_keys': json.dumps({
                    'nfl_game_id': str(game.get('game_id', '')),
                    'week': game.get('week', 0),
                    'season_type': game.get('season_type', 'REG'),
                    'game_type': game.get('game_type', '')
                })
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing schedule game: {str(e)}")
            return {}
    
    def get_todays_games(self, game_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get today's NFL games (or for a specific date).
        
        Args:
            game_date: Optional date to get games for. If None, uses current UTC date - 5 hours (US time)
        
        Returns:
            DataFrame with NFL games
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
        Get NFL games for a specific date with scores.
        
        Args:
            game_date: Date to get games for
            
        Returns:
            DataFrame with games including scores
        """
        try:
            self.logger.info(f"Getting NFL games for {game_date}")
            
            # First, check database for stored games
            games_from_db = self._get_games_from_db(game_date)
            if not games_from_db.empty:
                self.logger.info(f"Found {len(games_from_db)} NFL games in database for {game_date}")
                # Add team names from mapping
                return self._enrich_with_team_names(games_from_db)
            
            # If not in database, fetch from API
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
                        # Get season for this game
                        game_season = game['season']
                        
                        # Get schedule with results for this season
                        season_schedule = nfl.import_schedules([game_season])
                        
                        if not season_schedule.empty:
                            # Find this specific game
                            game_match = season_schedule[
                                season_schedule['game_id'] == game['game_id']
                            ].copy()
                            
                            if len(game_match) > 0:
                                game_row = game_match.iloc[0]  # type: ignore
                                
                                # Extract scores if available
                                if not pd.isna(game_row.get('home_score')):
                                    game_dict['home_score'] = int(game_row['home_score'])
                                if not pd.isna(game_row.get('away_score')):
                                    game_dict['away_score'] = int(game_row['away_score'])
                                
                                # Update status if we have result
                                if not pd.isna(game_row.get('result')):
                                    game_dict['status'] = 'final'
                    
                    except Exception as e:
                        self.logger.warning(f"Could not get scores for game {game['game_id']}: {str(e)}")
                
                enhanced_games.append(game_dict)
            
            result_df = pd.DataFrame(enhanced_games)
            self.logger.info(f"Retrieved {len(result_df)} NFL games with scores")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NFL games for {game_date}: {str(e)}")
            return self._create_empty_games_df()
    
    def _get_games_from_db(self, game_date: date) -> pd.DataFrame:
        """Get games from database for a specific date"""
        import sqlite3
        try:
            conn = sqlite3.connect('sports_predictions.db')
            query = """
                SELECT * FROM games 
                WHERE sport = 'NFL' AND DATE(game_date) = DATE(?)
                ORDER BY game_id
            """
            games_df = pd.read_sql_query(query, conn, params=[game_date.strftime('%Y-%m-%d')])
            conn.close()
            return games_df
        except Exception as e:
            self.logger.warning(f"Error querying database: {str(e)}")
            return pd.DataFrame()
    
    def _enrich_with_team_names(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add team names to games dataframe using team ID mapping"""
        if games_df.empty:
            return games_df
        
        games_list = []
        for _, game in games_df.iterrows():
            home_id = str(game['home_team_id'])
            away_id = str(game['away_team_id'])
            
            # Get team names from mapping (full names like "Kansas City Chiefs")
            home_name = self._team_id_map.get(home_id, {}).get('full_name', home_id)
            away_name = self._team_id_map.get(away_id, {}).get('full_name', away_id)
            
            game_dict = game.to_dict()
            game_dict['home_team_name'] = home_name
            game_dict['away_team_name'] = away_name
            games_list.append(game_dict)
        
        return pd.DataFrame(games_list)
    
    def get_team_stats(self, season: int, rolling_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get NFL team statistics for a season.
        
        Args:
            season: Season year (e.g., 2024 for 2024 season)
            rolling_days: If provided, get rolling stats for last N days
            
        Returns:
            DataFrame with team statistics
        """
        try:
            self.logger.info(f"Getting NFL team stats for season {season}")
            
            if rolling_days:
                return self._get_rolling_team_stats(season, rolling_days)
            else:
                return self._get_season_team_stats(season)
            
        except Exception as e:
            self.logger.error(f"Error getting NFL team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def _get_season_team_stats(self, season: int) -> pd.DataFrame:
        """Get full season team statistics"""
        try:
            # Get play-by-play data to calculate team stats
            pbp_data = nfl.import_pbp_data([season], downcast=True)
            
            if pbp_data.empty:
                self.logger.warning(f"No play-by-play data available for season {season}")
                return self._create_empty_stats_df()
            
            # Filter for regular season games only
            regular_season = pbp_data[pbp_data['season_type'] == 'REG']
            
            stats_list = []
            
            # Calculate stats for each team
            for team_abbrev in self._team_id_map.keys():
                try:
                    # Get team's offensive plays
                    team_offense = regular_season[regular_season['posteam'] == team_abbrev]
                    # Get team's defensive plays
                    team_defense = regular_season[regular_season['defteam'] == team_abbrev]
                    
                    # Get unique games for wins/losses
                    team_games = regular_season[
                        (regular_season['home_team'] == team_abbrev) | 
                        (regular_season['away_team'] == team_abbrev)
                    ]['game_id'].unique()
                    
                    # Calculate wins/losses from schedule
                    try:
                        schedule = nfl.import_schedules([season])
                        if not schedule.empty:
                            team_schedule = schedule[
                                (schedule['home_team'] == team_abbrev) | 
                                (schedule['away_team'] == team_abbrev)
                            ].copy()
                            
                            wins = 0
                            losses = 0
                            
                            for _, game in team_schedule.iterrows():  # type: ignore
                                if not pd.isna(game.get('result')):  # type: ignore
                                    is_home = game['home_team'] == team_abbrev
                                    home_score = game.get('home_score')
                                    away_score = game.get('away_score')
                                    
                                    # Only count if we have valid scores
                                    if home_score is not None and away_score is not None:
                                        home_score = float(home_score)
                                        away_score = float(away_score)
                                        
                                        if is_home:
                                            if home_score > away_score:
                                                wins += 1
                                            else:
                                                losses += 1
                                        else:
                                            if away_score > home_score:
                                                wins += 1
                                            else:
                                                losses += 1
                        else:
                            wins = 0
                            losses = 0
                    
                    except Exception as e:
                        self.logger.warning(f"Could not calculate wins/losses for {team_abbrev}: {str(e)}")
                        wins = 0
                        losses = 0
                    
                    # Calculate offensive stats
                    total_plays = len(team_offense)
                    total_yards = team_offense['yards_gained'].sum() if total_plays > 0 else 0
                    passing_yards = team_offense[team_offense['play_type'] == 'pass']['yards_gained'].sum()
                    rushing_yards = team_offense[team_offense['play_type'] == 'run']['yards_gained'].sum()
                    
                    # Calculate defensive stats
                    yards_allowed = team_defense['yards_gained'].sum() if len(team_defense) > 0 else 0
                    pass_yards_allowed = team_defense[team_defense['play_type'] == 'pass']['yards_gained'].sum()
                    rush_yards_allowed = team_defense[team_defense['play_type'] == 'run']['yards_gained'].sum()
                    
                    # EPA (Expected Points Added) stats
                    offensive_epa = team_offense['epa'].mean() if total_plays > 0 else 0
                    defensive_epa = team_defense['epa'].mean() if len(team_defense) > 0 else 0
                    
                    team_name = self._team_id_map[team_abbrev]['full_name']
                    
                    stats_list.append({
                        'sport': self.sport,
                        'league': self.league,
                        'team_id': team_abbrev,
                        'team_name': team_name,
                        'season': season,
                        'date': None,  # Full season stats
                        'games_played': len(team_games),
                        'wins': wins,
                        'losses': losses,
                        # NFL-specific stats
                        'total_yards_per_game': total_yards / max(len(team_games), 1),
                        'passing_yards_per_game': passing_yards / max(len(team_games), 1),
                        'rushing_yards_per_game': rushing_yards / max(len(team_games), 1),
                        'yards_allowed_per_game': yards_allowed / max(len(team_games), 1),
                        'pass_yards_allowed_per_game': pass_yards_allowed / max(len(team_games), 1),
                        'rush_yards_allowed_per_game': rush_yards_allowed / max(len(team_games), 1),
                        'offensive_epa': float(offensive_epa),
                        'defensive_epa': float(defensive_epa),
                        'net_epa': float(offensive_epa - defensive_epa),
                        'total_plays': total_plays
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating stats for team {team_abbrev}: {str(e)}")
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
            
            # Get recent games with scores by collecting daily data
            all_games = []
            current_date = start_date
            while current_date <= end_date:
                daily_games = self.get_games(current_date)
                if not daily_games.empty:
                    all_games.append(daily_games)
                current_date += timedelta(days=1)
            
            if not all_games:
                return self._create_empty_stats_df()
            
            games_df = pd.concat(all_games, ignore_index=True)
            
            # Calculate rolling stats from recent games
            stats_list = []
            
            for team_abbrev in self._team_id_map.keys():
                team_games = games_df[
                    (games_df['home_team_id'] == team_abbrev) | 
                    (games_df['away_team_id'] == team_abbrev)
                ]
                
                if len(team_games) > 0:
                    team_name = self._team_id_map[team_abbrev]['full_name']
                    
                    # Calculate rolling stats from recent games
                    wins = 0
                    losses = 0
                    total_points_scored = 0
                    total_points_allowed = 0
                    games_with_scores = 0
                    
                    for _, game in team_games.iterrows():
                        if game['status'] == 'final':
                            is_home = game['home_team_id'] == team_abbrev
                            
                            home_score = game.get('home_score')
                            away_score = game.get('away_score')
                            
                            if home_score is not None and away_score is not None:
                                team_score = home_score if is_home else away_score
                                opp_score = away_score if is_home else home_score
                                
                                total_points_scored += team_score
                                total_points_allowed += opp_score
                                games_with_scores += 1
                                
                                if team_score > opp_score:
                                    wins += 1
                                else:
                                    losses += 1
                    
                    # Calculate averages
                    avg_points_scored = total_points_scored / max(games_with_scores, 1)
                    avg_points_allowed = total_points_allowed / max(games_with_scores, 1)
                    point_differential = avg_points_scored - avg_points_allowed
                    
                    # Estimate yards based on NFL averages (rough approximation)
                    # NFL average is about 350-400 total yards per game
                    # Use point scoring as a proxy for offensive performance
                    estimated_total_yards = avg_points_scored * 15  # Rough ratio: 15 yards per point
                    estimated_yards_allowed = avg_points_allowed * 15
                    
                    stats_list.append({
                        'sport': self.sport,
                        'league': self.league,
                        'team_id': team_abbrev,
                        'team_name': team_name,
                        'season': season,
                        'date': end_date,
                        'games_played': games_with_scores,
                        'wins': wins,
                        'losses': losses,
                        # Rolling stats based on game results
                        'points_per_game': round(avg_points_scored, 1),
                        'points_allowed_per_game': round(avg_points_allowed, 1),
                        'point_differential': round(point_differential, 1),
                        'total_yards_per_game': round(estimated_total_yards, 1),
                        'yards_allowed_per_game': round(estimated_yards_allowed, 1),
                        # Estimated breakdowns (using NFL averages)
                        'passing_yards_per_game': round(estimated_total_yards * 0.65, 1),  # ~65% passing
                        'rushing_yards_per_game': round(estimated_total_yards * 0.35, 1),  # ~35% rushing
                        'pass_yards_allowed_per_game': round(estimated_yards_allowed * 0.65, 1),
                        'rush_yards_allowed_per_game': round(estimated_yards_allowed * 0.35, 1),
                        # EPA approximations based on scoring efficiency
                        'offensive_epa': round((avg_points_scored - 20) / 10, 2),  # Rough EPA estimate
                        'defensive_epa': round((20 - avg_points_allowed) / 10, 2),
                        'net_epa': round((avg_points_scored - avg_points_allowed) / 10, 2),
                        'total_plays': games_with_scores * 65  # Rough estimate of plays per game
                    })
            
            return pd.DataFrame(stats_list)
            
        except Exception as e:
            self.logger.error(f"Error getting rolling team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def get_recent_form(self, team_id: str, games: int = 10) -> Dict:
        """
        Get recent form/performance for an NFL team.
        
        Args:
            team_id: NFL team identifier (abbreviation)
            games: Number of recent games to analyze
            
        Returns:
            Dictionary with recent performance metrics
        """
        try:
            self.logger.info(f"Getting recent form for NFL team {team_id}")
            
            # Get recent games from the last 60 days (NFL games are weekly)
            end_date = date.today()
            start_date = end_date - timedelta(days=60)
            
            # Get recent games with scores by collecting daily data
            all_games = []
            current_date = start_date
            while current_date <= end_date:
                daily_games = self.get_games(current_date)
                if not daily_games.empty:
                    all_games.append(daily_games)
                current_date += timedelta(days=1)
            
            if not all_games:
                return {
                    'sport': self.sport, 
                    'league': self.league, 
                    'team_id': team_id, 
                    'error': 'No recent games found'
                }
            
            games_df = pd.concat(all_games, ignore_index=True)
            
            # Filter games for this team
            team_games = games_df[
                (games_df['home_team_id'] == team_id) | 
                (games_df['away_team_id'] == team_id)
            ]  # type: ignore
            if not team_games.empty:  # type: ignore
                team_games = team_games.sort_values(by=['game_date'], ascending=False).head(games)  # type: ignore
            
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
            
            for _, game in team_games.iterrows():
                if game['status'] == 'final':
                    is_home = game['home_team_id'] == team_id
                    
                    if game.get('home_score') is not None and game.get('away_score') is not None:
                        team_score = game['home_score'] if is_home else game['away_score']
                        opp_score = game['away_score'] if is_home else game['home_score']
                        
                        if team_score > opp_score:
                            wins += 1
                        else:
                            losses += 1
            
            form_dict = {
                'sport': self.sport,
                'league': self.league,
                'team_id': team_id,
                'games_analyzed': len(team_games),
                'wins': wins,
                'losses': losses,
                'win_percentage': wins / len(team_games) if len(team_games) > 0 else 0,
                'last_5_record': f"{wins}-{losses}" if len(team_games) <= 5 else f"{wins}-{losses}",
                'form_trend': 'improving' if wins > losses else 'declining' if losses > wins else 'stable'
            }
            
            return form_dict
            
        except Exception as e:
            self.logger.error(f"Error getting recent form for team {team_id}: {str(e)}")
            return {'sport': self.sport, 'league': self.league, 'team_id': team_id, 'error': str(e)}
    
    def get_head_to_head(self, team1_id: str, team2_id: str, seasons: int = 3) -> pd.DataFrame:
        """
        Get head-to-head matchup history between two NFL teams.
        
        Args:
            team1_id: First team identifier (abbreviation)
            team2_id: Second team identifier (abbreviation)
            seasons: Number of seasons to look back
            
        Returns:
            DataFrame with historical matchup data
        """
        try:
            self.logger.info(f"Getting NFL head-to-head: {team1_id} vs {team2_id}")
            
            matchups = []
            
            # Get seasons to analyze
            seasons_to_check = [self._current_season - i for i in range(seasons)]
            
            for season in seasons_to_check:
                try:
                    # Get schedule for this season
                    season_schedule = nfl.import_schedules([season])
                    
                    if not season_schedule.empty:
                        # Filter for matchups between the two teams
                        team_matchups = season_schedule[
                            ((season_schedule['home_team'] == team1_id) & (season_schedule['away_team'] == team2_id)) |
                            ((season_schedule['home_team'] == team2_id) & (season_schedule['away_team'] == team1_id))
                        ].copy()
                        
                        for _, game in team_matchups.iterrows():  # type: ignore
                            game_result = game.get('result')
                            home_score = game.get('home_score')
                            if not pd.isna(game_result) and not pd.isna(home_score):  # type: ignore
                                # Determine which team is which
                                if game['home_team'] == team1_id:
                                    team1_score = int(game['home_score'])
                                    team2_score = int(game['away_score'])
                                    location = 'home'
                                else:
                                    team1_score = int(game['away_score'])
                                    team2_score = int(game['home_score'])
                                    location = 'away'
                                
                                winner_id = team1_id if team1_score > team2_score else team2_id
                                
                                matchups.append({
                                    'sport': self.sport,
                                    'league': self.league,
                                    'game_id': str(game['game_id']),
                                    'game_date': game['gameday'],
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
            self.logger.error(f"Error getting NFL head-to-head data: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that collected NFL data meets requirements.
        
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
        if 'sport' in df.columns and not all(df['sport'] == 'NFL'):
            issues.append("Sport column should contain only 'NFL' values")
        
        if 'league' in df.columns and not all(df['league'] == 'NFL'):
            issues.append("League column should contain only 'NFL' values")
        
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
        """Get list of NFL statistics this collector supports"""
        return [
            'total_yards_per_game', 'passing_yards_per_game', 'rushing_yards_per_game',
            'yards_allowed_per_game', 'pass_yards_allowed_per_game', 'rush_yards_allowed_per_game',
            'offensive_epa', 'defensive_epa', 'net_epa', 'total_plays',
            'wins', 'losses', 'win_percentage'
        ]
    
    def get_data_source(self) -> str:
        """Get the name of the data source this collector uses"""
        return "nflverse (via nfl_data_py)"
    
    def _create_empty_schedule_df(self) -> pd.DataFrame:
        """Create empty DataFrame with schedule columns"""
        return pd.DataFrame({
            'sport': pd.Series(dtype=str),
            'league': pd.Series(dtype=str), 
            'game_id': pd.Series(dtype=str),
            'game_date': pd.Series(dtype='datetime64[ns]'),
            'home_team_id': pd.Series(dtype=str),
            'home_team_name': pd.Series(dtype=str),
            'away_team_id': pd.Series(dtype=str), 
            'away_team_name': pd.Series(dtype=str),
            'season': pd.Series(dtype='int64'),
            'status': pd.Series(dtype=str),
            'source_keys': pd.Series(dtype=str)
        })
    
    def _create_empty_games_df(self) -> pd.DataFrame:
        """Create empty DataFrame with games columns"""
        return pd.DataFrame({
            'sport': pd.Series(dtype=str),
            'league': pd.Series(dtype=str),
            'game_id': pd.Series(dtype=str), 
            'game_date': pd.Series(dtype='datetime64[ns]'),
            'home_team_id': pd.Series(dtype=str),
            'home_team_name': pd.Series(dtype=str),
            'away_team_id': pd.Series(dtype=str),
            'away_team_name': pd.Series(dtype=str),
            'season': pd.Series(dtype='int64'),
            'status': pd.Series(dtype=str),
            'home_score': pd.Series(dtype='float64'),
            'away_score': pd.Series(dtype='float64'),
            'source_keys': pd.Series(dtype=str)
        })
    
    def _create_empty_stats_df(self) -> pd.DataFrame:
        """Create empty DataFrame with stats columns"""
        return pd.DataFrame({
            'sport': pd.Series(dtype=str),
            'league': pd.Series(dtype=str),
            'team_id': pd.Series(dtype=str),
            'team_name': pd.Series(dtype=str),
            'season': pd.Series(dtype='int64'),
            'date': pd.Series(dtype='datetime64[ns]'),
            'games_played': pd.Series(dtype='int64'),
            'wins': pd.Series(dtype='int64'),
            'losses': pd.Series(dtype='int64'),
            'total_yards_per_game': pd.Series(dtype='float64'),
            'passing_yards_per_game': pd.Series(dtype='float64'),
            'rushing_yards_per_game': pd.Series(dtype='float64'),
            'yards_allowed_per_game': pd.Series(dtype='float64'),
            'pass_yards_allowed_per_game': pd.Series(dtype='float64'),
            'rush_yards_allowed_per_game': pd.Series(dtype='float64'),
            'offensive_epa': pd.Series(dtype='float64'),
            'defensive_epa': pd.Series(dtype='float64'),
            'net_epa': pd.Series(dtype='float64'),
            'total_plays': pd.Series(dtype='int64')
        })