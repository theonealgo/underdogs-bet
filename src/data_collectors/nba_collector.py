import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json

from nba_api.stats.endpoints import (
    leaguegamefinder, scoreboardv2, leaguestandings, 
    teamgamelogs, teamdashboardbygeneralsplits
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.static import teams

from src.interfaces.base_collector import BaseDataCollector


class NBADataCollector(BaseDataCollector):
    """
    NBA data collector using nba_api library.
    
    Provides access to NBA game schedules, scores, team statistics,
    and historical data through the official NBA.com APIs.
    """
    
    def __init__(self):
        super().__init__(sport='NBA', league='NBA')
        self.logger = logging.getLogger(__name__)
        
        # Cache for NBA teams
        self._teams_cache = None
        self._team_id_map = {}
        
        # Initialize team data
        self._initialize_teams()
    
    def _initialize_teams(self):
        """Initialize NBA teams data and create ID mappings"""
        try:
            self._teams_cache = teams.get_teams()
            
            # Create mappings for easier lookups
            for team in self._teams_cache:
                team_id = str(team['id'])
                abbreviation = team['abbreviation']
                full_name = team['full_name']
                
                self._team_id_map[team_id] = {
                    'abbreviation': abbreviation,
                    'full_name': full_name,
                    'city': team.get('city', ''),
                    'nickname': team.get('nickname', '')
                }
                
            self.logger.info(f"Initialized {len(self._teams_cache)} NBA teams")
            
        except Exception as e:
            self.logger.error(f"Error initializing NBA teams: {str(e)}")
            self._teams_cache = []
    
    def _normalize_season_id(self, season_id) -> int:
        """
        Normalize NBA season ID to proper season year.
        
        NBA API sometimes returns season IDs like 22024 which should be 2024.
        Also handles regular year formats.
        
        Args:
            season_id: Season identifier from NBA API
            
        Returns:
            Normalized season year as integer
        """
        try:
            season_id_str = str(season_id)
            
            # Handle NBA's SEASON_ID format like 22024 -> 2024
            if len(season_id_str) == 5 and season_id_str.startswith('2'):
                return int(season_id_str[-4:])
            
            # Handle regular 4-digit years
            if len(season_id_str) == 4:
                return int(season_id_str)
            
            # Fallback to current year
            return datetime.now().year
            
        except (ValueError, TypeError):
            self.logger.warning(f"Could not normalize season_id: {season_id}, using current year")
            return datetime.now().year
    
    def _calculate_opponent_points(self, recent_games: pd.DataFrame) -> float:
        """
        Calculate average opponent points from recent games.
        
        TeamGameLogs doesn't always include OPP_PTS, so we calculate it
        by looking at the opponent's score in each game.
        
        Args:
            recent_games: DataFrame with recent game logs
            
        Returns:
            Average opponent points per game
        """
        if recent_games.empty:
            return 0.0
            
        try:
            # If OPP_PTS column exists, use it directly
            if 'OPP_PTS' in recent_games.columns:
                opp_pts_series = recent_games['OPP_PTS']
                # Remove any null/NaN values
                valid_opp_pts = opp_pts_series.dropna()
                if len(valid_opp_pts) > 0:
                    return float(valid_opp_pts.mean())
            
            # Fallback: try to calculate from available data
            # This would require more complex logic to match opponent scores
            # For now, return 0 and log that calculation wasn't possible
            self.logger.warning("Could not calculate opponent points - OPP_PTS not available")
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating opponent points: {str(e)}")
            return 0.0
    
    def get_schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Get NBA game schedule for a date range.
        
        Args:
            start_date: Start date for schedule
            end_date: End date for schedule
            
        Returns:
            DataFrame with standardized schedule columns
        """
        try:
            self.logger.info(f"Getting NBA schedule from {start_date} to {end_date}")
            
            # Use leaguegamefinder to get games in date range
            game_finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=start_date.strftime('%m/%d/%Y'),
                date_to_nullable=end_date.strftime('%m/%d/%Y'),
                league_id_nullable='00'  # NBA
            )
            
            games_df = game_finder.get_data_frames()[0]
            
            if games_df.empty:
                self.logger.warning(f"No NBA games found for date range {start_date} to {end_date}")
                return self._create_empty_schedule_df()
            
            # Transform to standardized format
            schedule_data = []
            
            # Group by GAME_ID to get matchups (each game appears twice in raw data)
            for game_id, game_group in games_df.groupby('GAME_ID'):
                if len(game_group) != 2:
                    continue  # Skip incomplete game data
                
                # Determine home/away teams
                home_row = game_group[game_group['MATCHUP'].str.contains(' vs. ')].iloc[0] if len(game_group[game_group['MATCHUP'].str.contains(' vs. ')]) > 0 else game_group.iloc[0]
                away_row = game_group[game_group['MATCHUP'].str.contains(' @ ')].iloc[0] if len(game_group[game_group['MATCHUP'].str.contains(' @ ')]) > 0 else game_group.iloc[1]
                
                game_date = datetime.strptime(home_row['GAME_DATE'], '%Y-%m-%d').date()
                
                schedule_data.append({
                    'sport': self.sport,
                    'league': self.league,
                    'game_id': str(game_id),
                    'game_date': game_date,
                    'home_team_id': str(home_row['TEAM_ID']),
                    'home_team_name': home_row['TEAM_NAME'],
                    'away_team_id': str(away_row['TEAM_ID']),
                    'away_team_name': away_row['TEAM_NAME'],
                    'season': self._normalize_season_id(home_row['SEASON_ID']),
                    'status': 'final' if home_row.get('WL') else 'scheduled',
                    'source_keys': json.dumps({
                        'nba_game_id': str(game_id),
                        'season_id': str(home_row['SEASON_ID'])
                    })
                })
            
            result_df = pd.DataFrame(schedule_data)
            self.logger.info(f"Retrieved {len(result_df)} NBA games")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting NBA schedule: {str(e)}")
            return self._create_empty_schedule_df()
    
    def get_todays_games(self, game_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get today's NBA games (or for a specific date).
        
        Args:
            game_date: Optional date to get games for. If None, uses current UTC date - 5 hours (US time)
        
        Returns:
            DataFrame with NBA games
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
        Get NBA games for a specific date with scores.
        
        Args:
            game_date: Date to get games for
            
        Returns:
            DataFrame with games including scores
        """
        try:
            self.logger.info(f"Getting NBA games for {game_date}")
            
            # First, check database for stored games
            games_from_db = self._get_games_from_db(game_date)
            if not games_from_db.empty:
                self.logger.info(f"Found {len(games_from_db)} NBA games in database for {game_date}")
                # Add team names from mapping
                return self._enrich_with_team_names(games_from_db)
            
            # If not in database, fetch from API
            # Get today's games using live scoreboard first (for current games)
            if game_date == date.today():
                try:
                    live_board = live_scoreboard.ScoreBoard()
                    live_games = live_board.get_dict()
                    
                    if live_games and 'scoreboard' in live_games and 'games' in live_games['scoreboard']:
                        return self._parse_live_games(live_games['scoreboard']['games'], game_date)
                except Exception as e:
                    self.logger.warning(f"Live scoreboard failed, falling back to historical: {str(e)}")
            
            # Fallback to historical data using scoreboard endpoint
            try:
                board = scoreboardv2.ScoreboardV2(
                    game_date=game_date.strftime('%m/%d/%Y'),
                    league_id='00'
                )
                games_df = board.get_data_frames()[0]  # GameHeader
                
                if games_df.empty:
                    self.logger.info(f"No NBA games found for {game_date}")
                    return self._create_empty_games_df()
                
                return self._parse_historical_games(games_df, game_date)
                
            except Exception as e:
                self.logger.warning(f"Historical scoreboard failed: {str(e)}")
                
                # Final fallback to schedule data
                return self.get_schedule(game_date, game_date)
            
        except Exception as e:
            self.logger.error(f"Error getting NBA games for {game_date}: {str(e)}")
            return self._create_empty_games_df()
    
    def _get_games_from_db(self, game_date: date) -> pd.DataFrame:
        """Get games from database for a specific date"""
        try:
            import sqlite3
            conn = sqlite3.connect('sports_predictions.db')
            query = """
                SELECT * FROM games 
                WHERE sport = 'NBA' AND DATE(game_date) = DATE(?)
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
            
            # Get team names from mapping
            home_name = self._team_id_map.get(home_id, {}).get('full_name', home_id)
            away_name = self._team_id_map.get(away_id, {}).get('full_name', away_id)
            
            game_dict = game.to_dict()
            game_dict['home_team_name'] = home_name
            game_dict['away_team_name'] = away_name
            games_list.append(game_dict)
        
        return pd.DataFrame(games_list)
    
    def _parse_live_games(self, games_data: List[Dict], game_date: date) -> pd.DataFrame:
        """Parse live scoreboard games data"""
        games_list = []
        
        for game in games_data:
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
            status = 'scheduled'
            if game.get('gameStatus') == 3:  # Final
                status = 'final'
            elif game.get('gameStatus') == 2:  # In progress
                status = 'in_progress'
            
            games_list.append({
                'sport': self.sport,
                'league': self.league,
                'game_id': str(game.get('gameId', '')),
                'game_date': game_date,
                'home_team_id': str(home_team.get('teamId', '')),
                'home_team_name': home_team.get('teamName', ''),
                'away_team_id': str(away_team.get('teamId', '')),
                'away_team_name': away_team.get('teamName', ''),
                'season': int(game.get('seasonYear', datetime.now().year)),
                'status': status,
                'home_score': int(home_team.get('score', 0)) if status != 'scheduled' else None,
                'away_score': int(away_team.get('score', 0)) if status != 'scheduled' else None,
                'source_keys': json.dumps({
                    'nba_game_id': str(game.get('gameId', '')),
                    'game_status': game.get('gameStatus')
                })
            })
        
        return pd.DataFrame(games_list)
    
    def _parse_historical_games(self, games_df: pd.DataFrame, game_date: date) -> pd.DataFrame:
        """Parse historical scoreboard games data"""
        games_list = []
        
        for _, game in games_df.iterrows():
            # Determine status
            status_text = game.get('GAME_STATUS_TEXT', '')
            status = 'final' if status_text and status_text.lower() in ['final', 'final/ot'] else 'scheduled'
            
            # Get team names with fallback to ID mapping
            home_team_id = str(game['HOME_TEAM_ID'])
            away_team_id = str(game['VISITOR_TEAM_ID'])
            
            home_team_name = game.get('HOME_TEAM_NAME', '')
            if not home_team_name and home_team_id in self._team_id_map:
                home_team_name = self._team_id_map[home_team_id]['full_name']
                
            away_team_name = game.get('VISITOR_TEAM_NAME', '')
            if not away_team_name and away_team_id in self._team_id_map:
                away_team_name = self._team_id_map[away_team_id]['full_name']

            games_list.append({
                'sport': self.sport,
                'league': self.league,
                'game_id': str(game['GAME_ID']),
                'game_date': game_date,
                'home_team_id': home_team_id,
                'home_team_name': home_team_name,
                'away_team_id': away_team_id,
                'away_team_name': away_team_name,
                'season': self._normalize_season_id(game.get('SEASON', datetime.now().year)),
                'status': status,
                'home_score': int(game.get('PTS_HOME', 0) or 0) if status == 'final' and game.get('PTS_HOME') is not None else None,
                'away_score': int(game.get('PTS_VISITOR', 0) or 0) if status == 'final' and game.get('PTS_VISITOR') is not None else None,
                'source_keys': json.dumps({
                    'nba_game_id': str(game['GAME_ID']),
                    'game_status_text': game.get('GAME_STATUS_TEXT', '')
                })
            })
        
        return pd.DataFrame(games_list)
    
    def get_team_stats(self, season: int, rolling_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get NBA team statistics for a season.
        
        Args:
            season: Season year (e.g., 2024 for 2024-25 season)
            rolling_days: If provided, get rolling stats for last N days
            
        Returns:
            DataFrame with team statistics
        """
        try:
            self.logger.info(f"Getting NBA team stats for season {season}")
            
            # Convert season year to NBA season format (e.g., 2024 -> "2024-25")
            season_str = f"{season}-{str(season + 1)[-2:]}"
            
            if rolling_days:
                # For rolling stats, get recent game logs
                return self._get_rolling_team_stats(season_str, rolling_days)
            else:
                # Get full season stats
                return self._get_season_team_stats(season_str)
            
        except Exception as e:
            self.logger.error(f"Error getting NBA team stats: {str(e)}")
            return self._create_empty_stats_df()
    
    def _get_season_team_stats(self, season_str: str) -> pd.DataFrame:
        """Get full season team statistics"""
        stats_list = []
        
        for team in self._teams_cache or []:
            try:
                team_id = team['id']
                
                # Get team dashboard stats
                dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
                    team_id=team_id,
                    season=season_str,
                    season_type_all_star='Regular Season'
                )
                
                team_stats_df = dashboard.get_data_frames()[0]  # OverallTeamDashboard
                
                if not team_stats_df.empty:
                    stats = team_stats_df.iloc[0]
                    
                    stats_list.append({
                        'sport': self.sport,
                        'league': self.league,
                        'team_id': str(team_id),
                        'team_name': team['full_name'],
                        'season': int(season_str.split('-')[0]),
                        'date': None,  # Full season stats
                        'games_played': int(stats.get('GP', 0)),
                        'wins': int(stats.get('W', 0)),
                        'losses': int(stats.get('L', 0)),
                        # NBA-specific stats
                        'points_per_game': float(stats.get('PTS', 0)),
                        'rebounds_per_game': float(stats.get('REB', 0)),
                        'assists_per_game': float(stats.get('AST', 0)),
                        'field_goal_pct': float(stats.get('FG_PCT', 0)),
                        'three_point_pct': float(stats.get('FG3_PCT', 0)),
                        'free_throw_pct': float(stats.get('FT_PCT', 0)),
                        'steals_per_game': float(stats.get('STL', 0)),
                        'blocks_per_game': float(stats.get('BLK', 0)),
                        'turnovers_per_game': float(stats.get('TOV', 0)),
                        'plus_minus': float(stats.get('PLUS_MINUS', 0))
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error getting stats for team {team['full_name']}: {str(e)}")
                continue
        
        return pd.DataFrame(stats_list)
    
    def _get_rolling_team_stats(self, season_str: str, rolling_days: int) -> pd.DataFrame:
        """Get rolling team statistics for recent games using TeamGameLogs"""
        stats_list = []
        end_date = date.today()
        cutoff_date = end_date - timedelta(days=rolling_days)
        
        # Get stats for each team using their game logs
        for team in self._teams_cache or []:
            team_id = team.get('id', 'unknown')
            try:
                team_name = team['full_name']
                
                # Get recent game logs for this team
                game_logs = teamgamelogs.TeamGameLogs(
                    team_id_nullable=team_id,
                    season_nullable=season_str,
                    season_type_nullable='Regular Season'
                )
                
                logs_df = game_logs.get_data_frames()[0]
                
                if logs_df.empty:
                    continue
                    
                # Filter to recent games within rolling window
                logs_df['GAME_DATE_PARSED'] = pd.to_datetime(logs_df['GAME_DATE'])
                recent_logs = logs_df[logs_df['GAME_DATE_PARSED'] >= pd.Timestamp(cutoff_date)]
                
                if recent_logs.empty:
                    continue
                
                # Calculate rolling stats from recent games
                wins = len(recent_logs[recent_logs['WL'] == 'W'])
                losses = len(recent_logs[recent_logs['WL'] == 'L'])
                games_played = len(recent_logs)
                
                stats_list.append({
                    'sport': self.sport,
                    'league': self.league,
                    'team_id': str(team_id),
                    'team_name': team_name,
                    'season': int(season_str.split('-')[0]),
                    'date': end_date,
                    'games_played': games_played,
                    'wins': wins,
                    'losses': losses,
                    # Calculate actual NBA stats from game logs
                    'points_per_game': float(recent_logs['PTS'].mean()) if games_played > 0 else 0.0,
                    'rebounds_per_game': float(recent_logs['REB'].mean()) if games_played > 0 else 0.0,
                    'assists_per_game': float(recent_logs['AST'].mean()) if games_played > 0 else 0.0,
                    'field_goal_pct': float(recent_logs['FG_PCT'].mean()) if games_played > 0 else 0.0,
                    'three_point_pct': float(recent_logs['FG3_PCT'].mean()) if games_played > 0 else 0.0,
                    'free_throw_pct': float(recent_logs['FT_PCT'].mean()) if games_played > 0 else 0.0,
                    'steals_per_game': float(recent_logs['STL'].mean()) if games_played > 0 else 0.0,
                    'blocks_per_game': float(recent_logs['BLK'].mean()) if games_played > 0 else 0.0,
                    'turnovers_per_game': float(recent_logs['TOV'].mean()) if games_played > 0 else 0.0,
                    'plus_minus': float(recent_logs['PLUS_MINUS'].mean()) if games_played > 0 else 0.0
                })
                
            except Exception as e:
                self.logger.warning(f"Error getting rolling stats for team {team.get('full_name', team_id)}: {str(e)}")
                continue
        
        return pd.DataFrame(stats_list)
    
    def get_recent_form(self, team_id: str, games: int = 10) -> Dict:
        """
        Get recent form/performance for an NBA team.
        
        Args:
            team_id: NBA team identifier
            games: Number of recent games to analyze
            
        Returns:
            Dictionary with recent performance metrics
        """
        try:
            self.logger.info(f"Getting recent form for NBA team {team_id}")
            
            # Get recent game logs
            current_season = f"{datetime.now().year}-{str(datetime.now().year + 1)[-2:]}"
            
            game_logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=current_season,
                season_type_nullable='Regular Season'
            )
            
            logs_df = game_logs.get_data_frames()[0]
            
            if logs_df.empty:
                return {'sport': self.sport, 'league': self.league, 'team_id': team_id, 'error': 'No recent games found'}
            
            # Get last N games
            recent_games = logs_df.head(games)
            
            # Calculate form metrics
            wins = len(recent_games[recent_games['WL'] == 'W'])
            losses = len(recent_games[recent_games['WL'] == 'L'])
            
            form_dict = {
                'sport': self.sport,
                'league': self.league,
                'team_id': team_id,
                'games_analyzed': len(recent_games),
                'wins': wins,
                'losses': losses,
                'win_percentage': wins / len(recent_games) if len(recent_games) > 0 else 0,
                'avg_points': recent_games['PTS'].mean(),
                'avg_points_allowed': self._calculate_opponent_points(recent_games),
                'avg_rebounds': recent_games['REB'].mean(),
                'avg_assists': recent_games['AST'].mean(),
                'last_5_record': f"{len(recent_games.head(5)[recent_games.head(5)['WL'] == 'W'])}-{len(recent_games.head(5)[recent_games.head(5)['WL'] == 'L'])}",
                'form_trend': 'improving' if wins > losses else 'declining' if losses > wins else 'stable'
            }
            
            return form_dict
            
        except Exception as e:
            self.logger.error(f"Error getting recent form for team {team_id}: {str(e)}")
            return {'sport': self.sport, 'league': self.league, 'team_id': team_id, 'error': str(e)}
    
    def get_head_to_head(self, team1_id: str, team2_id: str, seasons: int = 3) -> pd.DataFrame:
        """
        Get head-to-head matchup history between two NBA teams.
        
        Args:
            team1_id: First team identifier
            team2_id: Second team identifier
            seasons: Number of seasons to look back
            
        Returns:
            DataFrame with historical matchup data
        """
        try:
            self.logger.info(f"Getting NBA head-to-head: {team1_id} vs {team2_id}")
            
            matchups = []
            current_year = datetime.now().year
            
            for i in range(seasons):
                season_year = current_year - i
                season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
                
                try:
                    # Get games where these teams played each other
                    game_finder = leaguegamefinder.LeagueGameFinder(
                        team_id_nullable=team1_id,
                        vs_team_id_nullable=team2_id,
                        season_nullable=season_str,
                        season_type_nullable='Regular Season'
                    )
                    
                    games_df = game_finder.get_data_frames()[0]
                    
                    for _, game in games_df.iterrows():
                        matchups.append({
                            'sport': self.sport,
                            'league': self.league,
                            'game_id': str(game['GAME_ID']),
                            'game_date': datetime.strptime(game['GAME_DATE'], '%Y-%m-%d').date(),
                            'season': season_year,
                            'team1_id': team1_id,
                            'team1_score': int(game['PTS']),
                            'team2_id': team2_id,
                            'team2_score': int(game.get('OPP_PTS', 0)) if 'OPP_PTS' in game else 0,
                            'winner_id': team1_id if game['WL'] == 'W' else team2_id,
                            'location': 'home' if '@' not in game['MATCHUP'] else 'away'
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Error getting {season_str} matchups: {str(e)}")
                    continue
            
            if not matchups:
                self.logger.warning(f"No head-to-head data found for teams {team1_id} vs {team2_id}")
                return pd.DataFrame()
            
            return pd.DataFrame(matchups)
            
        except Exception as e:
            self.logger.error(f"Error getting NBA head-to-head data: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that collected NBA data meets requirements.
        
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
        if 'sport' in df.columns and not all(df['sport'] == 'NBA'):
            issues.append("Sport column should contain only 'NBA' values")
        
        if 'league' in df.columns and not all(df['league'] == 'NBA'):
            issues.append("League column should contain only 'NBA' values")
        
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
        """Get list of NBA statistics this collector supports"""
        return [
            'points_per_game', 'rebounds_per_game', 'assists_per_game',
            'field_goal_pct', 'three_point_pct', 'free_throw_pct',
            'steals_per_game', 'blocks_per_game', 'turnovers_per_game',
            'plus_minus', 'wins', 'losses', 'win_percentage'
        ]
    
    def get_data_source(self) -> str:
        """Get the name of the data source this collector uses"""
        return "NBA.com (via nba_api)"
    
    def _create_empty_schedule_df(self) -> pd.DataFrame:
        """Create empty DataFrame with schedule columns"""
        cols = [
            'sport', 'league', 'game_id', 'game_date', 
            'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
            'season', 'status', 'source_keys'
        ]
        return pd.DataFrame(columns=cols)  # type: ignore
    
    def _create_empty_games_df(self) -> pd.DataFrame:
        """Create empty DataFrame with games columns"""
        cols = [
            'sport', 'league', 'game_id', 'game_date',
            'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
            'season', 'status', 'home_score', 'away_score', 'source_keys'
        ]
        return pd.DataFrame(columns=cols)  # type: ignore
    
    def _create_empty_stats_df(self) -> pd.DataFrame:
        """Create empty DataFrame with stats columns"""
        cols = [
            'sport', 'league', 'team_id', 'team_name', 'season', 'date',
            'games_played', 'wins', 'losses', 'points_per_game',
            'rebounds_per_game', 'assists_per_game', 'field_goal_pct',
            'three_point_pct', 'free_throw_pct', 'steals_per_game',
            'blocks_per_game', 'turnovers_per_game', 'plus_minus'
        ]
        return pd.DataFrame(columns=cols)  # type: ignore