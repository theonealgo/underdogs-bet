"""
NHL API Integration Module
Fetches goalie data, team stats, and game information from the free NHL API
No API key required - uses official NHL public endpoints
"""

import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHLAPIClient:
    """Client for NHL official API (free, no authentication required)"""
    
    BASE_URL = "https://api-web.nhle.com/v1"
    STATS_URL = "https://api.nhle.com/stats/rest/en"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_todays_schedule(self) -> List[Dict]:
        """Get today's NHL schedule with game IDs"""
        try:
            url = f"{self.BASE_URL}/schedule/now"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            if 'gameWeek' in data:
                for day in data['gameWeek']:
                    if 'games' in day:
                        games.extend(day['games'])
            
            logger.info(f"Retrieved {len(games)} games from NHL API")
            return games
            
        except Exception as e:
            logger.error(f"Error fetching NHL schedule: {e}")
            return []
    
    def get_game_details(self, game_id: int) -> Optional[Dict]:
        """Get detailed game information including starting goalies"""
        try:
            url = f"{self.BASE_URL}/gamecenter/{game_id}/landing"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching game {game_id} details: {e}")
            return None
    
    def get_starting_goalies(self, game_id: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Get starting goalies for a game
        Returns: (home_goalie_name, away_goalie_name)
        """
        game_data = self.get_game_details(game_id)
        
        if not game_data:
            return None, None
        
        home_goalie = None
        away_goalie = None
        
        try:
            # Check for starting goalies in game data
            if 'homeTeam' in game_data and 'goalies' in game_data.get('homeTeam', {}):
                goalies = game_data['homeTeam']['goalies']
                if goalies:
                    home_goalie = goalies[0].get('name', {}).get('default')
            
            if 'awayTeam' in game_data and 'goalies' in game_data.get('awayTeam', {}):
                goalies = game_data['awayTeam']['goalies']
                if goalies:
                    away_goalie = goalies[0].get('name', {}).get('default')
            
            logger.info(f"Game {game_id}: Home goalie: {home_goalie}, Away goalie: {away_goalie}")
            
        except Exception as e:
            logger.error(f"Error parsing goalie data for game {game_id}: {e}")
        
        return home_goalie, away_goalie
    
    def get_goalie_stats(self, season: str = "20252026") -> Dict[str, Dict]:
        """
        Get goalie statistics for current season
        Returns: {goalie_name: {save_pct, gaa, wins, losses, ...}}
        """
        try:
            url = f"{self.STATS_URL}/goalie/summary"
            params = {
                'limit': 100,
                'sort': 'wins',
                'cayenneExp': f'seasonId={season}'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            goalie_stats = {}
            
            if 'data' in data:
                for goalie in data['data']:
                    name = f"{goalie.get('firstName', '')} {goalie.get('lastName', '')}".strip()
                    goalie_stats[name] = {
                        'save_pct': goalie.get('savePct', 0.0),
                        'gaa': goalie.get('goalsAgainstAverage', 0.0),
                        'wins': goalie.get('wins', 0),
                        'losses': goalie.get('losses', 0),
                        'games_played': goalie.get('gamesPlayed', 0),
                        'shutouts': goalie.get('shutouts', 0)
                    }
            
            logger.info(f"Retrieved stats for {len(goalie_stats)} goalies")
            return goalie_stats
            
        except Exception as e:
            logger.error(f"Error fetching goalie stats: {e}")
            return {}
    
    def get_team_stats(self, season: str = "20252026") -> Dict[str, Dict]:
        """
        Get team statistics for current season
        Returns: {team_name: {goals_for, goals_against, ...}}
        """
        try:
            url = f"{self.STATS_URL}/team/summary"
            params = {
                'cayenneExp': f'seasonId={season}'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            team_stats = {}
            
            if 'data' in data:
                for team in data['data']:
                    team_name = team.get('teamFullName', '')
                    team_stats[team_name] = {
                        'games_played': team.get('gamesPlayed', 0),
                        'wins': team.get('wins', 0),
                        'losses': team.get('losses', 0),
                        'goals_for': team.get('goalsFor', 0),
                        'goals_against': team.get('goalsAgainst', 0),
                        'power_play_pct': team.get('powerPlayPct', 0.0),
                        'penalty_kill_pct': team.get('penaltyKillPct', 0.0)
                    }
            
            logger.info(f"Retrieved stats for {len(team_stats)} teams")
            return team_stats
            
        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            return {}


def test_nhl_api():
    """Test the NHL API integration"""
    client = NHLAPIClient()
    
    print("\n=== Testing NHL API ===")
    
    # Test 1: Get today's schedule
    print("\n1. Testing schedule fetch...")
    games = client.get_todays_schedule()
    print(f"Found {len(games)} games")
    if games:
        print(f"Sample game: {games[0].get('awayTeam', {}).get('abbrev')} @ {games[0].get('homeTeam', {}).get('abbrev')}")
    
    # Test 2: Get goalie stats
    print("\n2. Testing goalie stats fetch...")
    goalie_stats = client.get_goalie_stats()
    print(f"Found {len(goalie_stats)} goalies")
    if goalie_stats:
        sample = list(goalie_stats.items())[0]
        print(f"Sample: {sample[0]} - SV%: {sample[1]['save_pct']:.3f}, GAA: {sample[1]['gaa']:.2f}")
    
    # Test 3: Get team stats
    print("\n3. Testing team stats fetch...")
    team_stats = client.get_team_stats()
    print(f"Found {len(team_stats)} teams")
    if team_stats:
        sample = list(team_stats.items())[0]
        print(f"Sample: {sample[0]} - W: {sample[1]['wins']}, L: {sample[1]['losses']}")
    
    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    test_nhl_api()
