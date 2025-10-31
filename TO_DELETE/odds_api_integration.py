"""
The Odds API Integration Module
Fetches betting odds, spreads, and totals for NHL games
Requires ODDS_API_KEY environment variable
"""

import requests
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsAPIClient:
    """Client for The Odds API (requires API key)"""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "icehockey_nhl"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ODDS_API_KEY')
        if not self.api_key:
            logger.warning("No ODDS_API_KEY found. Odds features will be disabled.")
        
        self.session = requests.Session()
    
    def get_odds(self, markets: str = "h2h,spreads,totals") -> List[Dict]:
        """
        Get NHL betting odds
        markets: comma-separated list of markets (h2h=moneyline, spreads, totals)
        """
        if not self.api_key:
            logger.warning("Cannot fetch odds: No API key")
            return []
        
        try:
            url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': markets,
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            games = response.json()
            
            # Log remaining requests
            remaining = response.headers.get('x-requests-remaining')
            if remaining:
                logger.info(f"API requests remaining: {remaining}")
            
            logger.info(f"Retrieved odds for {len(games)} games")
            return games
            
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return []
    
    def parse_odds_for_game(self, home_team: str, away_team: str, odds_data: List[Dict]) -> Dict:
        """
        Parse odds data for a specific game
        Returns: {
            'home_moneyline': -150,
            'away_moneyline': +130,
            'spread': -1.5,
            'total': 6.5,
            'home_spread_odds': -110,
            'away_spread_odds': -110,
            'over_odds': -110,
            'under_odds': -110,
            'num_bookmakers': 5
        }
        """
        result = {
            'home_moneyline': None,
            'away_moneyline': None,
            'spread': None,
            'total': None,
            'home_spread_odds': None,
            'away_spread_odds': None,
            'over_odds': None,
            'under_odds': None,
            'num_bookmakers': 0
        }
        
        # Find matching game
        matching_game = None
        for game in odds_data:
            if (self._team_matches(game.get('home_team', ''), home_team) and 
                self._team_matches(game.get('away_team', ''), away_team)):
                matching_game = game
                break
        
        if not matching_game:
            return result
        
        bookmakers = matching_game.get('bookmakers', [])
        result['num_bookmakers'] = len(bookmakers)
        
        if not bookmakers:
            return result
        
        # Aggregate odds from all bookmakers (take average)
        moneylines_home = []
        moneylines_away = []
        spreads = []
        totals = []
        spread_odds_home = []
        spread_odds_away = []
        over_odds = []
        under_odds = []
        
        for bookmaker in bookmakers:
            markets = bookmaker.get('markets', [])
            
            for market in markets:
                market_key = market.get('key')
                outcomes = market.get('outcomes', [])
                
                if market_key == 'h2h':
                    for outcome in outcomes:
                        if self._team_matches(outcome.get('name', ''), home_team):
                            moneylines_home.append(outcome.get('price'))
                        elif self._team_matches(outcome.get('name', ''), away_team):
                            moneylines_away.append(outcome.get('price'))
                
                elif market_key == 'spreads':
                    for outcome in outcomes:
                        if self._team_matches(outcome.get('name', ''), home_team):
                            point = outcome.get('point')
                            if point is not None:
                                spreads.append(point)
                            spread_odds_home.append(outcome.get('price'))
                        elif self._team_matches(outcome.get('name', ''), away_team):
                            spread_odds_away.append(outcome.get('price'))
                
                elif market_key == 'totals':
                    for outcome in outcomes:
                        if outcome.get('name') == 'Over':
                            totals.append(outcome.get('point'))
                            over_odds.append(outcome.get('price'))
                        elif outcome.get('name') == 'Under':
                            under_odds.append(outcome.get('price'))
        
        # Calculate averages
        if moneylines_home:
            result['home_moneyline'] = sum(moneylines_home) / len(moneylines_home)
        if moneylines_away:
            result['away_moneyline'] = sum(moneylines_away) / len(moneylines_away)
        if spreads:
            result['spread'] = sum(spreads) / len(spreads)
        if totals:
            result['total'] = sum(totals) / len(totals)
        if spread_odds_home:
            result['home_spread_odds'] = sum(spread_odds_home) / len(spread_odds_home)
        if spread_odds_away:
            result['away_spread_odds'] = sum(spread_odds_away) / len(spread_odds_away)
        if over_odds:
            result['over_odds'] = sum(over_odds) / len(over_odds)
        if under_odds:
            result['under_odds'] = sum(under_odds) / len(under_odds)
        
        return result
    
    def _team_matches(self, api_team_name: str, our_team_name: str) -> bool:
        """Check if team names match (handles abbreviations and full names)"""
        api_team = api_team_name.lower().strip()
        our_team = our_team_name.lower().strip()
        
        # Direct match
        if api_team == our_team:
            return True
        
        # Check if one contains the other (e.g., "Tampa Bay Lightning" contains "Lightning")
        if api_team in our_team or our_team in api_team:
            return True
        
        # Common abbreviations mapping
        abbrev_map = {
            'boston bruins': ['bruins', 'bos'],
            'buffalo sabres': ['sabres', 'buf'],
            'detroit red wings': ['red wings', 'det'],
            'florida panthers': ['panthers', 'fla'],
            'montreal canadiens': ['canadiens', 'mtl'],
            'ottawa senators': ['senators', 'ott'],
            'tampa bay lightning': ['lightning', 'tb', 'tbl'],
            'toronto maple leafs': ['maple leafs', 'tor'],
            'carolina hurricanes': ['hurricanes', 'car'],
            'columbus blue jackets': ['blue jackets', 'cbj'],
            'new jersey devils': ['devils', 'nj', 'njd'],
            'new york islanders': ['islanders', 'nyi'],
            'new york rangers': ['rangers', 'nyr'],
            'philadelphia flyers': ['flyers', 'phi'],
            'pittsburgh penguins': ['penguins', 'pit'],
            'washington capitals': ['capitals', 'wsh'],
            'chicago blackhawks': ['blackhawks', 'chi'],
            'colorado avalanche': ['avalanche', 'col'],
            'dallas stars': ['stars', 'dal'],
            'minnesota wild': ['wild', 'min'],
            'nashville predators': ['predators', 'nsh'],
            'st. louis blues': ['blues', 'stl'],
            'winnipeg jets': ['jets', 'wpg'],
            'anaheim ducks': ['ducks', 'ana'],
            'calgary flames': ['flames', 'cgy'],
            'edmonton oilers': ['oilers', 'edm'],
            'los angeles kings': ['kings', 'la', 'lak'],
            'san jose sharks': ['sharks', 'sj', 'sjs'],
            'seattle kraken': ['kraken', 'sea'],
            'vancouver canucks': ['canucks', 'van'],
            'vegas golden knights': ['golden knights', 'vgk', 'vegas']
        }
        
        for full_name, abbreviations in abbrev_map.items():
            if (api_team in [full_name] + abbreviations and 
                our_team in [full_name] + abbreviations):
                return True
        
        return False
    
    def get_usage(self) -> Optional[Dict]:
        """Get API usage statistics"""
        if not self.api_key:
            return None
        
        try:
            url = f"{self.BASE_URL}/sports"
            params = {'apiKey': self.api_key}
            
            response = self.session.get(url, params=params, timeout=10)
            
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            
            return {
                'remaining': remaining,
                'used': used
            }
            
        except Exception as e:
            logger.error(f"Error checking usage: {e}")
            return None


def test_odds_api():
    """Test The Odds API integration"""
    client = OddsAPIClient()
    
    print("\n=== Testing The Odds API ===")
    
    # Test 1: Check API key
    if not client.api_key:
        print("ERROR: No API key found. Set ODDS_API_KEY environment variable.")
        return
    
    print(f"API Key found: {client.api_key[:10]}...")
    
    # Test 2: Get usage
    print("\n1. Checking API usage...")
    usage = client.get_usage()
    if usage:
        print(f"Requests remaining: {usage.get('remaining')}")
        print(f"Requests used: {usage.get('used')}")
    
    # Test 3: Get odds
    print("\n2. Fetching NHL odds...")
    odds = client.get_odds()
    print(f"Found odds for {len(odds)} games")
    
    if odds:
        game = odds[0]
        print(f"\nSample game: {game.get('away_team')} @ {game.get('home_team')}")
        print(f"Commence time: {game.get('commence_time')}")
        print(f"Bookmakers: {len(game.get('bookmakers', []))}")
        
        # Parse the sample game
        parsed = client.parse_odds_for_game(
            game.get('home_team'),
            game.get('away_team'),
            odds
        )
        print(f"\nParsed odds:")
        print(f"  Home ML: {parsed['home_moneyline']}")
        print(f"  Away ML: {parsed['away_moneyline']}")
        print(f"  Spread: {parsed['spread']}")
        print(f"  Total: {parsed['total']}")
    
    print("\n=== Test Complete ===\n")


if __name__ == "__main__":
    test_odds_api()
