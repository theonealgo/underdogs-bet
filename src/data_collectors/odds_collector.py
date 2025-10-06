"""
Odds data collector for real betting lines
"""
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

class OddsCollector:
    """Collects real betting odds from The Odds API"""
    
    def __init__(self):
        self.api_key = os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            self.logger.warning("ODDS_API_KEY not found - odds collection disabled")
    
    def get_mlb_odds(self) -> pd.DataFrame:
        """Get current MLB odds including moneyline, spreads, and totals"""
        if not self.api_key:
            self.logger.warning("No API key - returning empty odds data")
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/sports/baseball_mlb/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',  # Get all three markets
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            odds_data = response.json()
            
            if not odds_data:
                self.logger.info("No MLB odds available")
                return pd.DataFrame()
            
            # Parse odds data
            parsed_odds = []
            for game in odds_data:
                game_date = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00')).date()
                away_team = game['away_team']
                home_team = game['home_team']
                
                # Get best odds from all bookmakers for all markets
                best_away_ml = None
                best_home_ml = None
                best_away_spread = None
                best_away_spread_odds = None
                best_home_spread = None
                best_home_spread_odds = None
                best_over_line = None
                best_over_odds = None
                best_under_line = None
                best_under_odds = None
                bookmaker_count = 0
                
                for bookmaker in game['bookmakers']:
                    bookmaker_count += 1
                    for market in bookmaker['markets']:
                        # Moneyline (h2h)
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == away_team:
                                    if best_away_ml is None or outcome['price'] > best_away_ml:
                                        best_away_ml = outcome['price']
                                elif outcome['name'] == home_team:
                                    if best_home_ml is None or outcome['price'] > best_home_ml:
                                        best_home_ml = outcome['price']
                        
                        # Spreads
                        elif market['key'] == 'spreads':
                            for outcome in market['outcomes']:
                                if outcome['name'] == away_team:
                                    if best_away_spread_odds is None or outcome['price'] > best_away_spread_odds:
                                        best_away_spread = outcome.get('point')
                                        best_away_spread_odds = outcome['price']
                                elif outcome['name'] == home_team:
                                    if best_home_spread_odds is None or outcome['price'] > best_home_spread_odds:
                                        best_home_spread = outcome.get('point')
                                        best_home_spread_odds = outcome['price']
                        
                        # Totals (over/under)
                        elif market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over':
                                    if best_over_odds is None or outcome['price'] > best_over_odds:
                                        best_over_line = outcome.get('point')
                                        best_over_odds = outcome['price']
                                elif outcome['name'] == 'Under':
                                    if best_under_odds is None or outcome['price'] > best_under_odds:
                                        best_under_line = outcome.get('point')
                                        best_under_odds = outcome['price']
                
                if best_away_ml is not None and best_home_ml is not None:
                    parsed_odds.append({
                        'game_date': game_date,
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_odds': best_away_ml,
                        'home_odds': best_home_ml,
                        'away_implied_prob': self._american_to_probability(best_away_ml),
                        'home_implied_prob': self._american_to_probability(best_home_ml),
                        'away_spread': best_away_spread,
                        'away_spread_odds': best_away_spread_odds,
                        'home_spread': best_home_spread,
                        'home_spread_odds': best_home_spread_odds,
                        'total_line': best_over_line or best_under_line,
                        'over_odds': best_over_odds,
                        'under_odds': best_under_odds,
                        'bookmaker_count': bookmaker_count,
                        'sport': 'MLB',
                        'collected_at': datetime.now().isoformat()
                    })
            
            df = pd.DataFrame(parsed_odds)
            self.logger.info(f"Collected odds for {len(df)} MLB games")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error collecting odds: {str(e)}")
            return pd.DataFrame()
    
    def get_sport_odds(self, sport_key: str) -> pd.DataFrame:
        """Get odds for any sport supported by The Odds API"""
        if not self.api_key:
            return pd.DataFrame()
        
        sport_mappings = {
            'MLB': 'baseball_mlb',
            'NBA': 'basketball_nba',
            'NFL': 'americanfootball_nfl',
            'NHL': 'icehockey_nhl',
            'NCAAF': 'americanfootball_ncaaf',
            'NCAAB': 'basketball_ncaab',
            'WNBA': 'basketball_wnba'
        }
        
        api_sport_key = sport_mappings.get(sport_key)
        if not api_sport_key:
            self.logger.warning(f"Sport {sport_key} not supported for odds")
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/sports/{api_sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',  # Get all three markets
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            odds_data = response.json()
            
            if not odds_data:
                return pd.DataFrame()
            
            # Parse odds (same logic as MLB)
            parsed_odds = []
            for game in odds_data:
                game_date = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00')).date()
                away_team = game['away_team']
                home_team = game['home_team']
                
                best_away_ml = None
                best_home_ml = None
                best_away_spread = None
                best_away_spread_odds = None
                best_home_spread = None
                best_home_spread_odds = None
                best_over_line = None
                best_over_odds = None
                best_under_line = None
                best_under_odds = None
                bookmaker_count = 0
                
                for bookmaker in game['bookmakers']:
                    bookmaker_count += 1
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == away_team:
                                    if best_away_ml is None or outcome['price'] > best_away_ml:
                                        best_away_ml = outcome['price']
                                elif outcome['name'] == home_team:
                                    if best_home_ml is None or outcome['price'] > best_home_ml:
                                        best_home_ml = outcome['price']
                        elif market['key'] == 'spreads':
                            for outcome in market['outcomes']:
                                if outcome['name'] == away_team:
                                    if best_away_spread_odds is None or outcome['price'] > best_away_spread_odds:
                                        best_away_spread = outcome.get('point')
                                        best_away_spread_odds = outcome['price']
                                elif outcome['name'] == home_team:
                                    if best_home_spread_odds is None or outcome['price'] > best_home_spread_odds:
                                        best_home_spread = outcome.get('point')
                                        best_home_spread_odds = outcome['price']
                        elif market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over':
                                    if best_over_odds is None or outcome['price'] > best_over_odds:
                                        best_over_line = outcome.get('point')
                                        best_over_odds = outcome['price']
                                elif outcome['name'] == 'Under':
                                    if best_under_odds is None or outcome['price'] > best_under_odds:
                                        best_under_line = outcome.get('point')
                                        best_under_odds = outcome['price']
                
                if best_away_ml is not None and best_home_ml is not None:
                    parsed_odds.append({
                        'game_date': game_date,
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_odds': best_away_ml,
                        'home_odds': best_home_ml,
                        'away_implied_prob': self._american_to_probability(best_away_ml),
                        'home_implied_prob': self._american_to_probability(best_home_ml),
                        'away_spread': best_away_spread,
                        'away_spread_odds': best_away_spread_odds,
                        'home_spread': best_home_spread,
                        'home_spread_odds': best_home_spread_odds,
                        'total_line': best_over_line or best_under_line,
                        'over_odds': best_over_odds,
                        'under_odds': best_under_odds,
                        'bookmaker_count': bookmaker_count,
                        'sport': sport_key,
                        'collected_at': datetime.now().isoformat()
                    })
            
            df = pd.DataFrame(parsed_odds)
            self.logger.info(f"Collected odds for {len(df)} {sport_key} games")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting {sport_key} odds: {str(e)}")
            return pd.DataFrame()
    
    def _american_to_probability(self, american_odds: int) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def calculate_roi_potential(self, prediction_prob: float, market_odds: int) -> Dict:
        """Calculate ROI potential for a prediction vs market odds"""
        if market_odds is None or prediction_prob is None:
            return {'roi_potential': None, 'edge': None, 'profitable': False}
        
        market_prob = self._american_to_probability(market_odds)
        edge = prediction_prob - market_prob
        
        # Calculate expected ROI for $100 bet
        if market_odds > 0:
            potential_profit = 100 * (market_odds / 100)
        else:
            potential_profit = 100 * (100 / abs(market_odds))
        
        expected_roi = (prediction_prob * potential_profit) - (100 * (1 - prediction_prob))
        roi_percentage = expected_roi / 100
        
        return {
            'roi_potential': roi_percentage,
            'edge': edge,
            'profitable': edge > 0.02,  # Need 2% edge minimum
            'market_prob': market_prob,
            'prediction_prob': prediction_prob
        }