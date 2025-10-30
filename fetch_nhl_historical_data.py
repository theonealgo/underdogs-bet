#!/usr/bin/env python3
"""
Fetch NHL Historical Data from API (2022-23, 2023-24, 2024-25 seasons)
Provides 3+ seasons of training data for improved model accuracy
"""

import logging
import requests
import pickle
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "https://api-web.nhle.com/v1/"

# Map API city names to full team names (matching schedule database)
TEAM_NAME_MAP = {
    'Anaheim': 'Anaheim Ducks',
    'Arizona': 'Arizona Coyotes',
    'Boston': 'Boston Bruins',
    'Buffalo': 'Buffalo Sabres',
    'Calgary': 'Calgary Flames',
    'Carolina': 'Carolina Hurricanes',
    'Chicago': 'Chicago Blackhawks',
    'Colorado': 'Colorado Avalanche',
    'Columbus': 'Columbus Blue Jackets',
    'Dallas': 'Dallas Stars',
    'Detroit': 'Detroit Red Wings',
    'Edmonton': 'Edmonton Oilers',
    'Florida': 'Florida Panthers',
    'Los Angeles': 'Los Angeles Kings',
    'Minnesota': 'Minnesota Wild',
    'Montréal': 'Montreal Canadiens',
    'Montreal': 'Montreal Canadiens',
    'Nashville': 'Nashville Predators',
    'New Jersey': 'New Jersey Devils',
    'New York': 'New York Islanders',  # Default for NY
    'Islanders': 'New York Islanders',
    'Rangers': 'New York Rangers',
    'Ottawa': 'Ottawa Senators',
    'Philadelphia': 'Philadelphia Flyers',
    'Pittsburgh': 'Pittsburgh Penguins',
    'San Jose': 'San Jose Sharks',
    'Seattle': 'Seattle Kraken',
    'St. Louis': 'St Louis Blues',
    'Tampa Bay': 'Tampa Bay Lightning',
    'Toronto': 'Toronto Maple Leafs',
    'Utah': 'Utah Mammoth',
    'Vancouver': 'Vancouver Canucks',
    'Vegas': 'Vegas Golden Knights',
    'Washington': 'Washington Capitals',
    'Winnipeg': 'Winnipeg Jets'
}

def fetch_season_games(start_date, end_date, season_name):
    """Fetch all completed games for a specific season by date range"""
    logger.info(f"Fetching {season_name} season data ({start_date} to {end_date})...")
    
    from datetime import datetime, timedelta
    
    games = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            url = f"{API_URL}schedule/{date_str}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                schedule = response.json()
                
                game_weeks = schedule.get('gameWeek', [])
                
                for week in game_weeks:
                    for game in week.get('games', []):
                        game_state = game.get('gameState', '')
                        
                        if game_state in ['OFF', 'FINAL']:
                            away_team = game.get('awayTeam', {})
                            home_team = game.get('homeTeam', {})
                            
                            away_score = away_team.get('score')
                            home_score = home_team.get('score')
                            
                            if away_score is not None and home_score is not None:
                                game_date_str = game.get('gameDate', '')
                                if 'T' in game_date_str:
                                    game_date_formatted = game_date_str.split('T')[0]
                                else:
                                    game_date_formatted = date_str
                                
                                # Get city names from API
                                home_city = home_team.get('placeName', {}).get('default', '')
                                away_city = away_team.get('placeName', {}).get('default', '')
                                
                                # Map to full team names
                                home_full = TEAM_NAME_MAP.get(home_city, home_city)
                                away_full = TEAM_NAME_MAP.get(away_city, away_city)
                                
                                game_data = {
                                    'match_id': game.get('id'),
                                    'date': game_date_formatted,
                                    'home_team': home_full,
                                    'away_team': away_full,
                                    'home_score': int(home_score),
                                    'away_score': int(away_score),
                                    'season': season_name
                                }
                                games.append(game_data)
                
                time.sleep(0.1)
            
        except Exception as e:
            pass
        
        current_date += timedelta(days=7)
    
    logger.info(f"  ✓ Fetched {len(games)} completed games from {season_name}")
    return games

def main():
    """Fetch historical NHL data from multiple seasons"""
    logger.info("="*70)
    logger.info("FETCHING NHL HISTORICAL DATA FROM API")
    logger.info("="*70)
    
    all_games = []
    
    seasons = [
        ("2022-10-07", "2023-04-13", "2022-23"),
        ("2023-10-10", "2024-04-18", "2023-24"),
        ("2024-10-04", "2025-04-17", "2024-25")
    ]
    
    for start_date, end_date, season_name in seasons:
        season_games = fetch_season_games(start_date, end_date, season_name)
        all_games.extend(season_games)
    
    logger.info("="*70)
    logger.info(f"TOTAL HISTORICAL GAMES FETCHED: {len(all_games)}")
    logger.info("="*70)
    
    if len(all_games) > 0:
        output_file = 'nhl_historical_data.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(all_games, f)
        logger.info(f"✓ Saved historical data to {output_file}")
        
        logger.info("\nBreakdown by season:")
        for season_data in seasons:
            season_name = season_data[2]
            season_count = len([g for g in all_games if g.get('season') == season_name])
            logger.info(f"  {season_name}: {season_count} games")
        
        logger.info("\nNext step: Update train_nhl_models.py to use this historical data")
    else:
        logger.error("No historical data fetched")

if __name__ == "__main__":
    main()
