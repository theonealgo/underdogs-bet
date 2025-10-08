"""
Utility to parse pasted game data from ESPN and other sources.
"""

import re
from datetime import datetime, date
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def parse_pasted_games(pasted_data: str, sport: str, game_date: Optional[date] = None) -> List[Dict]:
    """
    Parse pasted game data from ESPN or other sources.
    
    Extracts team matchups in format like:
    - "Montreal @ Toronto"
    - "SEA @ DET"  
    - "Mariners @ Tigers"
    
    Args:
        pasted_data: Raw pasted text from ESPN or official sources
        sport: Sport code (NHL, MLB, NBA, etc.)
        game_date: Date for the games (defaults to today)
    
    Returns:
        List of game dictionaries with home/away teams
    """
    if not pasted_data or not pasted_data.strip():
        return []
    
    if game_date is None:
        game_date = date.today()
    
    games = []
    
    # Split into lines and look for matchup patterns
    lines = pasted_data.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for "@" pattern indicating away @ home
        if '@' in line:
            # Try to extract teams around the @
            parts = line.split('@')
            if len(parts) == 2:
                away_team = parts[0].strip()
                home_team = parts[1].strip()
                
                # Clean up team names - remove extra text
                away_team = clean_team_name(away_team)
                home_team = clean_team_name(home_team)
                
                if away_team and home_team:
                    game_id = f"{sport}_{game_date.strftime('%Y%m%d')}_{away_team}_{home_team}"
                    
                    games.append({
                        'sport': sport,
                        'league': sport,
                        'game_id': game_id,
                        'game_date': game_date,
                        'home_team_id': home_team,
                        'away_team_id': away_team,
                        'home_team_name': home_team,
                        'away_team_name': away_team,
                        'season': game_date.year,
                        'status': 'scheduled',
                        'source_keys': '{"source": "manual_paste"}'
                    })
                    logger.info(f"Parsed game: {away_team} @ {home_team}")
    
    return games


def clean_team_name(team: str) -> str:
    """
    Clean up team name extracted from pasted data.
    Remove extra whitespace, parentheses content, and common noise.
    """
    if not team:
        return ''
    
    # Remove content in parentheses (like records)
    team = re.sub(r'\([^)]*\)', '', team)
    
    # Remove common noise words at the start
    team = re.sub(r'^(tickets|line|ml|odds|gamecast|time|tv)\s*:?\s*', '', team, flags=re.IGNORECASE)
    
    # Remove trailing noise
    team = re.sub(r'\s+(tickets|as|low|gamecast).*$', '', team, flags=re.IGNORECASE)
    
    # Clean up whitespace
    team = ' '.join(team.split())
    
    return team.strip()
