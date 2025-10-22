#!/usr/bin/env python3
"""
Map NHL teams to their starting goalies using roster data
Uses nhl-api-py to get actual team rosters
"""

import sqlite3
from nhlpy import NHLClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions.db'

# Team abbreviation mapping (NHL API abbrev -> Database team name)
TEAM_NAME_MAPPING = {
    'ANA': 'Anaheim Ducks',
    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',
    'CAR': 'Carolina Hurricanes',
    'CBJ': 'Columbus Blue Jackets',
    'CGY': 'Calgary Flames',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'DAL': 'Dallas Stars',
    'DET': 'Detroit Red Wings',
    'EDM': 'Edmonton Oilers',
    'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings',
    'MIN': 'Minnesota Wild',
    'MTL': 'Montreal Canadiens',
    'NJD': 'New Jersey Devils',
    'NSH': 'Nashville Predators',
    'NYI': 'New York Islanders',
    'NYR': 'New York Rangers',
    'OTT': 'Ottawa Senators',
    'PHI': 'Philadelphia Flyers',
    'PIT': 'Pittsburgh Penguins',
    'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks',
    'STL': 'St. Louis Blues',
    'TBL': 'Tampa Bay Lightning',
    'TOR': 'Toronto Maple Leafs',
    'UTA': 'Utah Mammoth',
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights',
    'WPG': 'Winnipeg Jets',
    'WSH': 'Washington Capitals'
}

def create_team_goalie_table():
    """Create table mapping teams to their primary goalies"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_goalies (
            team_name TEXT PRIMARY KEY,
            goalie_name TEXT,
            team_abbr TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def map_teams_to_goalies():
    """Map teams to their starting goalies using roster data"""
    client = NHLClient()
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing mappings
    cursor.execute('DELETE FROM team_goalies')
    
    # Get all teams
    teams = client.teams.teams()
    
    mapped_count = 0
    for team in teams:
        team_abbr = team.get('abbrev', team.get('abbr', ''))
        team_full_name = TEAM_NAME_MAPPING.get(team_abbr)
        
        if not team_full_name:
            logger.warning(f"No mapping for {team_abbr}")
            continue
        
        try:
            # Get team roster
            roster = client.teams.team_roster(team_abbr=team_abbr, season="20252026")
            goalies = roster.get('goalies', [])
            
            if not goalies:
                logger.warning(f"No goalies found for {team_abbr}")
                continue
            
            # Try to find a goalie with stats (starter with most games played)
            mapped = False
            for goalie in goalies:
                first_name = goalie.get('firstName', {}).get('default', '')
                last_name = goalie.get('lastName', {}).get('default', '')
                goalie_full_name = f"{first_name} {last_name}".strip()
                
                # Check if this goalie exists in goalie_stats
                goalie_stats = cursor.execute(
                    'SELECT games_played FROM goalie_stats WHERE goalie_name = ?',
                    (goalie_full_name,)
                ).fetchone()
                
                if goalie_stats:
                    cursor.execute('''
                        INSERT INTO team_goalies (team_name, goalie_name, team_abbr)
                        VALUES (?, ?, ?)
                    ''', (team_full_name, goalie_full_name, team_abbr))
                    
                    print(f"✓ {team_abbr} ({team_full_name}): {goalie_full_name} ({goalie_stats[0]} GP)")
                    mapped_count += 1
                    mapped = True
                    break
            
            if not mapped:
                logger.warning(f"No goalie with stats found for {team_abbr}")
                
        except Exception as e:
            logger.error(f"Error mapping {team_abbr}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    return mapped_count

if __name__ == '__main__':
    print("\n" + "="*70)
    print("MAPPING NHL TEAMS TO STARTING GOALIES")
    print("="*70)
    
    create_team_goalie_table()
    count = map_teams_to_goalies()
    
    print(f"\n✅ Mapped {count}/32 teams to their starting goalies")
