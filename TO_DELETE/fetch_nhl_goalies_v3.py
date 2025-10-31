#!/usr/bin/env python3
"""
Fetch ALL NHL Goalies using nhl-api-py with roster data
Combines stats for goalies with games + roster data for all others
"""

import sqlite3
from nhlpy import NHLClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions.db'

# League average stats for goalies without game data
LEAGUE_AVG_SAVE_PCT = 0.91
LEAGUE_AVG_GAA = 2.80

def create_goalie_stats_table():
    """Create table for goalie statistics"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goalie_stats (
            goalie_name TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            save_pct REAL,
            gaa REAL,
            wins INTEGER,
            losses INTEGER,
            games_played INTEGER,
            shutouts INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✓ Goalie stats table ready")

def fetch_and_store_goalie_stats():
    """
    Fetch goalie stats from NHL API and rosters
    Uses stats API for goalies with games, fills gaps from team rosters
    """
    client = NHLClient()
    
    try:
        # Step 1: Get goalie stats for those who have played
        goalie_stats_list = client.stats.goalie_stats_summary(
            start_season="20252026", 
            end_season="20252026"
        )
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM goalie_stats')
        
        # Insert goalies with stats
        stats_count = 0
        goalie_names_with_stats = set()
        
        for goalie in goalie_stats_list:
            full_name = goalie.get('goalieFullName', '')
            goalie_names_with_stats.add(full_name)
            
            # Split name into first and last
            name_parts = full_name.split()
            first_name = name_parts[0] if len(name_parts) > 0 else ''
            last_name = name_parts[-1] if len(name_parts) > 0 else ''
            
            cursor.execute('''
                INSERT OR REPLACE INTO goalie_stats 
                (goalie_name, first_name, last_name, save_pct, gaa, wins, losses, games_played, shutouts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                full_name,
                first_name,
                last_name,
                goalie.get('savePct', LEAGUE_AVG_SAVE_PCT),
                goalie.get('goalsAgainstAverage', LEAGUE_AVG_GAA),
                goalie.get('wins', 0),
                goalie.get('losses', 0),
                goalie.get('gamesPlayed', 0),
                goalie.get('shutouts', 0)
            ))
            
            stats_count += 1
        
        logger.info(f"✓ Added {stats_count} goalies with game stats")
        
        # Step 2: Get all team rosters and add goalies without stats
        teams = client.teams.teams()
        roster_count = 0
        
        for team in teams:
            team_abbr = team.get('abbrev', team.get('abbr', ''))
            if not team_abbr:
                continue
            
            try:
                roster = client.teams.team_roster(team_abbr=team_abbr, season="20252026")
                goalies = roster.get('goalies', [])
                
                for goalie in goalies:
                    first_name = goalie.get('firstName', {}).get('default', '')
                    last_name = goalie.get('lastName', {}).get('default', '')
                    full_name = f"{first_name} {last_name}".strip()
                    
                    if full_name not in goalie_names_with_stats:
                        # Add goalie with league average stats
                        cursor.execute('''
                            INSERT OR IGNORE INTO goalie_stats 
                            (goalie_name, first_name, last_name, save_pct, gaa, wins, losses, games_played, shutouts)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            full_name,
                            first_name,
                            last_name,
                            LEAGUE_AVG_SAVE_PCT,
                            LEAGUE_AVG_GAA,
                            0,
                            0,
                            0,
                            0
                        ))
                        roster_count += 1
                        goalie_names_with_stats.add(full_name)
                
            except Exception as e:
                logger.warning(f"Error fetching roster for {team_abbr}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        total = stats_count + roster_count
        logger.info(f"✓ Added {roster_count} additional goalies from rosters (league avg stats)")
        logger.info(f"\n✅ Total: {total} goalies ({stats_count} with stats, {roster_count} from rosters)")
        
        # Show first 25 goalies with stats
        print(f"\n📊 Top goalies by save %:")
        for i, goalie in enumerate(goalie_stats_list[:25]):
            name = goalie.get('goalieFullName', 'Unknown')
            sv_pct = goalie.get('savePct', 0)
            gaa = goalie.get('goalsAgainstAverage', 0)
            wins = goalie.get('wins', 0)
            losses = goalie.get('losses', 0)
            print(f"  {i+1}. {name}: {sv_pct:.3f} SV%, {gaa:.2f} GAA, {wins}W-{losses}L")
        
        return total
        
    except Exception as e:
        logger.error(f"Error fetching goalie data: {e}")
        return 0

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FETCHING NHL GOALIES USING NHL-API-PY")
    print("Combining stats API + roster data for complete coverage")
    print("="*70)
    
    create_goalie_stats_table()
    count = fetch_and_store_goalie_stats()
    
    if count > 0:
        print(f"\n✅ Successfully fetched {count} goalie records")
    else:
        print("\n❌ Failed to fetch goalie data")
