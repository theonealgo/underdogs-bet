#!/usr/bin/env python3
"""
Import comprehensive goalie data from user-provided file
Uses advanced metrics for better prediction accuracy
"""

import sqlite3
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions.db'
DATA_FILE = 'attached_assets/Pasted-playerId-season-name-team-position-situation-games-played-icetime-xGoals-goals-unblocked-shot-attemp-1761174113062_1761174113063.txt'

# NHL team abbreviation to full name mapping
TEAM_MAPPING = {
    'NYR': 'New York Rangers',
    'VAN': 'Vancouver Canucks',
    'UTA': 'Utah Hockey Club',
    'DAL': 'Dallas Stars',
    'TBL': 'Tampa Bay Lightning',
    'PIT': 'Pittsburgh Penguins',
    'OTT': 'Ottawa Senators',
    'NYI': 'New York Islanders',
    'WPG': 'Winnipeg Jets',
    'BOS': 'Boston Bruins',
    'CHI': 'Chicago Blackhawks',
    'DET': 'Detroit Red Wings',
    'NJD': 'New Jersey Devils',
    'ANA': 'Anaheim Ducks',
    'ARI': 'Arizona Coyotes',
    'BUF': 'Buffalo Sabres',
    'CGY': 'Calgary Flames',
    'CAR': 'Carolina Hurricanes',
    'COL': 'Colorado Avalanche',
    'CBJ': 'Columbus Blue Jackets',
    'EDM': 'Edmonton Oilers',
    'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings',
    'MIN': 'Minnesota Wild',
    'MTL': 'Montreal Canadiens',
    'NSH': 'Nashville Predators',
    'PHI': 'Philadelphia Flyers',
    'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks',
    'STL': 'St. Louis Blues',
    'TOR': 'Toronto Maple Leafs',
    'VGK': 'Vegas Golden Knights',
    'WSH': 'Washington Capitals'
}

def create_advanced_goalie_table():
    """Create table for advanced goalie statistics"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Drop existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS advanced_goalie_stats')
    
    cursor.execute('''
        CREATE TABLE advanced_goalie_stats (
            player_id INTEGER,
            goalie_name TEXT,
            team_abbr TEXT,
            team_name TEXT,
            season INTEGER,
            games_played INTEGER,
            icetime REAL,
            
            -- Basic stats
            goals_against INTEGER,
            shots_on_goal INTEGER,
            save_pct REAL,
            gaa REAL,
            
            -- Expected goals
            xgoals_against REAL,
            goals_saved_above_expected REAL,
            
            -- Rebound control
            xrebounds REAL,
            rebounds INTEGER,
            rebound_control_pct REAL,
            
            -- Danger level stats
            low_danger_shots INTEGER,
            medium_danger_shots INTEGER,
            high_danger_shots INTEGER,
            low_danger_sv_pct REAL,
            medium_danger_sv_pct REAL,
            high_danger_sv_pct REAL,
            
            -- Advanced metrics
            high_danger_goals_allowed INTEGER,
            freeze_pct REAL,
            
            PRIMARY KEY (player_id, team_abbr)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✓ Advanced goalie stats table created")

def import_goalie_data():
    """Import goalie data from TSV file"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    goalies_imported = 0
    goalies_skipped = 0
    
    with open(DATA_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            # Only import "all" situation (overall stats)
            if row['situation'] != 'all':
                continue
            
            # Skip if position is not goalie
            if row['position'] != 'G':
                continue
            
            player_id = int(row['playerId'])
            goalie_name = row['name'].strip()
            team_abbr = row['team'].strip()
            team_name = TEAM_MAPPING.get(team_abbr, team_abbr)
            season = int(row['season'])
            games_played = int(row['games_played'])
            
            # Skip goalies with 0 games
            if games_played == 0:
                goalies_skipped += 1
                continue
            
            # Parse numeric fields (convert floats to ints where needed)
            icetime = float(row['icetime'])
            goals_against = int(float(row['goals']))
            shots_on_goal = int(float(row['ongoal']))
            xgoals = float(row['xGoals'])
            
            # Calculate save percentage
            if shots_on_goal > 0:
                save_pct = (shots_on_goal - goals_against) / shots_on_goal
                gaa = (goals_against / (icetime / 60)) * 60 if icetime > 0 else 0
            else:
                save_pct = 0.0
                gaa = 0.0
            
            # Goals saved above expected
            gsax = xgoals - goals_against
            
            # Rebound control
            xrebounds = float(row['xRebounds'])
            rebounds = int(float(row['rebounds']))
            rebound_control_pct = (xrebounds - rebounds) / xrebounds if xrebounds > 0 else 0
            
            # Danger level stats
            low_danger_shots = int(float(row['lowDangerShots']))
            medium_danger_shots = int(float(row['mediumDangerShots']))
            high_danger_shots = int(float(row['highDangerShots']))
            
            low_danger_goals = int(float(row['lowDangerGoals']))
            medium_danger_goals = int(float(row['mediumDangerGoals']))
            high_danger_goals = int(float(row['highDangerGoals']))
            
            # Calculate save % by danger level
            low_sv_pct = (low_danger_shots - low_danger_goals) / low_danger_shots if low_danger_shots > 0 else 0
            medium_sv_pct = (medium_danger_shots - medium_danger_goals) / medium_danger_shots if medium_danger_shots > 0 else 0
            high_sv_pct = (high_danger_shots - high_danger_goals) / high_danger_shots if high_danger_shots > 0 else 0
            
            # Freeze percentage
            xfreeze = float(row['xFreeze'])
            freeze = int(float(row['freeze']))
            freeze_pct = freeze / xfreeze if xfreeze > 0 else 0
            
            # Insert into database
            cursor.execute('''
                INSERT OR REPLACE INTO advanced_goalie_stats
                (player_id, goalie_name, team_abbr, team_name, season, games_played, icetime,
                 goals_against, shots_on_goal, save_pct, gaa,
                 xgoals_against, goals_saved_above_expected,
                 xrebounds, rebounds, rebound_control_pct,
                 low_danger_shots, medium_danger_shots, high_danger_shots,
                 low_danger_sv_pct, medium_danger_sv_pct, high_danger_sv_pct,
                 high_danger_goals_allowed, freeze_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_id, goalie_name, team_abbr, team_name, season, games_played, icetime,
                goals_against, shots_on_goal, save_pct, gaa,
                xgoals, gsax,
                xrebounds, rebounds, rebound_control_pct,
                low_danger_shots, medium_danger_shots, high_danger_shots,
                low_sv_pct, medium_sv_pct, high_sv_pct,
                high_danger_goals, freeze_pct
            ))
            
            goalies_imported += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"✓ Imported {goalies_imported} goalies")
    logger.info(f"  Skipped {goalies_skipped} goalies with 0 games")
    return goalies_imported

def update_goalie_stats_table():
    """Update the main goalie_stats table with data from advanced stats"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing goalie_stats
    cursor.execute('DELETE FROM goalie_stats')
    
    # Copy data from advanced_goalie_stats
    cursor.execute('''
        INSERT INTO goalie_stats
        (goalie_name, first_name, last_name, save_pct, gaa, wins, losses, games_played, shutouts)
        SELECT 
            goalie_name,
            SUBSTR(goalie_name, 1, INSTR(goalie_name || ' ', ' ') - 1) as first_name,
            SUBSTR(goalie_name, INSTR(goalie_name, ' ') + 1) as last_name,
            save_pct,
            gaa,
            0 as wins,
            0 as losses,
            games_played,
            0 as shutouts
        FROM advanced_goalie_stats
    ''')
    
    updated = cursor.rowcount
    conn.commit()
    conn.close()
    
    logger.info(f"✓ Updated goalie_stats table with {updated} records")
    return updated

def show_top_goalies():
    """Display top goalies by save percentage"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT goalie_name, team_abbr, games_played, save_pct, gaa, 
               goals_saved_above_expected, high_danger_sv_pct
        FROM advanced_goalie_stats
        ORDER BY save_pct DESC
        LIMIT 15
    ''')
    
    print("\n📊 Top 15 Goalies by Save %:")
    print(f"{'Name':<25} {'Team':<5} {'GP':<4} {'SV%':<6} {'GAA':<5} {'GSAX':<6} {'HD SV%':<6}")
    print("-" * 65)
    
    for row in cursor.fetchall():
        name, team, gp, sv_pct, gaa, gsax, hd_sv = row
        print(f"{name:<25} {team:<5} {gp:<4} {sv_pct:.3f}  {gaa:.2f}  {gsax:>5.1f}  {hd_sv:.3f}")
    
    conn.close()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("IMPORTING ADVANCED NHL GOALIE DATA")
    print("="*70)
    
    create_advanced_goalie_table()
    count = import_goalie_data()
    
    if count > 0:
        update_goalie_stats_table()
        show_top_goalies()
        
        print(f"\n✅ Successfully imported {count} goalie records")
        print("\n💡 Advanced metrics available:")
        print("   - Goals Saved Above Expected (GSAX)")
        print("   - High/Medium/Low Danger Save %")
        print("   - Rebound Control %")
        print("   - Freeze %")
    else:
        print("\n❌ Failed to import goalie data")
