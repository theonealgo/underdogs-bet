#!/usr/bin/env python3
"""
Import comprehensive NHL skater data from user-provided file
Uses advanced metrics for better team performance prediction
"""

import sqlite3
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions.db'
DATA_FILE = 'attached_assets/Pasted-playerId-season-name-team-position-situation-games-played-icetime-shifts-gameScore-onIce-xGoalsPerce-1761174989036_1761174989044.txt'

# NHL team abbreviation mapping (same as goalie data)
TEAM_MAPPING = {
    'NYR': 'New York Rangers', 'VAN': 'Vancouver Canucks', 'UTA': 'Utah Hockey Club',
    'DAL': 'Dallas Stars', 'TBL': 'Tampa Bay Lightning', 'PIT': 'Pittsburgh Penguins',
    'OTT': 'Ottawa Senators', 'NYI': 'New York Islanders', 'WPG': 'Winnipeg Jets',
    'BOS': 'Boston Bruins', 'CHI': 'Chicago Blackhawks', 'DET': 'Detroit Red Wings',
    'NJD': 'New Jersey Devils', 'ANA': 'Anaheim Ducks', 'BUF': 'Buffalo Sabres',
    'CGY': 'Calgary Flames', 'CAR': 'Carolina Hurricanes', 'COL': 'Colorado Avalanche',
    'CBJ': 'Columbus Blue Jackets', 'EDM': 'Edmonton Oilers', 'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings', 'MIN': 'Minnesota Wild', 'MTL': 'Montreal Canadiens',
    'NSH': 'Nashville Predators', 'PHI': 'Philadelphia Flyers', 'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks', 'STL': 'St. Louis Blues', 'TOR': 'Toronto Maple Leafs',
    'VGK': 'Vegas Golden Knights', 'WSH': 'Washington Capitals'
}

def create_skater_stats_table():
    """Create table for advanced skater statistics"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Drop existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS advanced_skater_stats')
    
    cursor.execute('''
        CREATE TABLE advanced_skater_stats (
            player_id INTEGER,
            player_name TEXT,
            team_abbr TEXT,
            team_name TEXT,
            position TEXT,
            season INTEGER,
            games_played INTEGER,
            icetime REAL,
            
            -- Performance metrics
            game_score REAL,
            onice_xgoals_pct REAL,
            onice_corsi_pct REAL,
            onice_fenwick_pct REAL,
            
            -- Offensive production
            goals INTEGER,
            primary_assists INTEGER,
            secondary_assists INTEGER,
            points INTEGER,
            shots_on_goal INTEGER,
            xgoals REAL,
            
            -- Plus/Minus style metrics
            office_xgoals_pct REAL,
            
            PRIMARY KEY (player_id, team_abbr)
        )
    ''')
    
    # Create team aggregation table
    cursor.execute('DROP TABLE IF EXISTS team_skater_aggregates')
    cursor.execute('''
        CREATE TABLE team_skater_aggregates (
            team_abbr TEXT PRIMARY KEY,
            team_name TEXT,
            
            -- Aggregate offensive metrics
            avg_game_score REAL,
            avg_onice_xgoals_pct REAL,
            avg_onice_corsi_pct REAL,
            total_goals INTEGER,
            total_points INTEGER,
            total_xgoals REAL,
            
            -- Top performer stats
            top_scorer_name TEXT,
            top_scorer_points INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("✓ Advanced skater stats tables created")

def import_skater_data():
    """Import skater data from TSV file"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    skaters_imported = 0
    skaters_skipped = 0
    
    with open(DATA_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            # Only import "all" situation (overall stats)
            if row['situation'] != 'all':
                continue
            
            player_id = int(row['playerId'])
            player_name = row['name'].strip()
            team_abbr = row['team'].strip()
            team_name = TEAM_MAPPING.get(team_abbr, team_abbr)
            position = row['position'].strip()
            season = int(row['season'])
            games_played = int(float(row['games_played']))
            
            # Skip players with 0 games
            if games_played == 0:
                skaters_skipped += 1
                continue
            
            # Parse key metrics
            icetime = float(row['icetime']) if row['icetime'] else 0.0
            game_score = float(row['gameScore']) if row['gameScore'] else 0.0
            
            # On-ice percentages
            onice_xgoals_pct = float(row['onIce_xGoalsPercentage']) if row['onIce_xGoalsPercentage'] else 0.5
            office_xgoals_pct = float(row['offIce_xGoalsPercentage']) if row['offIce_xGoalsPercentage'] else 0.5
            onice_corsi_pct = float(row['onIce_corsiPercentage']) if row['onIce_corsiPercentage'] else 0.5
            onice_fenwick_pct = float(row['onIce_fenwickPercentage']) if row['onIce_fenwickPercentage'] else 0.5
            
            # Offensive stats
            goals = int(float(row['I_F_goals'])) if row['I_F_goals'] else 0
            primary_assists = int(float(row['I_F_primaryAssists'])) if row['I_F_primaryAssists'] else 0
            secondary_assists = int(float(row['I_F_secondaryAssists'])) if row['I_F_secondaryAssists'] else 0
            points = goals + primary_assists + secondary_assists
            shots = int(float(row['I_F_shotsOnGoal'])) if row['I_F_shotsOnGoal'] else 0
            xgoals = float(row['I_F_xGoals']) if row['I_F_xGoals'] else 0.0
            
            # Insert into database
            cursor.execute('''
                INSERT OR REPLACE INTO advanced_skater_stats
                (player_id, player_name, team_abbr, team_name, position, season, games_played, icetime,
                 game_score, onice_xgoals_pct, onice_corsi_pct, onice_fenwick_pct,
                 goals, primary_assists, secondary_assists, points, shots_on_goal, xgoals,
                 office_xgoals_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                player_id, player_name, team_abbr, team_name, position, season, games_played, icetime,
                game_score, onice_xgoals_pct, onice_corsi_pct, onice_fenwick_pct,
                goals, primary_assists, secondary_assists, points, shots, xgoals,
                office_xgoals_pct
            ))
            
            skaters_imported += 1
    
    conn.commit()
    conn.close()
    
    logger.info(f"✓ Imported {skaters_imported} skaters")
    logger.info(f"  Skipped {skaters_skipped} skaters with 0 games")
    return skaters_imported

def aggregate_team_stats():
    """Aggregate skater stats by team"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing aggregates
    cursor.execute('DELETE FROM team_skater_aggregates')
    
    # Get all teams
    teams = cursor.execute('SELECT DISTINCT team_abbr, team_name FROM advanced_skater_stats').fetchall()
    
    for team_abbr, team_name in teams:
        # Calculate team aggregates
        stats = cursor.execute('''
            SELECT 
                AVG(game_score) as avg_game_score,
                AVG(onice_xgoals_pct) as avg_xgoals_pct,
                AVG(onice_corsi_pct) as avg_corsi_pct,
                SUM(goals) as total_goals,
                SUM(points) as total_points,
                SUM(xgoals) as total_xgoals
            FROM advanced_skater_stats
            WHERE team_abbr = ?
        ''', (team_abbr,)).fetchone()
        
        # Get top scorer
        top_scorer = cursor.execute('''
            SELECT player_name, points
            FROM advanced_skater_stats
            WHERE team_abbr = ?
            ORDER BY points DESC
            LIMIT 1
        ''', (team_abbr,)).fetchone()
        
        cursor.execute('''
            INSERT INTO team_skater_aggregates
            (team_abbr, team_name, avg_game_score, avg_onice_xgoals_pct, avg_onice_corsi_pct,
             total_goals, total_points, total_xgoals, top_scorer_name, top_scorer_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            team_abbr, team_name,
            stats[0], stats[1], stats[2], stats[3], stats[4], stats[5],
            top_scorer[0] if top_scorer else 'Unknown',
            top_scorer[1] if top_scorer else 0
        ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"✓ Aggregated stats for {len(teams)} teams")
    return len(teams)

def show_team_rankings():
    """Display team rankings by key metrics"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    print("\n📊 Top 10 Teams by Expected Goals %:")
    print(f"{'Team':<5} {'Team Name':<25} {'xG%':<6} {'Corsi%':<7} {'Goals':<6} {'Top Scorer':<25}")
    print("-" * 80)
    
    teams = cursor.execute('''
        SELECT team_abbr, team_name, avg_onice_xgoals_pct, avg_onice_corsi_pct,
               total_goals, top_scorer_name, top_scorer_points
        FROM team_skater_aggregates
        ORDER BY avg_onice_xgoals_pct DESC
        LIMIT 10
    ''').fetchall()
    
    for team in teams:
        abbr, name, xg_pct, corsi, goals, scorer, pts = team
        print(f"{abbr:<5} {name:<25} {xg_pct:.3f}  {corsi:.3f}   {goals:<6} {scorer:<20} ({pts} pts)")
    
    conn.close()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("IMPORTING ADVANCED NHL SKATER DATA (2025 SEASON)")
    print("="*70)
    
    create_skater_stats_table()
    count = import_skater_data()
    
    if count > 0:
        team_count = aggregate_team_stats()
        show_team_rankings()
        
        print(f"\n✅ Successfully imported {count} skater records")
        print(f"✅ Aggregated stats for {team_count} teams")
        print("\n💡 Advanced team metrics now available:")
        print("   - Average GameScore per team")
        print("   - On-Ice Expected Goals %")
        print("   - On-Ice Corsi % (shot attempts)")
        print("   - Team offensive production")
    else:
        print("\n❌ Failed to import skater data")
