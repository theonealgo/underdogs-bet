"""
Populate team_stats table from games data to enable rolling features.
Creates game-by-game team statistics for all sports.
"""

import sqlite3
import json
from datetime import datetime
from collections import defaultdict

def populate_nfl_team_stats(conn):
    """Populate NFL team stats from completed games"""
    cursor = conn.cursor()
    
    # Get all completed NFL games in chronological order
    cursor.execute('''
        SELECT game_date, home_team_id, away_team_id, home_score, away_score, season
        FROM games
        WHERE sport = 'NFL' AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY game_date ASC, game_id ASC
    ''')
    games = cursor.fetchall()
    
    print(f"Processing {len(games)} NFL games...")
    
    # Track cumulative stats per team
    team_records = defaultdict(lambda: {'games': 0, 'wins': 0, 'losses': 0})
    
    inserted = 0
    for game_date, home_team, away_team, home_score, away_score, season in games:
        # Determine winner
        home_won = home_score > away_score
        
        # Calculate game-level metrics
        point_diff_home = home_score - away_score
        point_diff_away = away_score - home_score
        total_points = home_score + away_score
        
        # Update home team stats
        team_records[home_team]['games'] += 1
        if home_won:
            team_records[home_team]['wins'] += 1
        else:
            team_records[home_team]['losses'] += 1
        
        home_metrics = {
            'points_scored': home_score,
            'points_allowed': away_score,
            'point_differential': point_diff_home,
            'total_points': total_points,
            'is_home': 1,
            'win': 1 if home_won else 0,
            'opponent': away_team,
            'opponent_score': away_score
        }
        
        # Insert home team stat
        cursor.execute('''
            INSERT INTO team_stats (sport, league, team_id, season, date, games_played, wins, losses, metrics, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'NFL', 'NFL', home_team, season, game_date,
            team_records[home_team]['games'],
            team_records[home_team]['wins'],
            team_records[home_team]['losses'],
            json.dumps(home_metrics),
            datetime.now().isoformat()
        ))
        inserted += 1
        
        # Update away team stats
        team_records[away_team]['games'] += 1
        if not home_won:
            team_records[away_team]['wins'] += 1
        else:
            team_records[away_team]['losses'] += 1
        
        away_metrics = {
            'points_scored': away_score,
            'points_allowed': home_score,
            'point_differential': point_diff_away,
            'total_points': total_points,
            'is_home': 0,
            'win': 0 if home_won else 1,
            'opponent': home_team,
            'opponent_score': home_score
        }
        
        # Insert away team stat
        cursor.execute('''
            INSERT INTO team_stats (sport, league, team_id, season, date, games_played, wins, losses, metrics, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'NFL', 'NFL', away_team, season, game_date,
            team_records[away_team]['games'],
            team_records[away_team]['wins'],
            team_records[away_team]['losses'],
            json.dumps(away_metrics),
            datetime.now().isoformat()
        ))
        inserted += 1
    
    conn.commit()
    print(f"✅ Inserted {inserted} NFL team_stats records")
    return inserted

def populate_all_sports(conn):
    """Populate team stats for all sports"""
    cursor = conn.cursor()
    
    # Clear existing team_stats
    cursor.execute('DELETE FROM team_stats')
    conn.commit()
    print("Cleared existing team_stats")
    
    # Populate NFL
    nfl_count = populate_nfl_team_stats(conn)
    
    # Could add other sports here (NBA, NHL, MLB, NCAAF)
    # For now, focus on NFL
    
    return nfl_count

if __name__ == '__main__':
    conn = sqlite3.connect('sports_predictions.db')
    
    print("="*60)
    print("POPULATING TEAM_STATS FROM GAMES DATA")
    print("="*60)
    
    total = populate_all_sports(conn)
    
    # Verify
    cursor = conn.cursor()
    cursor.execute('SELECT sport, COUNT(*) FROM team_stats GROUP BY sport')
    results = cursor.fetchall()
    
    print("\nVerification:")
    for sport, count in results:
        print(f"  {sport}: {count} team_stats records")
    
    conn.close()
    print("\n✅ Team stats population complete!")
