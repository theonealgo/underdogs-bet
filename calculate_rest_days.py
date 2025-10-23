#!/usr/bin/env python3
"""
Calculate rest days and back-to-back indicators for NHL teams
Adds new columns to games table for feature engineering
"""

import sqlite3
from datetime import datetime, timedelta

def parse_date(date_str):
    """Parse DD/MM/YYYY date string"""
    try:
        return datetime.strptime(date_str, '%d/%m/%Y')
    except:
        try:
            return datetime.strptime(date_str, '%m/%d/%Y')
        except:
            return None

def calculate_rest_days_for_team(cursor, team_name, game_date, game_id):
    """Calculate rest days for a team before a specific game"""
    parsed_date = parse_date(game_date)
    if not parsed_date:
        return 3  # Default rest days
    
    # Get previous game for this team (either home or away)
    cursor.execute('''
        SELECT game_date FROM games
        WHERE (home_team_id = ? OR away_team_id = ?)
        AND game_id != ?
        ORDER BY ABS(julianday(?) - julianday(game_date))
        LIMIT 1
    ''', (team_name, team_name, game_id, game_date))
    
    prev_game = cursor.fetchone()
    
    if prev_game:
        prev_date = parse_date(prev_game[0])
        if prev_date:
            rest_days = (parsed_date - prev_date).days - 1
            return max(0, rest_days)  # Can't have negative rest
    
    return 3  # Default if no previous game found

def add_rest_columns():
    """Add rest day columns to games table"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(games)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'home_rest_days' not in columns:
        cursor.execute('ALTER TABLE games ADD COLUMN home_rest_days INTEGER DEFAULT 3')
        print("✅ Added home_rest_days column")
    
    if 'away_rest_days' not in columns:
        cursor.execute('ALTER TABLE games ADD COLUMN away_rest_days INTEGER DEFAULT 3')
        print("✅ Added away_rest_days column")
    
    if 'home_back_to_back' not in columns:
        cursor.execute('ALTER TABLE games ADD COLUMN home_back_to_back INTEGER DEFAULT 0')
        print("✅ Added home_back_to_back column")
    
    if 'away_back_to_back' not in columns:
        cursor.execute('ALTER TABLE games ADD COLUMN away_back_to_back INTEGER DEFAULT 0')
        print("✅ Added away_back_to_back column")
    
    conn.commit()
    conn.close()

def calculate_all_rest_days():
    """Calculate rest days for all NHL games"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    # Get all NHL games sorted by date
    cursor.execute('''
        SELECT game_id, game_date, home_team_id, away_team_id
        FROM games
        WHERE sport = 'NHL'
        ORDER BY game_date ASC
    ''')
    
    games = cursor.fetchall()
    print(f"\n📊 Calculating rest days for {len(games)} NHL games...")
    
    # Track last game date for each team
    team_last_game = {}
    updates = []
    
    for game in games:
        game_id, game_date, home_team, away_team = game
        parsed_date = parse_date(game_date)
        
        if not parsed_date:
            continue
        
        # Calculate home team rest
        if home_team in team_last_game:
            home_rest = (parsed_date - team_last_game[home_team]).days - 1
            home_b2b = 1 if home_rest == 0 else 0
        else:
            home_rest = 3
            home_b2b = 0
        
        # Calculate away team rest
        if away_team in team_last_game:
            away_rest = (parsed_date - team_last_game[away_team]).days - 1
            away_b2b = 1 if away_rest == 0 else 0
        else:
            away_rest = 3
            away_b2b = 0
        
        # Update team last game dates
        team_last_game[home_team] = parsed_date
        team_last_game[away_team] = parsed_date
        
        # Store update
        updates.append((max(0, home_rest), max(0, away_rest), home_b2b, away_b2b, game_id))
    
    # Batch update all games
    cursor.executemany('''
        UPDATE games
        SET home_rest_days = ?, away_rest_days = ?, 
            home_back_to_back = ?, away_back_to_back = ?
        WHERE game_id = ?
    ''', updates)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Updated rest days for {len(updates)} games")

def verify_rest_calculations():
    """Verify rest day calculations"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    # Check some examples
    cursor.execute('''
        SELECT home_team_id, away_team_id, game_date, 
               home_rest_days, away_rest_days,
               home_back_to_back, away_back_to_back
        FROM games
        WHERE sport = 'NHL' AND home_score IS NOT NULL
        ORDER BY game_date DESC
        LIMIT 10
    ''')
    
    print("\n📋 Sample rest day calculations:")
    print(f"{'Home Team':<25} {'Away Team':<25} {'Date':<12} {'H-Rest':<7} {'A-Rest':<7} {'H-B2B':<6} {'A-B2B'}")
    print("-" * 110)
    
    for row in cursor.fetchall():
        home, away, date, h_rest, a_rest, h_b2b, a_b2b = row
        print(f"{home[:24]:<25} {away[:24]:<25} {date:<12} {h_rest:<7} {a_rest:<7} {h_b2b:<6} {a_b2b}")
    
    # Count back-to-backs
    cursor.execute('''
        SELECT COUNT(*) FROM games
        WHERE sport = 'NHL' AND (home_back_to_back = 1 OR away_back_to_back = 1)
    ''')
    b2b_count = cursor.fetchone()[0]
    print(f"\n📊 Total games with back-to-back situation: {b2b_count}")
    
    conn.close()

if __name__ == '__main__':
    print("="*70)
    print("NHL REST DAYS CALCULATOR")
    print("="*70)
    
    add_rest_columns()
    calculate_all_rest_days()
    verify_rest_calculations()
    
    print("\n" + "="*70)
    print("✅ Rest day calculation complete!")
    print("="*70 + "\n")
