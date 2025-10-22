#!/usr/bin/env python3
"""
Fetch NHL Goalie Statistics using nhl-api-py library
Replaces custom API integration with professional library
"""

import sqlite3
from nhlpy import NHLClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions.db'

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
    """Fetch goalie stats from NHL API and store in database"""
    client = NHLClient()
    
    try:
        # Get current season goalie stats (all goalies, not just top 25)
        all_goalies = []
        limit = 100
        start = 0
        
        while True:
            # Fetch goalies in batches
            batch = client.stats.goalie_stats_summary(
                start_season="20252026", 
                end_season="20252026"
            )
            
            if not batch:
                break
                
            all_goalies.extend(batch)
            
            # NHL API returns max 100 at a time, but usually less
            if len(batch) < limit:
                break
            
            start += limit
        
        goalie_stats = all_goalies
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM goalie_stats')
        
        # Insert goalie stats
        count = 0
        for goalie in goalie_stats:
            full_name = goalie.get('goalieFullName', '')
            
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
                goalie.get('savePct', 0.0),
                goalie.get('goalsAgainstAverage', 0.0),
                goalie.get('wins', 0),
                goalie.get('losses', 0),
                goalie.get('gamesPlayed', 0),
                goalie.get('shutouts', 0)
            ))
            
            count += 1
            if count <= 25:
                print(f"✓ {full_name}: {goalie.get('savePct', 0):.3f} SV%, {goalie.get('goalsAgainstAverage', 0):.2f} GAA, {goalie.get('wins', 0)}W-{goalie.get('losses', 0)}L")
        
        conn.commit()
        conn.close()
        
        logger.info(f"\n✅ Stored stats for {count} goalies from 2025-26 season")
        return count
        
    except Exception as e:
        logger.error(f"Error fetching goalie stats: {e}")
        return 0

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FETCHING NHL GOALIE STATS USING NHL-API-PY")
    print("="*70)
    
    create_goalie_stats_table()
    count = fetch_and_store_goalie_stats()
    
    if count > 0:
        print(f"\n✅ Successfully fetched and stored {count} goalie records")
    else:
        print("\n❌ Failed to fetch goalie stats")
