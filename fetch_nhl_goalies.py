"""
Fetch NHL goalie stats from the official NHL API
"""

import sqlite3
from nhl_api_integration import NHLAPIClient

DATABASE = 'sports_predictions.db'

def setup_goalie_table():
    """Create goalie_stats table if it doesn't exist"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS goalie_stats (
            goalie_name TEXT PRIMARY KEY,
            save_pct REAL,
            gaa REAL,
            wins INTEGER,
            losses INTEGER,
            games_played INTEGER,
            shutouts INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Goalie stats table ready")

def fetch_and_store_goalies():
    """Fetch goalie stats from NHL API and store in database"""
    
    print("\n" + "="*70)
    print("FETCHING NHL GOALIE STATS FROM OFFICIAL NHL API")
    print("="*70)
    
    # Setup table
    setup_goalie_table()
    
    # Fetch goalie stats
    client = NHLAPIClient()
    goalie_stats = client.get_goalie_stats(season="20252026")
    
    if not goalie_stats:
        print("⚠ No goalie stats retrieved")
        return
    
    # Store in database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    for goalie_name, stats in goalie_stats.items():
        cursor.execute('''
            INSERT OR REPLACE INTO goalie_stats 
            (goalie_name, save_pct, gaa, wins, losses, games_played, shutouts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            goalie_name,
            stats['save_pct'],
            stats['gaa'],
            stats['wins'],
            stats['losses'],
            stats['games_played'],
            stats['shutouts']
        ))
        
        print(f"✓ {goalie_name}: {stats['save_pct']:.3f} SV%, {stats['gaa']:.2f} GAA, {stats['wins']}W-{stats['losses']}L")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: Fetched and stored {len(goalie_stats)} goalies")
    print(f"{'='*70}\n")
    
    return len(goalie_stats)

if __name__ == "__main__":
    fetch_and_store_goalies()
