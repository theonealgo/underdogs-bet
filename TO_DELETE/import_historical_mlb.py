"""
Import MLB 2025 Historical Results
Parses the historical results file and populates database with actual scores
"""

import re
import sqlite3
from datetime import datetime
from src.data_storage.database import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_mlb_historical_data(filepath: str):
    """Parse MLB historical results from text file"""
    games = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern: Date	Start Time (Sask)	Start Time (EDT)	Away	Away Score	Home	Home Score
    # Example: 2025-03-18	4:10 AM	6:10 AM	Los Angeles Dodgers	4	Chicago Cubs	1
    pattern = r'(\d{4}-\d{2}-\d{2})\t[^\t]+\t[^\t]+\t([^\t]+)\t(\d+)\t([^\t]+)\t(\d+)'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        date_str, away_team, away_score, home_team, home_score = match
        
        # Parse date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        game_date = date_obj.strftime('%d/%m/%Y')  # Convert to DD/MM/YYYY
        
        # Clean team names
        away_team = away_team.strip()
        home_team = home_team.strip()
        
        # Season is 2025
        season = 2025
        
        games.append({
            'game_date': game_date,
            'away_team': away_team,
            'home_team': home_team,
            'away_score': int(away_score),
            'home_score': int(home_score),
            'sport': 'MLB',
            'league': 'MLB',
            'season': season,
            'status': 'final'
        })
    
    logger.info(f"Parsed {len(games)} MLB games from {filepath}")
    return games


def import_to_database(games):
    """Import games into database"""
    db = DatabaseManager()
    
    inserted = 0
    updated = 0
    
    with sqlite3.connect(db.db_path) as conn:
        for game in games:
            # Generate game_id
            game_id = f"MLB_{game['game_date'].replace('/', '')}_{game['away_team'][:3]}_{game['home_team'][:3]}"
            
            # Check if game exists
            cursor = conn.execute(
                'SELECT id FROM games WHERE game_id = ?',
                (game_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update with scores
                conn.execute('''
                    UPDATE games 
                    SET home_score = ?, away_score = ?, status = ?
                    WHERE game_id = ?
                ''', (game['home_score'], game['away_score'], game['status'], game_id))
                updated += 1
            else:
                # Insert new game
                conn.execute('''
                    INSERT INTO games (
                        sport, league, game_id, game_date, season,
                        home_team_id, away_team_id, home_score, away_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    game['sport'], game['league'], game_id, game['game_date'], game['season'],
                    game['home_team'], game['away_team'], 
                    game['home_score'], game['away_score'], game['status']
                ))
                inserted += 1
        
        conn.commit()
    
    logger.info(f"Imported MLB games: {inserted} new, {updated} updated")
    return inserted, updated


def main():
    """Main import function"""
    filepath = 'attached_assets/Pasted-Date-Start-Time-Sask-Start-Time-EDT-Away-Away-Score-Home-Home-Score-2025-03-18-4-10-AM-6-10-AM--1760701187672_1760701187673.txt'
    
    logger.info("="*70)
    logger.info("IMPORTING MLB 2025 HISTORICAL RESULTS")
    logger.info("="*70)
    
    # Parse data
    games = parse_mlb_historical_data(filepath)
    
    # Import to database
    inserted, updated = import_to_database(games)
    
    logger.info("="*70)
    logger.info(f"IMPORT COMPLETE: {inserted + updated} total games")
    logger.info("="*70)


if __name__ == "__main__":
    main()
