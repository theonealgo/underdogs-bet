"""
Import NFL 2024 Historical Results
Parses the historical results file and populates database with actual scores
"""

import re
import sqlite3
from datetime import datetime
from src.data_storage.database import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_nfl_historical_data(filepath: str):
    """Parse NFL historical results from text file"""
    games = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern: Date Visitor VisitorScore Home HomeScore (OT optional) Box Score
    # Example: 09/05/2024	Baltimore Ravens	20	Kansas City Chiefs	27		Box Score
    pattern = r'(\d{2}/\d{2}/\d{4})\t([^\t]+)\t(\d+)\t([^\t]+)\t(\d+)\t?(OT)?\t'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        date_str, away_team, away_score, home_team, home_score, overtime = match
        
        # Parse date
        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
        game_date = date_obj.strftime('%d/%m/%Y')  # Convert to DD/MM/YYYY
        
        # Clean team names
        away_team = away_team.strip()
        home_team = home_team.strip()
        
        games.append({
            'game_date': game_date,
            'away_team': away_team,
            'home_team': home_team,
            'away_score': int(away_score),
            'home_score': int(home_score),
            'overtime': bool(overtime),
            'sport': 'NFL',
            'league': 'NFL',
            'season': 2024,
            'status': 'final'
        })
    
    logger.info(f"Parsed {len(games)} NFL games from {filepath}")
    return games


def import_to_database(games):
    """Import games into database"""
    db = DatabaseManager()
    
    inserted = 0
    updated = 0
    
    with sqlite3.connect(db.db_path) as conn:
        for game in games:
            # Generate game_id
            game_id = f"NFL_{game['game_date'].replace('/', '')}_{game['away_team'][:3]}_{game['home_team'][:3]}"
            
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
    
    logger.info(f"Imported NFL games: {inserted} new, {updated} updated")
    return inserted, updated


def main():
    """Main import function"""
    filepath = 'attached_assets/Pasted-Week-1-Date-Visitor-Home-Box-Score-09-05-2024-Baltimore-Ravens-20-Kansas-City-Chiefs-27-Box-Scor-1760699160743_1760699160744.txt'
    
    logger.info("="*70)
    logger.info("IMPORTING NFL 2024 HISTORICAL RESULTS")
    logger.info("="*70)
    
    # Parse data
    games = parse_nfl_historical_data(filepath)
    
    # Import to database
    inserted, updated = import_to_database(games)
    
    logger.info("="*70)
    logger.info(f"IMPORT COMPLETE: {inserted + updated} total games")
    logger.info("="*70)


if __name__ == "__main__":
    main()
