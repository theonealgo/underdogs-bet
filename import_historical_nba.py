"""
Import NBA 2024-2025 Historical Results
Parses the historical results files and populates database with actual scores
"""

import re
import sqlite3
from datetime import datetime
from src.data_storage.database import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_nba_historical_data(filepaths: list):
    """Parse NBA historical results from text files"""
    games = []
    
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern: Date	Start (ET)	Visitor/Neutral	PTS	Home/Neutral	PTS
        # Example: Tue, Oct 22, 2024	7:30p	New York Knicks	109	Boston Celtics	132
        pattern = r'([A-Za-z]+, [A-Za-z]+ \d+, \d{4})\t[^\t]+\t([^\t]+)\t(\d+)\t([^\t]+)\t(\d+)\t'
        
        matches = re.findall(pattern, content)
        
        for match in matches:
            date_str, away_team, away_score, home_team, home_score = match
            
            # Parse date - format: "Tue, Oct 22, 2024"
            date_obj = datetime.strptime(date_str, '%a, %b %d, %Y')
            game_date = date_obj.strftime('%d/%m/%Y')  # Convert to DD/MM/YYYY
            
            # Clean team names
            away_team = away_team.strip()
            home_team = home_team.strip()
            
            # Determine season (Oct-Apr = current season year, May-Sep = previous season year)
            season = 2025 if date_obj.month >= 10 else 2024
            
            games.append({
                'game_date': game_date,
                'away_team': away_team,
                'home_team': home_team,
                'away_score': int(away_score),
                'home_score': int(home_score),
                'sport': 'NBA',
                'league': 'NBA',
                'season': season,
                'status': 'final'
            })
    
    logger.info(f"Parsed {len(games)} NBA games from {len(filepaths)} files")
    return games


def import_to_database(games):
    """Import games into database"""
    db = DatabaseManager()
    
    inserted = 0
    updated = 0
    
    with sqlite3.connect(db.db_path) as conn:
        for game in games:
            # Generate game_id
            game_id = f"NBA_{game['game_date'].replace('/', '')}_{game['away_team'][:3]}_{game['home_team'][:3]}"
            
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
    
    logger.info(f"Imported NBA games: {inserted} new, {updated} updated")
    return inserted, updated


def main():
    """Main import function"""
    filepaths = [
        'attached_assets/Pasted-October-Schedule-Share-Export-Modify-Export-Share-Table-Get-as-Excel-Workbook-Get-table-as-CSV-1760699716850_1760699716851.txt',
        'attached_assets/Pasted-Date-Start-ET-Visitor-Neutral-PTS-Home-Neutral-PTS-Attend-LOG-Arena-Notes-Fri-Nov-1-2024-7--1760699744846_1760699744847.txt'
    ]
    
    logger.info("="*70)
    logger.info("IMPORTING NBA 2024-2025 HISTORICAL RESULTS")
    logger.info("="*70)
    
    # Parse data
    games = parse_nba_historical_data(filepaths)
    
    # Import to database
    inserted, updated = import_to_database(games)
    
    logger.info("="*70)
    logger.info(f"IMPORT COMPLETE: {inserted + updated} total games")
    logger.info("="*70)


if __name__ == "__main__":
    main()
