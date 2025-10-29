#!/usr/bin/env python3
"""
Import NHL October 2025 Results
Parses the NHL results file and populates database with actual scores for training
"""

import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions_original.db'
FILEPATH = 'attached_assets/Pasted-10-07-2025-Chicago-Blackhawks-2-Florida-Panthers-3-10-07-2025-Pittsburgh-Penguins-3-New-York-Rangers-1761230963496_1761230963497.txt'

def parse_nhl_results():
    """Parse NHL results from text file (tab-delimited)"""
    games = []
    
    with open(FILEPATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # File is TAB-DELIMITED: Date\tAway Team\tAway Score\tHome Team\tHome Score
            parts = line.split('\t')
            
            if len(parts) >= 5:
                date_str = parts[0]
                away_team = parts[1]
                away_score = int(parts[2])
                home_team = parts[3]
                home_score = int(parts[4])
                
                # Handle two date formats: MM/DD/YYYY and YYYY-MM-DD
                try:
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    else:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    game_date = date_obj.strftime('%d/%m/%Y')
                except ValueError as e:
                    logger.warning(f"Skipping line with invalid date '{date_str}': {e}")
                    continue
                
                games.append({
                    'game_date': game_date,
                    'away_team': away_team,
                    'home_team': home_team,
                    'away_score': away_score,
                    'home_score': home_score,
                    'sport': 'NHL',
                    'league': 'NHL',
                    'season': 2024,  # 2024-25 season for training
                    'status': 'final'
                })
    
    logger.info(f"Parsed {len(games)} NHL games from {FILEPATH}")
    return games

def import_to_database(games):
    """Import games into database"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    inserted = 0
    updated = 0
    
    for game in games:
        # Generate game_id
        date_part = game['game_date'].replace('/', '')
        game_id = f"nhl_2024_{date_part}_{game['away_team'][:3].replace(' ', '')}_{game['home_team'][:3].replace(' ', '')}"
        
        # Check if game exists
        cursor.execute('SELECT game_id FROM games WHERE game_id = ?', (game_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update with scores
            cursor.execute('''
                UPDATE games 
                SET home_score = ?, away_score = ?, status = ?
                WHERE game_id = ?
            ''', (game['home_score'], game['away_score'], game['status'], game_id))
            updated += 1
        else:
            # Insert new game
            cursor.execute('''
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
    conn.close()
    
    logger.info(f"Imported NHL games: {inserted} new, {updated} updated")
    return inserted, updated

def main():
    """Main import function"""
    logger.info("="*70)
    logger.info("IMPORTING NHL OCTOBER 2025 RESULTS (for 2024-25 season training)")
    logger.info("="*70)
    
    # Parse data
    games = parse_nhl_results()
    
    if not games:
        logger.error("No games found in file")
        return
    
    # Import to database
    inserted, updated = import_to_database(games)
    
    logger.info("="*70)
    logger.info(f"✅ IMPORT COMPLETE: {inserted} inserted, {updated} updated")
    logger.info("="*70)
    logger.info("You can now run 'python train_nhl_models.py' to train models")

if __name__ == "__main__":
    main()
