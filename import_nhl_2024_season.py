#!/usr/bin/env python3
"""
Import Complete NHL 2024-25 Season for Training
================================================

Imports 1,312 games with actual results from nhlschedules.py
This gives models a full season of data to learn from!
"""

import sqlite3
from datetime import datetime
import logging
from nhlschedules import get_nhl_2024_schedule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions_original.db'

def import_2024_season():
    """Import full 2024-25 NHL season"""
    logger.info("="*70)
    logger.info("IMPORTING NHL 2024-25 SEASON")
    logger.info("="*70)
    
    # Get schedule
    schedule = get_nhl_2024_schedule()
    logger.info(f"Loaded {len(schedule)} games from 2024-25 season")
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Clear existing 2024 season data
    cursor.execute("DELETE FROM games WHERE sport='NHL' AND season=2024")
    logger.info(f"Cleared old 2024 season data")
    
    inserted = 0
    skipped = 0
    
    for game in schedule:
        # Only import games with completed results
        if 'home_score' not in game or 'away_score' not in game:
            skipped += 1
            continue
            
        if game['home_score'] is None or game['away_score'] is None:
            skipped += 1
            continue
        
        # Generate game_id
        date_str = game['date']  # DD/MM/YYYY format
        home_short = game['home_team'][:3].replace(' ', '')
        away_short = game['away_team'][:3].replace(' ', '')
        game_id = f"nhl_2024_{date_str.replace('/', '')}_{away_short}_{home_short}"
        
        # Insert into database
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO games (
                    game_id, sport, league, season, game_date,
                    home_team_id, away_team_id, 
                    home_score, away_score,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                'NHL',
                'NHL', 
                2024,  # Season 2024 = training data
                date_str,  # DD/MM/YYYY
                game['home_team'],
                game['away_team'],
                game['home_score'],
                game['away_score'],
                'final',
                datetime.now().isoformat()
            ))
            inserted += 1
        except Exception as e:
            logger.warning(f"Error inserting game {game_id}: {e}")
            skipped += 1
    
    conn.commit()
    conn.close()
    
    logger.info("\n" + "="*70)
    logger.info("IMPORT COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Inserted: {inserted} games")
    logger.info(f"⏭️  Skipped:  {skipped} games")
    logger.info("="*70)
    
    if inserted > 0:
        logger.info("\n🎯 Next Steps:")
        logger.info("  1. Run 'python train_nhl_models.py' to retrain with full season")
        logger.info("  2. Run 'python generate_nhl_predictions.py' to generate new predictions")
        logger.info("\nYour models will now learn from 1,300+ games instead of just 92!")

if __name__ == "__main__":
    import_2024_season()
