"""
Populate NFL betting odds data for model meta features
Fetches current NFL odds and stores them for feature engineering
"""
import sqlite3
import logging
from datetime import datetime
from src.data_collectors.odds_collector import OddsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_odds_features_table():
    """Create table for odds-based meta features"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sport TEXT NOT NULL,
            game_id TEXT,
            game_date DATE NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_moneyline INTEGER,
            away_moneyline INTEGER,
            home_implied_prob REAL,
            away_implied_prob REAL,
            spread_line REAL,
            home_spread_odds INTEGER,
            away_spread_odds INTEGER,
            total_line REAL,
            over_odds INTEGER,
            under_odds INTEGER,
            bookmaker_count INTEGER,
            market_vig REAL,
            collected_at TIMESTAMP,
            UNIQUE(sport, game_date, home_team_id, away_team_id)
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("✅ Created market_odds table")

def populate_nfl_odds():
    """Fetch and store current NFL odds"""
    collector = OddsCollector()
    
    # Get NFL odds from The Odds API
    logger.info("Fetching NFL odds...")
    nfl_odds = collector.get_sport_odds('NFL')
    
    if nfl_odds.empty:
        logger.warning("No NFL odds available")
        return 0
    
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    games_added = 0
    for _, row in nfl_odds.iterrows():
        # Calculate market vig (overround)
        total_prob = row['home_implied_prob'] + row['away_implied_prob']
        market_vig = (total_prob - 1.0) if total_prob > 1.0 else 0.0
        
        # Format date as DD/MM/YYYY to match games table
        game_date_str = row['game_date'].strftime('%d/%m/%Y')
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO market_odds (
                    sport, game_date, home_team_id, away_team_id,
                    home_moneyline, away_moneyline,
                    home_implied_prob, away_implied_prob,
                    spread_line, home_spread_odds, away_spread_odds,
                    total_line, over_odds, under_odds,
                    bookmaker_count, market_vig, collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'NFL', game_date_str, row['home_team'], row['away_team'],
                row['home_odds'], row['away_odds'],
                row['home_implied_prob'], row['away_implied_prob'],
                row['home_spread'], row['home_spread_odds'], row['away_spread_odds'],
                row['total_line'], row['over_odds'], row['under_odds'],
                row['bookmaker_count'], market_vig, datetime.now()
            ))
            games_added += 1
        except Exception as e:
            logger.error(f"Error inserting odds for {row['away_team']} @ {row['home_team']}: {e}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Added odds for {games_added} NFL games")
    return games_added

if __name__ == "__main__":
    print("="*60)
    print("POPULATING NFL BETTING ODDS DATA")
    print("="*60)
    
    create_odds_features_table()
    total = populate_nfl_odds()
    
    print(f"\n✅ Total NFL games with odds: {total}")
    print("\nOdds data ready for model integration!")
