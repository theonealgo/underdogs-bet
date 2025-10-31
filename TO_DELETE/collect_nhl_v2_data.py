"""
NHL V2 Data Collection Script
Fetches goalie stats and betting odds from APIs and stores them in the database
Designed for modular integration with V2 predictor
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from nhl_api_integration import NHLAPIClient
from odds_api_integration import OddsAPIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NHLV2DataCollector:
    """Collects enhanced data for NHL V2 predictions"""
    
    def __init__(self, db_path='backups/v2/sports_predictions_nhl_v2.db'):
        self.db_path = db_path
        self.nhl_client = NHLAPIClient()
        self.odds_client = OddsAPIClient()
    
    def collect_all_data(self):
        """Run full data collection pipeline"""
        logger.info("Starting NHL V2 data collection")
        
        # Step 1: Collect and store goalie stats
        self.collect_goalie_stats()
        
        # Step 2: Collect and store betting odds for upcoming games
        self.collect_betting_odds()
        
        # Step 3: Link goalies to upcoming games
        self.link_goalies_to_games()
        
        logger.info("NHL V2 data collection complete")
    
    def collect_goalie_stats(self):
        """Fetch and store goalie statistics"""
        logger.info("Collecting goalie stats...")
        
        goalie_stats = self.nhl_client.get_goalie_stats(season="20252026")
        
        if not goalie_stats:
            logger.warning("No goalie stats retrieved")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for goalie_name, stats in goalie_stats.items():
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO goalie_stats 
                    (goalie_name, season, save_pct, gaa, wins, losses, games_played, shutouts, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    goalie_name,
                    '2025-2026',
                    stats['save_pct'],
                    stats['gaa'],
                    stats['wins'],
                    stats['losses'],
                    stats['games_played'],
                    stats['shutouts'],
                    datetime.now()
                ))
                count += 1
            except Exception as e:
                logger.error(f"Error storing stats for {goalie_name}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored stats for {count} goalies")
    
    def collect_betting_odds(self):
        """Fetch and store betting odds for upcoming games"""
        logger.info("Collecting betting odds...")
        
        odds_data = self.odds_client.get_odds()
        
        if not odds_data:
            logger.warning("No odds data retrieved")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get upcoming games from database
        cursor.execute('''
            SELECT id, home_team_id, away_team_id, game_date
            FROM games
            WHERE sport = 'NHL' AND home_score IS NULL
            ORDER BY game_date
        ''')
        upcoming_games = cursor.fetchall()
        
        count = 0
        for game_id, home_team, away_team, game_date in upcoming_games:
            # Parse odds for this game
            parsed_odds = self.odds_client.parse_odds_for_game(
                home_team, away_team, odds_data
            )
            
            if parsed_odds['home_moneyline'] is not None:
                try:
                    # Calculate implied probabilities from moneyline
                    home_prob = self._moneyline_to_probability(parsed_odds['home_moneyline'])
                    away_prob = self._moneyline_to_probability(parsed_odds['away_moneyline'])
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO betting_odds
                        (game_id, home_moneyline, away_moneyline, spread, total,
                         home_spread_odds, away_spread_odds, over_odds, under_odds,
                         num_bookmakers, home_implied_prob, away_implied_prob, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        game_id,
                        parsed_odds['home_moneyline'],
                        parsed_odds['away_moneyline'],
                        parsed_odds['spread'],
                        parsed_odds['total'],
                        parsed_odds['home_spread_odds'],
                        parsed_odds['away_spread_odds'],
                        parsed_odds['over_odds'],
                        parsed_odds['under_odds'],
                        parsed_odds['num_bookmakers'],
                        home_prob,
                        away_prob,
                        datetime.now()
                    ))
                    count += 1
                except Exception as e:
                    logger.error(f"Error storing odds for game {game_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored odds for {count} games")
    
    def link_goalies_to_games(self):
        """Link starting goalies to upcoming games"""
        logger.info("Linking goalies to games...")
        
        # Get today's games from NHL API
        schedule = self.nhl_client.get_todays_schedule()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for game in schedule:
            try:
                game_id_api = game.get('id')
                home_team = game.get('homeTeam', {}).get('placeName', {}).get('default', '')
                away_team = game.get('awayTeam', {}).get('placeName', {}).get('default', '')
                
                # Find matching game in our database
                cursor.execute('''
                    SELECT id FROM games
                    WHERE sport = 'NHL' 
                    AND (home_team_id LIKE ? OR home_team_id LIKE ?)
                    AND (away_team_id LIKE ? OR away_team_id LIKE ?)
                    AND home_score IS NULL
                    LIMIT 1
                ''', (
                    f'%{home_team}%', f'%{home_team.split()[-1]}%',
                    f'%{away_team}%', f'%{away_team.split()[-1]}%'
                ))
                
                result = cursor.fetchone()
                if not result:
                    continue
                
                db_game_id = result[0]
                
                # Get starting goalies
                home_goalie, away_goalie = self.nhl_client.get_starting_goalies(game_id_api)
                
                if home_goalie or away_goalie:
                    # Look up goalie stats
                    home_save_pct, home_gaa = self._get_goalie_stats(cursor, home_goalie)
                    away_save_pct, away_gaa = self._get_goalie_stats(cursor, away_goalie)
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO game_goalies
                        (game_id, home_goalie, away_goalie, 
                         home_goalie_save_pct, away_goalie_save_pct,
                         home_goalie_gaa, away_goalie_gaa, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        db_game_id,
                        home_goalie,
                        away_goalie,
                        home_save_pct,
                        away_save_pct,
                        home_gaa,
                        away_gaa,
                        datetime.now()
                    ))
                    count += 1
                
            except Exception as e:
                logger.error(f"Error linking goalies for game: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Linked goalies to {count} games")
    
    def _get_goalie_stats(self, cursor, goalie_name):
        """Look up goalie stats from database"""
        if not goalie_name:
            return None, None
        
        cursor.execute('''
            SELECT save_pct, gaa
            FROM goalie_stats
            WHERE goalie_name = ? AND season = '2025-2026'
        ''', (goalie_name,))
        
        result = cursor.fetchone()
        if result:
            return result[0], result[1]
        return None, None
    
    def _moneyline_to_probability(self, moneyline):
        """Convert American odds to implied probability"""
        if moneyline is None:
            return None
        
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)
    
    def get_summary(self):
        """Get summary of collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM goalie_stats WHERE season = "2025-2026"')
        goalie_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM betting_odds')
        odds_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM game_goalies')
        goalie_links = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'goalies': goalie_count,
            'games_with_odds': odds_count,
            'games_with_goalie_info': goalie_links
        }


def main():
    """Run data collection"""
    print("\n=== NHL V2 Data Collection ===\n")
    
    collector = NHLV2DataCollector()
    
    # Collect all data
    collector.collect_all_data()
    
    # Show summary
    summary = collector.get_summary()
    print("\n=== Collection Summary ===")
    print(f"Goalies in database: {summary['goalies']}")
    print(f"Games with betting odds: {summary['games_with_odds']}")
    print(f"Games with goalie info: {summary['games_with_goalie_info']}")
    print("\nData collection complete!\n")


if __name__ == "__main__":
    main()
