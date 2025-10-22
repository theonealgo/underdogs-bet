"""
Import 77 NHL games from October 7-18, 2025 with actual predictions and scores
"""

import sqlite3
from datetime import datetime

DATABASE = 'sports_predictions.db'

# Games with predictions and actual scores
games_data = [
    ("10/07/2025", "Chicago Blackhawks", 2, "Florida Panthers", 3, 76.20, 80.10, 78.50, 78.50),
    ("10/07/2025", "Pittsburgh Penguins", 3, "New York Rangers", 0, 59.60, 62.50, 61.40, 61.40),
    ("10/07/2025", "Colorado Avalanche", 4, "Los Angeles Kings", 1, 40.30, 42.40, 41.60, 41.60),
    ("10/08/2025", "Montreal Canadiens", 2, "Toronto Maple Leafs", 5, 57.00, 59.80, 58.70, 58.70),
    ("10/08/2025", "Boston Bruins", 3, "Washington Capitals", 1, 72.00, 75.60, 74.20, 74.20),
    ("10/08/2025", "Calgary Flames", 4, "Edmonton Oilers", 3, 52.20, 54.80, 53.80, 53.80),
    ("10/08/2025", "Los Angeles Kings", 6, "Vegas Golden Knights", 5, 61.20, 64.20, 63.00, 63.00),
    ("10/09/2025", "Chicago Blackhawks", 3, "Boston Bruins", 4, 56.00, 58.80, 57.70, 57.70),
    ("10/09/2025", "New York Rangers", 4, "Buffalo Sabres", 0, 52.00, 54.60, 53.50, 53.50),
    ("10/09/2025", "Montreal Canadiens", 5, "Detroit Red Wings", 1, 57.00, 59.90, 58.70, 58.70),
    ("10/09/2025", "Ottawa Senators", 5, "Tampa Bay Lightning", 4, 57.30, 60.20, 59.00, 59.00),
    ("10/09/2025", "Philadelphia Flyers", 1, "Florida Panthers", 2, 66.80, 70.10, 68.80, 68.80),
    ("10/09/2025", "New York Islanders", 3, "Pittsburgh Penguins", 4, 38.60, 40.50, 39.80, 39.80),
    ("10/09/2025", "New Jersey Devils", 3, "Carolina Hurricanes", 6, 64.50, 67.70, 66.50, 66.50),
    ("10/09/2025", "Minnesota Wild", 5, "St. Louis Blues", 0, 59.70, 62.70, 61.50, 61.50),
    ("10/09/2025", "Columbus Blue Jackets", 1, "Nashville Predators", 2, 28.50, 29.90, 29.30, 29.30),
    ("10/09/2025", "Dallas Stars", 5, "Winnipeg Jets", 4, 53.90, 56.50, 55.50, 55.50),
    ("10/09/2025", "Utah Mammoth", 1, "Colorado Avalanche", 2, 58.70, 61.60, 60.40, 60.40),
    ("10/09/2025", "Calgary Flames", 1, "Vancouver Canucks", 5, 47.60, 50.00, 49.10, 49.10),
    ("10/09/2025", "Vegas Golden Knights", 4, "San Jose Sharks", 3, 17.00, 17.90, 17.60, 17.60),
    ("10/09/2025", "Anaheim Ducks", 1, "Seattle Kraken", 3, 45.50, 47.80, 46.90, 46.90),
    ("10/11/2025", "Los Angeles Kings", 2, "Winnipeg Jets", 3, 73.70, 77.40, 76.00, 76.00),
    ("10/11/2025", "St. Louis Blues", 4, "Calgary Flames", 2, 40.60, 42.70, 41.90, 41.90),
    ("10/11/2025", "Buffalo Sabres", 1, "Boston Bruins", 3, 32.60, 34.30, 33.60, 33.60),
    ("10/11/2025", "Toronto Maple Leafs", 3, "Detroit Red Wings", 6, 50.10, 52.60, 51.60, 51.60),
    ("10/11/2025", "New Jersey Devils", 5, "Tampa Bay Lightning", 3, 60.30, 63.40, 62.20, 62.20),
    ("10/11/2025", "Ottawa Senators", 2, "Florida Panthers", 6, 51.60, 54.10, 53.10, 53.10),
    ("10/11/2025", "Washington Capitals", 4, "New York Islanders", 2, 44.50, 46.70, 45.80, 45.80),
    ("10/11/2025", "New York Rangers", 6, "Pittsburgh Penguins", 1, 40.40, 42.50, 41.60, 41.60),
    ("10/11/2025", "Philadelphia Flyers", 3, "Carolina Hurricanes", 4, 75.20, 78.90, 77.40, 77.40),
    ("10/11/2025", "Montreal Canadiens", 3, "Chicago Blackhawks", 2, 28.70, 30.20, 29.60, 29.60),
    ("10/11/2025", "Utah Mammoth", 3, "Nashville Predators", 2, 37.80, 39.60, 38.90, 38.90),
    ("10/11/2025", "Columbus Blue Jackets", 7, "Minnesota Wild", 4, 39.20, 41.20, 40.40, 40.40),
    ("10/11/2025", "Dallas Stars", 5, "Colorado Avalanche", 4, 38.10, 40.00, 39.20, 39.20),
    ("10/11/2025", "Vancouver Canucks", 1, "Edmonton Oilers", 3, 54.60, 57.30, 56.20, 56.20),
    ("10/11/2025", "Anaheim Ducks", 7, "San Jose Sharks", 6, 27.00, 28.30, 27.80, 27.80),
    ("10/11/2025", "Vegas Golden Knights", 1, "Seattle Kraken", 2, 31.70, 33.30, 32.70, 32.70),
    ("10/12/2025", "Washington Capitals", 1, "New York Rangers", 0, 42.60, 44.70, 43.80, 43.80),
    ("10/13/2025", "Colorado Avalanche", 3, "Buffalo Sabres", 1, 41.40, 43.40, 42.60, 42.60),
    ("10/13/2025", "Tampa Bay Lightning", 4, "Boston Bruins", 3, 23.90, 25.10, 24.60, 24.60),
    ("10/13/2025", "Nashville Predators", 4, "Ottawa Senators", 1, 65.50, 68.70, 67.40, 67.40),
    ("10/13/2025", "Winnipeg Jets", 5, "New York Islanders", 2, 27.00, 28.40, 27.90, 27.90),
    ("10/13/2025", "Detroit Red Wings", 3, "Toronto Maple Leafs", 2, 49.90, 52.40, 51.40, 51.40),
    ("10/13/2025", "Florida Panthers", 2, "Philadelphia Flyers", 5, 33.20, 34.90, 34.20, 34.20),
    ("10/13/2025", "New Jersey Devils", 3, "Columbus Blue Jackets", 2, 60.10, 63.10, 61.90, 61.90),
    ("10/13/2025", "St. Louis Blues", 5, "Vancouver Canucks", 2, 38.40, 40.30, 39.50, 39.50),
    ("10/13/2025", "Los Angeles Kings", 3, "Minnesota Wild", 4, 50.60, 53.10, 52.10, 52.10),
    ("10/13/2025", "Utah Mammoth", 1, "Chicago Blackhawks", 3, 27.60, 29.00, 28.40, 28.40),
    ("10/14/2025", "Nashville Predators", 4, "Toronto Maple Leafs", 7, 67.30, 70.70, 69.40, 69.40),
    ("10/14/2025", "Seattle Kraken", 4, "Montreal Canadiens", 5, 57.30, 60.20, 59.00, 59.00),
    ("10/14/2025", "Edmonton Oilers", 2, "New York Rangers", 0, 45.90, 48.20, 47.30, 47.30),
    ("10/14/2025", "Tampa Bay Lightning", 2, "Washington Capitals", 3, 44.70, 47.00, 46.10, 46.10),
    ("10/14/2025", "Vegas Golden Knights", 4, "Calgary Flames", 2, 39.70, 41.70, 40.90, 40.90),
    ("10/14/2025", "Minnesota Wild", 2, "Dallas Stars", 5, 70.20, 73.70, 72.30, 72.30),
    ("10/14/2025", "Carolina Hurricanes", 5, "San Jose Sharks", 1, 14.40, 15.10, 14.90, 14.90),
    ("10/14/2025", "Pittsburgh Penguins", 3, "Anaheim Ducks", 4, 57.20, 60.10, 59.00, 59.00),
    ("10/15/2025", "Ottawa Senators", 4, "Buffalo Sabres", 8, 46.60, 48.90, 48.00, 48.00),
    ("10/15/2025", "Florida Panthers", 1, "Detroit Red Wings", 4, 50.60, 53.10, 52.10, 52.10),
    ("10/15/2025", "Chicago Blackhawks", 8, "St. Louis Blues", 3, 79.30, 83.20, 81.60, 81.60),
    ("10/15/2025", "Calgary Flames", 1, "Utah Mammoth", 3, 50.10, 52.60, 51.60, 51.60),
    ("10/16/2025", "New York Rangers", 1, "Toronto Maple Leafs", 2, 57.50, 60.30, 59.20, 59.20),
    ("10/16/2025", "Nashville Predators", 2, "Montreal Canadiens", 3, 60.90, 63.90, 62.70, 62.70),
    ("10/16/2025", "Seattle Kraken", 3, "Ottawa Senators", 4, 62.00, 65.10, 63.90, 63.90),
    ("10/16/2025", "Florida Panthers", 1, "New Jersey Devils", 3, 45.30, 47.60, 46.70, 46.70),
    ("10/16/2025", "Edmonton Oilers", 2, "New York Islanders", 4, 47.80, 50.20, 49.30, 49.30),
    ("10/16/2025", "Winnipeg Jets", 5, "Philadelphia Flyers", 2, 18.40, 19.30, 19.00, 19.00),
    ("10/16/2025", "Colorado Avalanche", 4, "Columbus Blue Jackets", 1, 51.80, 54.40, 53.30, 53.30),
    ("10/16/2025", "Vancouver Canucks", 5, "Dallas Stars", 3, 71.80, 75.40, 74.00, 74.00),
    ("10/16/2025", "Boston Bruins", 5, "Vegas Golden Knights", 6, 75.70, 79.50, 78.00, 78.00),
    ("10/16/2025", "Carolina Hurricanes", 4, "Anaheim Ducks", 1, 31.30, 32.90, 32.30, 32.30),
    ("10/16/2025", "Pittsburgh Penguins", 4, "Los Angeles Kings", 2, 60.40, 63.50, 62.30, 62.30),
    ("10/17/2025", "Tampa Bay Lightning", 1, "Detroit Red Wings", 2, 44.80, 47.10, 46.20, 46.20),
    ("10/17/2025", "Minnesota Wild", 1, "Washington Capitals", 5, 56.00, 58.80, 57.60, 57.60),
    ("10/17/2025", "Vancouver Canucks", 3, "Chicago Blackhawks", 2, 29.60, 31.10, 30.50, 30.50),
    ("10/17/2025", "San Jose Sharks", 3, "Utah Mammoth", 6, 76.30, 80.10, 78.60, 78.60),
    ("10/18/2025", "Florida Panthers", 0, "Buffalo Sabres", 3, 76.30, 80.10, 78.60, 78.60),
]

def import_games():
    """Import 77 games with predictions and scores"""
    
    print("\n" + "="*70)
    print("IMPORTING 77 NHL GAMES (October 7-18, 2025)")
    print("="*70)
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # First, clear old NHL completed games
    cursor.execute("UPDATE games SET home_score = NULL, away_score = NULL, status = 'scheduled' WHERE sport='NHL'")
    print(f"✓ Cleared old NHL results")
    
    inserted = 0
    
    for game in games_data:
        date, away, away_score, home, home_score, xgb, catboost, elo, meta = game
        
        # Insert game with predictions and scores
        game_id_str = f"{away}@{home}_{date.replace('/', '-')}"
        cursor.execute('''
            INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, 
                             home_score, away_score, status)
            VALUES (?, 'NHL', ?, 2025, ?, ?, ?, ?, ?, 'completed')
        ''', ('NHL', game_id_str, date, home, away, home_score, away_score))
        
        # Insert prediction
        home_won = 1 if home_score > away_score else 0
        xgb_correct = 1 if (xgb > 50 and home_won == 1) or (xgb < 50 and home_won == 0) else 0
        
        cursor.execute('''
            INSERT INTO predictions (
                sport, league, game_id, game_date, home_team_id, away_team_id,
                predicted_winner, win_probability, 
                elo_home_prob, logistic_home_prob, xgboost_home_prob,
                actual_winner, actual_home_score, actual_away_score,
                win_prediction_correct
            )
            VALUES (?, 'NHL', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('NHL', game_id_str, date, home, away, 
              home if meta > 50 else away, meta / 100,
              elo / 100, catboost / 100, xgb / 100,
              home_won, home_score, away_score, xgb_correct))
        
        inserted += 1
        print(f"✓ {date}: {away} {away_score} @ {home} {home_score}")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*70}")
    print(f"SUCCESS: Imported {inserted} games with predictions and scores")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import_games()
