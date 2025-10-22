"""
Import actual NHL game results from October 7-18, 2025
Updates the database with actual scores for performance tracking
"""

import sqlite3
from datetime import datetime

DATABASE = 'sports_predictions.db'

# Actual game results from October 7-18, 2025
results = """10/07/2025	Chicago Blackhawks	2	Florida Panthers	3
10/07/2025	Pittsburgh Penguins	3	New York Rangers	0
10/07/2025	Colorado Avalanche	4	Los Angeles Kings	1
10/08/2025	Montreal Canadiens	2	Toronto Maple Leafs	5
10/08/2025	Boston Bruins	3	Washington Capitals	1
10/08/2025	Calgary Flames	4	Edmonton Oilers	3
10/08/2025	Los Angeles Kings	6	Vegas Golden Knights	5
10/09/2025	Chicago Blackhawks	3	Boston Bruins	4
10/09/2025	New York Rangers	4	Buffalo Sabres	0
10/09/2025	Montreal Canadiens	5	Detroit Red Wings	1
10/09/2025	Ottawa Senators	5	Tampa Bay Lightning	4
10/09/2025	Philadelphia Flyers	1	Florida Panthers	2
10/09/2025	New York Islanders	3	Pittsburgh Penguins	4
10/09/2025	New Jersey Devils	3	Carolina Hurricanes	6
10/09/2025	Minnesota Wild	5	St. Louis Blues	0
10/09/2025	Columbus Blue Jackets	1	Nashville Predators	2
10/09/2025	Dallas Stars	5	Winnipeg Jets	4
10/09/2025	Utah Mammoth	1	Colorado Avalanche	2
10/09/2025	Calgary Flames	1	Vancouver Canucks	5
10/09/2025	Vegas Golden Knights	4	San Jose Sharks	3
10/09/2025	Anaheim Ducks	1	Seattle Kraken	3
10/11/2025	Los Angeles Kings	2	Winnipeg Jets	3
10/11/2025	St. Louis Blues	4	Calgary Flames	2
10/11/2025	Buffalo Sabres	1	Boston Bruins	3
10/11/2025	Toronto Maple Leafs	3	Detroit Red Wings	6
10/11/2025	New Jersey Devils	5	Tampa Bay Lightning	3
10/11/2025	Ottawa Senators	2	Florida Panthers	6
10/11/2025	Washington Capitals	4	New York Islanders	2
10/11/2025	New York Rangers	6	Pittsburgh Penguins	1
10/11/2025	Philadelphia Flyers	3	Carolina Hurricanes	4
10/11/2025	Montreal Canadiens	3	Chicago Blackhawks	2
10/11/2025	Utah Mammoth	3	Nashville Predators	2
10/11/2025	Columbus Blue Jackets	7	Minnesota Wild	4
10/11/2025	Dallas Stars	5	Colorado Avalanche	4
10/11/2025	Vancouver Canucks	1	Edmonton Oilers	3
10/11/2025	Anaheim Ducks	7	San Jose Sharks	6
10/11/2025	Vegas Golden Knights	1	Seattle Kraken	2
10/12/2025	Washington Capitals	1	New York Rangers	0
10/13/2025	Colorado Avalanche	3	Buffalo Sabres	1
10/13/2025	Tampa Bay Lightning	4	Boston Bruins	3
10/13/2025	Nashville Predators	4	Ottawa Senators	1
10/13/2025	Winnipeg Jets	5	New York Islanders	2
10/13/2025	Detroit Red Wings	3	Toronto Maple Leafs	2
10/13/2025	Florida Panthers	2	Philadelphia Flyers	5
10/13/2025	New Jersey Devils	3	Columbus Blue Jackets	2
10/13/2025	St. Louis Blues	5	Vancouver Canucks	2
10/13/2025	Los Angeles Kings	3	Minnesota Wild	4
10/13/2025	Utah Mammoth	1	Chicago Blackhawks	3
10/14/2025	Nashville Predators	4	Toronto Maple Leafs	7
10/14/2025	Seattle Kraken	4	Montreal Canadiens	5
10/14/2025	Edmonton Oilers	2	New York Rangers	0
10/14/2025	Tampa Bay Lightning	2	Washington Capitals	3
10/14/2025	Vegas Golden Knights	4	Calgary Flames	2
10/14/2025	Minnesota Wild	2	Dallas Stars	5
10/14/2025	Carolina Hurricanes	5	San Jose Sharks	1
10/14/2025	Pittsburgh Penguins	3	Anaheim Ducks	4
10/15/2025	Ottawa Senators	4	Buffalo Sabres	8
10/15/2025	Florida Panthers	1	Detroit Red Wings	4
10/15/2025	Chicago Blackhawks	8	St. Louis Blues	3
10/15/2025	Calgary Flames	1	Utah Mammoth	3
10/16/2025	New York Rangers	1	Toronto Maple Leafs	2
10/16/2025	Nashville Predators	2	Montreal Canadiens	3
10/16/2025	Seattle Kraken	3	Ottawa Senators	4
10/16/2025	Florida Panthers	1	New Jersey Devils	3
10/16/2025	Edmonton Oilers	2	New York Islanders	4
10/16/2025	Winnipeg Jets	5	Philadelphia Flyers	2
10/16/2025	Colorado Avalanche	4	Columbus Blue Jackets	1
10/16/2025	Vancouver Canucks	5	Dallas Stars	3
10/16/2025	Boston Bruins	5	Vegas Golden Knights	6
10/16/2025	Carolina Hurricanes	4	Anaheim Ducks	1
10/16/2025	Pittsburgh Penguins	4	Los Angeles Kings	2
10/17/2025	Tampa Bay Lightning	1	Detroit Red Wings	2
10/17/2025	Minnesota Wild	1	Washington Capitals	5
10/17/2025	Vancouver Canucks	3	Chicago Blackhawks	2
10/17/2025	San Jose Sharks	3	Utah Mammoth	6
10/18/2025	Florida Panthers	0	Buffalo Sabres	3"""

def import_results():
    """Import actual game results into database"""
    
    print("\n" + "="*70)
    print("IMPORTING NHL ACTUAL GAME RESULTS (October 7-18, 2025)")
    print("="*70)
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    updated = 0
    not_found = 0
    
    for line in results.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        
        date = parts[0]
        away_team = parts[1]
        away_score = int(parts[2])
        home_team = parts[3]
        home_score = int(parts[4])
        
        # Find and update the game
        cursor.execute('''
            UPDATE games
            SET home_score = ?, away_score = ?, status = 'completed'
            WHERE sport = 'NHL'
            AND game_date = ?
            AND home_team_id = ?
            AND away_team_id = ?
        ''', (home_score, away_score, date, home_team, away_team))
        
        if cursor.rowcount > 0:
            updated += 1
            print(f"✓ {date}: {away_team} {away_score} @ {home_team} {home_score}")
        else:
            not_found += 1
            print(f"⚠ Not found: {date}: {away_team} @ {home_team}")
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"{'='*70}")
    print(f"Updated: {updated} games")
    print(f"Not found: {not_found} games")
    print(f"\n✅ Results imported successfully!\n")
    
    return updated

if __name__ == "__main__":
    import_results()
