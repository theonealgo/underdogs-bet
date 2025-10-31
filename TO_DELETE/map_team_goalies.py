"""
Create team-to-goalie mappings for NHL teams
Maps each team to their primary starting goalie based on stats
"""

import sqlite3

DATABASE = 'sports_predictions.db'

# Team-to-goalie mappings (last names only to match NHL API data)
# Note: Where multiple goalies share last name (e.g., Thompson), using most games played
TEAM_GOALIES = {
    'Toronto Maple Leafs': 'Stolarz',
    'Boston Bruins': 'Swayman',
    'Florida Panthers': 'Bobrovsky',
    'Tampa Bay Lightning': 'Vasilevskiy',
    'Buffalo Sabres': 'Thompson',  # Logan Thompson
    'Detroit Red Wings': 'Talbot',
    'Ottawa Senators': 'Forsberg',
    'Montreal Canadiens': 'Montembeault',
    
    # Metropolitan
    'New York Rangers': 'Shesterkin',
    'Carolina Hurricanes': 'Andersen',
    'New Jersey Devils': 'Markstrom',
    'New York Islanders': 'Sorokin',
    'Washington Capitals': 'Thompson',  # Logan Thompson (same as Buffalo - will use same stats)
    'Pittsburgh Penguins': 'Jarry',
    'Columbus Blue Jackets': 'Merzlikins',
    'Philadelphia Flyers': 'Ersson',
    
    # Central
    'Winnipeg Jets': 'Hellebuyck',
    'Dallas Stars': 'Oettinger',
    'Colorado Avalanche': 'Georgiev',  # FIXED: Was incorrectly "Wedgewood"
    'Minnesota Wild': 'Gustavsson',
    'St. Louis Blues': 'Binnington',
    'Nashville Predators': 'Saros',
    'Utah Mammoth': 'Vejmelka',
    'Chicago Blackhawks': 'Mrazek',
    
    # Pacific  
    'Vegas Golden Knights': 'Hill',
    'Edmonton Oilers': 'Skinner',
    'Los Angeles Kings': 'Kuemper',
    'Vancouver Canucks': 'Demko',
    'Calgary Flames': 'Vladar',
    'Seattle Kraken': 'Daccord',
    'Anaheim Ducks': 'Dostal',
    'San Jose Sharks': 'Vanecek'
}

def create_team_goalie_table():
    """Create table mapping teams to their primary goalies"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS team_goalies (
            team_name TEXT PRIMARY KEY,
            goalie_name TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert mappings
    for team, goalie in TEAM_GOALIES.items():
        cursor.execute('''
            INSERT OR REPLACE INTO team_goalies (team_name, goalie_name)
            VALUES (?, ?)
        ''', (team, goalie))
        print(f"✓ {team}: {goalie}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Mapped {len(TEAM_GOALIES)} teams to their primary goalies")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CREATING TEAM-TO-GOALIE MAPPINGS")
    print("="*70)
    create_team_goalie_table()
