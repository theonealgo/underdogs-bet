import sqlite3
import re
from datetime import datetime

# 2025 NFL game results from user's data
game_results_text = """
09/04/2025Dallas Cowboys20Philadelphia Eagles24
09/05/2025Kansas City Chiefs21Los Angeles Chargers27
09/07/2025Arizona Cardinals20New Orleans Saints13
09/07/2025Pittsburgh Steelers34New York Jets32
09/07/2025Miami Dolphins8Indianapolis Colts33
09/07/2025Tampa Bay Buccaneers23Atlanta Falcons20
09/07/2025New York Giants6Washington Commanders21
09/07/2025Carolina Panthers10Jacksonville Jaguars26
09/07/2025Cincinnati Bengals17Cleveland Browns16
09/07/2025Las Vegas Raiders20New England Patriots13
09/07/2025San Francisco 49ers17Seattle Seahawks13
09/07/2025Tennessee Titans12Denver Broncos20
09/07/2025Houston Texans9Los Angeles Rams14
09/07/2025Detroit Lions13Green Bay Packers27
09/07/2025Baltimore Ravens40Buffalo Bills41
09/08/2025Minnesota Vikings27Chicago Bears24
09/11/2025Washington Commanders18Green Bay Packers27
09/14/2025Los Angeles Rams33Tennessee Titans19
09/14/2025New York Giants37Dallas Cowboys40
09/14/2025San Francisco 49ers26New Orleans Saints21
09/14/2025Buffalo Bills30New York Jets10
09/14/2025New England Patriots33Miami Dolphins27
09/14/2025Seattle Seahawks31Pittsburgh Steelers17
09/14/2025Chicago Bears21Detroit Lions52
09/14/2025Jacksonville Jaguars27Cincinnati Bengals31
09/14/2025Cleveland Browns17Baltimore Ravens41
09/14/2025Carolina Panthers22Arizona Cardinals27
09/14/2025Denver Broncos28Indianapolis Colts29
09/14/2025Philadelphia Eagles20Kansas City Chiefs17
09/14/2025Atlanta Falcons22Minnesota Vikings6
09/15/2025Tampa Bay Buccaneers20Houston Texans19
09/15/2025Los Angeles Chargers20Las Vegas Raiders9
09/18/2025Miami Dolphins21Buffalo Bills31
09/21/2025Indianapolis Colts41Tennessee Titans20
09/21/2025Cincinnati Bengals10Minnesota Vikings48
09/21/2025Pittsburgh Steelers21New England Patriots14
09/21/2025Los Angeles Rams26Philadelphia Eagles33
09/21/2025New York Jets27Tampa Bay Buccaneers29
09/21/2025Las Vegas Raiders24Washington Commanders41
09/21/2025Atlanta Falcons0Carolina Panthers30
09/21/2025Houston Texans10Jacksonville Jaguars17
09/21/2025Green Bay Packers10Cleveland Browns13
09/21/2025New Orleans Saints13Seattle Seahawks44
09/21/2025Denver Broncos20Los Angeles Chargers23
09/21/2025Arizona Cardinals15San Francisco 49ers16
09/21/2025Dallas Cowboys14Chicago Bears31
09/21/2025Kansas City Chiefs22New York Giants9
09/22/2025Detroit Lions38Baltimore Ravens30
09/25/2025Seattle Seahawks23Arizona Cardinals20
09/28/2025Minnesota Vikings21Pittsburgh Steelers24
09/28/2025Tennessee Titans0Houston Texans26
09/28/2025Philadelphia Eagles31Tampa Bay Buccaneers25
09/28/2025Carolina Panthers13New England Patriots42
09/28/2025Los Angeles Chargers18New York Giants21
09/28/2025Washington Commanders27Atlanta Falcons34
09/28/2025New Orleans Saints19Buffalo Bills31
09/28/2025Cleveland Browns10Detroit Lions34
09/28/2025Indianapolis Colts20Los Angeles Rams27
09/28/2025Jacksonville Jaguars26San Francisco 49ers21
09/28/2025Chicago Bears25Las Vegas Raiders24
09/28/2025Baltimore Ravens20Kansas City Chiefs37
09/28/2025Green Bay Packers40Dallas Cowboys40
09/29/2025New York Jets21Miami Dolphins27
09/29/2025Cincinnati Bengals3Denver Broncos28
10/02/2025San Francisco 49ers26Los Angeles Rams23
10/05/2025Minnesota Vikings21Cleveland Browns17
10/05/2025New York Giants14New Orleans Saints26
10/05/2025Dallas Cowboys37New York Jets22
10/05/2025Denver Broncos21Philadelphia Eagles17
10/05/2025Houston Texans44Baltimore Ravens10
10/05/2025Las Vegas Raiders6Indianapolis Colts40
10/05/2025Miami Dolphins24Carolina Panthers27
10/05/2025Tampa Bay Buccaneers38Seattle Seahawks35
10/05/2025Tennessee Titans22Arizona Cardinals21
10/05/2025Washington Commanders27Los Angeles Chargers10
10/05/2025Detroit Lions37Cincinnati Bengals24
10/05/2025New England Patriots23Buffalo Bills20
10/06/2025Kansas City Chiefs28Jacksonville Jaguars31
10/09/2025Philadelphia Eagles17New York Giants34
10/12/2025Denver Broncos13New York Jets11
10/12/2025Cleveland Browns9Pittsburgh Steelers23
10/12/2025Dallas Cowboys27Carolina Panthers30
10/12/2025Seattle Seahawks20Jacksonville Jaguars12
10/12/2025Los Angeles Rams17Baltimore Ravens3
10/12/2025Arizona Cardinals27Indianapolis Colts31
10/12/2025Los Angeles Chargers29Miami Dolphins27
10/12/2025New England Patriots25New Orleans Saints19
10/12/2025Tennessee Titans10Las Vegas Raiders20
10/12/2025San Francisco 49ers19Tampa Bay Buccaneers30
10/12/2025Cincinnati Bengals18Green Bay Packers27
10/12/2025Detroit Lions17Kansas City Chiefs30
10/13/2025Buffalo Bills14Atlanta Falcons24
10/13/2025Chicago Bears25Washington Commanders24
10/16/2025Pittsburgh Steelers31Cincinnati Bengals33
"""

def parse_games(text):
    """Parse game results from text"""
    games = []
    pattern = r'(\d{2}/\d{2}/\d{4})([A-Za-z\s]+?)(\d+)([A-Za-z\s]+?)(\d+)'
    
    for line in text.strip().split('\n'):
        if not line.strip():
            continue
        match = re.match(pattern, line)
        if match:
            date_str = match.group(1)
            away_team = match.group(2).strip()
            away_score = int(match.group(3))
            home_team = match.group(4).strip()
            home_score = int(match.group(5))
            
            # Convert date from MM/DD/YYYY to DD/MM/YYYY
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
            date_formatted = date_obj.strftime('%d/%m/%Y')
            
            games.append({
                'date': date_formatted,
                'away_team': away_team,
                'home_team': home_team,
                'away_score': away_score,
                'home_score': home_score
            })
    
    return games

def import_to_database(games):
    """Import games into database"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    imported = 0
    updated = 0
    skipped = 0
    
    for game in games:
        # Generate game_id
        game_id = f"NFL_{game['date']}_{game['away_team'].replace(' ', '_')}_{game['home_team'].replace(' ', '_')}"
        
        # Check if game exists
        cursor.execute('''
            SELECT id, home_score, away_score FROM games 
            WHERE sport = 'NFL' AND home_team_id = ? AND away_team_id = ? AND game_date = ?
        ''', (game['home_team'], game['away_team'], game['date']))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update scores if they're missing or different
            if existing[1] is None or existing[2] is None:
                cursor.execute('''
                    UPDATE games 
                    SET home_score = ?, away_score = ?, status = 'final', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (game['home_score'], game['away_score'], existing[0]))
                updated += 1
                print(f"✓ Updated: {game['date']} {game['away_team']} @ {game['home_team']} ({game['away_score']}-{game['home_score']})")
            else:
                skipped += 1
        else:
            # Insert new game
            cursor.execute('''
                INSERT INTO games (sport, league, game_id, season, game_date, home_team_id, away_team_id, 
                                 home_score, away_score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', ('NFL', 'NFL', game_id, 2025, game['date'], game['home_team'], game['away_team'],
                  game['home_score'], game['away_score'], 'final'))
            imported += 1
            print(f"+ Added: {game['date']} {game['away_team']} @ {game['home_team']} ({game['away_score']}-{game['home_score']})")
    
    conn.commit()
    conn.close()
    
    return imported, updated, skipped

if __name__ == '__main__':
    print("🏈 IMPORTING 2025 NFL GAME RESULTS")
    print("="*80)
    
    games = parse_games(game_results_text)
    print(f"\nParsed {len(games)} games from 2025 season")
    print()
    
    imported, updated, skipped = import_to_database(games)
    
    print()
    print("="*80)
    print(f"✅ IMPORT COMPLETE")
    print(f"   New games added: {imported}")
    print(f"   Existing games updated: {updated}")
    print(f"   Skipped (already complete): {skipped}")
    print(f"   Total 2025 games: {imported + updated + skipped}")
