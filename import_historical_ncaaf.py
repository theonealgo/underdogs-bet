"""
Import NCAAF 2025 Historical Results
Parses the historical results and populates database with actual scores
"""

import re
import sqlite3
from datetime import datetime
from src.data_storage.database import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_ncaaf_historical_data(text_data: str):
    """Parse NCAAF historical results from text"""
    games = []
    
    # Split by date headers
    date_pattern = r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})'
    
    # Find all dates
    dates = re.findall(date_pattern, text_data)
    sections = re.split(date_pattern, text_data)
    
    current_date = None
    for i, section in enumerate(sections):
        # Check if this is a date
        if re.match(date_pattern, section):
            current_date = section
        elif current_date and 'FINAL' in section:
            # Parse games in this section
            # Pattern: FINAL\n\nTeam1\nScore1\n\nTeam2\nScore2
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            
            j = 0
            while j < len(lines):
                if lines[j] == 'FINAL':
                    # Next non-empty lines should be: away_team, away_score, home_team, home_score
                    try:
                        j += 1
                        # Skip empty or ranking lines, get team name
                        away_team = ''
                        while j < len(lines) and (not lines[j] or lines[j].isdigit()):
                            if lines[j] and not lines[j].isdigit():
                                away_team = lines[j]
                                break
                            j += 1
                        else:
                            if j < len(lines):
                                away_team = lines[j]
                                j += 1
                        
                        # Get away score
                        if j < len(lines):
                            away_score = lines[j]
                            j += 1
                        
                        # Skip empty or ranking lines, get home team
                        home_team = ''
                        while j < len(lines) and (not lines[j] or lines[j].isdigit()):
                            if lines[j] and not lines[j].isdigit():
                                home_team = lines[j]
                                break
                            j += 1
                        else:
                            if j < len(lines):
                                home_team = lines[j]
                                j += 1
                        
                        # Get home score
                        if j < len(lines):
                            home_score = lines[j]
                            j += 1
                        
                        # Clean team names (remove rankings)
                        away_team = re.sub(r'^\d+\s*', '', away_team).strip()
                        home_team = re.sub(r'^\d+\s*', '', home_team).strip()
                        
                        if away_team and home_team and away_score.isdigit() and home_score.isdigit():
                            # Parse date
                            date_obj = datetime.strptime(current_date, '%A, %B %d, %Y')
                            game_date = date_obj.strftime('%d/%m/%Y')
                            
                            games.append({
                                'game_date': game_date,
                                'away_team': away_team,
                                'home_team': home_team,
                                'away_score': int(away_score),
                                'home_score': int(home_score),
                                'sport': 'NCAAF',
                                'league': 'NCAAF',
                                'season': 2025,
                                'status': 'final'
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse game: {e}")
                        j += 1
                else:
                    j += 1
    
    logger.info(f"Parsed {len(games)} NCAAF games")
    return games


def import_to_database(games):
    """Import games into database"""
    db = DatabaseManager()
    
    inserted = 0
    updated = 0
    
    with sqlite3.connect(db.db_path) as conn:
        for game in games:
            # Generate game_id
            game_id = f"NCAAF_{game['game_date'].replace('/', '')}_{game['away_team'][:3]}_{game['home_team'][:3]}"
            
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
    
    logger.info(f"Imported NCAAF games: {inserted} new, {updated} updated")
    return inserted, updated


def main():
    """Main import function"""
    # Read the attached file
    with open('attached_assets/Pasted-Wednesday-October-8-2025-FINAL-Missouri-St-22-Middle-Tenn-20-FINAL-Liberty-19-UTEP-8-Thursda-1760701979349_1760701979349.txt', 'r') as f:
        file_data = f.read()
    
    # Also parse the inline text data from user message
    inline_data = """Thursday, October 2, 2025
FINAL
Sam Houston
10
New Mexico St.
37
Friday, October 3, 2025
FINAL
West Virginia
24
23
BYU
38
FINAL
Charlotte
26
South Fla.
54
FINAL
Western Ky.
27
Delaware
24
FINAL
New Mexico
28
San Jose St.
35
FINAL
Colorado St.
24
San Diego St.
45
Saturday, October 4, 2025
FINAL
Minnesota
3
1
Ohio St.
42
FINAL
3
Miami (FL)
28
18
Florida St.
22
FINAL
Kent St.
0
5
Oklahoma
44
FINAL
Mississippi St.
9
6
Texas A&M
31
FINAL
7
Penn St.
37
UCLA
42
FINAL
9
Texas
21
Florida
29
FINAL
16
Vanderbilt
14
10
Alabama
30
FINAL
11
Texas Tech
35
Houston
11
FINAL
Kentucky
14
12
Georgia
35
FINAL
14
Iowa St.
30
Cincinnati
38
FINAL
Wisconsin
10
20
Michigan
24
FINAL
Boise St.
7
21
Notre Dame
28
FINAL
22
Illinois
43
Purdue
27
FINAL
24
Virginia
30
Louisville
27
FINAL
Kansas St.
34
Baylor
35
FINAL
Air Force
31
Navy
34
FINAL
Clemson
38
North Carolina
10
FINAL
Boston College
7
Pittsburgh
48
FINAL
Ohio
14
Ball St.
20
Thursday, September 25, 2025
FINAL
Army West Point
6
East Carolina
28
Friday, September 26, 2025
FINAL
8
Florida St.
38
Virginia
46
FINAL
24
TCU
24
Arizona St.
27
FINAL
Houston
27
Oregon St.
24
Saturday, September 27, 2025
FINAL
1
Ohio St.
24
Washington
6
FINAL
6
Oregon
30
3
Penn St.
24
FINAL
4
LSU
19
13
Ole Miss
24
FINAL
17
Alabama
24
5
Georgia
21
FINAL
Auburn
10
9
Texas A&M
16
FINAL
11
Indiana
20
Iowa
15
FINAL
Arizona
14
14
Iowa St.
39
Thursday, September 18, 2025
FINAL
Rice
28
Charlotte
17
Friday, September 19, 2025
FINAL
Tulsa
19
Oklahoma St.
12
FINAL
Iowa
38
Rutgers
28
Saturday, September 20, 2025
FINAL
Southeastern La.
10
3
LSU
56
Thursday, September 11, 2025
FINAL
NC State
34
Wake Forest
24"""
    
    logger.info("="*70)
    logger.info("IMPORTING NCAAF 2025 HISTORICAL RESULTS")
    logger.info("="*70)
    
    # Parse both sources
    all_data = inline_data + "\n" + file_data
    games = parse_ncaaf_historical_data(all_data)
    
    # Import to database
    inserted, updated = import_to_database(games)
    
    logger.info("="*70)
    logger.info(f"IMPORT COMPLETE: {inserted + updated} total games")
    logger.info("="*70)


if __name__ == "__main__":
    main()
