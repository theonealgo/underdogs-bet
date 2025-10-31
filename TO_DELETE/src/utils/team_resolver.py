"""
Team name resolver for normalizing team names across data sources
"""
import logging
from typing import Dict, Optional, List
import json

class TeamResolver:
    """Resolves team names from various sources to canonical team IDs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Canonical team mappings with aliases
        self.team_mappings = {
            'MLB': {
                'Arizona Diamondbacks': {'id': 'ARI', 'aliases': ['Arizona', 'D-backs', 'Diamondbacks']},
                'Atlanta Braves': {'id': 'ATL', 'aliases': ['Atlanta', 'Braves']},
                'Baltimore Orioles': {'id': 'BAL', 'aliases': ['Baltimore', 'Orioles']},
                'Boston Red Sox': {'id': 'BOS', 'aliases': ['Boston', 'Red Sox']},
                'Chicago Cubs': {'id': 'CHC', 'aliases': ['Chicago Cubs', 'Cubs']},
                'Chicago White Sox': {'id': 'CWS', 'aliases': ['Chicago White Sox', 'White Sox', 'Chi White Sox']},
                'Cincinnati Reds': {'id': 'CIN', 'aliases': ['Cincinnati', 'Reds']},
                'Cleveland Guardians': {'id': 'CLE', 'aliases': ['Cleveland', 'Guardians', 'Cleveland Indians']},
                'Colorado Rockies': {'id': 'COL', 'aliases': ['Colorado', 'Rockies']},
                'Detroit Tigers': {'id': 'DET', 'aliases': ['Detroit', 'Tigers']},
                'Houston Astros': {'id': 'HOU', 'aliases': ['Houston', 'Astros']},
                'Kansas City Royals': {'id': 'KC', 'aliases': ['Kansas City', 'Royals', 'KC Royals']},
                'Los Angeles Angels': {'id': 'LAA', 'aliases': ['LA Angels', 'Angels', 'Los Angeles Angels']},
                'Los Angeles Dodgers': {'id': 'LAD', 'aliases': ['LA Dodgers', 'Dodgers', 'Los Angeles Dodgers']},
                'Miami Marlins': {'id': 'MIA', 'aliases': ['Miami', 'Marlins', 'Florida Marlins']},
                'Milwaukee Brewers': {'id': 'MIL', 'aliases': ['Milwaukee', 'Brewers']},
                'Minnesota Twins': {'id': 'MIN', 'aliases': ['Minnesota', 'Twins']},
                'New York Mets': {'id': 'NYM', 'aliases': ['NY Mets', 'Mets', 'New York Mets']},
                'New York Yankees': {'id': 'NYY', 'aliases': ['NY Yankees', 'Yankees', 'New York Yankees']},
                'Oakland Athletics': {'id': 'OAK', 'aliases': ['Oakland', 'Athletics', "A's", 'Oakland A\'s']},
                'Philadelphia Phillies': {'id': 'PHI', 'aliases': ['Philadelphia', 'Phillies']},
                'Pittsburgh Pirates': {'id': 'PIT', 'aliases': ['Pittsburgh', 'Pirates']},
                'San Diego Padres': {'id': 'SD', 'aliases': ['San Diego', 'Padres', 'SD Padres']},
                'San Francisco Giants': {'id': 'SF', 'aliases': ['San Francisco', 'Giants', 'SF Giants']},
                'Seattle Mariners': {'id': 'SEA', 'aliases': ['Seattle', 'Mariners']},
                'St. Louis Cardinals': {'id': 'STL', 'aliases': ['St Louis', 'Cardinals', 'St. Louis Cardinals', 'Saint Louis Cardinals']},
                'Tampa Bay Rays': {'id': 'TB', 'aliases': ['Tampa Bay', 'Rays', 'Tampa']},
                'Texas Rangers': {'id': 'TEX', 'aliases': ['Texas', 'Rangers']},
                'Toronto Blue Jays': {'id': 'TOR', 'aliases': ['Toronto', 'Blue Jays', 'Jays']},
                'Washington Nationals': {'id': 'WSH', 'aliases': ['Washington', 'Nationals', 'Nats']},
            },
            'NBA': {
                'Atlanta Hawks': {'id': 'ATL', 'aliases': ['Atlanta', 'Hawks']},
                'Boston Celtics': {'id': 'BOS', 'aliases': ['Boston', 'Celtics']},
                'Brooklyn Nets': {'id': 'BKN', 'aliases': ['Brooklyn', 'Nets']},
                'Charlotte Hornets': {'id': 'CHA', 'aliases': ['Charlotte', 'Hornets']},
                'Chicago Bulls': {'id': 'CHI', 'aliases': ['Chicago', 'Bulls']},
                'Cleveland Cavaliers': {'id': 'CLE', 'aliases': ['Cleveland', 'Cavaliers', 'Cavs']},
                'Dallas Mavericks': {'id': 'DAL', 'aliases': ['Dallas', 'Mavericks', 'Mavs']},
                'Denver Nuggets': {'id': 'DEN', 'aliases': ['Denver', 'Nuggets']},
                'Detroit Pistons': {'id': 'DET', 'aliases': ['Detroit', 'Pistons']},
                'Golden State Warriors': {'id': 'GSW', 'aliases': ['Golden State', 'Warriors', 'GS Warriors']},
                'Houston Rockets': {'id': 'HOU', 'aliases': ['Houston', 'Rockets']},
                'Indiana Pacers': {'id': 'IND', 'aliases': ['Indiana', 'Pacers']},
                'Los Angeles Clippers': {'id': 'LAC', 'aliases': ['LA Clippers', 'Clippers', 'Los Angeles Clippers']},
                'Los Angeles Lakers': {'id': 'LAL', 'aliases': ['LA Lakers', 'Lakers', 'Los Angeles Lakers']},
                'Memphis Grizzlies': {'id': 'MEM', 'aliases': ['Memphis', 'Grizzlies']},
                'Miami Heat': {'id': 'MIA', 'aliases': ['Miami', 'Heat']},
                'Milwaukee Bucks': {'id': 'MIL', 'aliases': ['Milwaukee', 'Bucks']},
                'Minnesota Timberwolves': {'id': 'MIN', 'aliases': ['Minnesota', 'Timberwolves', 'T-Wolves']},
                'New Orleans Pelicans': {'id': 'NOP', 'aliases': ['New Orleans', 'Pelicans', 'NO Pelicans']},
                'New York Knicks': {'id': 'NYK', 'aliases': ['New York', 'Knicks', 'NY Knicks']},
                'Oklahoma City Thunder': {'id': 'OKC', 'aliases': ['Oklahoma City', 'Thunder', 'OKC Thunder']},
                'Orlando Magic': {'id': 'ORL', 'aliases': ['Orlando', 'Magic']},
                'Philadelphia 76ers': {'id': 'PHI', 'aliases': ['Philadelphia', '76ers', 'Sixers', 'Philly']},
                'Phoenix Suns': {'id': 'PHX', 'aliases': ['Phoenix', 'Suns']},
                'Portland Trail Blazers': {'id': 'POR', 'aliases': ['Portland', 'Trail Blazers', 'Blazers']},
                'Sacramento Kings': {'id': 'SAC', 'aliases': ['Sacramento', 'Kings']},
                'San Antonio Spurs': {'id': 'SAS', 'aliases': ['San Antonio', 'Spurs', 'SA Spurs']},
                'Toronto Raptors': {'id': 'TOR', 'aliases': ['Toronto', 'Raptors']},
                'Utah Jazz': {'id': 'UTA', 'aliases': ['Utah', 'Jazz']},
                'Washington Wizards': {'id': 'WAS', 'aliases': ['Washington', 'Wizards']},
            },
            'NFL': {
                'Arizona Cardinals': {'id': 'ARI', 'aliases': ['Arizona', 'Cardinals']},
                'Atlanta Falcons': {'id': 'ATL', 'aliases': ['Atlanta', 'Falcons']},
                'Baltimore Ravens': {'id': 'BAL', 'aliases': ['Baltimore', 'Ravens']},
                'Buffalo Bills': {'id': 'BUF', 'aliases': ['Buffalo', 'Bills']},
                'Carolina Panthers': {'id': 'CAR', 'aliases': ['Carolina', 'Panthers']},
                'Chicago Bears': {'id': 'CHI', 'aliases': ['Chicago', 'Bears']},
                'Cincinnati Bengals': {'id': 'CIN', 'aliases': ['Cincinnati', 'Bengals']},
                'Cleveland Browns': {'id': 'CLE', 'aliases': ['Cleveland', 'Browns']},
                'Dallas Cowboys': {'id': 'DAL', 'aliases': ['Dallas', 'Cowboys']},
                'Denver Broncos': {'id': 'DEN', 'aliases': ['Denver', 'Broncos']},
                'Detroit Lions': {'id': 'DET', 'aliases': ['Detroit', 'Lions']},
                'Green Bay Packers': {'id': 'GB', 'aliases': ['Green Bay', 'Packers']},
                'Houston Texans': {'id': 'HOU', 'aliases': ['Houston', 'Texans']},
                'Indianapolis Colts': {'id': 'IND', 'aliases': ['Indianapolis', 'Colts']},
                'Jacksonville Jaguars': {'id': 'JAX', 'aliases': ['Jacksonville', 'Jaguars', 'Jax']},
                'Kansas City Chiefs': {'id': 'KC', 'aliases': ['Kansas City', 'Chiefs', 'KC Chiefs']},
                'Las Vegas Raiders': {'id': 'LV', 'aliases': ['Las Vegas', 'Raiders', 'Oakland Raiders']},
                'Los Angeles Chargers': {'id': 'LAC', 'aliases': ['LA Chargers', 'Chargers', 'Los Angeles Chargers']},
                'Los Angeles Rams': {'id': 'LAR', 'aliases': ['LA Rams', 'Rams', 'Los Angeles Rams']},
                'Miami Dolphins': {'id': 'MIA', 'aliases': ['Miami', 'Dolphins']},
                'Minnesota Vikings': {'id': 'MIN', 'aliases': ['Minnesota', 'Vikings']},
                'New England Patriots': {'id': 'NE', 'aliases': ['New England', 'Patriots']},
                'New Orleans Saints': {'id': 'NO', 'aliases': ['New Orleans', 'Saints']},
                'New York Giants': {'id': 'NYG', 'aliases': ['NY Giants', 'Giants', 'New York Giants']},
                'New York Jets': {'id': 'NYJ', 'aliases': ['NY Jets', 'Jets', 'New York Jets']},
                'Philadelphia Eagles': {'id': 'PHI', 'aliases': ['Philadelphia', 'Eagles']},
                'Pittsburgh Steelers': {'id': 'PIT', 'aliases': ['Pittsburgh', 'Steelers']},
                'San Francisco 49ers': {'id': 'SF', 'aliases': ['San Francisco', '49ers', 'SF 49ers']},
                'Seattle Seahawks': {'id': 'SEA', 'aliases': ['Seattle', 'Seahawks']},
                'Tampa Bay Buccaneers': {'id': 'TB', 'aliases': ['Tampa Bay', 'Buccaneers', 'Bucs']},
                'Tennessee Titans': {'id': 'TEN', 'aliases': ['Tennessee', 'Titans']},
                'Washington Commanders': {'id': 'WAS', 'aliases': ['Washington', 'Commanders', 'Washington Football Team']},
            },
            'NHL': {
                'Anaheim Ducks': {'id': 'ANA', 'aliases': ['Anaheim', 'Ducks']},
                'Arizona Coyotes': {'id': 'ARI', 'aliases': ['Arizona', 'Coyotes']},
                'Boston Bruins': {'id': 'BOS', 'aliases': ['Boston', 'Bruins']},
                'Buffalo Sabres': {'id': 'BUF', 'aliases': ['Buffalo', 'Sabres']},
                'Calgary Flames': {'id': 'CGY', 'aliases': ['Calgary', 'Flames']},
                'Carolina Hurricanes': {'id': 'CAR', 'aliases': ['Carolina', 'Hurricanes']},
                'Chicago Blackhawks': {'id': 'CHI', 'aliases': ['Chicago', 'Blackhawks']},
                'Colorado Avalanche': {'id': 'COL', 'aliases': ['Colorado', 'Avalanche', 'Avs']},
                'Columbus Blue Jackets': {'id': 'CBJ', 'aliases': ['Columbus', 'Blue Jackets']},
                'Dallas Stars': {'id': 'DAL', 'aliases': ['Dallas', 'Stars']},
                'Detroit Red Wings': {'id': 'DET', 'aliases': ['Detroit', 'Red Wings']},
                'Edmonton Oilers': {'id': 'EDM', 'aliases': ['Edmonton', 'Oilers']},
                'Florida Panthers': {'id': 'FLA', 'aliases': ['Florida', 'Panthers']},
                'Los Angeles Kings': {'id': 'LAK', 'aliases': ['LA Kings', 'Kings', 'Los Angeles Kings']},
                'Minnesota Wild': {'id': 'MIN', 'aliases': ['Minnesota', 'Wild']},
                'Montreal Canadiens': {'id': 'MTL', 'aliases': ['Montreal', 'Canadiens', 'Habs']},
                'Nashville Predators': {'id': 'NSH', 'aliases': ['Nashville', 'Predators', 'Preds']},
                'New Jersey Devils': {'id': 'NJD', 'aliases': ['New Jersey', 'Devils', 'NJ Devils']},
                'New York Islanders': {'id': 'NYI', 'aliases': ['NY Islanders', 'Islanders', 'New York Islanders']},
                'New York Rangers': {'id': 'NYR', 'aliases': ['NY Rangers', 'Rangers', 'New York Rangers']},
                'Ottawa Senators': {'id': 'OTT', 'aliases': ['Ottawa', 'Senators', 'Sens']},
                'Philadelphia Flyers': {'id': 'PHI', 'aliases': ['Philadelphia', 'Flyers']},
                'Pittsburgh Penguins': {'id': 'PIT', 'aliases': ['Pittsburgh', 'Penguins', 'Pens']},
                'San Jose Sharks': {'id': 'SJS', 'aliases': ['San Jose', 'Sharks', 'SJ Sharks']},
                'Seattle Kraken': {'id': 'SEA', 'aliases': ['Seattle', 'Kraken']},
                'St. Louis Blues': {'id': 'STL', 'aliases': ['St Louis', 'Blues', 'St. Louis Blues']},
                'Tampa Bay Lightning': {'id': 'TBL', 'aliases': ['Tampa Bay', 'Lightning', 'Tampa']},
                'Toronto Maple Leafs': {'id': 'TOR', 'aliases': ['Toronto', 'Maple Leafs', 'Leafs']},
                'Vancouver Canucks': {'id': 'VAN', 'aliases': ['Vancouver', 'Canucks']},
                'Vegas Golden Knights': {'id': 'VGK', 'aliases': ['Vegas', 'Golden Knights', 'Las Vegas Golden Knights']},
                'Washington Capitals': {'id': 'WSH', 'aliases': ['Washington', 'Capitals', 'Caps']},
                'Winnipeg Jets': {'id': 'WPG', 'aliases': ['Winnipeg', 'Jets']},
            },
            'NCAAF': {
                'Alabama Crimson Tide': {'id': 'ALA', 'aliases': ['Alabama', 'Crimson Tide']},
                'Auburn Tigers': {'id': 'AUB', 'aliases': ['Auburn', 'Tigers']},
                'Georgia Bulldogs': {'id': 'UGA', 'aliases': ['Georgia', 'Bulldogs', 'UGA']},
                'LSU Tigers': {'id': 'LSU', 'aliases': ['LSU']},
                'Texas A&M Aggies': {'id': 'TAMU', 'aliases': ['Texas A&M', 'Aggies', 'A&M']},
                'Florida Gators': {'id': 'FLA', 'aliases': ['Florida', 'Gators']},
                'Tennessee Volunteers': {'id': 'TENN', 'aliases': ['Tennessee', 'Volunteers', 'Vols']},
                'South Carolina Gamecocks': {'id': 'SC', 'aliases': ['South Carolina', 'Gamecocks']},
                'Ohio State Buckeyes': {'id': 'OSU', 'aliases': ['Ohio State', 'Buckeyes', 'OSU']},
                'Michigan Wolverines': {'id': 'MICH', 'aliases': ['Michigan', 'Wolverines']},
                'Penn State Nittany Lions': {'id': 'PSU', 'aliases': ['Penn State', 'Nittany Lions', 'PSU']},
                'USC Trojans': {'id': 'USC', 'aliases': ['USC', 'Southern California', 'Trojans']},
                'Clemson Tigers': {'id': 'CLEM', 'aliases': ['Clemson', 'Tigers']},
                'Notre Dame Fighting Irish': {'id': 'ND', 'aliases': ['Notre Dame', 'Fighting Irish', 'ND']},
                'Oklahoma Sooners': {'id': 'OU', 'aliases': ['Oklahoma', 'Sooners', 'OU']},
                'Texas Longhorns': {'id': 'TEX', 'aliases': ['Texas', 'Longhorns']},
                'Oregon Ducks': {'id': 'ORE', 'aliases': ['Oregon', 'Ducks']},
                'Washington Huskies': {'id': 'WASH', 'aliases': ['Washington', 'Huskies']},
            },
            'NCAAB': {
                'Duke Blue Devils': {'id': 'DUKE', 'aliases': ['Duke', 'Blue Devils']},
                'North Carolina Tar Heels': {'id': 'UNC', 'aliases': ['North Carolina', 'Tar Heels', 'UNC']},
                'Kentucky Wildcats': {'id': 'UK', 'aliases': ['Kentucky', 'Wildcats', 'UK']},
                'Kansas Jayhawks': {'id': 'KU', 'aliases': ['Kansas', 'Jayhawks', 'KU']},
                'Villanova Wildcats': {'id': 'NOVA', 'aliases': ['Villanova', 'Wildcats', 'Nova']},
                'Gonzaga Bulldogs': {'id': 'GONZ', 'aliases': ['Gonzaga', 'Bulldogs']},
                'Michigan State Spartans': {'id': 'MSU', 'aliases': ['Michigan State', 'Spartans', 'MSU']},
            },
            'WNBA': {
                'Atlanta Dream': {'id': 'ATL', 'aliases': ['Atlanta', 'Dream']},
                'Chicago Sky': {'id': 'CHI', 'aliases': ['Chicago', 'Sky']},
                'Connecticut Sun': {'id': 'CON', 'aliases': ['Connecticut', 'Sun']},
                'Dallas Wings': {'id': 'DAL', 'aliases': ['Dallas', 'Wings']},
                'Indiana Fever': {'id': 'IND', 'aliases': ['Indiana', 'Fever']},
                'Las Vegas Aces': {'id': 'LV', 'aliases': ['Las Vegas', 'Aces']},
                'Los Angeles Sparks': {'id': 'LA', 'aliases': ['Los Angeles', 'Sparks', 'LA Sparks']},
                'Minnesota Lynx': {'id': 'MIN', 'aliases': ['Minnesota', 'Lynx']},
                'New York Liberty': {'id': 'NY', 'aliases': ['New York', 'Liberty', 'NY Liberty']},
                'Phoenix Mercury': {'id': 'PHX', 'aliases': ['Phoenix', 'Mercury']},
                'Seattle Storm': {'id': 'SEA', 'aliases': ['Seattle', 'Storm']},
                'Washington Mystics': {'id': 'WAS', 'aliases': ['Washington', 'Mystics']},
            }
        }
        
        # Build reverse lookup cache (alias -> team_id)
        self._build_lookup_cache()
    
    def _build_lookup_cache(self):
        """Build reverse lookup cache for fast resolution"""
        self.lookup_cache = {}
        
        for sport, teams in self.team_mappings.items():
            self.lookup_cache[sport] = {}
            
            for full_name, data in teams.items():
                team_id = data['id']
                
                # Add full name
                self.lookup_cache[sport][full_name.lower()] = team_id
                
                # Add team ID itself
                self.lookup_cache[sport][team_id.lower()] = team_id
                
                # Add all aliases
                for alias in data['aliases']:
                    self.lookup_cache[sport][alias.lower()] = team_id
    
    def resolve(self, sport: str, team_name: str) -> Optional[str]:
        """
        Resolve team name to canonical team ID
        
        Args:
            sport: Sport code (MLB, NBA, etc.)
            team_name: Team name from any source
            
        Returns:
            Canonical team ID or None if not found
        """
        if not team_name:
            return None
        
        # Check cache
        if sport in self.lookup_cache:
            team_id = self.lookup_cache[sport].get(team_name.lower())
            if team_id:
                return team_id
        
        # Try partial matching as fallback
        team_name_lower = team_name.lower()
        if sport in self.lookup_cache:
            for key, team_id in self.lookup_cache[sport].items():
                if team_name_lower in key or key in team_name_lower:
                    self.logger.info(f"Partial match: {team_name} -> {team_id}")
                    return team_id
        
        self.logger.warning(f"Could not resolve team: {sport}/{team_name}")
        return None
    
    def get_full_name(self, sport: str, team_id: str) -> Optional[str]:
        """Get full team name from team ID"""
        if sport not in self.team_mappings:
            return team_id
        
        for full_name, data in self.team_mappings[sport].items():
            if data['id'] == team_id:
                return full_name
        
        return team_id
