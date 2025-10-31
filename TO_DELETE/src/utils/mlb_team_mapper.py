"""MLB Team Name Mapping Utility"""

class MLBTeamMapper:
    """Maps between MLB team full names and abbreviations"""
    
    # Map abbreviations to full names
    ABBREV_TO_FULL = {
        'ARI': 'Arizona Diamondbacks',
        'ATL': 'Atlanta Braves',
        'BAL': 'Baltimore Orioles',
        'BOS': 'Boston Red Sox',
        'CHC': 'Chicago Cubs',
        'CHW': 'Chicago White Sox',
        'CIN': 'Cincinnati Reds',
        'CLE': 'Cleveland Guardians',
        'COL': 'Colorado Rockies',
        'DET': 'Detroit Tigers',
        'HOU': 'Houston Astros',
        'KC': 'Kansas City Royals',
        'LAA': 'Los Angeles Angels',
        'LAD': 'Los Angeles Dodgers',
        'MIA': 'Miami Marlins',
        'MIL': 'Milwaukee Brewers',
        'MIN': 'Minnesota Twins',
        'NYM': 'New York Mets',
        'NYY': 'New York Yankees',
        'OAK': 'Oakland Athletics',
        'PHI': 'Philadelphia Phillies',
        'PIT': 'Pittsburgh Pirates',
        'SD': 'San Diego Padres',
        'SEA': 'Seattle Mariners',
        'SF': 'San Francisco Giants',
        'STL': 'St. Louis Cardinals',
        'TB': 'Tampa Bay Rays',
        'TEX': 'Texas Rangers',
        'TOR': 'Toronto Blue Jays',
        'WSH': 'Washington Nationals'
    }
    
    # Reverse map for full name to abbreviation
    FULL_TO_ABBREV = {v: k for k, v in ABBREV_TO_FULL.items()}
    
    @classmethod
    def abbrev_to_full(cls, abbrev: str) -> str:
        """Convert abbreviation to full name"""
        return cls.ABBREV_TO_FULL.get(abbrev, abbrev)
    
    @classmethod
    def full_to_abbrev(cls, full_name: str) -> str:
        """Convert full name to abbreviation"""
        return cls.FULL_TO_ABBREV.get(full_name, full_name)
    
    @classmethod
    def normalize_team_name(cls, name: str) -> str:
        """
        Normalize team name to abbreviation
        Handles both full names and abbreviations
        """
        # Check if it's already an abbreviation
        if name in cls.ABBREV_TO_FULL:
            return name
        
        # Try full name lookup
        if name in cls.FULL_TO_ABBREV:
            return cls.FULL_TO_ABBREV[name]
        
        # Return as-is if not found
        return name
