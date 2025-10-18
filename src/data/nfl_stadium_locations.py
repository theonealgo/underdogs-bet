"""
NFL Stadium Locations and Weather Configuration

Coordinates for all 32 NFL stadiums to fetch weather data.
Dome/retractable roof stadiums marked to skip weather impact.
"""

NFL_STADIUMS = {
    # AFC East
    'Buffalo Bills': {
        'stadium': 'Highmark Stadium',
        'latitude': 42.7738,
        'longitude': -78.7870,
        'is_dome': False,
        'elevation': 187
    },
    'Miami Dolphins': {
        'stadium': 'Hard Rock Stadium',
        'latitude': 25.9580,
        'longitude': -80.2389,
        'is_dome': False,  # Open air with partial roof
        'elevation': 3
    },
    'New England Patriots': {
        'stadium': 'Gillette Stadium',
        'latitude': 42.0909,
        'longitude': -71.2643,
        'is_dome': False,
        'elevation': 48
    },
    'New York Jets': {
        'stadium': 'MetLife Stadium',
        'latitude': 40.8128,
        'longitude': -74.0742,
        'is_dome': False,
        'elevation': 3
    },
    
    # AFC North
    'Baltimore Ravens': {
        'stadium': 'M&T Bank Stadium',
        'latitude': 39.2780,
        'longitude': -76.6227,
        'is_dome': False,
        'elevation': 12
    },
    'Cincinnati Bengals': {
        'stadium': 'Paycor Stadium',
        'latitude': 39.0954,
        'longitude': -84.5160,
        'is_dome': False,
        'elevation': 146
    },
    'Cleveland Browns': {
        'stadium': 'Cleveland Browns Stadium',
        'latitude': 41.5061,
        'longitude': -81.6995,
        'is_dome': False,
        'elevation': 177
    },
    'Pittsburgh Steelers': {
        'stadium': 'Acrisure Stadium',
        'latitude': 40.4468,
        'longitude': -80.0158,
        'is_dome': False,
        'elevation': 227
    },
    
    # AFC South
    'Houston Texans': {
        'stadium': 'NRG Stadium',
        'latitude': 29.6847,
        'longitude': -95.4107,
        'is_dome': True,  # Retractable roof dome
        'elevation': 15
    },
    'Indianapolis Colts': {
        'stadium': 'Lucas Oil Stadium',
        'latitude': 39.7601,
        'longitude': -86.1639,
        'is_dome': True,  # Retractable roof dome
        'elevation': 217
    },
    'Jacksonville Jaguars': {
        'stadium': 'EverBank Stadium',
        'latitude': 30.3240,
        'longitude': -81.6373,
        'is_dome': False,
        'elevation': 5
    },
    'Tennessee Titans': {
        'stadium': 'Nissan Stadium',
        'latitude': 36.1665,
        'longitude': -86.7713,
        'is_dome': False,
        'elevation': 117
    },
    
    # AFC West
    'Denver Broncos': {
        'stadium': 'Empower Field at Mile High',
        'latitude': 39.7439,
        'longitude': -105.0201,
        'is_dome': False,
        'elevation': 1609  # Mile high elevation
    },
    'Kansas City Chiefs': {
        'stadium': 'GEHA Field at Arrowhead Stadium',
        'latitude': 39.0489,
        'longitude': -94.4839,
        'is_dome': False,
        'elevation': 287
    },
    'Las Vegas Raiders': {
        'stadium': 'Allegiant Stadium',
        'latitude': 36.0908,
        'longitude': -115.1833,
        'is_dome': True,  # Fixed roof dome
        'elevation': 610
    },
    'Los Angeles Chargers': {
        'stadium': 'SoFi Stadium',
        'latitude': 33.9535,
        'longitude': -118.3392,
        'is_dome': True,  # Fixed roof dome
        'elevation': 32
    },
    
    # NFC East
    'Dallas Cowboys': {
        'stadium': 'AT&T Stadium',
        'latitude': 32.7473,
        'longitude': -97.0945,
        'is_dome': True,  # Retractable roof dome
        'elevation': 171
    },
    'New York Giants': {
        'stadium': 'MetLife Stadium',
        'latitude': 40.8128,
        'longitude': -74.0742,
        'is_dome': False,
        'elevation': 3
    },
    'Philadelphia Eagles': {
        'stadium': 'Lincoln Financial Field',
        'latitude': 39.9008,
        'longitude': -75.1675,
        'is_dome': False,
        'elevation': 5
    },
    'Washington Commanders': {
        'stadium': 'Northwest Stadium',
        'latitude': 38.9076,
        'longitude': -76.8645,
        'is_dome': False,
        'elevation': 61
    },
    
    # NFC North
    'Chicago Bears': {
        'stadium': 'Soldier Field',
        'latitude': 41.8623,
        'longitude': -87.6167,
        'is_dome': False,
        'elevation': 180
    },
    'Detroit Lions': {
        'stadium': 'Ford Field',
        'latitude': 42.3400,
        'longitude': -83.0456,
        'is_dome': True,  # Fixed roof dome
        'elevation': 189
    },
    'Green Bay Packers': {
        'stadium': 'Lambeau Field',
        'latitude': 44.5013,
        'longitude': -88.0622,
        'is_dome': False,  # Famous for cold weather games
        'elevation': 195
    },
    'Minnesota Vikings': {
        'stadium': 'U.S. Bank Stadium',
        'latitude': 44.9738,
        'longitude': -93.2577,
        'is_dome': True,  # Fixed roof dome
        'elevation': 264
    },
    
    # NFC South
    'Atlanta Falcons': {
        'stadium': 'Mercedes-Benz Stadium',
        'latitude': 33.7553,
        'longitude': -84.4006,
        'is_dome': True,  # Retractable roof dome
        'elevation': 320
    },
    'Carolina Panthers': {
        'stadium': 'Bank of America Stadium',
        'latitude': 35.2258,
        'longitude': -80.8528,
        'is_dome': False,
        'elevation': 215
    },
    'New Orleans Saints': {
        'stadium': 'Caesars Superdome',
        'latitude': 29.9511,
        'longitude': -90.0812,
        'is_dome': True,  # Fixed roof dome
        'elevation': 2
    },
    'Tampa Bay Buccaneers': {
        'stadium': 'Raymond James Stadium',
        'latitude': 27.9759,
        'longitude': -82.5033,
        'is_dome': False,
        'elevation': 5
    },
    
    # NFC West
    'Arizona Cardinals': {
        'stadium': 'State Farm Stadium',
        'latitude': 33.5276,
        'longitude': -112.2626,
        'is_dome': True,  # Retractable roof dome
        'elevation': 345
    },
    'Los Angeles Rams': {
        'stadium': 'SoFi Stadium',
        'latitude': 33.9535,
        'longitude': -118.3392,
        'is_dome': True,  # Fixed roof dome
        'elevation': 32
    },
    'San Francisco 49ers': {
        'stadium': "Levi's Stadium",
        'latitude': 37.4032,
        'longitude': -121.9698,
        'is_dome': False,
        'elevation': 12
    },
    'Seattle Seahawks': {
        'stadium': 'Lumen Field',
        'latitude': 47.5952,
        'longitude': -122.3316,
        'is_dome': False,  # Partial roof but open air
        'elevation': 4
    }
}

def get_stadium_info(team_name: str) -> dict:
    """Get stadium information for a team"""
    return NFL_STADIUMS.get(team_name, {
        'latitude': 40.0,
        'longitude': -95.0,
        'is_dome': False,
        'elevation': 0
    })

def is_outdoor_stadium(team_name: str) -> bool:
    """Check if team plays in outdoor stadium (weather matters)"""
    stadium_info = get_stadium_info(team_name)
    return not stadium_info.get('is_dome', False)
