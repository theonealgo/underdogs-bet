from typing import Dict, List, Tuple


def _canon_name(v: str) -> str:
    return " ".join((v or "").strip().split())


_TEAM_ALIASES: Dict[str, str] = {
    # NBA full names
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}


def _canon_team(v: str) -> str:
    raw = (v or "").strip()
    up = raw.upper()
    return _TEAM_ALIASES.get(up, up)


# Strict Top-50 pools supplied by user.
_TOP50_BY_LEAGUE: Dict[str, Dict[str, str]] = {
    "NBA": {
        "Nikola Jokić": "DEN",
        "Luka Dončić": "DAL",
        "Shai Gilgeous-Alexander": "OKC",
        "Giannis Antetokounmpo": "MIL",
        "Victor Wembanyama": "SAS",
        "Jayson Tatum": "BOS",
        "Anthony Edwards": "MIN",
        "Joel Embiid": "PHI",
        "Kevin Durant": "PHR",
        "Stephen Curry": "GSW",
        "Devin Booker": "PHR",
        "Anthony Davis": "LAL",
        "Jalen Brunson": "NYK",
        "Tyrese Haliburton": "IND",
        "LeBron James": "LAL",
        "Ja Morant": "MEM",
        "Donovan Mitchell": "CLE",
        "Kawhi Leonard": "LAC",
        "Kyrie Irving": "DAL",
        "Trae Young": "ATL",
        "Cooper Flagg": "BKN",
        "Lauri Markkanen": "UTA",
        "Chet Holmgren": "OKC",
        "Franz Wagner": "ORL",
        "Dejounte Murray": "NOP",
        "Paolo Banchero": "ORL",
        "Tyrese Maxey": "PHI",
        "Bam Adebayo": "MIA",
        "De'Aaron Fox": "SAC",
        "Jaylen Brown": "BOS",
        "Jimmy Butler": "MIA",
        "Paul George": "PHI",
        "Domantas Sabonis": "SAC",
        "Jalen Williams": "OKC",
        "Damian Lillard": "MIL",
        "Pascal Siakam": "IND",
        "Cade Cunningham": "DET",
        "Scottie Barnes": "TOR",
        "Brandon Miller": "CHA",
        "Alperen Şengün": "HOU",
        "LaMelo Ball": "CHA",
        "Zion Williamson": "NOP",
        "Jamal Murray": "DEN",
        "Karl-Anthony Towns": "NYK",
        "Evan Mobley": "CLE",
        "Rudy Gobert": "MIN",
        "Kristaps Porziņģis": "BOS",
        "Brandon Ingram": "NOP",
        "Julius Randle": "MIN",
    },
}


def is_scoped_player(league: str, player_name: str, team: str) -> bool:
    pool = _TOP50_BY_LEAGUE.get((league or "").upper())
    if not pool:
        return True
    return pool.get(_canon_name(player_name)) == _canon_team(team)


def filter_scoped_players(league: str, players: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    pool = _TOP50_BY_LEAGUE.get((league or "").upper())
    if not pool:
        return players or [], []
    in_scope: List[Dict] = []
    excluded: List[Dict] = []
    for p in players or []:
        if is_scoped_player(league, p.get("name", ""), p.get("team", "")):
            in_scope.append(p)
        else:
            excluded.append(
                {
                    "player_id": p.get("player_id"),
                    "name": p.get("name"),
                    "team": p.get("team"),
                    "reasons": ["PLAYER_OUT_OF_SCOPE"],
                }
            )
    return in_scope, excluded
