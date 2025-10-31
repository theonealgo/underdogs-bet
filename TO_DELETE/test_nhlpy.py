#!/usr/bin/env python3
"""
Test nhl-api-py library to explore available data
"""

from nhlpy import NHLClient

client = NHLClient()

print("="*70)
print("TESTING NHL-API-PY LIBRARY")
print("="*70)

# Test 1: Get goalie stats
print("\n1. GOALIE STATS (Current Season)")
print("-"*70)
goalie_stats = client.stats.goalie_stats_summary(
    start_season="20252026", 
    end_season="20252026"
)

# Show first 5 goalies
for i, goalie in enumerate(goalie_stats[:5]):
    name = goalie.get('goalieFullName', 'Unknown')
    save_pct = goalie.get('savePct', 0)
    gaa = goalie.get('goalsAgainstAverage', 0)
    wins = goalie.get('wins', 0)
    games = goalie.get('gamesPlayed', 0)
    print(f"{i+1}. {name}: {save_pct:.3f} SV%, {gaa:.2f} GAA, {wins}W in {games}GP")

# Test 2: Get teams
print("\n2. NHL TEAMS")
print("-"*70)
teams = client.teams.teams()
print(f"Found {len(teams)} teams")
for team in teams[:5]:
    name = team.get('fullName', team.get('name', 'Unknown'))
    abbr = team.get('abbrev', team.get('abbr', '???'))
    print(f"  {abbr}: {name}")

# Test 3: Get team roster (with goalies)
print("\n3. TEAM ROSTER - Toronto Maple Leafs")
print("-"*70)
roster = client.teams.team_roster(team_abbr="TOR", season="20252026")
print(f"Forwards: {len(roster.get('forwards', []))}")
print(f"Defensemen: {len(roster.get('defensemen', []))}")
print(f"Goalies: {len(roster.get('goalies', []))}")

if roster.get('goalies'):
    print("\nGoalies on roster:")
    for goalie in roster['goalies']:
        name = goalie.get('firstName', {}).get('default', '') + ' ' + goalie.get('lastName', {}).get('default', '')
        print(f"  - {name.strip()}")

# Test 4: Get team stats
print("\n4. TEAM STATS (Current Season)")
print("-"*70)
team_stats = client.stats.team_summary(
    start_season="20252026",
    end_season="20252026"
)

# Show first 3 teams
for i, team in enumerate(team_stats[:3]):
    name = team.get('teamFullName', 'Unknown')
    gp = team.get('gamesPlayed', 0)
    gf = team.get('goalsFor', 0)
    ga = team.get('goalsAgainst', 0)
    print(f"{i+1}. {name}: {gf} GF, {ga} GA in {gp} GP")

print("\n" + "="*70)
print("✓ Library test complete!")
print("="*70)
