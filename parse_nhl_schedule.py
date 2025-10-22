#!/usr/bin/env python3
"""
Parse NHL schedule from TSV file and generate nhlschedules.py code
"""

# Read the schedule file
with open('attached_assets/Pasted-Date-Start-Time-Sask-Start-Time-ET-Visitor-Score-Home-Score-Status-Visitor-Goalie-Home-Goalie-2-1761156364118_1761156364120.txt', 'r') as f:
    lines = f.readlines()

# Skip header
games = []
match_id = 1

for line in lines[1:]:  # Skip header row
    parts = line.strip().split('\t')
    if len(parts) < 6:
        continue
    
    date = parts[0]  # YYYY-MM-DD format
    visitor = parts[3]
    visitor_score = parts[4] if len(parts) > 4 and parts[4].strip() else None
    home = parts[5]
    home_score = parts[6] if len(parts) > 6 and parts[6].strip() else None
    
    # Convert date from YYYY-MM-DD to DD/MM/YYYY
    if date and '-' in date:
        year, month, day = date.split('-')
        formatted_date = f"{day}/{month}/{year}"
        
        game = {
            'match_id': match_id,
            'date': formatted_date,
            'home_team': home,
            'away_team': visitor
        }
        
        # Add scores if available
        if visitor_score and home_score:
            try:
                game['home_score'] = int(home_score)
                game['away_score'] = int(visitor_score)
            except ValueError:
                pass
        
        games.append(game)
        match_id += 1

print(f"Total games parsed: {len(games)}")
print(f"\nFirst 5 games:")
for game in games[:5]:
    print(game)

print(f"\nLast 5 games:")
for game in games[-5:]:
    print(game)

# Generate the Python code
print("\n\n# Generating Python code for nhlschedules.py...")
print(f"# Total games: {len(games)}")

# Write to output file
with open('nhl_2025_schedule_code.txt', 'w') as f:
    f.write("    nhl_2025_schedule = [\n")
    for game in games:
        f.write(f"        {game},\n")
    f.write("    ]\n")
    f.write("    return nhl_2025_schedule\n")

print("\n✅ Schedule code written to: nhl_2025_schedule_code.txt")
print(f"✅ Total: {len(games)} games")
