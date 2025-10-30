import pandas as pd

# Parse the NHL schedule file
schedule_file = "attached_assets/Pasted-Date-Time-Visitor-G-Home-G-Att-LOG-Notes-2025-10-07-5-00-PM-Chicago-Blackhawks-2-Florida-Panthers--1761842548476_1761842548477.txt"

# Read the file
with open(schedule_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse games (skip header rows that repeat)
games = []
match_id = 1

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    parts = line.split('\t')
    
    # Skip header rows
    if len(parts) > 0 and (parts[0] == 'Date' or parts[0].strip() == 'Date'):
        continue
    
    # Valid game row has at least 3 columns (Date, Time, Visitor, Home)
    if len(parts) >= 5:
        date = parts[0].strip()
        time = parts[1].strip() if len(parts) > 1 else ''
        visitor = parts[2].strip() if len(parts) > 2 else ''
        home = parts[4].strip() if len(parts) > 4 else ''
        
        # Skip if missing critical data
        if not date or not visitor or not home:
            continue
        
        # Skip if this looks like a header
        if date == 'Date' or visitor == 'Visitor':
            continue
        
        games.append({
            'match_id': match_id,
            'date': date,
            'time': time,
            'visitor': visitor,
            'home': home
        })
        match_id += 1

print(f"TOTAL GAMES PARSED: {len(games)}\n")
print("="*80)

# Show first 10 games
print("\nFIRST 10 GAMES (Match IDs 1-10):")
print("-"*80)
for g in games[:10]:
    print(f"ID {g['match_id']:4d} | {g['date']:12s} | {g['visitor']:25s} @ {g['home']:25s}")

# Show games 139-152 (the critical 148-game range)
print("\n\nGAMES 139-152 (End of first 148 games):")
print("-"*80)
for g in games[138:152]:
    print(f"ID {g['match_id']:4d} | {g['date']:12s} | {g['visitor']:25s} @ {g['home']:25s}")

# Show last 10 games
print(f"\n\nLAST 10 GAMES (Match IDs {len(games)-9}-{len(games)}):")
print("-"*80)
for g in games[-10:]:
    print(f"ID {g['match_id']:4d} | {g['date']:12s} | {g['visitor']:25s} @ {g['home']:25s}")

print("\n" + "="*80)
print(f"\nSCHEDULE VERIFICATION COMPLETE: {len(games)} games with Match IDs 1-{len(games)}")

# Save to CSV for user review
output_file = "nhl_2025_schedule_verified.csv"
df = pd.DataFrame(games)
df.to_csv(output_file, index=False)
print(f"\nFull schedule saved to: {output_file}")
print("Review this file to confirm all games are correct before proceeding.")
