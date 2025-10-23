# Sports Schedule Files

This directory contains sport-specific schedules in two formats:

## 1. Python Schedule Files (New)
Sport-specific `.py` files with hardcoded schedule data:
- `nfl_schedule.py` - NFL 2025 season (94 games)
- `nhl_schedule.py` - NHL placeholder (uses Excel/API)
- More sports coming soon...

## 2. Excel Schedule Files (Legacy)
The system automatically uses:
- **Excel files** (this directory) for regular season games
- **API data** for playoff games

## File Naming

Place Excel files here with these naming patterns:
- `MLB.xlsx` or `mlb.xlsx`
- `NBA.xlsx` or `nba.xlsx`  
- `NHL.xlsx` or `nhl.xlsx`
- `NFL.xlsx` or `nfl.xlsx`
- `NCAAF.xlsx` or `ncaaf.xlsx`
- `NCAAB.xlsx` or `ncaab.xlsx`
- `WNBA.xlsx` or `wnba.xlsx`

You can also use season-specific files:
- `MLB_2025.xlsx`
- `NBA_2024.xlsx`

## Excel Format

Your Excel file must have these columns (case-insensitive):

### Required Columns
- **Date** (or "Game Date", "game_date", "DATE")
  - Format: YYYY-MM-DD, MM/DD/YYYY, or MM/DD/YY
  - Example: `2025-10-15` or `10/15/2025`

- **Away** (or "Away Team", "away_team", "Visitor", "AWAY")
  - Team abbreviation (2-4 letters)
  - Example: `NYY`, `LAL`, `BOS`

- **Home** (or "Home Team", "home_team", "HOME")
  - Team abbreviation (2-4 letters)
  - Example: `BOS`, `LAL`, `NYY`

### Optional Columns
- **Time** (or "Game Time", "time")
  - Example: `7:00 PM`, `19:00`

## Example Excel Layout

| Date       | Away | Home | Time    |
|------------|------|------|---------|
| 2025-10-15 | NYY  | BOS  | 7:05 PM |
| 2025-10-15 | LAD  | SF   | 10:10 PM |
| 2025-10-16 | CHC  | STL  | 1:20 PM |

## How It Works

1. **Regular Season**: System checks for Excel file first
   - If found, uses Excel data
   - Perfect for full season schedules

2. **Playoffs**: System uses API data
   - If no Excel file, falls back to API
   - Automatically gets playoff games

3. **Data Priority**: Excel → API
   - Place Excel file here for regular season
   - Remove Excel file to use API for playoffs

## Tips

- Use team abbreviations from the league (not full names)
- Make sure dates are properly formatted
- One row per game
- The system will auto-detect column names (flexible)
