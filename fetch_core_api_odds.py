#!/usr/bin/env python3
"""
Fetch betting lines from ESPN Core API (sports.core.api.espn.com)
This endpoint has odds data that the site API doesn't provide
"""
import sqlite3
import requests
from datetime import datetime, timedelta
from colorama import Fore, Style, init

init(autoreset=True)

DB_PATH = "sports_predictions_original.db"

CORE_API_ENDPOINTS = {
    'NBA': 'http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events',
    'NHL': 'http://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/events',
    'NFL': 'http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events',
    'MLB': 'http://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events',
    'WNBA': 'http://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba/events',
    'NCAAB': 'http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/events',
    'NCAAF': 'http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events',
}

# Map sport to Core API sport/league slugs
SPORT_TO_API_PATH = {
    'NBA': ('basketball', 'nba'),
    'NHL': ('hockey', 'nhl'),
    'NFL': ('football', 'nfl'),
    'MLB': ('baseball', 'mlb'),
    'WNBA': ('basketball', 'wnba'),
    'NCAAB': ('basketball', 'mens-college-basketball'),
    'NCAAF': ('football', 'college-football'),
}


def fetch_odds_for_sport(sport):
    """Fetch odds from Core API for a sport"""
    print(f"\n{Fore.CYAN}Fetching {sport} Odds from Core API{Style.RESET_ALL}")
    
    if sport not in CORE_API_ENDPOINTS:
        print(f"  {Fore.RED}No endpoint for {sport}{Style.RESET_ALL}")
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    added = 0
    
    try:
        # Get list of events
        events_url = CORE_API_ENDPOINTS[sport]
        response = requests.get(events_url, timeout=10)
        response.raise_for_status()
        events_data = response.json()
        
        events = events_data.get('items', [])
        
        for event_ref in events[:20]:  # Limit to first 20 events
            event_url = event_ref.get('$ref')
            if not event_url:
                continue
            
            # Get event details
            event_resp = requests.get(event_url, timeout=10)
            event = event_resp.json()
            
            # Get game date - MUST convert UTC to Eastern Time (same as schedule updater)
            # ESPN API returns UTC, but our DB stores games in ET (UTC-5)
            date_str = event.get('date', '')
            if not date_str:
                continue
            try:
                # Parse UTC datetime and convert to ET
                game_date_utc = datetime.strptime(date_str, '%Y-%m-%dT%H:%M%SZ')
                game_date_et = game_date_utc - timedelta(hours=5)  # UTC to ET
                game_date_local = game_date_et.strftime('%Y-%m-%d')
            except:
                # Fallback to just the date portion
                game_date_local = date_str[:10]
            
            # Get competitors
            comps = event.get('competitions', [])
            if not comps:
                continue
            
            comp_ref = comps[0].get('$ref')
            if not comp_ref:
                continue
            
            # Get competition details for teams
            comp_resp = requests.get(comp_ref, timeout=10)
            comp = comp_resp.json()
            
            competitors = comp.get('competitors', [])
            if len(competitors) != 2:
                continue
            
            home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            
            if not home or not away:
                continue
            
            # Get team names from team reference
            home_team_ref = home.get('team', {}).get('$ref')
            away_team_ref = away.get('team', {}).get('$ref')
            
            home_team_resp = requests.get(home_team_ref, timeout=10)
            away_team_resp = requests.get(away_team_ref, timeout=10)
            
            home_team = home_team_resp.json().get('displayName', '')
            away_team = away_team_resp.json().get('displayName', '')
            
            # Get odds
            event_id = event.get('id')
            sport_slug, league_slug = SPORT_TO_API_PATH[sport]
            odds_url = f"http://sports.core.api.espn.com/v2/sports/{sport_slug}/leagues/{league_slug}/events/{event_id}/competitions/{event_id}/odds"
            
            try:
                odds_resp = requests.get(odds_url, timeout=10)
                odds_data = odds_resp.json()
                
                odds_items = odds_data.get('items', [])
                if not odds_items:
                    continue
                
                # Use first provider (usually ESPN BET)
                primary_odds = odds_items[0]
                
                spread = primary_odds.get('spread')
                over_under = primary_odds.get('overUnder')
                
                # Get moneylines
                home_ml = primary_odds.get('homeTeamOdds', {}).get('moneyLine')
                away_ml = primary_odds.get('awayTeamOdds', {}).get('moneyLine')
                
                if spread is None and over_under is None:
                    continue
                
                # Find game_id in database
                cursor.execute("""
                    SELECT game_id FROM games
                    WHERE sport = ? AND home_team_id = ? AND away_team_id = ?
                    AND date(game_date) = ?
                """, (sport, home_team, away_team, game_date_local))
                
                result = cursor.fetchone()
                if not result:
                    print(f"  ⚠ No match: {away_team} @ {home_team}")
                    continue
                
                game_id = result[0]
                
                # Insert/update betting lines
                cursor.execute("""
                    INSERT OR REPLACE INTO betting_lines
                    (sport, game_id, game_date, home_team, away_team, spread, total, home_moneyline, away_moneyline, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ESPN Core API')
                """, (sport, game_id, game_date_local, home_team, away_team,
                      spread, over_under, home_ml, away_ml))
                
                added += 1
                print(f"  ✓ {away_team} @ {home_team}: Spread {spread}, O/U {over_under}")
                
            except Exception as e:
                print(f"  ! Error processing {away_team} @ {home_team} on {game_date_local}: {e}")
        
    except Exception as e:
        print(f"  {Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    conn.commit()
    conn.close()
    
    print(f"  {Fore.GREEN}✓ Added {added} betting lines{Style.RESET_ALL}")
    return added


def main():
    print(f"{Fore.CYAN}{'='*60}")
    print(f"ESPN Core API Odds Fetcher - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    total = 0
    for sport in ['NBA', 'NHL', 'NFL', 'NCAAB', 'NCAAF']:
        total += fetch_odds_for_sport(sport)
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"✓ Complete - {total} betting lines fetched")
    print(f"{'='*60}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
