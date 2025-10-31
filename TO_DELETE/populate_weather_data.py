"""
Populate Historical Weather Data for NFL Games

Fetches weather conditions for all existing NFL games in the database
using the Open-Meteo API (free, no API key required).
"""

import sqlite3
import logging
from src.collectors.weather_collector import WeatherCollector
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_weather_data():
    """Populate weather data for all NFL games"""
    
    collector = WeatherCollector()
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    # Get all NFL games without weather data
    cursor.execute("""
        SELECT id, home_team_id, game_date, status
        FROM games
        WHERE sport = 'NFL'
        AND temperature IS NULL
        ORDER BY game_date DESC
    """)
    
    games = cursor.fetchall()
    logger.info(f"Found {len(games)} NFL games needing weather data")
    
    if len(games) == 0:
        logger.info("✅ All NFL games already have weather data")
        conn.close()
        return
    
    # Process each game
    updated_count = 0
    error_count = 0
    
    for game_id, home_team, game_date, status in tqdm(games, desc="Fetching weather"):
        try:
            # Extract game time if available (default 1pm ET)
            game_time = "13:00"
            
            # Get weather for game
            weather = collector.get_game_weather(home_team, game_date, game_time)
            impact_score = collector.get_weather_impact_score(weather)
            
            # Update database
            cursor.execute("""
                UPDATE games
                SET temperature = ?,
                    precipitation = ?,
                    wind_speed = ?,
                    wind_gusts = ?,
                    is_dome = ?,
                    weather_impact_score = ?
                WHERE id = ?
            """, (
                weather['temperature'],
                weather['precipitation'],
                weather['wind_speed'],
                weather['wind_gusts'],
                1 if weather['is_dome'] else 0,
                impact_score,
                game_id
            ))
            
            updated_count += 1
            
            # Commit every 10 games to avoid losing progress
            if updated_count % 10 == 0:
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error processing game {game_id}: {e}")
            error_count += 1
            continue
    
    # Final commit
    conn.commit()
    conn.close()
    
    logger.info(f"\n✅ Weather data population complete!")
    logger.info(f"   Updated: {updated_count} games")
    logger.info(f"   Errors: {error_count} games")
    
    # Show sample data
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT home_team_id, game_date, temperature, wind_speed, precipitation, is_dome, weather_impact_score
        FROM games
        WHERE sport = 'NFL' AND temperature IS NOT NULL
        ORDER BY weather_impact_score DESC
        LIMIT 5
    """)
    
    print("\n🌡️ Worst Weather Conditions (Top 5):")
    print("="*80)
    for row in cursor.fetchall():
        home, date, temp, wind, precip, dome, impact = row
        print(f"{date}: {home}")
        print(f"  Temp: {temp:.1f}°C, Wind: {wind:.1f} m/s, Precip: {precip:.1f}mm")
        print(f"  Dome: {'Yes' if dome else 'No'}, Impact Score: {impact:.1f}")
        print()
    
    conn.close()

if __name__ == '__main__':
    populate_weather_data()
