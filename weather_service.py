"""
Weather Service Wrapper
Provides simple interface for NFL72FINAL.py to access weather data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.collectors.weather_collector import WeatherCollector

# Initialize weather collector
_weather_collector = WeatherCollector()

def get_weather_for_game(team_name, game_date=None, game_time="13:00"):
    """
    Get weather for a game
    
    Args:
        team_name: Home team name
        game_date: Game date (optional, defaults to today)
        game_time: Game time (default 1pm)
    
    Returns:
        Dict with temperature, wind_speed, precipitation, etc.
    """
    from datetime import datetime
    
    if game_date is None:
        game_date = datetime.now().strftime('%d/%m/%Y')
    
    try:
        weather = _weather_collector.get_game_weather(team_name, game_date, game_time)
        return weather
    except Exception as e:
        # Return default weather on error
        return {
            'temperature': 65.0,
            'precipitation': 0.0,
            'wind_speed': 5.0,
            'wind_gusts': 10.0,
            'is_dome': False
        }

def get_weather_impact(weather_data):
    """
    Calculate weather impact score
    
    Args:
        weather_data: Dict from get_weather_for_game
        
    Returns:
        Float impact score (0-100, higher = worse conditions)
    """
    try:
        return _weather_collector.get_weather_impact_score(weather_data)
    except Exception:
        return 0.0
