"""
NFL Weather Data Collector using Open-Meteo API

Fetches weather conditions for NFL games to improve prediction accuracy.
Weather significantly impacts outdoor games (wind affects passing/kicking, 
temperature affects ball handling, precipitation impacts field conditions).
"""

import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import pandas as pd
from src.data.nfl_stadium_locations import get_stadium_info, is_outdoor_stadium

class WeatherCollector:
    """Collects weather data for NFL game locations using Open-Meteo API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Setup Open-Meteo API client with cache and retry
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
    def get_game_weather(self, team_name: str, game_date: str, game_time: str = "13:00") -> Dict:
        """
        Get weather conditions for a specific NFL game
        
        Args:
            team_name: Home team name
            game_date: Game date in format 'DD/MM/YYYY' or 'YYYY-MM-DD'
            game_time: Game time in format 'HH:MM' (default 1pm ET)
            
        Returns:
            Dictionary with weather features or neutral values for domes
        """
        # Check if dome stadium (weather doesn't matter)
        if not is_outdoor_stadium(team_name):
            return self._get_dome_weather()
        
        # Get stadium coordinates
        stadium_info = get_stadium_info(team_name)
        
        # Parse game datetime
        try:
            game_datetime = self._parse_game_datetime(game_date, game_time)
        except Exception as e:
            self.logger.warning(f"Could not parse date {game_date} {game_time}: {e}")
            return self._get_default_weather()
        
        # Fetch weather data from Open-Meteo
        try:
            weather_data = self._fetch_weather_data(
                stadium_info['latitude'],
                stadium_info['longitude'],
                game_datetime
            )
            return weather_data
        except Exception as e:
            self.logger.error(f"Weather API error for {team_name} on {game_date}: {e}")
            return self._get_default_weather()
    
    def _parse_game_datetime(self, game_date: str, game_time: str) -> datetime:
        """Parse game date and time into datetime object"""
        # Try DD/MM/YYYY format first (our database format)
        try:
            date_part = datetime.strptime(game_date, '%d/%m/%Y')
        except ValueError:
            # Try YYYY-MM-DD format
            try:
                date_part = datetime.strptime(game_date, '%Y-%m-%d')
            except ValueError:
                # Try ISO format with time
                date_part = datetime.strptime(game_date.split()[0], '%Y-%m-%d')
        
        # Parse time
        try:
            time_parts = game_time.split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
        except:
            hour, minute = 13, 0  # Default 1pm
        
        return datetime(date_part.year, date_part.month, date_part.day, hour, minute)
    
    def _fetch_weather_data(self, latitude: float, longitude: float, game_datetime: datetime) -> Dict:
        """Fetch weather data from Open-Meteo API"""
        # Determine if historical or forecast
        now = datetime.now()
        is_historical = game_datetime < now - timedelta(days=1)
        
        if is_historical:
            # Use historical API for past games
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": game_datetime.strftime('%Y-%m-%d'),
                "end_date": game_datetime.strftime('%Y-%m-%d'),
                "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"]
            }
        else:
            # Use forecast API for future games
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"],
                "forecast_days": 16
            }
        
        # Make API request
        responses = self.client.weather_api(url if is_historical else self.base_url, params=params)
        response = responses[0]
        
        # Extract hourly data
        hourly = response.Hourly()
        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "precipitation": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(2).ValuesAsNumpy(),
            "wind_gusts": hourly.Variables(3).ValuesAsNumpy()
        }
        
        df = pd.DataFrame(hourly_data)
        
        # Find closest hour to game time
        game_hour = game_datetime.replace(tzinfo=None)
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        df['time_diff'] = abs(df['time'] - game_hour)
        closest_idx = df['time_diff'].idxmin()
        
        weather_row = df.loc[closest_idx]
        
        return {
            'temperature': float(weather_row['temperature']),
            'precipitation': float(weather_row['precipitation']),
            'wind_speed': float(weather_row['wind_speed']),
            'wind_gusts': float(weather_row['wind_gusts']),
            'is_dome': False
        }
    
    def _get_dome_weather(self) -> Dict:
        """Return neutral weather for dome stadiums"""
        return {
            'temperature': 72.0,  # Climate controlled
            'precipitation': 0.0,
            'wind_speed': 0.0,
            'wind_gusts': 0.0,
            'is_dome': True
        }
    
    def _get_default_weather(self) -> Dict:
        """Return default neutral weather when API fails"""
        return {
            'temperature': 65.0,  # Mild conditions
            'precipitation': 0.0,
            'wind_speed': 5.0,
            'wind_gusts': 10.0,
            'is_dome': False
        }
    
    def get_weather_impact_score(self, weather: Dict) -> float:
        """
        Calculate weather impact score for NFL game
        Higher score = worse conditions for offense
        
        Factors:
        - Extreme cold/heat affects ball handling
        - High wind affects passing and kicking
        - Precipitation affects ball control
        """
        if weather['is_dome']:
            return 0.0
        
        score = 0.0
        temp = weather['temperature']
        
        # Temperature impact (Celsius)
        # Extreme cold (<0°C / 32°F) or extreme heat (>35°C / 95°F)
        if temp < 0:
            score += abs(temp) * 0.5  # Cold impact
        elif temp > 35:
            score += (temp - 35) * 0.3  # Heat impact
        
        # Wind impact (mph -> m/s conversion: 1 mph = 0.447 m/s)
        # High wind (>15 mph / 6.7 m/s) significantly impacts passing
        wind_mph = weather['wind_speed'] * 2.237  # Convert m/s to mph
        if wind_mph > 15:
            score += (wind_mph - 15) * 0.8
        
        # Precipitation impact (mm)
        # Any precipitation affects ball control
        if weather['precipitation'] > 0:
            score += weather['precipitation'] * 2.0
        
        return min(score, 100.0)  # Cap at 100
