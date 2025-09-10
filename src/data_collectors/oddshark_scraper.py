import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import trafilatura

class OddsSharkScraper:
    """
    Scraper for OddsShark.com sports betting trends and data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://www.oddsshark.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_mlb_trends(self, date: datetime = None) -> List[Dict]:
        """
        Get MLB betting trends from OddsShark
        
        Args:
            date: Date to get trends for (defaults to today)
            
        Returns:
            List of dictionaries with trend data
        """
        if date is None:
            date = datetime.now()
        
        try:
            # Try different MLB pages on OddsShark
            urls_to_try = [
                f"{self.base_url}/mlb/odds",
                f"{self.base_url}/mlb/scores",
                f"{self.base_url}/mlb/database"
            ]
            
            trends_data = []
            
            for url in urls_to_try:
                try:
                    self.logger.info(f"Fetching MLB trends from: {url}")
                    
                    # Get page content using trafilatura for better text extraction
                    content = self._get_website_content(url)
                    
                    if content:
                        # Parse content for betting trends
                        parsed_trends = self._parse_mlb_content(content, url)
                        trends_data.extend(parsed_trends)
                    
                    time.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching from {url}: {str(e)}")
                    continue
            
            # Remove duplicates and return
            unique_trends = self._deduplicate_trends(trends_data)
            self.logger.info(f"Successfully fetched {len(unique_trends)} unique trends")
            
            return unique_trends
            
        except Exception as e:
            self.logger.error(f"Error getting MLB trends: {str(e)}")
            return []
    
    def get_mlb_odds(self, date: datetime = None) -> List[Dict]:
        """
        Get current MLB odds and lines
        
        Args:
            date: Date to get odds for (defaults to today)
            
        Returns:
            List of dictionaries with odds data
        """
        if date is None:
            date = datetime.now()
        
        try:
            odds_url = f"{self.base_url}/mlb/odds"
            self.logger.info(f"Fetching MLB odds from: {odds_url}")
            
            response = self.session.get(odds_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for odds data in various formats
            odds_data = []
            
            # Try to find JSON data in script tags
            script_tags = soup.find_all('script', type='application/json')
            for script in script_tags:
                try:
                    json_data = json.loads(script.string)
                    if self._contains_mlb_data(json_data):
                        parsed_odds = self._parse_odds_json(json_data)
                        odds_data.extend(parsed_odds)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            # Try to parse HTML tables
            tables = soup.find_all('table')
            for table in tables:
                if self._is_odds_table(table):
                    parsed_table = self._parse_odds_table(table)
                    odds_data.extend(parsed_table)
            
            # Parse general content for odds information
            content = self._get_website_content(odds_url)
            if content:
                content_odds = self._parse_odds_from_content(content)
                odds_data.extend(content_odds)
            
            return self._deduplicate_odds(odds_data)
            
        except Exception as e:
            self.logger.error(f"Error getting MLB odds: {str(e)}")
            return []
    
    def get_team_trends(self, team: str, opponent: str = None) -> Dict:
        """
        Get specific team trends from OddsShark database
        
        Args:
            team: Team name or abbreviation
            opponent: Opponent team (optional)
            
        Returns:
            Dictionary with team trend data
        """
        try:
            # Construct database search URL
            database_url = f"{self.base_url}/mlb/database"
            
            # Try to access the database page
            content = self._get_website_content(database_url)
            
            if content:
                team_trends = self._parse_team_trends(content, team, opponent)
                return team_trends
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting team trends for {team}: {str(e)}")
            return {}
    
    def _get_website_content(self, url: str) -> str:
        """
        Get website text content using trafilatura
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Extracted text content
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text if text else ""
            return ""
        except Exception as e:
            self.logger.warning(f"Trafilatura extraction failed for {url}: {str(e)}")
            # Fallback to regular requests
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except:
                return ""
    
    def _parse_mlb_content(self, content: str, source_url: str) -> List[Dict]:
        """
        Parse MLB content for betting trends and data
        
        Args:
            content: Text content from webpage
            source_url: Source URL for context
            
        Returns:
            List of parsed trend dictionaries
        """
        trends = []
        
        try:
            # Look for common betting patterns in text
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Pattern for team vs team matchups
                team_pattern = r'([A-Z]{2,3})\s+(?:vs|@)\s+([A-Z]{2,3})'
                team_match = re.search(team_pattern, line)
                
                if team_match:
                    away_team, home_team = team_match.groups()
                    
                    # Look for odds information in the same line or nearby
                    odds_info = self._extract_odds_from_line(line)
                    
                    trend = {
                        'away_team': away_team,
                        'home_team': home_team,
                        'source_url': source_url,
                        'extracted_at': datetime.now().isoformat(),
                        **odds_info
                    }
                    
                    trends.append(trend)
                
                # Pattern for betting trends (e.g., "Team is 4-1 ATS")
                trend_pattern = r'([A-Za-z\s]+)\s+(?:is|are)\s+(\d+-\d+)\s+(ATS|O/U|SU)'
                trend_match = re.search(trend_pattern, line)
                
                if trend_match:
                    team_info, record, bet_type = trend_match.groups()
                    
                    trend = {
                        'team_info': team_info.strip(),
                        'record': record,
                        'bet_type': bet_type,
                        'trend_text': line,
                        'source_url': source_url,
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    trends.append(trend)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error parsing MLB content: {str(e)}")
            return []
    
    def _extract_odds_from_line(self, line: str) -> Dict:
        """
        Extract odds information from a text line
        
        Args:
            line: Text line to parse
            
        Returns:
            Dictionary with odds information
        """
        odds_info = {}
        
        try:
            # Pattern for moneyline odds (+150, -120)
            moneyline_pattern = r'([+-]\d{3,4})'
            moneyline_matches = re.findall(moneyline_pattern, line)
            
            if len(moneyline_matches) >= 2:
                odds_info['away_moneyline'] = moneyline_matches[0]
                odds_info['home_moneyline'] = moneyline_matches[1]
            
            # Pattern for point spreads (+1.5, -1.5)
            spread_pattern = r'([+-]\d+\.?\d*)\s*runs?'
            spread_match = re.search(spread_pattern, line)
            
            if spread_match:
                odds_info['spread'] = spread_match.group(1)
            
            # Pattern for totals (O/U 8.5)
            total_pattern = r'(?:O/U|total)\s*(\d+\.?\d*)'
            total_match = re.search(total_pattern, line, re.IGNORECASE)
            
            if total_match:
                odds_info['total'] = total_match.group(1)
            
            return odds_info
            
        except Exception as e:
            self.logger.warning(f"Error extracting odds from line: {str(e)}")
            return {}
    
    def _contains_mlb_data(self, data: Dict) -> bool:
        """
        Check if JSON data contains MLB information
        
        Args:
            data: JSON data to check
            
        Returns:
            True if contains MLB data
        """
        try:
            # Convert to string and check for MLB-related keywords
            data_str = json.dumps(data).lower()
            mlb_keywords = ['mlb', 'baseball', 'american league', 'national league', 'yankees', 'dodgers']
            
            return any(keyword in data_str for keyword in mlb_keywords)
            
        except:
            return False
    
    def _parse_odds_json(self, json_data: Dict) -> List[Dict]:
        """
        Parse JSON data for odds information
        
        Args:
            json_data: JSON data containing odds
            
        Returns:
            List of parsed odds dictionaries
        """
        odds_list = []
        
        try:
            # This is a generic parser - actual structure will depend on OddsShark's JSON format
            if isinstance(json_data, dict):
                if 'games' in json_data:
                    for game in json_data['games']:
                        odds_list.append(self._parse_game_data(game))
                elif 'odds' in json_data:
                    odds_list.extend(self._parse_odds_array(json_data['odds']))
            
            return odds_list
            
        except Exception as e:
            self.logger.warning(f"Error parsing odds JSON: {str(e)}")
            return []
    
    def _parse_game_data(self, game_data: Dict) -> Dict:
        """
        Parse individual game data from JSON
        
        Args:
            game_data: Individual game dictionary
            
        Returns:
            Parsed game dictionary
        """
        try:
            parsed_game = {
                'game_id': game_data.get('id', ''),
                'away_team': game_data.get('away_team', ''),
                'home_team': game_data.get('home_team', ''),
                'game_time': game_data.get('game_time', ''),
                'away_moneyline': game_data.get('away_ml', ''),
                'home_moneyline': game_data.get('home_ml', ''),
                'spread': game_data.get('spread', ''),
                'total': game_data.get('total', ''),
                'extracted_at': datetime.now().isoformat()
            }
            
            return parsed_game
            
        except Exception as e:
            self.logger.warning(f"Error parsing game data: {str(e)}")
            return {}
    
    def _is_odds_table(self, table) -> bool:
        """
        Check if HTML table contains odds data
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            True if table contains odds
        """
        try:
            # Look for common odds table headers
            headers = table.find_all(['th', 'td'])
            header_text = ' '.join([h.get_text().lower() for h in headers[:10]])
            
            odds_keywords = ['team', 'spread', 'total', 'moneyline', 'ml', 'o/u', 'odds']
            
            return any(keyword in header_text for keyword in odds_keywords)
            
        except:
            return False
    
    def _parse_odds_table(self, table) -> List[Dict]:
        """
        Parse HTML table for odds data
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            List of parsed odds dictionaries
        """
        odds_data = []
        
        try:
            rows = table.find_all('tr')
            
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                
                if len(cells) >= 3:
                    row_data = {
                        'team_info': cells[0].get_text().strip(),
                        'odds_info': ' '.join([cell.get_text().strip() for cell in cells[1:]]),
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    # Try to parse structured odds from the row
                    structured_odds = self._extract_odds_from_line(row_data['odds_info'])
                    row_data.update(structured_odds)
                    
                    odds_data.append(row_data)
            
            return odds_data
            
        except Exception as e:
            self.logger.warning(f"Error parsing odds table: {str(e)}")
            return []
    
    def _parse_odds_from_content(self, content: str) -> List[Dict]:
        """
        Parse odds information from general content
        
        Args:
            content: Text content to parse
            
        Returns:
            List of odds dictionaries
        """
        odds_data = []
        
        try:
            # Split content into lines and look for game information
            lines = content.split('\n')
            
            current_game = {}
            
            for line in lines:
                line = line.strip()
                
                # Look for game matchups
                if ' vs ' in line or ' @ ' in line:
                    if current_game:
                        odds_data.append(current_game)
                    
                    current_game = {
                        'matchup': line,
                        'extracted_at': datetime.now().isoformat()
                    }
                
                # Look for odds in the current line
                odds_info = self._extract_odds_from_line(line)
                if odds_info and current_game:
                    current_game.update(odds_info)
            
            # Add the last game
            if current_game:
                odds_data.append(current_game)
            
            return odds_data
            
        except Exception as e:
            self.logger.warning(f"Error parsing odds from content: {str(e)}")
            return []
    
    def _parse_team_trends(self, content: str, team: str, opponent: str = None) -> Dict:
        """
        Parse team-specific trends from content
        
        Args:
            content: Text content to parse
            team: Team to look for
            opponent: Opponent team (optional)
            
        Returns:
            Dictionary with team trends
        """
        try:
            trends = {
                'team': team,
                'opponent': opponent,
                'trends': [],
                'extracted_at': datetime.now().isoformat()
            }
            
            lines = content.split('\n')
            
            for line in lines:
                if team.lower() in line.lower():
                    # Look for trend patterns
                    trend_matches = re.findall(r'(\d+-\d+)\s+(ATS|O/U|SU)', line)
                    
                    for record, bet_type in trend_matches:
                        trends['trends'].append({
                            'record': record,
                            'type': bet_type,
                            'context': line.strip()
                        })
            
            return trends
            
        except Exception as e:
            self.logger.warning(f"Error parsing team trends: {str(e)}")
            return {}
    
    def _deduplicate_trends(self, trends: List[Dict]) -> List[Dict]:
        """
        Remove duplicate trend entries
        
        Args:
            trends: List of trend dictionaries
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique_trends = []
        
        for trend in trends:
            # Create a key based on relevant fields
            key_fields = ['away_team', 'home_team', 'record', 'bet_type']
            key_values = tuple(trend.get(field, '') for field in key_fields)
            
            if key_values not in seen:
                seen.add(key_values)
                unique_trends.append(trend)
        
        return unique_trends
    
    def _deduplicate_odds(self, odds: List[Dict]) -> List[Dict]:
        """
        Remove duplicate odds entries
        
        Args:
            odds: List of odds dictionaries
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique_odds = []
        
        for odd in odds:
            # Create a key based on relevant fields
            key_fields = ['away_team', 'home_team', 'matchup']
            key_values = tuple(odd.get(field, '') for field in key_fields)
            
            if key_values not in seen:
                seen.add(key_values)
                unique_odds.append(odd)
        
        return unique_odds
    
    def _parse_odds_array(self, odds_array: List) -> List[Dict]:
        """
        Parse array of odds data
        
        Args:
            odds_array: Array of odds objects
            
        Returns:
            List of parsed odds dictionaries
        """
        parsed_odds = []
        
        for odds_item in odds_array:
            if isinstance(odds_item, dict):
                parsed_odds.append(self._parse_game_data(odds_item))
        
        return parsed_odds
