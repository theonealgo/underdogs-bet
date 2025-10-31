#!/usr/bin/env python3
"""Import the exact NFL schedule provided by user"""
import sqlite3
import sys

# Paste the exact schedule data
nfl_schedule = [
    {'match_id': 1, 'round': 1, 'date': '05/09/2025 00:20', 'venue': 'Lincoln Financial Field', 'home_team': 'Philadelphia Eagles', 'away_team': 'Dallas Cowboys', 'result': '24 - 20'},
    {'match_id': 2, 'round': 1, 'date': '06/09/2025 00:00', 'venue': 'Arena Corinthians', 'home_team': 'Los Angeles Chargers', 'away_team': 'Kansas City Chiefs', 'result': '27 - 21'},
    {'match_id': 3, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'Mercedes-Benz Stadium', 'home_team': 'Atlanta Falcons', 'away_team': 'Tampa Bay Buccaneers', 'result': '20 - 23'},
    {'match_id': 4, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'Huntington Bank Field', 'home_team': 'Cleveland Browns', 'away_team': 'Cincinnati Bengals', 'result': '16 - 17'},
    {'match_id': 5, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'Lucas Oil Stadium', 'home_team': 'Indianapolis Colts', 'away_team': 'Miami Dolphins', 'result': '33 - 8'},
    {'match_id': 6, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'EverBank Stadium', 'home_team': 'Jacksonville Jaguars', 'away_team': 'Carolina Panthers', 'result': '26 - 10'},
    {'match_id': 7, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'Gillette Stadium', 'home_team': 'New England Patriots', 'away_team': 'Las Vegas Raiders', 'result': '13 - 20'},
    {'match_id': 8, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'Caesars Superdome', 'home_team': 'New Orleans Saints', 'away_team': 'Arizona Cardinals', 'result': '13 - 20'},
    {'match_id': 9, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'MetLife Stadium', 'home_team': 'New York Jets', 'away_team': 'Pittsburgh Steelers', 'result': '32 - 34'},
    {'match_id': 10, 'round': 1, 'date': '07/09/2025 17:00', 'venue': 'Northwest Stadium', 'home_team': 'Washington Commanders', 'away_team': 'New York Giants', 'result': '21 - 6'},
    {'match_id': 11, 'round': 1, 'date': '07/09/2025 20:05', 'venue': 'Empower Field at Mile High', 'home_team': 'Denver Broncos', 'away_team': 'Tennessee Titans', 'result': '20 - 12'},
    {'match_id': 12, 'round': 1, 'date': '07/09/2025 20:05', 'venue': 'Lumen Field', 'home_team': 'Seattle Seahawks', 'away_team': 'San Francisco 49ers', 'result': '13 - 17'},
    {'match_id': 13, 'round': 1, 'date': '07/09/2025 20:25', 'venue': 'Lambeau Field', 'home_team': 'Green Bay Packers', 'away_team': 'Detroit Lions', 'result': '27 - 13'},
    {'match_id': 14, 'round': 1, 'date': '07/09/2025 20:25', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Rams', 'away_team': 'Houston Texans', 'result': '14 - 9'},
    {'match_id': 15, 'round': 1, 'date': '08/09/2025 00:20', 'venue': 'Highmark Stadium', 'home_team': 'Buffalo Bills', 'away_team': 'Baltimore Ravens', 'result': '41 - 40'},
    {'match_id': 16, 'round': 1, 'date': '09/09/2025 00:15', 'venue': 'Soldier Field', 'home_team': 'Chicago Bears', 'away_team': 'Minnesota Vikings', 'result': '24 - 27'},
    {'match_id': 17, 'round': 2, 'date': '12/09/2025 00:15', 'venue': 'Lambeau Field', 'home_team': 'Green Bay Packers', 'away_team': 'Washington Commanders', 'result': '27 - 18'},
    {'match_id': 18, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'M&T Bank Stadium', 'home_team': 'Baltimore Ravens', 'away_team': 'Cleveland Browns', 'result': '41 - 17'},
    {'match_id': 19, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'Paycor Stadium', 'home_team': 'Cincinnati Bengals', 'away_team': 'Jacksonville Jaguars', 'result': '31 - 27'},
    {'match_id': 20, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'AT&T Stadium', 'home_team': 'Dallas Cowboys', 'away_team': 'New York Giants', 'result': '40 - 37'},
    {'match_id': 21, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'Ford Field', 'home_team': 'Detroit Lions', 'away_team': 'Chicago Bears', 'result': '52 - 21'},
    {'match_id': 22, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'Hard Rock Stadium', 'home_team': 'Miami Dolphins', 'away_team': 'New England Patriots', 'result': '27 - 33'},
    {'match_id': 23, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'Caesars Superdome', 'home_team': 'New Orleans Saints', 'away_team': 'San Francisco 49ers', 'result': '21 - 26'},
    {'match_id': 24, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'MetLife Stadium', 'home_team': 'New York Jets', 'away_team': 'Buffalo Bills', 'result': '10 - 30'},
    {'match_id': 25, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'Acrisure Stadium', 'home_team': 'Pittsburgh Steelers', 'away_team': 'Seattle Seahawks', 'result': '17 - 31'},
    {'match_id': 26, 'round': 2, 'date': '14/09/2025 17:00', 'venue': 'Nissan Stadium', 'home_team': 'Tennessee Titans', 'away_team': 'Los Angeles Rams', 'result': '19 - 33'},
    {'match_id': 27, 'round': 2, 'date': '14/09/2025 20:05', 'venue': 'State Farm Stadium', 'home_team': 'Arizona Cardinals', 'away_team': 'Carolina Panthers', 'result': '27 - 22'},
    {'match_id': 28, 'round': 2, 'date': '14/09/2025 20:05', 'venue': 'Lucas Oil Stadium', 'home_team': 'Indianapolis Colts', 'away_team': 'Denver Broncos', 'result': '29 - 28'},
    {'match_id': 29, 'round': 2, 'date': '14/09/2025 20:25', 'venue': 'GEHA Field at Arrowhead Stadium', 'home_team': 'Kansas City Chiefs', 'away_team': 'Philadelphia Eagles', 'result': '17 - 20'},
    {'match_id': 30, 'round': 2, 'date': '15/09/2025 00:20', 'venue': 'U.S. Bank Stadium', 'home_team': 'Minnesota Vikings', 'away_team': 'Atlanta Falcons', 'result': '6 - 22'},
    {'match_id': 31, 'round': 2, 'date': '15/09/2025 23:00', 'venue': 'NRG Stadium', 'home_team': 'Houston Texans', 'away_team': 'Tampa Bay Buccaneers', 'result': '19 - 20'},
    {'match_id': 32, 'round': 2, 'date': '16/09/2025 02:00', 'venue': 'Allegiant Stadium', 'home_team': 'Las Vegas Raiders', 'away_team': 'Los Angeles Chargers', 'result': '9 - 20'},
    {'match_id': 33, 'round': 3, 'date': '19/09/2025 00:15', 'venue': 'Highmark Stadium', 'home_team': 'Buffalo Bills', 'away_team': 'Miami Dolphins', 'result': '31 - 21'},
    {'match_id': 34, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'Bank of America Stadium', 'home_team': 'Carolina Panthers', 'away_team': 'Atlanta Falcons', 'result': '30 - 0'},
    {'match_id': 35, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'Huntington Bank Field', 'home_team': 'Cleveland Browns', 'away_team': 'Green Bay Packers', 'result': '13 - 10'},
    {'match_id': 36, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'EverBank Stadium', 'home_team': 'Jacksonville Jaguars', 'away_team': 'Houston Texans', 'result': '17 - 10'},
    {'match_id': 37, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'U.S. Bank Stadium', 'home_team': 'Minnesota Vikings', 'away_team': 'Cincinnati Bengals', 'result': '48 - 10'},
    {'match_id': 38, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'Gillette Stadium', 'home_team': 'New England Patriots', 'away_team': 'Pittsburgh Steelers', 'result': '14 - 21'},
    {'match_id': 39, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'Lincoln Financial Field', 'home_team': 'Philadelphia Eagles', 'away_team': 'Los Angeles Rams', 'result': '33 - 26'},
    {'match_id': 40, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'Raymond James Stadium', 'home_team': 'Tampa Bay Buccaneers', 'away_team': 'New York Jets', 'result': '29 - 27'},
    {'match_id': 41, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'Nissan Stadium', 'home_team': 'Tennessee Titans', 'away_team': 'Indianapolis Colts', 'result': '20 - 41'},
    {'match_id': 42, 'round': 3, 'date': '21/09/2025 17:00', 'venue': 'Northwest Stadium', 'home_team': 'Washington Commanders', 'away_team': 'Las Vegas Raiders', 'result': '41 - 24'},
    {'match_id': 43, 'round': 3, 'date': '21/09/2025 20:05', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Chargers', 'away_team': 'Denver Broncos', 'result': '23 - 20'},
    {'match_id': 44, 'round': 3, 'date': '21/09/2025 20:05', 'venue': 'Lumen Field', 'home_team': 'Seattle Seahawks', 'away_team': 'New Orleans Saints', 'result': '44 - 13'},
    {'match_id': 45, 'round': 3, 'date': '21/09/2025 20:25', 'venue': 'Soldier Field', 'home_team': 'Chicago Bears', 'away_team': 'Dallas Cowboys', 'result': '31 - 14'},
    {'match_id': 46, 'round': 3, 'date': '21/09/2025 20:25', 'venue': "Levi's Stadium", 'home_team': 'San Francisco 49ers', 'away_team': 'Arizona Cardinals', 'result': '16 - 15'},
    {'match_id': 47, 'round': 3, 'date': '22/09/2025 00:20', 'venue': 'MetLife Stadium', 'home_team': 'New York Giants', 'away_team': 'Kansas City Chiefs', 'result': '9 - 22'},
    {'match_id': 48, 'round': 3, 'date': '23/09/2025 00:15', 'venue': 'M&T Bank Stadium', 'home_team': 'Baltimore Ravens', 'away_team': 'Detroit Lions', 'result': '30 - 38'},
    {'match_id': 49, 'round': 4, 'date': '26/09/2025 00:15', 'venue': 'State Farm Stadium', 'home_team': 'Arizona Cardinals', 'away_team': 'Seattle Seahawks', 'result': '20 - 23'},
    {'match_id': 50, 'round': 4, 'date': '28/09/2025 13:30', 'venue': 'Croke Park', 'home_team': 'Pittsburgh Steelers', 'away_team': 'Minnesota Vikings', 'result': '24 - 21'},
    {'match_id': 51, 'round': 4, 'date': '28/09/2025 17:00', 'venue': 'Mercedes-Benz Stadium', 'home_team': 'Atlanta Falcons', 'away_team': 'Washington Commanders', 'result': '34 - 27'},
    {'match_id': 52, 'round': 4, 'date': '28/09/2025 17:00', 'venue': 'Highmark Stadium', 'home_team': 'Buffalo Bills', 'away_team': 'New Orleans Saints', 'result': '31 - 19'},
    {'match_id': 53, 'round': 4, 'date': '28/09/2025 17:00', 'venue': 'Ford Field', 'home_team': 'Detroit Lions', 'away_team': 'Cleveland Browns', 'result': '34 - 10'},
    {'match_id': 54, 'round': 4, 'date': '28/09/2025 17:00', 'venue': 'NRG Stadium', 'home_team': 'Houston Texans', 'away_team': 'Tennessee Titans', 'result': '26 - 0'},
    {'match_id': 55, 'round': 4, 'date': '28/09/2025 17:00', 'venue': 'Gillette Stadium', 'home_team': 'New England Patriots', 'away_team': 'Carolina Panthers', 'result': '42 - 13'},
    {'match_id': 56, 'round': 4, 'date': '28/09/2025 17:00', 'venue': 'MetLife Stadium', 'home_team': 'New York Giants', 'away_team': 'Los Angeles Chargers', 'result': '21 - 18'},
    {'match_id': 57, 'round': 4, 'date': '28/09/2025 17:00', 'venue': 'Raymond James Stadium', 'home_team': 'Tampa Bay Buccaneers', 'away_team': 'Philadelphia Eagles', 'result': '25 - 31'},
    {'match_id': 58, 'round': 4, 'date': '28/09/2025 20:05', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Rams', 'away_team': 'Indianapolis Colts', 'result': '27 - 20'},
    {'match_id': 59, 'round': 4, 'date': '28/09/2025 20:05', 'venue': "Levi's Stadium", 'home_team': 'San Francisco 49ers', 'away_team': 'Jacksonville Jaguars', 'result': '21 - 26'},
    {'match_id': 60, 'round': 4, 'date': '28/09/2025 20:25', 'venue': 'GEHA Field at Arrowhead Stadium', 'home_team': 'Kansas City Chiefs', 'away_team': 'Baltimore Ravens', 'result': '37 - 20'},
    {'match_id': 61, 'round': 4, 'date': '28/09/2025 20:25', 'venue': 'Allegiant Stadium', 'home_team': 'Las Vegas Raiders', 'away_team': 'Chicago Bears', 'result': '24 - 25'},
    {'match_id': 62, 'round': 4, 'date': '29/09/2025 00:20', 'venue': 'AT&T Stadium', 'home_team': 'Dallas Cowboys', 'away_team': 'Green Bay Packers', 'result': '40 - 40'},
    {'match_id': 63, 'round': 4, 'date': '29/09/2025 23:15', 'venue': 'Hard Rock Stadium', 'home_team': 'Miami Dolphins', 'away_team': 'New York Jets', 'result': '27 - 21'},
    {'match_id': 64, 'round': 4, 'date': '30/09/2025 00:15', 'venue': 'Empower Field at Mile High', 'home_team': 'Denver Broncos', 'away_team': 'Cincinnati Bengals', 'result': '28 - 3'},
    {'match_id': 65, 'round': 5, 'date': '03/10/2025 00:15', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Rams', 'away_team': 'San Francisco 49ers', 'result': '23 - 26'},
    {'match_id': 66, 'round': 5, 'date': '05/10/2025 13:30', 'venue': 'Tottenham Hotspur Stadium', 'home_team': 'Cleveland Browns', 'away_team': 'Minnesota Vikings', 'result': None},
    {'match_id': 67, 'round': 5, 'date': '05/10/2025 17:00', 'venue': 'M&T Bank Stadium', 'home_team': 'Baltimore Ravens', 'away_team': 'Houston Texans', 'result': None},
    {'match_id': 68, 'round': 5, 'date': '05/10/2025 17:00', 'venue': 'Bank of America Stadium', 'home_team': 'Carolina Panthers', 'away_team': 'Miami Dolphins', 'result': None},
    {'match_id': 69, 'round': 5, 'date': '05/10/2025 17:00', 'venue': 'Lucas Oil Stadium', 'home_team': 'Indianapolis Colts', 'away_team': 'Las Vegas Raiders', 'result': None},
    {'match_id': 70, 'round': 5, 'date': '05/10/2025 17:00', 'venue': 'Caesars Superdome', 'home_team': 'New Orleans Saints', 'away_team': 'New York Giants', 'result': None},
    {'match_id': 71, 'round': 5, 'date': '05/10/2025 17:00', 'venue': 'MetLife Stadium', 'home_team': 'New York Jets', 'away_team': 'Dallas Cowboys', 'result': None},
    {'match_id': 72, 'round': 5, 'date': '05/10/2025 17:00', 'venue': 'Lincoln Financial Field', 'home_team': 'Philadelphia Eagles', 'away_team': 'Denver Broncos', 'result': None},
    {'match_id': 73, 'round': 5, 'date': '05/10/2025 20:05', 'venue': 'State Farm Stadium', 'home_team': 'Arizona Cardinals', 'away_team': 'Tennessee Titans', 'result': None},
    {'match_id': 74, 'round': 5, 'date': '05/10/2025 20:05', 'venue': 'Lumen Field', 'home_team': 'Seattle Seahawks', 'away_team': 'Tampa Bay Buccaneers', 'result': None},
    {'match_id': 75, 'round': 5, 'date': '05/10/2025 20:25', 'venue': 'Paycor Stadium', 'home_team': 'Cincinnati Bengals', 'away_team': 'Detroit Lions', 'result': None},
    {'match_id': 76, 'round': 5, 'date': '05/10/2025 20:25', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Chargers', 'away_team': 'Washington Commanders', 'result': None},
    {'match_id': 77, 'round': 5, 'date': '06/10/2025 00:20', 'venue': 'Highmark Stadium', 'home_team': 'Buffalo Bills', 'away_team': 'New England Patriots', 'result': None},
    {'match_id': 78, 'round': 5, 'date': '07/10/2025 00:15', 'venue': 'EverBank Stadium', 'home_team': 'Jacksonville Jaguars', 'away_team': 'Kansas City Chiefs', 'result': None},
    {'match_id': 79, 'round': 6, 'date': '10/10/2025 00:15', 'venue': 'MetLife Stadium', 'home_team': 'New York Giants', 'away_team': 'Philadelphia Eagles', 'result': None},
    {'match_id': 80, 'round': 6, 'date': '12/10/2025 13:30', 'venue': 'Tottenham Hotspur Stadium', 'home_team': 'New York Jets', 'away_team': 'Denver Broncos', 'result': None},
    {'match_id': 81, 'round': 6, 'date': '12/10/2025 17:00', 'venue': 'M&T Bank Stadium', 'home_team': 'Baltimore Ravens', 'away_team': 'Los Angeles Rams', 'result': None},
    {'match_id': 82, 'round': 6, 'date': '12/10/2025 17:00', 'venue': 'Bank of America Stadium', 'home_team': 'Carolina Panthers', 'away_team': 'Dallas Cowboys', 'result': None},
    {'match_id': 83, 'round': 6, 'date': '12/10/2025 17:00', 'venue': 'Lucas Oil Stadium', 'home_team': 'Indianapolis Colts', 'away_team': 'Arizona Cardinals', 'result': None},
    {'match_id': 84, 'round': 6, 'date': '12/10/2025 17:00', 'venue': 'EverBank Stadium', 'home_team': 'Jacksonville Jaguars', 'away_team': 'Seattle Seahawks', 'result': None},
    {'match_id': 85, 'round': 6, 'date': '12/10/2025 17:00', 'venue': 'Hard Rock Stadium', 'home_team': 'Miami Dolphins', 'away_team': 'Los Angeles Chargers', 'result': None},
    {'match_id': 86, 'round': 6, 'date': '12/10/2025 17:00', 'venue': 'Acrisure Stadium', 'home_team': 'Pittsburgh Steelers', 'away_team': 'Cleveland Browns', 'result': None},
    {'match_id': 90, 'round': 6, 'date': '12/10/2025 17:00', 'venue': 'Caesars Superdome', 'home_team': 'New Orleans Saints', 'away_team': 'New England Patriots', 'result': None},
    {'match_id': 88, 'round': 6, 'date': '12/10/2025 20:05', 'venue': 'Allegiant Stadium', 'home_team': 'Las Vegas Raiders', 'away_team': 'Tennessee Titans', 'result': None},
    {'match_id': 87, 'round': 6, 'date': '12/10/2025 20:25', 'venue': 'Raymond James Stadium', 'home_team': 'Tampa Bay Buccaneers', 'away_team': 'San Francisco 49ers', 'result': None},
    {'match_id': 89, 'round': 6, 'date': '12/10/2025 20:25', 'venue': 'Lambeau Field', 'home_team': 'Green Bay Packers', 'away_team': 'Cincinnati Bengals', 'result': None},
    {'match_id': 91, 'round': 6, 'date': '13/10/2025 00:20', 'venue': 'GEHA Field at Arrowhead Stadium', 'home_team': 'Kansas City Chiefs', 'away_team': 'Detroit Lions', 'result': None},
    {'match_id': 92, 'round': 6, 'date': '13/10/2025 23:15', 'venue': 'Mercedes-Benz Stadium', 'home_team': 'Atlanta Falcons', 'away_team': 'Buffalo Bills', 'result': None},
    {'match_id': 93, 'round': 6, 'date': '14/10/2025 00:15', 'venue': 'Northwest Stadium', 'home_team': 'Washington Commanders', 'away_team': 'Chicago Bears', 'result': None},
    {'match_id': 94, 'round': 7, 'date': '17/10/2025 00:15', 'venue': 'Paycor Stadium', 'home_team': 'Cincinnati Bengals', 'away_team': 'Pittsburgh Steelers', 'result': None},
    {'match_id': 95, 'round': 7, 'date': '19/10/2025 13:30', 'venue': 'Wembley Stadium', 'home_team': 'Jacksonville Jaguars', 'away_team': 'Los Angeles Rams', 'result': None},
    {'match_id': 96, 'round': 7, 'date': '19/10/2025 17:00', 'venue': 'Soldier Field', 'home_team': 'Chicago Bears', 'away_team': 'New Orleans Saints', 'result': None},
    {'match_id': 97, 'round': 7, 'date': '19/10/2025 17:00', 'venue': 'Huntington Bank Field', 'home_team': 'Cleveland Browns', 'away_team': 'Miami Dolphins', 'result': None},
    {'match_id': 98, 'round': 7, 'date': '19/10/2025 17:00', 'venue': 'GEHA Field at Arrowhead Stadium', 'home_team': 'Kansas City Chiefs', 'away_team': 'Las Vegas Raiders', 'result': None},
    {'match_id': 99, 'round': 7, 'date': '19/10/2025 17:00', 'venue': 'U.S. Bank Stadium', 'home_team': 'Minnesota Vikings', 'away_team': 'Philadelphia Eagles', 'result': None},
    {'match_id': 100, 'round': 7, 'date': '19/10/2025 17:00', 'venue': 'MetLife Stadium', 'home_team': 'New York Jets', 'away_team': 'Carolina Panthers', 'result': None},
    {'match_id': 101, 'round': 7, 'date': '19/10/2025 17:00', 'venue': 'Nissan Stadium', 'home_team': 'Tennessee Titans', 'away_team': 'New England Patriots', 'result': None},
    {'match_id': 102, 'round': 7, 'date': '19/10/2025 20:05', 'venue': 'Empower Field at Mile High', 'home_team': 'Denver Broncos', 'away_team': 'New York Giants', 'result': None},
    {'match_id': 103, 'round': 7, 'date': '19/10/2025 20:05', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Chargers', 'away_team': 'Indianapolis Colts', 'result': None},
    {'match_id': 104, 'round': 7, 'date': '19/10/2025 20:25', 'venue': 'State Farm Stadium', 'home_team': 'Arizona Cardinals', 'away_team': 'Green Bay Packers', 'result': None},
    {'match_id': 105, 'round': 7, 'date': '19/10/2025 20:25', 'venue': 'AT&T Stadium', 'home_team': 'Dallas Cowboys', 'away_team': 'Washington Commanders', 'result': None},
    {'match_id': 106, 'round': 7, 'date': '20/10/2025 00:20', 'venue': "Levi's Stadium", 'home_team': 'San Francisco 49ers', 'away_team': 'Atlanta Falcons', 'result': None},
    {'match_id': 107, 'round': 7, 'date': '20/10/2025 23:00', 'venue': 'Ford Field', 'home_team': 'Detroit Lions', 'away_team': 'Tampa Bay Buccaneers', 'result': None},
    {'match_id': 108, 'round': 7, 'date': '21/10/2025 02:00', 'venue': 'Lumen Field', 'home_team': 'Seattle Seahawks', 'away_team': 'Houston Texans', 'result': None},
    {'match_id': 109, 'round': 8, 'date': '24/10/2025 00:15', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Chargers', 'away_team': 'Minnesota Vikings', 'result': None},
    {'match_id': 110, 'round': 8, 'date': '26/10/2025 17:00', 'venue': 'Mercedes-Benz Stadium', 'home_team': 'Atlanta Falcons', 'away_team': 'Miami Dolphins', 'result': None},
    {'match_id': 111, 'round': 8, 'date': '26/10/2025 17:00', 'venue': 'M&T Bank Stadium', 'home_team': 'Baltimore Ravens', 'away_team': 'Chicago Bears', 'result': None},
    {'match_id': 112, 'round': 8, 'date': '26/10/2025 17:00', 'venue': 'Bank of America Stadium', 'home_team': 'Carolina Panthers', 'away_team': 'Buffalo Bills', 'result': None},
    {'match_id': 113, 'round': 8, 'date': '26/10/2025 17:00', 'venue': 'Paycor Stadium', 'home_team': 'Cincinnati Bengals', 'away_team': 'New York Jets', 'result': None},
    {'match_id': 114, 'round': 8, 'date': '26/10/2025 17:00', 'venue': 'NRG Stadium', 'home_team': 'Houston Texans', 'away_team': 'San Francisco 49ers', 'result': None},
    {'match_id': 115, 'round': 8, 'date': '26/10/2025 17:00', 'venue': 'Gillette Stadium', 'home_team': 'New England Patriots', 'away_team': 'Cleveland Browns', 'result': None},
    {'match_id': 116, 'round': 8, 'date': '26/10/2025 17:00', 'venue': 'Lincoln Financial Field', 'home_team': 'Philadelphia Eagles', 'away_team': 'New York Giants', 'result': None},
    {'match_id': 117, 'round': 8, 'date': '26/10/2025 20:05', 'venue': 'Caesars Superdome', 'home_team': 'New Orleans Saints', 'away_team': 'Tampa Bay Buccaneers', 'result': None},
    {'match_id': 118, 'round': 8, 'date': '26/10/2025 20:25', 'venue': 'Empower Field at Mile High', 'home_team': 'Denver Broncos', 'away_team': 'Dallas Cowboys', 'result': None},
    {'match_id': 119, 'round': 8, 'date': '26/10/2025 20:25', 'venue': 'Lucas Oil Stadium', 'home_team': 'Indianapolis Colts', 'away_team': 'Tennessee Titans', 'result': None},
    {'match_id': 120, 'round': 8, 'date': '27/10/2025 00:20', 'venue': 'Acrisure Stadium', 'home_team': 'Pittsburgh Steelers', 'away_team': 'Green Bay Packers', 'result': None},
    {'match_id': 121, 'round': 8, 'date': '28/10/2025 00:15', 'venue': 'GEHA Field at Arrowhead Stadium', 'home_team': 'Kansas City Chiefs', 'away_team': 'Washington Commanders', 'result': None},
    {'match_id': 122, 'round': 9, 'date': '31/10/2025 00:15', 'venue': 'Hard Rock Stadium', 'home_team': 'Miami Dolphins', 'away_team': 'Baltimore Ravens', 'result': None},
    {'match_id': 123, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'Paycor Stadium', 'home_team': 'Cincinnati Bengals', 'away_team': 'Chicago Bears', 'result': None},
    {'match_id': 124, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'Ford Field', 'home_team': 'Detroit Lions', 'away_team': 'Minnesota Vikings', 'result': None},
    {'match_id': 125, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'Lambeau Field', 'home_team': 'Green Bay Packers', 'away_team': 'Carolina Panthers', 'result': None},
    {'match_id': 126, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'NRG Stadium', 'home_team': 'Houston Texans', 'away_team': 'Denver Broncos', 'result': None},
    {'match_id': 127, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'Gillette Stadium', 'home_team': 'New England Patriots', 'away_team': 'Atlanta Falcons', 'result': None},
    {'match_id': 128, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'MetLife Stadium', 'home_team': 'New York Giants', 'away_team': 'San Francisco 49ers', 'result': None},
    {'match_id': 129, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'Acrisure Stadium', 'home_team': 'Pittsburgh Steelers', 'away_team': 'Indianapolis Colts', 'result': None},
    {'match_id': 130, 'round': 9, 'date': '02/11/2025 18:00', 'venue': 'Nissan Stadium', 'home_team': 'Tennessee Titans', 'away_team': 'Los Angeles Chargers', 'result': None},
    {'match_id': 131, 'round': 9, 'date': '02/11/2025 21:05', 'venue': 'SoFi Stadium', 'home_team': 'Los Angeles Rams', 'away_team': 'New Orleans Saints', 'result': None},
    {'match_id': 132, 'round': 9, 'date': '02/11/2025 21:05', 'venue': 'Allegiant Stadium', 'home_team': 'Las Vegas Raiders', 'away_team': 'Jacksonville Jaguars', 'result': None},
    {'match_id': 133, 'round': 9, 'date': '02/11/2025 21:25', 'venue': 'Highmark Stadium', 'home_team': 'Buffalo Bills', 'away_team': 'Kansas City Chiefs', 'result': None},
    {'match_id': 134, 'round': 9, 'date': '03/11/2025 01:20', 'venue': 'Northwest Stadium', 'home_team': 'Washington Commanders', 'away_team': 'Seattle Seahawks', 'result': None},
    {'match_id': 135, 'round': 9, 'date': '04/11/2025 01:15', 'venue': 'AT&T Stadium', 'home_team': 'Dallas Cowboys', 'away_team': 'Arizona Cardinals', 'result': None},
    {'match_id': 136, 'round': 10, 'date': '07/11/2025 01:15', 'venue': 'Empower Field at Mile High', 'home_team': 'Denver Broncos', 'away_team': 'Las Vegas Raiders', 'result': None},
    {'match_id': 137, 'round': 10, 'date': '09/11/2025 14:30', 'venue': 'Olympic Stadium', 'home_team': 'Indianapolis Colts', 'away_team': 'Atlanta Falcons', 'result': None},
    {'match_id': 138, 'round': 10, 'date': '09/11/2025 18:00', 'venue': 'Bank of America Stadium', 'home_team': 'Carolina Panthers', 'away_team': 'New Orleans Saints', 'result': None},
    {'match_id': 139, 'round': 10, 'date': '09/11/2025 18:00', 'venue': 'Soldier Field', 'home_team': 'Chicago Bears', 'away_team': 'New York Giants', 'result': None},
    {'match_id': 140, 'round': 10, 'date': '09/11/2025 18:00', 'venue': 'NRG Stadium', 'home_team': 'Houston Texans', 'away_team': 'Jacksonville Jaguars', 'result': None},
    {'match_id': 141, 'round': 10, 'date': '09/11/2025 18:00', 'venue': 'Hard Rock Stadium', 'home_team': 'Miami Dolphins', 'away_team': 'Buffalo Bills', 'result': None},
    {'match_id': 142, 'round': 10, 'date': '09/11/2025 18:00', 'venue': 'U.S. Bank Stadium', 'home_team': 'Minnesota Vikings', 'away_team': 'Baltimore Ravens', 'result': None},
    {'match_id': 143, 'round': 10, 'date': '09/11/2025 18:00', 'venue': 'MetLife Stadium', 'home_team': 'New York Jets', 'away_team': 'Cleveland Browns', 'result': None},
    {'match_id': 144, 'round': 10, 'date': '09/11/2025 18:00', 'venue': 'Raymond James Stadium', 'home_team': 'Tampa Bay Buccaneers', 'away_team': 'New England Patriots', 'result': None},
    {'match_id': 145, 'round': 10, 'date': '09/11/2025 21:05', 'venue': 'Lumen Field', 'home_team': 'Seattle Seahawks', 'away_team': 'Arizona Cardinals', 'result': None},
    {'match_id': 146, 'round': 10, 'date': '09/11/2025 21:25', 'venue': "Levi's Stadium", 'home_team': 'San Francisco 49ers', 'away_team': 'Los Angeles Rams', 'result': None},
    {'match_id': 147, 'round': 10, 'date': '09/11/2025 21:25', 'venue': 'Northwest Stadium', 'home_team': 'Washington Commanders', 'away_team': 'Detroit Lions', 'result': None},
    {'match_id': 148, 'round': 10, 'date': '10/11/2025 01:20', 'venue': 'AT&T Stadium', 'home_team': 'Dallas Cowboys', 'away_team': 'Pittsburgh Steelers', 'result': None},
    {'match_id': 149, 'round': 10, 'date': '11/11/2025 01:15', 'venue': 'Lambeau Field', 'home_team': 'Green Bay Packers', 'away_team': 'Los Angeles Chargers', 'result': None},
]

# Extract scores
def parse_result(result_str):
    """Parse '24 - 20' into (24, 20) or return (None, None)"""
    if result_str is None:
        return None, None
    try:
        parts = result_str.split('-')
        return int(parts[0].strip()), int(parts[1].strip())
    except:
        return None, None

# Connect to database
conn = sqlite3.connect('sports_predictions_original.db')
cursor = conn.cursor()

# Delete all NFL games and predictions
print("Deleting all current NFL games...")
cursor.execute("DELETE FROM predictions WHERE game_id IN (SELECT game_id FROM games WHERE sport='NFL')")
cursor.execute("DELETE FROM games WHERE sport='NFL'")
conn.commit()

# Insert schedule
print("Importing NFL schedule...")
inserted = 0
for game in nfl_schedule:
    home_score, away_score = parse_result(game['result'])
    status = 'completed' if home_score is not None else 'scheduled'
    
    game_id = f"NFL_2025_{game['match_id']}"
    
    cursor.execute("""
        INSERT INTO games (
            sport, league, game_id, season, game_date, 
            home_team_id, away_team_id, status, home_score, away_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        'NFL', 'NFL', game_id, 2025, game['date'],
        game['home_team'], game['away_team'], status, home_score, away_score
    ))
    inserted += 1

conn.commit()
conn.close()

print(f"✓ Imported {inserted} NFL games")
print(f"✓ Completed games: {sum(1 for g in nfl_schedule if g['result'] is not None)}")
print("Done!")
