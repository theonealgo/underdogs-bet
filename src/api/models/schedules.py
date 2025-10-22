#!/usr/bin/env python3
"""
Sports Schedules Module
Stub functions for all sports - NHL loads from database
"""

import sys
import os
import sqlite3
import pandas as pd

# Add models to path
models_path = os.path.join(os.path.dirname(__file__), '../../..')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

# Dynamic import to avoid circular dependencies
def _get_root_schedule_function(sport):
    """Dynamically import schedule function from root models/schedules.py"""
    try:
        import models.schedules as root_schedules
        return getattr(root_schedules, f'get_{sport.lower()}_schedule', None)
    except:
        return None

def get_nfl_schedule():
    """NFL Schedule from root models/schedules.py"""
    func = _get_root_schedule_function('NFL')
    return func() if func else []

def get_nba_schedule():
    """NBA Schedule from root models/schedules.py"""
    func = _get_root_schedule_function('NBA')
    return func() if func else []

def get_nhl_schedule():
    """NHL Schedule - Returns all games from database via nhlschedules.py"""
    # For now, return empty list - will be loaded from database
    # In production, this would read from schedules/NHL.xlsx or database
    return []

def get_mlb_schedule():
    """MLB Schedule - placeholder"""
    return []

def get_ncaaf_schedule():
    """NCAA Football Schedule - placeholder"""
    return []

def get_ncaab_schedule():
    """NCAA Basketball Schedule - placeholder"""
    return []

def get_wnba_schedule():
    """WNBA Schedule - placeholder"""
    return []

__all__ = [
    'get_nfl_schedule',
    'get_nba_schedule', 
    'get_nhl_schedule',
    'get_mlb_schedule',
    'get_ncaaf_schedule',
    'get_ncaab_schedule',
    'get_wnba_schedule'
]
