"""
Sport-agnostic interfaces for the multi-sport prediction system.

This module provides abstract base classes that define consistent interfaces
for data collection, feature engineering, and prediction across all sports.
"""

from .base_collector import BaseDataCollector
from .base_feature_engineer import BaseFeatureEngineer
from .base_predictor import BasePredictor

__all__ = [
    'BaseDataCollector',
    'BaseFeatureEngineer', 
    'BasePredictor'
]