from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import date

class BaseFeatureEngineer(ABC):
    """
    Abstract base class for sport-specific feature engineering.
    
    All sport feature engineers must implement these methods to ensure
    consistent feature creation across different sports.
    """
    
    def __init__(self, sport: str):
        """
        Initialize the feature engineer.
        
        Args:
            sport: Sport name (e.g., 'MLB', 'NBA', 'NFL', 'NHL', 'CFB', 'CBB')
        """
        self.sport = sport
        self.required_features = []
        self.feature_descriptions = {}
    
    @abstractmethod
    def create_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame, 
                       additional_data: Optional[Dict] = None, 
                       include_targets: bool = True) -> pd.DataFrame:
        """
        Create features for prediction models.
        
        Args:
            games_df: DataFrame with game information
            team_stats_df: DataFrame with team statistics
            additional_data: Optional dictionary with additional data sources
            include_targets: Whether to include target variables (False for inference)
            
        Returns:
            DataFrame with engineered features for each game
            Must include columns:
            - game_id: Game identifier
            - home_team_id: Home team identifier
            - away_team_id: Away team identifier
            - game_date: Date of game
            - target_winner: Target variable for winner prediction (1=home, 0=away) [if include_targets=True]
            - target_total: Target variable for total points/runs/goals [if include_targets=True]
            - [feature columns]: All engineered features
        """
        pass
    
    @abstractmethod
    def create_rolling_features(self, team_stats_df: pd.DataFrame, 
                               window_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create rolling/moving average features.
        
        Args:
            team_stats_df: DataFrame with team statistics
            window_sizes: List of window sizes for rolling calculations (default: [5, 10, 20])
            
        Returns:
            DataFrame with rolling features added
        """
        if window_sizes is None:
            window_sizes = [5, 10, 20]
        pass
    
    @abstractmethod
    def create_matchup_features(self, home_team_id: str, away_team_id: str,
                               team_stats_df: pd.DataFrame, 
                               h2h_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Create features specific to team matchup.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            team_stats_df: DataFrame with team statistics
            h2h_data: Optional head-to-head historical data
            
        Returns:
            Dictionary with matchup-specific features
        """
        pass
    
    @abstractmethod
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """
        Get features grouped by importance categories.
        
        Returns:
            Dictionary mapping category names to lists of feature names
            Categories should include: 'offense', 'defense', 'special', 'situational'
        """
        pass
    
    def create_team_strength_features(self, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create general team strength features (implemented by base class).
        
        Args:
            team_stats_df: DataFrame with team statistics
            
        Returns:
            DataFrame with team strength features
        """
        df = team_stats_df.copy()
        
        if 'wins' in df.columns and 'losses' in df.columns:
            df['win_percentage'] = df['wins'] / (df['wins'] + df['losses'])
            df['win_percentage'] = df['win_percentage'].fillna(0.5)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in feature data.
        
        Args:
            df: DataFrame with features
            strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'forward_fill')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'median':
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        elif strategy == 'mean':
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        elif strategy == 'forward_fill':
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
        
        # Fill any remaining NaNs with 0
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df
    
    def validate_features(self, features_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that created features meet requirements.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required_cols = ['game_id', 'home_team_id', 'away_team_id', 'game_date']
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for target columns only if they should be present
        target_cols = ['target_winner', 'target_total']
        has_any_targets = any(col in features_df.columns for col in target_cols)
        if has_any_targets:
            # If any targets present, validate them
            for col in target_cols:
                if col in features_df.columns and bool(features_df[col].isnull().all()):
                    issues.append(f"Target column '{col}' is present but entirely null")
        
        # Check for excessive missing values
        missing_pct = features_df.isnull().sum() / len(features_df)
        high_missing = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing:
            issues.append(f"Features with >50% missing values: {high_missing}")
        
        # Check for infinite values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(features_df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            issues.append(f"Features with infinite values: {inf_cols}")
        
        return len(issues) == 0, issues
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names this engineer creates.
        
        Returns:
            List of feature names
        """
        return self.required_features
    
    def get_feature_description(self, feature_name: str) -> str:
        """
        Get description of a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Description of the feature
        """
        return self.feature_descriptions.get(feature_name, "No description available")