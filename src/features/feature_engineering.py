import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    """
    Feature engineering for MLB prediction models
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for MLB prediction
        
        Args:
            data: Raw game data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if data.empty:
                return data
            
            features_df = data.copy()
            
            # Basic game features
            features_df = self._add_basic_features(features_df)
            
            # Team performance features
            features_df = self._add_team_features(features_df)
            
            # Pitching and batting features
            features_df = self._add_pitching_features(features_df)
            features_df = self._add_batting_features(features_df)
            
            # Situational features
            features_df = self._add_situational_features(features_df)
            
            # Rolling statistics
            features_df = self._add_rolling_stats(features_df)
            
            # Opponent-specific features
            features_df = self._add_matchup_features(features_df)
            
            self.logger.info(f"Created {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            return data
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic game-level features"""
        try:
            # Date features
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
                df['month'] = df['game_date'].dt.month
                df['day_of_week'] = df['game_date'].dt.dayofweek
                df['day_of_year'] = df['game_date'].dt.dayofyear
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Team encoding (convert team names to numeric)
            if 'home_team' in df.columns and 'away_team' in df.columns:
                if 'home_team_encoded' not in df.columns:
                    teams = pd.concat([df['home_team'], df['away_team']]).unique()
                    team_encoder = LabelEncoder()
                    team_encoder.fit(teams)
                    self.encoders['team'] = team_encoder
                    
                    df['home_team_encoded'] = team_encoder.transform(df['home_team'])
                    df['away_team_encoded'] = team_encoder.transform(df['away_team'])
            
            # Game context features
            if 'total_pitches' in df.columns:
                df['pitches_per_inning'] = df['total_pitches'] / 9
                df['game_pace'] = np.where(df['total_pitches'] > df['total_pitches'].median(), 1, 0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basic features: {str(e)}")
            return df
    
    def _add_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team-level performance features"""
        try:
            # Calculate team statistics over recent games
            for team_col in ['home_team', 'away_team']:
                if team_col in df.columns:
                    prefix = team_col.split('_')[0]  # 'home' or 'away'
                    
                    # Team scoring averages
                    if f'{prefix}_score' in df.columns:
                        # Rolling averages for team performance
                        df = df.sort_values('game_date')
                        df[f'{prefix}_score_avg_5'] = df.groupby(team_col)[f'{prefix}_score'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
                        df[f'{prefix}_score_avg_10'] = df.groupby(team_col)[f'{prefix}_score'].transform(
                            lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
                        )
                    
                    # Recent form (wins in last 5/10 games)
                    if 'home_win' in df.columns:
                        if prefix == 'home':
                            win_col = 'home_win'
                        else:
                            win_col = 'away_win'
                            if 'away_win' not in df.columns:
                                df['away_win'] = 1 - df['home_win']
                        
                        if win_col in df.columns:
                            df[f'{prefix}_wins_last_5'] = df.groupby(team_col)[win_col].transform(
                                lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
                            )
                            df[f'{prefix}_wins_last_10'] = df.groupby(team_col)[win_col].transform(
                                lambda x: x.rolling(window=10, min_periods=1).sum().shift(1)
                            )
            
            # Head-to-head record
            if 'home_team' in df.columns and 'away_team' in df.columns:
                df['matchup'] = df['home_team'] + '_vs_' + df['away_team']
                
                if 'home_win' in df.columns:
                    df['h2h_home_wins'] = df.groupby('matchup')['home_win'].transform(
                        lambda x: x.expanding().sum().shift(1)
                    )
                    df['h2h_total_games'] = df.groupby('matchup').cumcount()
                    df['h2h_home_win_pct'] = np.where(
                        df['h2h_total_games'] > 0,
                        df['h2h_home_wins'] / df['h2h_total_games'],
                        0.5
                    )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding team features: {str(e)}")
            return df
    
    def _add_pitching_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pitching-related features"""
        try:
            if 'avg_velocity' in df.columns:
                # Velocity features
                df['high_velocity'] = (df['avg_velocity'] > 95).astype(int)
                df['velocity_differential'] = df['avg_velocity'] - df['avg_velocity'].mean()
            
            # Pitching effectiveness metrics
            if 'hard_hits' in df.columns and 'total_pitches' in df.columns:
                df['hard_hit_rate'] = np.where(
                    df['total_pitches'] > 0,
                    df['hard_hits'] / df['total_pitches'],
                    0
                )
            
            if 'barrels' in df.columns and 'total_pitches' in df.columns:
                df['barrel_rate'] = np.where(
                    df['total_pitches'] > 0,
                    df['barrels'] / df['total_pitches'],
                    0
                )
            
            # Team pitching trends
            for team_col in ['home_team', 'away_team']:
                if team_col in df.columns:
                    prefix = team_col.split('_')[0]
                    
                    # Rolling pitching stats
                    if 'avg_velocity' in df.columns:
                        df[f'{prefix}_velocity_avg_5'] = df.groupby(team_col)['avg_velocity'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
                    
                    if 'hard_hit_rate' in df.columns:
                        df[f'{prefix}_hard_hit_rate_avg_5'] = df.groupby(team_col)['hard_hit_rate'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding pitching features: {str(e)}")
            return df
    
    def _add_batting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add batting-related features"""
        try:
            # Exit velocity features
            if 'avg_exit_velocity' in df.columns:
                df['high_exit_velocity'] = (df['avg_exit_velocity'] > 95).astype(int)
                df['exit_velocity_differential'] = df['avg_exit_velocity'] - df['avg_exit_velocity'].mean()
            
            # Quality of contact
            if 'hard_hits' in df.columns and 'barrels' in df.columns:
                df['quality_contact_rate'] = (df['hard_hits'] + df['barrels']) / (df['total_pitches'] + 1)
            
            # Team batting trends
            for team_col in ['home_team', 'away_team']:
                if team_col in df.columns:
                    prefix = team_col.split('_')[0]
                    
                    # Rolling batting stats
                    if 'avg_exit_velocity' in df.columns:
                        df[f'{prefix}_exit_velocity_avg_5'] = df.groupby(team_col)['avg_exit_velocity'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
                    
                    if 'quality_contact_rate' in df.columns:
                        df[f'{prefix}_quality_contact_avg_5'] = df.groupby(team_col)['quality_contact_rate'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding batting features: {str(e)}")
            return df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational and contextual features"""
        try:
            # Home field advantage proxy
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df['score_differential'] = df['home_score'] - df['away_score']
            
            # Rest days (simplified - would need actual schedule data)
            if 'game_date' in df.columns:
                df = df.sort_values(['home_team', 'game_date'])
                df['home_rest_days'] = df.groupby('home_team')['game_date'].diff().dt.days
                df['home_rest_days'] = df['home_rest_days'].fillna(3)  # Default rest
                
                df = df.sort_values(['away_team', 'game_date'])
                df['away_rest_days'] = df.groupby('away_team')['game_date'].diff().dt.days
                df['away_rest_days'] = df['away_rest_days'].fillna(3)
                
                # Rest advantage
                df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
            
            # Season timing features
            if 'day_of_year' in df.columns:
                df['early_season'] = (df['day_of_year'] < 120).astype(int)  # Before May
                df['late_season'] = (df['day_of_year'] > 240).astype(int)   # After August
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding situational features: {str(e)}")
            return df
    
    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        try:
            # Sort by date for proper rolling calculations
            df = df.sort_values('game_date')
            
            # Total runs trends
            if 'total_runs' in df.columns:
                for team_col in ['home_team', 'away_team']:
                    if team_col in df.columns:
                        prefix = team_col.split('_')[0]
                        
                        # Rolling totals for over/under prediction
                        df[f'{prefix}_total_runs_avg_5'] = df.groupby(team_col)['total_runs'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
                        df[f'{prefix}_total_runs_avg_10'] = df.groupby(team_col)['total_runs'].transform(
                            lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
                        )
                        
                        # Rolling standard deviation (volatility)
                        df[f'{prefix}_total_runs_std_5'] = df.groupby(team_col)['total_runs'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).std().shift(1)
                        )
            
            # Recent performance trends
            if 'home_score' in df.columns and 'away_score' in df.columns:
                for team_col, score_col in [('home_team', 'home_score'), ('away_team', 'away_score')]:
                    if team_col in df.columns and score_col in df.columns:
                        prefix = team_col.split('_')[0]
                        
                        # Runs scored trends
                        df[f'{prefix}_runs_scored_avg_5'] = df.groupby(team_col)[score_col].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
                        
                        # Runs allowed trends (opponent's scoring)
                        opp_score_col = 'away_score' if prefix == 'home' else 'home_score'
                        if opp_score_col in df.columns:
                            df[f'{prefix}_runs_allowed_avg_5'] = df.groupby(team_col)[opp_score_col].transform(
                                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding rolling stats: {str(e)}")
            return df
    
    def _add_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features"""
        try:
            if 'home_team' in df.columns and 'away_team' in df.columns:
                # Create unique matchup identifiers
                df['team_matchup'] = df['home_team'] + '_vs_' + df['away_team']
                df['division_matchup'] = 0  # Would need division data
                
                # Historical scoring in this matchup
                if 'total_runs' in df.columns:
                    df['matchup_avg_total'] = df.groupby('team_matchup')['total_runs'].transform(
                        lambda x: x.expanding().mean().shift(1)
                    )
                    df['matchup_total_std'] = df.groupby('team_matchup')['total_runs'].transform(
                        lambda x: x.expanding().std().shift(1)
                    )
                
                # Matchup win percentage
                if 'home_win' in df.columns:
                    df['matchup_home_wins'] = df.groupby('team_matchup')['home_win'].transform(
                        lambda x: x.expanding().sum().shift(1)
                    )
                    df['matchup_games'] = df.groupby('team_matchup').cumcount()
                    df['matchup_home_win_pct'] = np.where(
                        df['matchup_games'] > 0,
                        df['matchup_home_wins'] / df['matchup_games'],
                        0.5
                    )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding matchup features: {str(e)}")
            return df
    
    def prepare_features_for_training(self, df: pd.DataFrame, target_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for model training
        
        Args:
            df: DataFrame with features
            target_columns: List of target column names
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        try:
            # Separate features and targets
            feature_columns = [col for col in df.columns if col not in target_columns and col not in [
                'game_pk', 'game_date', 'home_team', 'away_team', 'matchup', 'team_matchup'
            ]]
            
            features = df[feature_columns].copy()
            targets = df[target_columns].copy()
            
            # Handle missing values
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            categorical_columns = features.select_dtypes(exclude=[np.number]).columns
            
            # Impute numeric columns
            if len(numeric_columns) > 0:
                if 'numeric' not in self.imputers:
                    self.imputers['numeric'] = SimpleImputer(strategy='median')
                    features[numeric_columns] = self.imputers['numeric'].fit_transform(features[numeric_columns])
                else:
                    features[numeric_columns] = self.imputers['numeric'].transform(features[numeric_columns])
            
            # Impute categorical columns
            if len(categorical_columns) > 0:
                if 'categorical' not in self.imputers:
                    self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                    features[categorical_columns] = self.imputers['categorical'].fit_transform(features[categorical_columns])
                else:
                    features[categorical_columns] = self.imputers['categorical'].transform(features[categorical_columns])
            
            # Scale features
            if 'features' not in self.scalers:
                self.scalers['features'] = StandardScaler()
                scaled_features = self.scalers['features'].fit_transform(features[numeric_columns])
            else:
                scaled_features = self.scalers['features'].transform(features[numeric_columns])
            
            # Create final feature DataFrame
            final_features = features.copy()
            final_features[numeric_columns] = scaled_features
            
            self.logger.info(f"Prepared {len(final_features.columns)} features for training")
            
            return final_features, targets
            
        except Exception as e:
            self.logger.error(f"Error preparing features for training: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get list of feature names for importance analysis
        
        Returns:
            List of feature names
        """
        # This would be populated after feature creation
        # For now, return common feature categories
        return [
            'Basic Features', 'Team Performance', 'Pitching Stats', 
            'Batting Stats', 'Situational', 'Rolling Stats', 'Matchup History'
        ]
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Create features
            features_df = self.create_features(df)
            
            # Apply same transformations as training data
            feature_columns = [col for col in features_df.columns if col not in [
                'game_pk', 'game_date', 'home_team', 'away_team', 'matchup', 'team_matchup',
                'home_win', 'total_runs'  # target columns
            ]]
            
            if not feature_columns:
                return features_df
            
            features = features_df[feature_columns].copy()
            
            # Apply imputers if fitted
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            categorical_columns = features.select_dtypes(exclude=[np.number]).columns
            
            if 'numeric' in self.imputers and len(numeric_columns) > 0:
                features[numeric_columns] = self.imputers['numeric'].transform(features[numeric_columns])
            
            if 'categorical' in self.imputers and len(categorical_columns) > 0:
                features[categorical_columns] = self.imputers['categorical'].transform(features[categorical_columns])
            
            # Apply scaler if fitted
            if 'features' in self.scalers and len(numeric_columns) > 0:
                features[numeric_columns] = self.scalers['features'].transform(features[numeric_columns])
            
            # Merge back with original DataFrame
            result_df = features_df.copy()
            result_df[feature_columns] = features
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error transforming new data: {str(e)}")
            return df
