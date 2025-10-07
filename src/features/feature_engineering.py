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
        
        # Deterministic pregame feature schema - only features available before games
        # Based on Bill James Pythagorean Theorem and high-correlation stats
        self.PREGAME_FEATURES = [
            # Basic temporal features
            'month', 'day_of_week', 'day_of_year', 'is_weekend',
            'home_team_encoded', 'away_team_encoded',
            
            # Pythagorean expectation features (highest correlation with wins)
            'home_pythag_win_pct', 'away_pythag_win_pct', 'pythag_matchup_diff',
            'home_pythag_14', 'away_pythag_14', 'pythag_14_diff',
            'home_pythag_30', 'away_pythag_30', 'pythag_30_diff',
            
            # Run differential (R² = 0.887 - highest correlation)
            'home_run_diff', 'away_run_diff', 'run_diff_matchup',
            'home_run_diff_14', 'away_run_diff_14', 'run_diff_14_matchup',
            'home_run_diff_30', 'away_run_diff_30', 'run_diff_30_matchup',
            
            # Top pitching stats (11 of top 19 correlations)
            'home_era', 'away_era', 'era_matchup_diff',
            'home_fip', 'away_fip', 'fip_matchup_diff', 
            'home_whip', 'away_whip', 'whip_matchup_diff',
            'home_h_per_9', 'away_h_per_9', 'h9_matchup_diff',
            'home_baa', 'away_baa', 'baa_matchup_diff',
            
            # Top hitting stats
            'home_obp', 'away_obp', 'obp_matchup_diff',
            'home_slg', 'away_slg', 'slg_matchup_diff',
            'home_ops', 'away_ops', 'ops_matchup_diff',
            
            # Composite strength indices (weighted by correlations)
            'home_pitching_index', 'away_pitching_index', 'pitching_matchup_diff',
            'home_hitting_index', 'away_hitting_index', 'hitting_matchup_diff',
            
            # Legacy features for compatibility
            'pitches_per_inning', 'game_pace'
        ]
        
        # Correlation weights for composite indices (from user's analysis)
        self.PITCHING_WEIGHTS = {
            'era': 0.790, 'fip': 0.770, 'whip': 0.691, 'h_per_9': 0.676, 'baa': 0.663
        }
        
        self.HITTING_WEIGHTS = {
            'obp': 0.343, 'slg': 0.399, 'ops': 0.422  # Normalized weights
        }
    
    def get_pregame_feature_columns(self):
        """Return the exact feature columns used for pregame predictions"""
        return self.PREGAME_FEATURES.copy()
    
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
    
    def create_pregame_features(self, data: pd.DataFrame, db_manager=None) -> pd.DataFrame:
        """
        Create features using only pregame-available data with Pythagorean statistics
        
        Args:
            data: Raw game data with schedule information
            db_manager: Database manager for team metrics lookup
            
        Returns:
            DataFrame with exactly PREGAME_FEATURES columns including Pythagorean stats
        """
        try:
            if data.empty:
                return pd.DataFrame(columns=self.PREGAME_FEATURES)
            
            self.logger.info(f"Creating advanced pregame features for {len(data)} games...")
            features_df = data.copy()
            
            # Add basic pregame features 
            features_df = self._add_basic_features(features_df)
            
            # Add advanced team metrics and Pythagorean features
            if db_manager is not None:
                features_df = self._add_team_metrics_features(features_df, db_manager)
            else:
                self.logger.warning("No database manager provided - using basic features only")
            
            # Select only the defined pregame features, fill missing with defaults
            result_df = pd.DataFrame(index=features_df.index)
            for feature in self.PREGAME_FEATURES:
                if feature in features_df.columns:
                    result_df[feature] = features_df[feature]
                else:
                    # Default values for missing pregame features
                    if 'encoded' in feature:
                        result_df[feature] = 0  # Default team encoding
                    elif feature in ['month', 'day_of_week', 'day_of_year']:
                        result_df[feature] = 1  # Default date values
                    elif 'pythag' in feature.lower():
                        result_df[feature] = 0.5  # Default win percentage
                    elif 'diff' in feature or 'matchup' in feature:
                        result_df[feature] = 0  # Default difference
                    elif 'era' in feature:
                        result_df[feature] = 4.5  # Default ERA
                    elif any(stat in feature for stat in ['obp', 'slg', 'ops']):
                        result_df[feature] = 0.75 if 'ops' in feature else 0.32  # Default hitting stats
                    else:
                        result_df[feature] = 0  # Default numeric values
            
            # Ensure all numeric and no NaN values
            result_df = result_df.fillna(0).astype(float)
            
            self.logger.info(f"Created pregame features with {len(result_df.columns)} columns including Pythagorean stats")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error creating pregame features: {str(e)}")
            return pd.DataFrame(columns=self.PREGAME_FEATURES)
    
    def _add_team_metrics_features(self, df: pd.DataFrame, db_manager) -> pd.DataFrame:
        """
        Add advanced team metrics features including Pythagorean calculations
        
        Args:
            df: DataFrame with game data
            db_manager: Database manager for metrics lookup
            
        Returns:
            DataFrame with team metrics features added
        """
        try:
            if 'game_date' not in df.columns or 'home_team' not in df.columns or 'away_team' not in df.columns:
                self.logger.warning("Missing required columns for team metrics")
                return df
            
            self.logger.info("Adding Pythagorean and advanced team statistics...")
            
            for idx, row in df.iterrows():
                game_date = pd.to_datetime(row['game_date'])
                home_team = row['home_team']
                away_team = row['away_team']
                
                # Get team metrics for both teams (from previous day to avoid data leakage)
                prev_date = game_date - timedelta(days=1)
                home_metrics = db_manager.get_team_metrics(home_team, prev_date)
                away_metrics = db_manager.get_team_metrics(away_team, prev_date)
                
                # Add Pythagorean features
                df = self._add_pythagorean_features(df, idx, home_metrics, away_metrics, 'home', 'away')
                
                # Add run differential features  
                df = self._add_run_differential_features(df, idx, home_metrics, away_metrics)
                
                # Add pitching statistics
                df = self._add_pitching_stat_features(df, idx, home_metrics, away_metrics)
                
                # Add hitting statistics
                df = self._add_hitting_stat_features(df, idx, home_metrics, away_metrics)
                
                # Calculate composite strength indices
                df = self._add_composite_indices(df, idx, home_metrics, away_metrics)
            
            self.logger.info("Successfully added team metrics features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding team metrics features: {str(e)}")
            return df
    
    def _safe_get_metric(self, metrics_df: pd.DataFrame, column: str, default_value):
        """Safely get a metric value, handling None/NaN"""
        if metrics_df.empty or column not in metrics_df.columns:
            return default_value
        value = metrics_df[column].iloc[0]
        return value if value is not None and not pd.isna(value) else default_value
    
    def _add_pythagorean_features(self, df: pd.DataFrame, idx: int, home_metrics: pd.DataFrame, 
                                 away_metrics: pd.DataFrame, home_prefix: str, away_prefix: str) -> pd.DataFrame:
        """Add Pythagorean win percentage features"""
        try:
            # Season Pythagorean win percentages
            home_pythag = self._safe_get_metric(home_metrics, 'pythag_win_pct', 0.5)
            away_pythag = self._safe_get_metric(away_metrics, 'pythag_win_pct', 0.5)
            
            df.loc[idx, f'{home_prefix}_pythag_win_pct'] = home_pythag
            df.loc[idx, f'{away_prefix}_pythag_win_pct'] = away_pythag
            df.loc[idx, 'pythag_matchup_diff'] = home_pythag - away_pythag
            
            # 14-day rolling Pythagorean
            home_pythag_14 = self._safe_get_metric(home_metrics, 'pythag_win_pct_14', home_pythag)
            away_pythag_14 = self._safe_get_metric(away_metrics, 'pythag_win_pct_14', away_pythag)
            
            df.loc[idx, f'{home_prefix}_pythag_14'] = home_pythag_14
            df.loc[idx, f'{away_prefix}_pythag_14'] = away_pythag_14
            df.loc[idx, 'pythag_14_diff'] = home_pythag_14 - away_pythag_14
            
            # 30-day rolling Pythagorean
            home_pythag_30 = self._safe_get_metric(home_metrics, 'pythag_win_pct_30', home_pythag)
            away_pythag_30 = self._safe_get_metric(away_metrics, 'pythag_win_pct_30', away_pythag)
            
            df.loc[idx, f'{home_prefix}_pythag_30'] = home_pythag_30
            df.loc[idx, f'{away_prefix}_pythag_30'] = away_pythag_30
            df.loc[idx, 'pythag_30_diff'] = home_pythag_30 - away_pythag_30
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding Pythagorean features: {str(e)}")
            return df
    
    def _add_run_differential_features(self, df: pd.DataFrame, idx: int, 
                                     home_metrics: pd.DataFrame, away_metrics: pd.DataFrame) -> pd.DataFrame:
        """Add run differential features (highest correlation R² = 0.887)"""
        try:
            # Season run differentials
            home_run_diff = self._safe_get_metric(home_metrics, 'run_differential', 0)
            away_run_diff = self._safe_get_metric(away_metrics, 'run_differential', 0)
            
            df.loc[idx, 'home_run_diff'] = home_run_diff
            df.loc[idx, 'away_run_diff'] = away_run_diff
            df.loc[idx, 'run_diff_matchup'] = home_run_diff - away_run_diff
            
            # 14-day run differentials
            home_run_diff_14 = self._safe_get_metric(home_metrics, 'run_diff_14', home_run_diff)
            away_run_diff_14 = self._safe_get_metric(away_metrics, 'run_diff_14', away_run_diff)
            
            df.loc[idx, 'home_run_diff_14'] = home_run_diff_14
            df.loc[idx, 'away_run_diff_14'] = away_run_diff_14
            df.loc[idx, 'run_diff_14_matchup'] = home_run_diff_14 - away_run_diff_14
            
            # 30-day run differentials  
            home_run_diff_30 = self._safe_get_metric(home_metrics, 'run_diff_30', home_run_diff)
            away_run_diff_30 = self._safe_get_metric(away_metrics, 'run_diff_30', away_run_diff)
            
            df.loc[idx, 'home_run_diff_30'] = home_run_diff_30
            df.loc[idx, 'away_run_diff_30'] = away_run_diff_30
            df.loc[idx, 'run_diff_30_matchup'] = home_run_diff_30 - away_run_diff_30
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding run differential features: {str(e)}")
            return df
    
    def _add_pitching_stat_features(self, df: pd.DataFrame, idx: int,
                                  home_metrics: pd.DataFrame, away_metrics: pd.DataFrame) -> pd.DataFrame:
        """Add top pitching statistics (11 of top 19 correlations)"""
        try:
            # ERA (R² = 0.790)
            home_era = self._safe_get_metric(home_metrics, 'era', 4.5)
            away_era = self._safe_get_metric(away_metrics, 'era', 4.5)
            df.loc[idx, 'home_era'] = home_era
            df.loc[idx, 'away_era'] = away_era
            df.loc[idx, 'era_matchup_diff'] = away_era - home_era
            
            # FIP (R² = 0.770)
            home_fip = self._safe_get_metric(home_metrics, 'fip', 4.5)
            away_fip = self._safe_get_metric(away_metrics, 'fip', 4.5)
            df.loc[idx, 'home_fip'] = home_fip
            df.loc[idx, 'away_fip'] = away_fip
            df.loc[idx, 'fip_matchup_diff'] = away_fip - home_fip
            
            # WHIP (R² = 0.691)
            home_whip = self._safe_get_metric(home_metrics, 'whip', 1.3)
            away_whip = self._safe_get_metric(away_metrics, 'whip', 1.3)
            df.loc[idx, 'home_whip'] = home_whip
            df.loc[idx, 'away_whip'] = away_whip
            df.loc[idx, 'whip_matchup_diff'] = away_whip - home_whip
            
            # H/9 (R² = 0.676)
            home_h9 = self._safe_get_metric(home_metrics, 'h_per_9', 9.0)
            away_h9 = self._safe_get_metric(away_metrics, 'h_per_9', 9.0)
            df.loc[idx, 'home_h_per_9'] = home_h9
            df.loc[idx, 'away_h_per_9'] = away_h9
            df.loc[idx, 'h9_matchup_diff'] = away_h9 - home_h9
            
            # BAA - Batting Average Against (R² = 0.663)
            home_baa = self._safe_get_metric(home_metrics, 'batting_avg_against', 0.250)
            away_baa = self._safe_get_metric(away_metrics, 'batting_avg_against', 0.250)
            df.loc[idx, 'home_baa'] = home_baa
            df.loc[idx, 'away_baa'] = away_baa
            df.loc[idx, 'baa_matchup_diff'] = away_baa - home_baa
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding pitching stat features: {str(e)}")
            return df
    
    def _add_hitting_stat_features(self, df: pd.DataFrame, idx: int,
                                 home_metrics: pd.DataFrame, away_metrics: pd.DataFrame) -> pd.DataFrame:
        """Add top hitting statistics"""
        try:
            # OBP (On-base percentage)
            home_obp = self._safe_get_metric(home_metrics, 'obp', 0.320)
            away_obp = self._safe_get_metric(away_metrics, 'obp', 0.320)
            df.loc[idx, 'home_obp'] = home_obp
            df.loc[idx, 'away_obp'] = away_obp
            df.loc[idx, 'obp_matchup_diff'] = home_obp - away_obp
            
            # SLG (Slugging percentage)
            home_slg = self._safe_get_metric(home_metrics, 'slg', 0.400)
            away_slg = self._safe_get_metric(away_metrics, 'slg', 0.400)
            df.loc[idx, 'home_slg'] = home_slg
            df.loc[idx, 'away_slg'] = away_slg
            df.loc[idx, 'slg_matchup_diff'] = home_slg - away_slg
            
            # OPS (On-base Plus Slugging)
            home_ops = self._safe_get_metric(home_metrics, 'ops', 0.720)
            away_ops = self._safe_get_metric(away_metrics, 'ops', 0.720)
            df.loc[idx, 'home_ops'] = home_ops
            df.loc[idx, 'away_ops'] = away_ops
            df.loc[idx, 'ops_matchup_diff'] = home_ops - away_ops
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding hitting stat features: {str(e)}")
            return df
    
    def _add_composite_indices(self, df: pd.DataFrame, idx: int,
                             home_metrics: pd.DataFrame, away_metrics: pd.DataFrame) -> pd.DataFrame:
        """Add composite strength indices weighted by correlation values"""
        try:
            # Calculate pitching strength index (weighted by correlations)
            home_pitching_stats = {
                'era': df.loc[idx, 'home_era'] if 'home_era' in df.columns else 4.5,
                'fip': df.loc[idx, 'home_fip'] if 'home_fip' in df.columns else 4.5,
                'whip': df.loc[idx, 'home_whip'] if 'home_whip' in df.columns else 1.3,
                'h_per_9': df.loc[idx, 'home_h_per_9'] if 'home_h_per_9' in df.columns else 9.0,
                'baa': df.loc[idx, 'home_baa'] if 'home_baa' in df.columns else 0.250
            }
            
            away_pitching_stats = {
                'era': df.loc[idx, 'away_era'] if 'away_era' in df.columns else 4.5,
                'fip': df.loc[idx, 'away_fip'] if 'away_fip' in df.columns else 4.5,
                'whip': df.loc[idx, 'away_whip'] if 'away_whip' in df.columns else 1.3,
                'h_per_9': df.loc[idx, 'away_h_per_9'] if 'away_h_per_9' in df.columns else 9.0,
                'baa': df.loc[idx, 'away_baa'] if 'away_baa' in df.columns else 0.250
            }
            
            # Calculate weighted pitching indices (normalize and invert for "lower is better" stats)
            home_pitching_index = self._calculate_pitching_index(home_pitching_stats)
            away_pitching_index = self._calculate_pitching_index(away_pitching_stats)
            
            df.loc[idx, 'home_pitching_index'] = home_pitching_index
            df.loc[idx, 'away_pitching_index'] = away_pitching_index
            df.loc[idx, 'pitching_matchup_diff'] = home_pitching_index - away_pitching_index
            
            # Calculate hitting strength index
            home_hitting_stats = {
                'obp': df.loc[idx, 'home_obp'] if 'home_obp' in df.columns else 0.320,
                'slg': df.loc[idx, 'home_slg'] if 'home_slg' in df.columns else 0.400,
                'ops': df.loc[idx, 'home_ops'] if 'home_ops' in df.columns else 0.720
            }
            
            away_hitting_stats = {
                'obp': df.loc[idx, 'away_obp'] if 'away_obp' in df.columns else 0.320,
                'slg': df.loc[idx, 'away_slg'] if 'away_slg' in df.columns else 0.400,
                'ops': df.loc[idx, 'away_ops'] if 'away_ops' in df.columns else 0.720
            }
            
            home_hitting_index = self._calculate_hitting_index(home_hitting_stats)
            away_hitting_index = self._calculate_hitting_index(away_hitting_stats)
            
            df.loc[idx, 'home_hitting_index'] = home_hitting_index
            df.loc[idx, 'away_hitting_index'] = away_hitting_index
            df.loc[idx, 'hitting_matchup_diff'] = home_hitting_index - away_hitting_index
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding composite indices: {str(e)}")
            return df
    
    def _calculate_pitching_index(self, stats: Dict[str, float]) -> float:
        """Calculate weighted pitching strength index"""
        try:
            # Normalize weights
            total_weight = sum(self.PITCHING_WEIGHTS.values())
            normalized_weights = {k: v/total_weight for k, v in self.PITCHING_WEIGHTS.items()}
            
            # Calculate weighted index (invert for lower-is-better stats)
            index = 0
            for stat, weight in normalized_weights.items():
                if stat in stats:
                    # Invert ERA, FIP, WHIP, H/9, BAA (lower is better)
                    normalized_stat = 1.0 / (1.0 + stats[stat]) if stats[stat] > 0 else 0
                    index += weight * normalized_stat
            
            return index
            
        except Exception as e:
            self.logger.error(f"Error calculating pitching index: {str(e)}")
            return 0.5
    
    def _calculate_hitting_index(self, stats: Dict[str, float]) -> float:
        """Calculate weighted hitting strength index"""
        try:
            # Normalize weights
            total_weight = sum(self.HITTING_WEIGHTS.values())
            normalized_weights = {k: v/total_weight for k, v in self.HITTING_WEIGHTS.items()}
            
            # Calculate weighted index (higher is better for hitting stats)
            index = 0
            for stat, weight in normalized_weights.items():
                if stat in stats:
                    # Normalize hitting stats to 0-1 scale
                    normalized_stat = min(stats[stat] / 0.5, 1.0) if stat == 'ops' else min(stats[stat] / 0.4, 1.0)
                    index += weight * normalized_stat
            
            return index
            
        except Exception as e:
            self.logger.error(f"Error calculating hitting index: {str(e)}")
            return 0.5
    
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
            
            # Game context features (only if statcast data is available)
            if 'total_pitches' in df.columns:
                df['pitches_per_inning'] = df['total_pitches'] / 9
                df['game_pace'] = np.where(df['total_pitches'] > df['total_pitches'].median(), 1, 0)
            else:
                # Default values for schedule data
                df['pitches_per_inning'] = 17.0  # Average pitches per inning
                df['game_pace'] = 0  # Normal pace
            
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
            
            # Team batting trends (only if statcast data exists)
            for team_col in ['home_team', 'away_team']:
                if team_col in df.columns:
                    prefix = team_col.split('_')[0]
                    
                    # Rolling batting stats (only if source data exists)
                    if 'avg_exit_velocity' in df.columns:
                        df[f'{prefix}_exit_velocity_avg_5'] = df.groupby(team_col)['avg_exit_velocity'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
                    else:
                        # Default batting performance for schedule data
                        df[f'{prefix}_exit_velocity_avg_5'] = 89.0  # League average exit velocity
                    
                    if 'quality_contact_rate' in df.columns:
                        df[f'{prefix}_quality_contact_avg_5'] = df.groupby(team_col)['quality_contact_rate'].transform(
                            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
                        )
                    else:
                        # Default quality contact rate
                        df[f'{prefix}_quality_contact_avg_5'] = 0.25  # Average quality contact rate
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding batting features: {str(e)}")
            return df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational and contextual features"""
        try:
            # Only add features if the required columns exist
            if df.empty:
                return df
                
            # Home field advantage proxy (only if we have game scores)
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df['score_differential'] = df['home_score'] - df['away_score']
            else:
                # Default home field advantage for schedule data
                df['score_differential'] = 0.5  # Slight home advantage
            
            # Rest days calculation (only if we have teams and dates)
            if 'game_date' in df.columns and 'home_team' in df.columns and 'away_team' in df.columns:
                df = df.copy()  # Ensure we're working with a copy
                
                # Convert game_date to datetime if it's not already
                df['game_date'] = pd.to_datetime(df['game_date'])
                
                # Calculate rest days with proper error handling
                try:
                    df = df.sort_values(['home_team', 'game_date'])
                    df['home_rest_days'] = df.groupby('home_team')['game_date'].diff().dt.days
                    df['home_rest_days'] = df['home_rest_days'].fillna(3).astype(float)  # Default rest
                    
                    df = df.sort_values(['away_team', 'game_date'])
                    df['away_rest_days'] = df.groupby('away_team')['game_date'].diff().dt.days
                    df['away_rest_days'] = df['away_rest_days'].fillna(3).astype(float)
                    
                    # Rest advantage
                    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
                except:
                    # Fallback if groupby fails
                    df['home_rest_days'] = 3.0
                    df['away_rest_days'] = 3.0
                    df['rest_advantage'] = 0.0
            else:
                # Default values for schedule data without historical context
                df['home_rest_days'] = 3.0
                df['away_rest_days'] = 3.0
                df['rest_advantage'] = 0.0
            
            # Season timing features (only if day_of_year exists)
            if 'day_of_year' in df.columns:
                df['early_season'] = (df['day_of_year'] < 120).astype(int)  # Before May
                df['late_season'] = (df['day_of_year'] > 240).astype(int)   # After August
            else:
                # Default mid-season values
                df['early_season'] = 0
                df['late_season'] = 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding situational features: {str(e)}")
            # Return original DataFrame with basic default features
            try:
                df['score_differential'] = 0.5
                df['home_rest_days'] = 3.0
                df['away_rest_days'] = 3.0
                df['rest_advantage'] = 0.0
                df['early_season'] = 0
                df['late_season'] = 0
            except:
                pass
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
                    features.loc[:, numeric_columns] = pd.DataFrame(
                        self.imputers['numeric'].fit_transform(features[numeric_columns]),
                        index=features.index,
                        columns=numeric_columns
                    )
                else:
                    features.loc[:, numeric_columns] = pd.DataFrame(
                        self.imputers['numeric'].transform(features[numeric_columns]),
                        index=features.index,
                        columns=numeric_columns
                    )
            
            # Impute categorical columns
            if len(categorical_columns) > 0:
                if 'categorical' not in self.imputers:
                    self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                    features.loc[:, categorical_columns] = pd.DataFrame(
                        self.imputers['categorical'].fit_transform(features[categorical_columns]),
                        index=features.index,
                        columns=categorical_columns
                    )
                else:
                    features.loc[:, categorical_columns] = pd.DataFrame(
                        self.imputers['categorical'].transform(features[categorical_columns]),
                        index=features.index,
                        columns=categorical_columns
                    )
            
            # Scale features
            if 'features' not in self.scalers:
                self.scalers['features'] = StandardScaler()
                scaled_features = pd.DataFrame(
                    self.scalers['features'].fit_transform(features[numeric_columns]),
                    index=features.index,
                    columns=numeric_columns
                )
            else:
                scaled_features = pd.DataFrame(
                    self.scalers['features'].transform(features[numeric_columns]),
                    index=features.index,
                    columns=numeric_columns
                )
            
            # Create final feature DataFrame
            final_features = features.copy()
            if len(numeric_columns) > 0:
                final_features.loc[:, numeric_columns] = scaled_features
            
            self.logger.info(f"Prepared {len(final_features.columns)} features for training")
            
            # Ensure we return proper DataFrames
            return pd.DataFrame(final_features), pd.DataFrame(targets)
            
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
