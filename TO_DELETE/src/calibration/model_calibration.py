#!/usr/bin/env python3
"""
Model Calibration System
Analyzes predicted probabilities vs actual outcomes to create reliability curves
and identify calibration gaps for dynamic weight adjustment
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCalibrator:
    """Calibrates model predictions based on historical performance"""
    
    def __init__(self, db_path: str = 'sports_predictions.db'):
        self.db_path = db_path
        self.calibration_data = {}
        self.team_bias_factors = {}
        
    def load_historical_predictions(self, sport: str = 'NFL') -> pd.DataFrame:
        """Load historical predictions with actual results"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            game_date,
            home_team_id,
            away_team_id,
            predicted_winner,
            win_probability,
            actual_winner,
            actual_home_score,
            actual_away_score,
            key_factors
        FROM predictions
        WHERE sport = ? AND actual_winner IS NOT NULL
        ORDER BY game_date
        """
        
        df = pd.DataFrame(conn.execute(query, (sport,)).fetchall(),
                         columns=['date', 'home_team', 'away_team', 'predicted_winner', 
                                 'win_probability', 'actual_winner', 'home_score', 'away_score', 'key_factors'])
        
        conn.close()
        
        # Parse key_factors JSON to extract individual model probabilities
        df['xgb_prob'] = df['key_factors'].apply(lambda x: self._extract_prob(x, 'xgb'))
        df['elo_prob'] = df['key_factors'].apply(lambda x: self._extract_prob(x, 'elo'))
        
        # Calculate prediction correctness
        df['correct'] = (df['predicted_winner'] == df['actual_winner']).astype(int)
        
        # Calculate home win
        df['home_won'] = (df['home_score'] > df['away_score']).astype(int)
        
        logger.info(f"Loaded {len(df)} historical predictions for {sport}")
        return df
    
    def _extract_prob(self, json_str: str, model: str) -> float:
        """Extract probability from JSON key_factors"""
        try:
            if not json_str:
                return None
            data = json.loads(json_str)
            if model == 'xgb':
                return data.get('xgboost_home_prob', None)
            elif model == 'elo':
                return data.get('elo_home_prob', None)
        except:
            return None
        return None
    
    def calculate_reliability_curve(self, df: pd.DataFrame, model: str = 'ensemble', bins: int = 10) -> Dict:
        """
        Calculate reliability curve: predicted probability vs actual win rate
        
        Args:
            df: DataFrame with predictions
            model: 'ensemble', 'xgb', or 'elo'
            bins: Number of probability bins
            
        Returns:
            Dict with bin centers, actual rates, counts, and calibration error
        """
        if model == 'ensemble':
            pred_probs = df['win_probability'].values
        elif model == 'xgb':
            pred_probs = df['xgb_prob'].dropna().values
            df = df[df['xgb_prob'].notna()].copy()
        elif model == 'elo':
            pred_probs = df['elo_prob'].dropna().values
            df = df[df['elo_prob'].notna()].copy()
        
        # Create probability bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Assign each prediction to a bin
        digitized = np.digitize(pred_probs, bin_edges) - 1
        digitized = np.clip(digitized, 0, bins - 1)
        
        # Calculate actual win rate in each bin
        bin_actual_rates = []
        bin_counts = []
        bin_predicted_rates = []
        
        for i in range(bins):
            mask = digitized == i
            count = np.sum(mask)
            bin_counts.append(count)
            
            if count > 0:
                # Actual win rate in this bin
                actual_rate = df[mask]['home_won'].mean()
                bin_actual_rates.append(actual_rate)
                
                # Average predicted probability in this bin
                predicted_rate = pred_probs[mask].mean()
                bin_predicted_rates.append(predicted_rate)
            else:
                bin_actual_rates.append(np.nan)
                bin_predicted_rates.append(np.nan)
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(pred_probs)
        for i in range(bins):
            if bin_counts[i] > 0 and not np.isnan(bin_actual_rates[i]):
                ece += (bin_counts[i] / total_samples) * abs(bin_predicted_rates[i] - bin_actual_rates[i])
        
        return {
            'model': model,
            'bin_centers': bin_centers.tolist(),
            'predicted_rates': bin_predicted_rates,
            'actual_rates': bin_actual_rates,
            'bin_counts': bin_counts,
            'ece': ece,
            'total_samples': total_samples
        }
    
    def calculate_team_accuracy(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate per-team prediction accuracy"""
        team_stats = {}
        
        # Analyze each team (both home and away)
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        
        for team in all_teams:
            # Games involving this team
            team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            
            if len(team_games) == 0:
                continue
            
            # Calculate accuracy when team is predicted to win
            predicted_wins = team_games[team_games['predicted_winner'] == team]
            actual_wins = team_games[team_games['actual_winner'] == team]
            
            accuracy = len(predicted_wins[predicted_wins['correct'] == 1]) / len(predicted_wins) if len(predicted_wins) > 0 else 0.5
            win_rate = len(actual_wins) / len(team_games)
            
            # Calculate bias: difference between predicted win rate and actual win rate
            predicted_win_rate = len(predicted_wins) / len(team_games)
            bias = predicted_win_rate - win_rate
            
            team_stats[team] = {
                'games': len(team_games),
                'accuracy': accuracy,
                'actual_win_rate': win_rate,
                'predicted_win_rate': predicted_win_rate,
                'bias': bias,
                'correction_factor': -bias * 0.5  # Nudge predictions opposite to bias
            }
        
        return team_stats
    
    def calculate_dynamic_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate dynamic ensemble weights based on calibration performance
        
        Returns:
            Dict with weights for XGB and Elo based on their calibration
        """
        # Get reliability curves for each model
        xgb_cal = self.calculate_reliability_curve(df, model='xgb', bins=10)
        elo_cal = self.calculate_reliability_curve(df, model='elo', bins=10)
        
        # Models with lower ECE (better calibration) get higher weight
        xgb_ece = xgb_cal['ece']
        elo_ece = elo_cal['ece']
        
        # Inverse ECE for weighting (add small epsilon to avoid division by zero)
        xgb_inv = 1 / (xgb_ece + 0.01)
        elo_inv = 1 / (elo_ece + 0.01)
        
        # Normalize to sum to 1
        total = xgb_inv + elo_inv
        xgb_weight = xgb_inv / total
        elo_weight = elo_inv / total
        
        # Also consider raw accuracy
        xgb_acc = df['xgb_prob'].notna().sum()
        elo_acc = df['elo_prob'].notna().sum()
        
        if xgb_acc > 0 and elo_acc > 0:
            # XGB predictions closer to 0.5 or 1.0 based on actual outcome
            xgb_correct_prob = df[df['xgb_prob'].notna()].apply(
                lambda row: row['xgb_prob'] if row['home_won'] == 1 else 1 - row['xgb_prob'], axis=1
            ).mean()
            
            elo_correct_prob = df[df['elo_prob'].notna()].apply(
                lambda row: row['elo_prob'] if row['home_won'] == 1 else 1 - row['elo_prob'], axis=1
            ).mean()
            
            # Blend calibration-based weights with accuracy-based weights
            accuracy_xgb = xgb_correct_prob / (xgb_correct_prob + elo_correct_prob)
            accuracy_elo = elo_correct_prob / (xgb_correct_prob + elo_correct_prob)
            
            # Final weights: 60% calibration-based, 40% accuracy-based
            final_xgb = 0.6 * xgb_weight + 0.4 * accuracy_xgb
            final_elo = 0.6 * elo_weight + 0.4 * accuracy_elo
            
            # Normalize
            total_final = final_xgb + final_elo
            final_xgb /= total_final
            final_elo /= total_final
        else:
            final_xgb = xgb_weight
            final_elo = elo_weight
        
        return {
            'xgb': float(final_xgb),
            'elo': float(final_elo),
            'xgb_ece': float(xgb_ece),
            'elo_ece': float(elo_ece),
            'xgb_calibration_weight': float(xgb_weight),
            'elo_calibration_weight': float(elo_weight)
        }
    
    def generate_calibration_report(self, sport: str = 'NFL') -> Dict:
        """Generate comprehensive calibration report"""
        df = self.load_historical_predictions(sport)
        
        if len(df) == 0:
            logger.warning(f"No historical data for {sport}")
            return {}
        
        # Calculate reliability curves
        ensemble_cal = self.calculate_reliability_curve(df, model='ensemble')
        xgb_cal = self.calculate_reliability_curve(df, model='xgb')
        elo_cal = self.calculate_reliability_curve(df, model='elo')
        
        # Calculate team-level accuracy
        team_stats = self.calculate_team_accuracy(df)
        
        # Calculate dynamic weights
        dynamic_weights = self.calculate_dynamic_weights(df)
        
        # Overall accuracy
        overall_accuracy = df['correct'].mean()
        
        report = {
            'sport': sport,
            'total_games': len(df),
            'overall_accuracy': float(overall_accuracy),
            'reliability_curves': {
                'ensemble': ensemble_cal,
                'xgb': xgb_cal,
                'elo': elo_cal
            },
            'team_stats': team_stats,
            'dynamic_weights': dynamic_weights,
            'recommendations': self._generate_recommendations(ensemble_cal, xgb_cal, elo_cal, dynamic_weights)
        }
        
        return report
    
    def _generate_recommendations(self, ensemble_cal, xgb_cal, elo_cal, weights) -> List[str]:
        """Generate actionable recommendations based on calibration analysis"""
        recs = []
        
        # Check ECE
        if ensemble_cal['ece'] > 0.1:
            recs.append(f"⚠️ High calibration error (ECE={ensemble_cal['ece']:.3f}). Model is poorly calibrated.")
        
        if xgb_cal['ece'] < elo_cal['ece']:
            recs.append(f"✅ XGBoost is better calibrated (ECE={xgb_cal['ece']:.3f}) than Elo (ECE={elo_cal['ece']:.3f})")
        else:
            recs.append(f"✅ Elo is better calibrated (ECE={elo_cal['ece']:.3f}) than XGBoost (ECE={xgb_cal['ece']:.3f})")
        
        # Recommend dynamic weights
        recs.append(f"📊 Suggested weights based on calibration: XGB={weights['xgb']:.1%}, Elo={weights['elo']:.1%}")
        
        return recs
    
    def save_calibration_data(self, report: Dict, filepath: str = 'calibration_report.json'):
        """Save calibration report to file"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Calibration report saved to {filepath}")


if __name__ == '__main__':
    calibrator = ModelCalibrator()
    report = calibrator.generate_calibration_report('NFL')
    
    print("\n" + "="*80)
    print("📊 MODEL CALIBRATION REPORT - NFL")
    print("="*80)
    print(f"\nTotal Games Analyzed: {report['total_games']}")
    print(f"Overall Accuracy: {report['overall_accuracy']:.1%}")
    
    print("\n📈 CALIBRATION METRICS (Expected Calibration Error):")
    print(f"  Ensemble ECE: {report['reliability_curves']['ensemble']['ece']:.3f}")
    print(f"  XGBoost ECE:  {report['reliability_curves']['xgb']['ece']:.3f}")
    print(f"  Elo ECE:      {report['reliability_curves']['elo']['ece']:.3f}")
    
    print("\n⚖️ DYNAMIC WEIGHTS (based on calibration):")
    weights = report['dynamic_weights']
    print(f"  XGBoost: {weights['xgb']:.1%}")
    print(f"  Elo:     {weights['elo']:.1%}")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    print("\n🏈 TOP 5 TEAMS BY PREDICTION ACCURACY:")
    team_stats = sorted(report['team_stats'].items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
    for team, stats in team_stats:
        print(f"  {team}: {stats['accuracy']:.1%} ({stats['games']} games)")
    
    print("\n⚠️ TOP 5 TEAMS WITH HIGHEST BIAS (Model Over/Under-predicts):")
    bias_teams = sorted(report['team_stats'].items(), key=lambda x: abs(x[1]['bias']), reverse=True)[:5]
    for team, stats in bias_teams:
        direction = "OVER-predicts" if stats['bias'] > 0 else "UNDER-predicts"
        print(f"  {team}: {direction} by {abs(stats['bias']):.1%} (correction: {stats['correction_factor']:+.3f})")
    
    calibrator.save_calibration_data(report)
