"""
Universal Sports CSV Predictor
Works with ANY sport: MLB, NFL, NBA, NHL, NCAA Football, NCAA Basketball, WNBA

Usage:
    python universal_sports_predictor.py <sport> <csv_file>

Examples:
    python universal_sports_predictor.py MLB mlb_schedule.csv
    python universal_sports_predictor.py NBA nba_schedule.csv
    python universal_sports_predictor.py NHL nhl_schedule.csv
"""

import sys
import pandas as pd
import numpy as np
from src.models.universal_ensemble_predictor import UniversalSportsEnsemble
from src.data_storage.database import DatabaseManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect column names from CSV
    
    Looks for common column names like:
    - Home/Away Team columns
    - Date columns
    - Result/Score columns
    """
    col_lower = {col.lower(): col for col in df.columns}
    
    mapping = {}
    
    # Detect home team (partial match for flexibility)
    for col_name_lower, col_name in col_lower.items():
        if 'home' in col_name_lower and 'home_team' not in mapping:
            mapping['home_team'] = col_name
            break
    
    # Detect away team (partial match for flexibility)
    for col_name_lower, col_name in col_lower.items():
        if any(pattern in col_name_lower for pattern in ['away', 'visitor', 'visiting']) and 'away_team' not in mapping:
            mapping['away_team'] = col_name
            break
    
    # Detect date
    for pattern in ['date', 'game date', 'game_date', 'gamedate']:
        if pattern in col_lower:
            mapping['date'] = col_lower[pattern]
            break
    
    # Detect result
    for pattern in ['result', 'score', 'final', 'final score', 'outcome']:
        if pattern in col_lower:
            mapping['result'] = col_lower[pattern]
            break
    
    # Detect match/game ID
    for pattern in ['match number', 'match_id', 'game_id', 'gameid', 'id', 'match']:
        if pattern in col_lower:
            mapping['match_id'] = col_lower[pattern]
            break
    
    return mapping


def parse_result(score_str: str) -> str:
    """
    Parse result from various formats to H/A/D
    
    Handles:
    - "24 - 20" (home win if first > second)
    - "H" / "A" / "D"
    - "Home" / "Away" / "Draw"
    - "1" / "2" / "0"
    """
    if not score_str or pd.isna(score_str):
        return ''
    
    score_str = str(score_str).strip()
    
    # Score format (e.g., "24 - 20")
    if '-' in score_str:
        try:
            parts = score_str.split('-')
            if len(parts) == 2:
                home_score = int(parts[0].strip())
                away_score = int(parts[1].strip())
                
                if home_score > away_score:
                    return 'H'
                elif away_score > home_score:
                    return 'A'
                else:
                    return 'D'
        except:
            pass
    
    # Text format
    score_upper = score_str.upper()
    if score_upper in ['HOME', 'H', '1', 'W']:
        return 'H'
    elif score_upper in ['AWAY', 'A', '2', 'L']:
        return 'A'
    elif score_upper in ['DRAW', 'D', 'TIE', '0', 'T']:
        return 'D'
    
    return ''


def load_csv(csv_path: str, sport: str) -> pd.DataFrame:
    """Load and preprocess CSV for any sport"""
    logger.info(f"Loading {sport} CSV from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    
    # Auto-detect columns
    col_mapping = detect_columns(df)
    logger.info(f"Detected columns: {col_mapping}")
    
    # Rename to standard format
    if 'home_team' in col_mapping:
        df = df.rename(columns={col_mapping['home_team']: 'home_team'})
    if 'away_team' in col_mapping:
        df = df.rename(columns={col_mapping['away_team']: 'away_team'})
    if 'result' in col_mapping:
        df = df.rename(columns={col_mapping['result']: 'result'})
    if 'date' in col_mapping:
        df = df.rename(columns={col_mapping['date']: 'date'})
    
    # Clean team names
    if 'home_team' in df.columns:
        df['home_team'] = df['home_team'].str.strip()
    if 'away_team' in df.columns:
        df['away_team'] = df['away_team'].str.strip()
    
    # Parse results
    if 'result' in df.columns:
        df['result'] = df['result'].apply(parse_result)
        result_counts = df['result'].value_counts()
        logger.info(f"Result distribution:\n{result_counts}")
    
    # Add sport column
    df['sport'] = sport
    
    return df


def train_and_predict(sport: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Train ensemble models and generate predictions
    
    Args:
        sport: Sport code (MLB, NFL, NBA, etc.)
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with predictions
    """
    # Split into train/test (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Split: {len(train_df)} training, {len(test_df)} test games")
    
    # Initialize predictor
    predictor = UniversalSportsEnsemble(sport=sport, k_factor=20)
    
    # Train on games with results
    train_with_results = train_df[train_df['result'].isin(['H', 'A'])].copy()
    
    if len(train_with_results) > 0:
        logger.info(f"Training on {len(train_with_results)} games with results...")
        training_results = predictor.train(train_with_results)
        logger.info(f"Training results: {training_results}")
    else:
        logger.warning("No training data with results")
    
    # Generate predictions for all games
    logger.info(f"Generating predictions for {len(df)} games...")
    predictions = predictor.predict_multiple_games(df)
    
    # Merge predictions with original data using index to avoid duplicate matchup explosion
    pred_df = pd.DataFrame(predictions)
    pred_df.index = df.index  # Align indices
    
    # Add prediction columns directly to avoid merge issues with duplicate matchups
    result_df = df.copy()
    result_df['elo_home_prob'] = pred_df['elo_home_prob']
    result_df['glmnet_home_prob'] = pred_df['glmnet_home_prob']
    result_df['xgboost_home_prob'] = pred_df['xgboost_home_prob']
    result_df['blended_home_prob'] = pred_df['blended_home_prob']
    result_df['predicted_winner'] = pred_df['predicted_winner']
    result_df['confidence'] = pred_df['confidence']
    
    # Evaluate on test set
    test_results = result_df[result_df.index.isin(test_df.index)].copy()
    test_with_results = test_results[test_results['result'].isin(['H', 'A'])].copy()
    
    if len(test_with_results) > 0:
        # Calculate accuracy
        test_with_results['correct'] = (
            ((test_with_results['blended_home_prob'] > 0.5) & (test_with_results['result'] == 'H')) |
            ((test_with_results['blended_home_prob'] <= 0.5) & (test_with_results['result'] == 'A'))
        )
        
        accuracy = test_with_results['correct'].mean()
        logger.info(f"\n{sport} Test Set Accuracy: {accuracy:.1%} ({len(test_with_results)} games)")
    
    # Save model
    model_path = f'models/{sport.lower()}_ensemble.pkl'
    predictor.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return result_df


def save_to_database(df: pd.DataFrame, sport: str):
    """Save predictions to database"""
    try:
        db = DatabaseManager()
        
        # Prepare games and predictions data together with consistent IDs
        games_data = []
        pred_data = []
        
        for idx, row in df.iterrows():
            # Create consistent game_id using match_id or row index
            match_id = row.get('match_id', idx)
            game_id = f"{sport}_{match_id}"
            
            # Parse date - use today's date as fallback
            game_date = row.get('date', '')
            if not game_date or pd.isna(game_date) or game_date == '':
                # Use today's date as fallback to avoid invalid dates
                from datetime import datetime
                game_date = datetime.now().strftime('%Y-%m-%d')
            
            # Store game data
            game = {
                'sport': sport,
                'league': sport,
                'game_id': game_id,
                'game_date': game_date,
                'home_team_id': row['home_team'],
                'away_team_id': row['away_team'],
                'status': 'Final' if row.get('result') in ['H', 'A', 'D'] else 'Scheduled'
            }
            games_data.append(game)
            
            # Store prediction data with same game_id
            if 'blended_home_prob' in row:
                pred = {
                    'sport': sport,
                    'league': sport,
                    'game_id': game_id,  # Use same game_id as games table
                    'game_date': game_date,
                    'home_team_id': row['home_team'],
                    'away_team_id': row['away_team'],
                    'predicted_winner': row['predicted_winner'],
                    'win_probability': float(row['blended_home_prob']),
                    'model_version': 'ensemble_v1'
                }
                pred_data.append(pred)
        
        # Store games
        if games_data:
            games_df = pd.DataFrame(games_data)
            db.store_games(games_df)
            logger.info(f"Stored {len(games_df)} games in database")
        
        # Store predictions
        if pred_data:
            db.store_predictions(pred_data)
            logger.info(f"Stored {len(pred_data)} predictions in database")
        
    except Exception as e:
        logger.error(f"Error saving to database: {e}")


def main():
    """Main execution"""
    if len(sys.argv) < 3:
        print("Usage: python universal_sports_predictor.py <sport> <csv_file>")
        print("\nExamples:")
        print("  python universal_sports_predictor.py MLB mlb_schedule.csv")
        print("  python universal_sports_predictor.py NBA nba_schedule.csv")
        print("  python universal_sports_predictor.py NHL nhl_schedule.csv")
        print("\nSupported sports: MLB, NFL, NBA, NHL, NCAAF, NCAAB, WNBA")
        sys.exit(1)
    
    sport = sys.argv[1].upper()
    csv_path = sys.argv[2]
    
    logger.info("=" * 70)
    logger.info(f"UNIVERSAL SPORTS PREDICTOR - {sport}")
    logger.info("=" * 70)
    
    # Load CSV
    df = load_csv(csv_path, sport)
    
    # Train and predict
    result_df = train_and_predict(sport, df)
    
    # Save results
    output_file = f'{sport.lower()}_predictions.csv'
    result_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Save to database
    save_to_database(result_df, sport)
    
    # Show sample predictions
    logger.info("\nSample Predictions:")
    sample_cols = ['away_team', 'home_team', 'blended_home_prob', 'predicted_winner']
    sample_cols = [col for col in sample_cols if col in result_df.columns]
    print(result_df[sample_cols].head(10).to_string(index=False))
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
