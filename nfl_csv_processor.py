"""
NFL CSV Processor and Prediction System
Processes NFL schedule CSV and generates ensemble predictions

Usage:
    python nfl_csv_processor.py <path_to_csv>

Example:
    python nfl_csv_processor.py nfl-2025-UTC.csv
"""

import sys
import pandas as pd
import numpy as np
from src.models.nfl_ensemble_predictor import NFLEnsemblePredictor
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_preprocess_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and map columns to standard format
    
    Column mapping:
    - Match Number → match_id
    - Round Number → round
    - Date → date
    - Location → venue
    - Home Team → home_team
    - Away Team → away_team
    - Result → result
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading CSV from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Column mapping
    column_mapping = {
        'Match Number': 'match_id',
        'Round Number': 'round',
        'Date': 'date',
        'Location': 'venue',
        'Home Team': 'home_team',
        'Away Team': 'away_team',
        'Result': 'result'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    logger.info(f"Columns after mapping: {df.columns.tolist()}")
    
    # Clean team names (remove extra spaces)
    if 'home_team' in df.columns:
        df['home_team'] = df['home_team'].str.strip()
    if 'away_team' in df.columns:
        df['away_team'] = df['away_team'].str.strip()
    
    # Handle date parsing
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            logger.warning(f"Could not parse dates: {e}")
    
    # Standardize result column
    # Result should be 'H' for home win, 'A' for away win, 'D' for draw
    if 'result' in df.columns:
        # Convert various formats to H/A/D
        df['result'] = df['result'].fillna('').astype(str).str.strip().str.upper()
        
        # Handle different result formats
        df['result'] = df['result'].replace({
            'HOME': 'H',
            'AWAY': 'A', 
            'DRAW': 'D',
            'TIE': 'D',
            '1': 'H',
            '2': 'A',
            '0': 'D'
        })
        
        result_counts = df['result'].value_counts()
        logger.info(f"Result distribution:\n{result_counts}")
    
    return df


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split data into training and test sets
    
    Args:
        df: Full dataset
        test_ratio: Fraction of data to use for testing
        
    Returns:
        train_df, test_df
    """
    # Sort by date if available
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Split data: {len(train_df)} training games, {len(test_df)} test games")
    
    return train_df, test_df


def run_predictions(predictor: NFLEnsemblePredictor, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions for all games
    
    Args:
        predictor: Trained predictor
        games_df: Games to predict
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Generating predictions for {len(games_df)} games...")
    
    predictions = []
    
    for idx, row in games_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get prediction
        pred = predictor.predict_game(home_team, away_team)
        
        # Combine with original row data
        result = row.to_dict()
        result.update(pred)
        predictions.append(result)
    
    return pd.DataFrame(predictions)


def evaluate_predictions(results_df: pd.DataFrame) -> dict:
    """
    Evaluate prediction accuracy
    
    Args:
        results_df: DataFrame with predictions and actual results
        
    Returns:
        Dictionary with evaluation metrics
    """
    if 'result' not in results_df.columns or results_df['result'].isna().all():
        logger.warning("No actual results available for evaluation")
        return {}
    
    # Filter only games with results
    eval_df = results_df[results_df['result'].isin(['H', 'A'])].copy()
    
    if len(eval_df) == 0:
        logger.warning("No games with H/A results for evaluation")
        return {}
    
    logger.info(f"Evaluating on {len(eval_df)} completed games")
    
    # Determine if prediction was correct
    eval_df['elo_correct'] = (
        ((eval_df['elo_home_prob'] > 0.5) & (eval_df['result'] == 'H')) |
        ((eval_df['elo_home_prob'] <= 0.5) & (eval_df['result'] == 'A'))
    )
    
    eval_df['glmnet_correct'] = (
        ((eval_df['glmnet_home_prob'] > 0.5) & (eval_df['result'] == 'H')) |
        ((eval_df['glmnet_home_prob'] <= 0.5) & (eval_df['result'] == 'A'))
    )
    
    eval_df['xgboost_correct'] = (
        ((eval_df['xgboost_home_prob'] > 0.5) & (eval_df['result'] == 'H')) |
        ((eval_df['xgboost_home_prob'] <= 0.5) & (eval_df['result'] == 'A'))
    )
    
    eval_df['blended_correct'] = (
        ((eval_df['blended_home_prob'] > 0.5) & (eval_df['result'] == 'H')) |
        ((eval_df['blended_home_prob'] <= 0.5) & (eval_df['result'] == 'A'))
    )
    
    metrics = {
        'elo_accuracy': eval_df['elo_correct'].mean(),
        'glmnet_accuracy': eval_df['glmnet_correct'].mean(),
        'xgboost_accuracy': eval_df['xgboost_correct'].mean(),
        'blended_accuracy': eval_df['blended_correct'].mean(),
        'total_games': len(eval_df)
    }
    
    return metrics


def main():
    """Main execution function"""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python nfl_csv_processor.py <path_to_csv>")
        print("Example: python nfl_csv_processor.py nfl-2025-UTC.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Step 1: Load and preprocess data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing CSV")
    logger.info("=" * 60)
    df = load_and_preprocess_csv(csv_path)
    
    # Display sample
    logger.info(f"\nSample data:\n{df.head()}")
    
    # Step 2: Split into train/test
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Splitting data into train/test sets")
    logger.info("=" * 60)
    train_df, test_df = split_train_test(df, test_ratio=0.2)
    
    # Step 3: Initialize and train predictor
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Training ensemble models")
    logger.info("=" * 60)
    predictor = NFLEnsemblePredictor()
    
    # Train on historical games with results
    train_with_results = train_df[train_df['result'].isin(['H', 'A', 'D'])].copy()
    
    if len(train_with_results) > 0:
        training_results = predictor.train(train_with_results)
        logger.info(f"\nTraining results: {training_results}")
    else:
        logger.warning("No training data with results available")
    
    # Step 4: Generate predictions for all games
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Generating predictions")
    logger.info("=" * 60)
    
    # Predict on full dataset
    results_df = run_predictions(predictor, df)
    
    # Step 5: Evaluate if test data has results
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Evaluating predictions")
    logger.info("=" * 60)
    
    test_results = results_df[results_df['match_id'].isin(test_df['match_id'])]
    metrics = evaluate_predictions(test_results)
    
    if metrics:
        logger.info("\nModel Performance (Test Set):")
        logger.info(f"  Elo Accuracy:     {metrics['elo_accuracy']:.1%}")
        logger.info(f"  GLMNet Accuracy:  {metrics['glmnet_accuracy']:.1%}")
        logger.info(f"  XGBoost Accuracy: {metrics['xgboost_accuracy']:.1%}")
        logger.info(f"  Blended Accuracy: {metrics['blended_accuracy']:.1%}")
        logger.info(f"  Total Test Games: {metrics['total_games']}")
    
    # Step 6: Save results
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Saving results")
    logger.info("=" * 60)
    
    output_file = 'nfl_predictions_output.csv'
    
    # Select columns for output
    output_columns = [
        'match_id', 'round', 'date', 'venue',
        'away_team', 'home_team', 'result',
        'elo_home_prob', 'glmnet_home_prob', 'xgboost_home_prob',
        'blended_home_prob', 'predicted_winner', 'confidence'
    ]
    
    # Only include columns that exist
    output_columns = [col for col in output_columns if col in results_df.columns]
    
    results_df[output_columns].to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")
    
    # Display sample predictions
    logger.info("\nSample Predictions:")
    sample_cols = ['away_team', 'home_team', 'blended_home_prob', 'predicted_winner']
    sample_cols = [col for col in sample_cols if col in results_df.columns]
    logger.info(f"\n{results_df[sample_cols].head(10).to_string(index=False)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
