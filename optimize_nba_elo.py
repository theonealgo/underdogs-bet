#!/usr/bin/env python3
"""
Optimize NBA Elo K-factor to improve prediction accuracy
Tests different K-factors on historical 2024-25 data
"""

import sqlite3
import pandas as pd
import numpy as np
from src.models.universal_ensemble_predictor import UniversalEloRatingSystem

DATABASE = 'sports_predictions_original.db'

def load_nba_training_data():
    """Load 2025-26 NBA season COMPLETED games for testing"""
    conn = sqlite3.connect(DATABASE)
    
    query = """
        SELECT 
            game_id,
            game_date,
            home_team_id as home_team,
            away_team_id as away_team,
            home_score,
            away_score,
            CASE 
                WHEN home_score > away_score THEN 1
                ELSE 0
            END as home_won
        FROM games
        WHERE sport = 'NBA' 
        AND season = 2025
        AND home_score IS NOT NULL
        ORDER BY game_date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} completed games from 2025-26 NBA season")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    return df

def test_elo_k_factor(df, k_factor, home_advantage=0):
    """Test Elo performance with given K-factor"""
    # Split into train (80%) and validation (20%)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()
    
    # Initialize Elo system
    elo_system = UniversalEloRatingSystem(sport='NBA', k_factor=k_factor)
    
    # Train on first 80% of games
    for idx, row in train_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_won = row['home_won'] == 1
        
        # Update ratings
        elo_system.update_ratings(home_team, away_team, home_won)
    
    # Test on remaining 20%
    correct = 0
    total = len(val_df)
    
    for idx, row in val_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        actual_home_won = row['home_won'] == 1
        
        # Get prediction
        home_prob = elo_system.predict_game(home_team, away_team)
        predicted_home_win = home_prob > 0.5
        
        if predicted_home_win == actual_home_won:
            correct += 1
        
        # Update ratings for next prediction
        elo_system.update_ratings(home_team, away_team, actual_home_won)
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

def main():
    print("=" * 70)
    print("NBA ELO K-FACTOR OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    df = load_nba_training_data()
    
    # Test different K-factors
    k_factors_to_test = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 35]
    
    print("\nTesting K-factors:")
    print("-" * 50)
    
    results = []
    for k in k_factors_to_test:
        accuracy, correct, total = test_elo_k_factor(df, k)
        results.append((k, accuracy, correct, total))
        print(f"K={k:2d}: {correct}/{total} = {accuracy:.2f}%")
    
    # Find best K-factor
    best_k, best_acc, best_correct, best_total = max(results, key=lambda x: x[1])
    
    print("\n" + "=" * 70)
    print(f"BEST K-FACTOR: {best_k}")
    print(f"Validation Accuracy: {best_correct}/{best_total} = {best_acc:.2f}%")
    print("=" * 70)
    
    # Compare to current K=18
    current_result = [r for r in results if r[0] == 18][0]
    print(f"\nCurrent K=18: {current_result[2]}/{current_result[3]} = {current_result[1]:.2f}%")
    print(f"Best K={best_k}: {best_correct}/{best_total} = {best_acc:.2f}%")
    print(f"Improvement: +{best_acc - current_result[1]:.2f}%")

if __name__ == "__main__":
    main()
