#!/usr/bin/env python3
"""
Backtesting Script for jackpotpicks.bet
Evaluates model predictions against actual game results
"""

import sys
sys.path.insert(0, 'models')

import sqlite3
import pandas as pd
from schedules import get_schedule
from import_schedules import calculate_predictions, parse_date

def parse_result_winner(result, home_team, away_team):
    """Determine actual winner from result string"""
    result = result.strip().lower()
    
    if ' - ' in result:
        # Score format: "24 - 20" (home - away)
        scores = result.split('-')
        home_score = int(scores[0].strip())
        away_score = int(scores[1].strip())
        return home_team if home_score > away_score else away_team
    elif 'home win' in result:
        return home_team
    elif 'away win' in result:
        return away_team
    
    return None

def backtest_sport(sport):
    """Backtest a specific sport"""
    print(f"\n{'='*60}")
    print(f"Backtesting {sport}")
    print(f"{'='*60}")
    
    # Get schedule with results
    schedule = get_schedule(sport)
    completed_games = [g for g in schedule if g.get('result')]
    
    if not completed_games:
        print(f"  No completed games for {sport}")
        return None
    
    print(f"  Found {len(completed_games)} completed games")
    
    # Prepare training data and test games
    training_data = []
    test_games = []
    
    for game in completed_games:
        result = game['result'].strip()
        home_score = None
        away_score = None
        
        # Parse result
        if ' - ' in result:
            scores = result.split('-')
            home_score = int(scores[0].strip())
            away_score = int(scores[1].strip())
        elif 'home win' in result.lower():
            home_score = 1
            away_score = 0
        elif 'away win' in result.lower():
            home_score = 0
            away_score = 1
        
        if home_score is not None and away_score is not None:
            training_data.append({
                'Home': game['home_team'],
                'Away': game['away_team'],
                'Home_Score': home_score,
                'Away_Score': away_score
            })
            
            test_games.append({
                'home': game['home_team'],
                'away': game['away_team'],
                'actual_winner': parse_result_winner(game['result'], game['home_team'], game['away_team']),
                'result': game['result']
            })
    
    if not test_games:
        print(f"  Could not parse any results")
        return None
    
    training_df = pd.DataFrame(training_data)
    
    # Generate predictions and compare
    results = {
        'elo_correct': 0,
        'consensus_correct': 0,
        'xgboost_correct': 0,
        'combined_correct': 0,
        'total': len(test_games)
    }
    
    for game in test_games:
        pred = calculate_predictions(training_df, game['home'], game['away'])
        
        # Determine predicted winners
        elo_pred = game['home'] if pred['elo_home_prob'] > 0.5 else game['away']
        consensus_pred = game['home'] if pred['logistic_home_prob'] > 0.5 else game['away']
        xgboost_pred = game['home'] if pred['xgboost_home_prob'] > 0.5 else game['away']
        combined_pred = game['home'] if pred['home_win_prob'] > 0.5 else game['away']
        
        # Check accuracy
        if elo_pred == game['actual_winner']:
            results['elo_correct'] += 1
        if consensus_pred == game['actual_winner']:
            results['consensus_correct'] += 1
        if xgboost_pred == game['actual_winner']:
            results['xgboost_correct'] += 1
        if combined_pred == game['actual_winner']:
            results['combined_correct'] += 1
    
    # Calculate percentages
    results['elo_accuracy'] = (results['elo_correct'] / results['total']) * 100
    results['consensus_accuracy'] = (results['consensus_correct'] / results['total']) * 100
    results['xgboost_accuracy'] = (results['xgboost_correct'] / results['total']) * 100
    results['combined_accuracy'] = (results['combined_correct'] / results['total']) * 100
    
    print(f"\n  Results for {sport}:")
    print(f"    Elo Rating: {results['elo_correct']}/{results['total']} ({results['elo_accuracy']:.1f}%)")
    print(f"    Consensus: {results['consensus_correct']}/{results['total']} ({results['consensus_accuracy']:.1f}%)")
    print(f"    XGBoost: {results['xgboost_correct']}/{results['total']} ({results['xgboost_accuracy']:.1f}%)")
    print(f"    Combined: {results['combined_correct']}/{results['total']} ({results['combined_accuracy']:.1f}%)")
    
    return {
        'sport': sport,
        **results
    }

def save_results_to_db(results_list):
    """Save backtesting results to database"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    # Create results table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_backtest_results (
            sport TEXT PRIMARY KEY,
            elo_correct INTEGER,
            consensus_correct INTEGER,
            xgboost_correct INTEGER,
            combined_correct INTEGER,
            total_games INTEGER,
            elo_accuracy REAL,
            consensus_accuracy REAL,
            xgboost_accuracy REAL,
            combined_accuracy REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    for result in results_list:
        cursor.execute('''
            INSERT OR REPLACE INTO model_backtest_results 
            (sport, elo_correct, consensus_correct, xgboost_correct, combined_correct, 
             total_games, elo_accuracy, consensus_accuracy, xgboost_accuracy, combined_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['sport'],
            result['elo_correct'],
            result['consensus_correct'],
            result['xgboost_correct'],
            result['combined_correct'],
            result['total'],
            result['elo_accuracy'],
            result['consensus_accuracy'],
            result['xgboost_accuracy'],
            result['combined_accuracy']
        ))
    
    conn.commit()
    conn.close()
    print("\n✅ Results saved to database")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("JACKPOTPICKS.BET - MODEL BACKTESTING")
    print("="*60)
    
    sports = ['NFL', 'NHL', 'MLB']
    all_results = []
    
    for sport in sports:
        result = backtest_sport(sport)
        if result:
            all_results.append(result)
    
    if all_results:
        save_results_to_db(all_results)
    
    print(f"\n{'='*60}")
    print("✅ BACKTESTING COMPLETE")
    print(f"{'='*60}")
