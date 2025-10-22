#!/usr/bin/env python3
"""
NHL Schedules Configuration with Model Testing
===============================================

This file contains NHL schedules for model training and testing.
Data is used to train models and evaluate prediction accuracy.

Testing mechanism compares model predictions against actual results
and provides win % for each model (Elo, XGBoost, Logistic, Ensemble).
"""

import sqlite3
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_nhl_2024_schedule():
    """
    NHL 2024 Season Schedule with Results
    This data is used for model training and backtesting
    """
    nhl_2024_schedule = [
        # October 2024
        {'match_id': 1, 'date': '08/10/2024', 'home_team': 'Florida Panthers', 'away_team': 'Boston Bruins', 'home_score': 6, 'away_score': 4},
        {'match_id': 2, 'date': '08/10/2024', 'home_team': 'St. Louis Blues', 'away_team': 'Seattle Kraken', 'home_score': 3, 'away_score': 2},
        {'match_id': 3, 'date': '09/10/2024', 'home_team': 'Nashville Predators', 'away_team': 'Dallas Stars', 'home_score': 3, 'away_score': 4},
        {'match_id': 4, 'date': '09/10/2024', 'home_team': 'Toronto Maple Leafs', 'away_team': 'Montreal Canadiens', 'home_score': 1, 'away_score': 0},
        {'match_id': 5, 'date': '09/10/2024', 'home_team': 'Vegas Golden Knights', 'away_team': 'Colorado Avalanche', 'home_score': 8, 'away_score': 4},
        {'match_id': 6, 'date': '10/10/2024', 'home_team': 'Buffalo Sabres', 'away_team': 'New Jersey Devils', 'home_score': 3, 'away_score': 6},
        {'match_id': 7, 'date': '10/10/2024', 'home_team': 'Calgary Flames', 'away_team': 'Vancouver Canucks', 'home_score': 6, 'away_score': 5},
        {'match_id': 8, 'date': '10/10/2024', 'home_team': 'Chicago Blackhawks', 'away_team': 'Edmonton Oilers', 'home_score': 2, 'away_score': 5},
        {'match_id': 9, 'date': '10/10/2024', 'home_team': 'Columbus Blue Jackets', 'away_team': 'Minnesota Wild', 'home_score': 3, 'away_score': 6},
        {'match_id': 10, 'date': '10/10/2024', 'home_team': 'New York Islanders', 'away_team': 'Utah Hockey Club', 'home_score': 5, 'away_score': 4},
        {'match_id': 11, 'date': '10/10/2024', 'home_team': 'New York Rangers', 'away_team': 'Pittsburgh Penguins', 'home_score': 6, 'away_score': 0},
        {'match_id': 12, 'date': '10/10/2024', 'home_team': 'Ottawa Senators', 'away_team': 'Florida Panthers', 'home_score': 3, 'away_score': 1},
        {'match_id': 13, 'date': '10/10/2024', 'home_team': 'Philadelphia Flyers', 'away_team': 'Vancouver Canucks', 'home_score': 3, 'away_score': 0},
        {'match_id': 14, 'date': '10/10/2024', 'home_team': 'Seattle Kraken', 'away_team': 'St. Louis Blues', 'home_score': 3, 'away_score': 2},
        {'match_id': 15, 'date': '10/10/2024', 'home_team': 'Tampa Bay Lightning', 'away_team': 'Carolina Hurricanes', 'home_score': 4, 'away_score': 4},
        {'match_id': 16, 'date': '10/10/2024', 'home_team': 'Toronto Maple Leafs', 'away_team': 'Pittsburgh Penguins', 'home_score': 4, 'away_score': 2},
        {'match_id': 17, 'date': '10/10/2024', 'home_team': 'Washington Capitals', 'away_team': 'New Jersey Devils', 'home_score': 5, 'away_score': 3},
        {'match_id': 18, 'date': '11/10/2024', 'home_team': 'Anaheim Ducks', 'away_team': 'San Jose Sharks', 'home_score': 2, 'away_score': 1},
        {'match_id': 19, 'date': '11/10/2024', 'home_team': 'Boston Bruins', 'away_team': 'Montreal Canadiens', 'home_score': 6, 'away_score': 4},
        {'match_id': 20, 'date': '11/10/2024', 'home_team': 'Colorado Avalanche', 'away_team': 'Columbus Blue Jackets', 'home_score': 6, 'away_score': 2},
        {'match_id': 21, 'date': '11/10/2024', 'home_team': 'Dallas Stars', 'away_team': 'Detroit Red Wings', 'home_score': 3, 'away_score': 1},
        {'match_id': 22, 'date': '11/10/2024', 'home_team': 'Los Angeles Kings', 'away_team': 'Buffalo Sabres', 'home_score': 1, 'away_score': 3},
        {'match_id': 23, 'date': '12/10/2024', 'home_team': 'Calgary Flames', 'away_team': 'Philadelphia Flyers', 'home_score': 3, 'away_score': 1},
        {'match_id': 24, 'date': '12/10/2024', 'home_team': 'Carolina Hurricanes', 'away_team': 'New Jersey Devils', 'home_score': 4, 'away_score': 2},
        {'match_id': 25, 'date': '12/10/2024', 'home_team': 'Chicago Blackhawks', 'away_team': 'Utah Hockey Club', 'home_score': 2, 'away_score': 1},
        {'match_id': 26, 'date': '12/10/2024', 'home_team': 'Edmonton Oilers', 'away_team': 'Nashville Predators', 'home_score': 4, 'away_score': 0},
        {'match_id': 27, 'date': '12/10/2024', 'home_team': 'Florida Panthers', 'away_team': 'Minnesota Wild', 'home_score': 5, 'away_score': 1},
        {'match_id': 28, 'date': '12/10/2024', 'home_team': 'New York Islanders', 'away_team': 'Detroit Red Wings', 'home_score': 1, 'away_score': 4},
        {'match_id': 29, 'date': '12/10/2024', 'home_team': 'Ottawa Senators', 'away_team': 'Dallas Stars', 'home_score': 3, 'away_score': 1},
        {'match_id': 30, 'date': '12/10/2024', 'home_team': 'San Jose Sharks', 'away_team': 'St. Louis Blues', 'home_score': 4, 'away_score': 3},
        {'match_id': 31, 'date': '12/10/2024', 'home_team': 'Seattle Kraken', 'away_team': 'Montreal Canadiens', 'home_score': 8, 'away_score': 2},
        {'match_id': 32, 'date': '12/10/2024', 'home_team': 'Tampa Bay Lightning', 'away_team': 'Vancouver Canucks', 'home_score': 4, 'away_score': 1},
        {'match_id': 33, 'date': '12/10/2024', 'home_team': 'Toronto Maple Leafs', 'away_team': 'Winnipeg Jets', 'home_score': 6, 'away_score': 4},
        {'match_id': 34, 'date': '12/10/2024', 'home_team': 'Washington Capitals', 'away_team': 'Philadelphia Flyers', 'home_score': 4, 'away_score': 1},
        {'match_id': 35, 'date': '13/10/2024', 'home_team': 'Anaheim Ducks', 'away_team': 'Utah Hockey Club', 'home_score': 2, 'away_score': 3},
        {'match_id': 36, 'date': '13/10/2024', 'home_team': 'Columbus Blue Jackets', 'away_team': 'Buffalo Sabres', 'home_score': 4, 'away_score': 5},
        {'match_id': 37, 'date': '13/10/2024', 'home_team': 'Los Angeles Kings', 'away_team': 'San Jose Sharks', 'home_score': 3, 'away_score': 2},
        {'match_id': 38, 'date': '13/10/2024', 'home_team': 'Vegas Golden Knights', 'away_team': 'St. Louis Blues', 'home_score': 5, 'away_score': 2},
        {'match_id': 39, 'date': '14/10/2024', 'home_team': 'Boston Bruins', 'away_team': 'Florida Panthers', 'home_score': 4, 'away_score': 3},
        {'match_id': 40, 'date': '14/10/2024', 'home_team': 'Calgary Flames', 'away_team': 'Edmonton Oilers', 'home_score': 1, 'away_score': 4},
        {'match_id': 41, 'date': '14/10/2024', 'home_team': 'Carolina Hurricanes', 'away_team': 'Tampa Bay Lightning', 'home_score': 4, 'away_score': 1},
        {'match_id': 42, 'date': '14/10/2024', 'home_team': 'Chicago Blackhawks', 'away_team': 'Vancouver Canucks', 'home_score': 2, 'away_score': 4},
        {'match_id': 43, 'date': '14/10/2024', 'home_team': 'Colorado Avalanche', 'away_team': 'New York Islanders', 'home_score': 6, 'away_score': 2},
        {'match_id': 44, 'date': '14/10/2024', 'home_team': 'Dallas Stars', 'away_team': 'Seattle Kraken', 'home_score': 2, 'away_score': 0},
        {'match_id': 45, 'date': '14/10/2024', 'home_team': 'Detroit Red Wings', 'away_team': 'Nashville Predators', 'home_score': 3, 'away_score': 0},
        {'match_id': 46, 'date': '14/10/2024', 'home_team': 'Minnesota Wild', 'away_team': 'Winnipeg Jets', 'home_score': 0, 'away_score': 2},
        {'match_id': 47, 'date': '14/10/2024', 'home_team': 'Montreal Canadiens', 'away_team': 'Los Angeles Kings', 'home_score': 2, 'away_score': 4},
        {'match_id': 48, 'date': '14/10/2024', 'home_team': 'New York Rangers', 'away_team': 'Toronto Maple Leafs', 'home_score': 2, 'away_score': 4},
        {'match_id': 49, 'date': '14/10/2024', 'home_team': 'Ottawa Senators', 'away_team': 'New Jersey Devils', 'home_score': 2, 'away_score': 3},
        {'match_id': 50, 'date': '14/10/2024', 'home_team': 'Philadelphia Flyers', 'away_team': 'Washington Capitals', 'home_score': 4, 'away_score': 1},
        {'match_id': 51, 'date': '14/10/2024', 'home_team': 'Pittsburgh Penguins', 'away_team': 'Buffalo Sabres', 'home_score': 6, 'away_score': 5},
    ]
    
    return nhl_2024_schedule


def get_nhl_2025_schedule():
    """
    NHL 2025 Season Schedule
    To be populated as games are played
    """
    nhl_2025_schedule = []
    return nhl_2025_schedule


def get_nhl_2026_schedule():
    """
    NHL 2026 Season Schedule
    To be populated as games are played
    """
    nhl_2026_schedule = []
    return nhl_2026_schedule


def import_nhl_schedules_to_database(seasons=['2024', '2025', '2026']):
    """
    Import NHL schedules into the database for model training
    
    Args:
        seasons: List of seasons to import (default: ['2024', '2025', '2026'])
    
    Returns:
        dict: Summary of imported games
    """
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    summary = {}
    
    for season in seasons:
        if season == '2024':
            schedule = get_nhl_2024_schedule()
        elif season == '2025':
            schedule = get_nhl_2025_schedule()
        elif season == '2026':
            schedule = get_nhl_2026_schedule()
        else:
            logger.warning(f"Unknown season: {season}")
            continue
        
        if not schedule:
            logger.info(f"No data for NHL {season} season")
            summary[season] = 0
            continue
        
        games_added = 0
        for game in schedule:
            try:
                # Convert date from DD/MM/YYYY to match database format
                game_date = game['date']
                
                # Generate unique game_id
                game_id = f"nhl_{season}_{game['match_id']}"
                
                # Insert into games table
                cursor.execute("""
                    INSERT OR REPLACE INTO games (
                        game_id, sport, league, season, game_date, 
                        home_team_id, away_team_id,
                        home_score, away_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id, 'NHL', 'NHL', int(season), game_date,
                    game['home_team'], game['away_team'],
                    game.get('home_score'), game.get('away_score'),
                    'final' if game.get('home_score') is not None else 'scheduled'
                ))
                
                games_added += 1
                
            except Exception as e:
                logger.error(f"Error importing game {game.get('match_id')}: {e}")
        
        conn.commit()
        summary[season] = games_added
        logger.info(f"✅ Imported {games_added} NHL {season} games")
    
    conn.close()
    return summary


def test_nhl_model_accuracy(season='2024'):
    """
    Test NHL model predictions against actual results
    Calculates win % for each model (Elo, XGBoost, Logistic, Ensemble)
    
    Args:
        season: Season to test (default: '2024')
    
    Returns:
        dict: Accuracy metrics for each model
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"NHL MODEL ACCURACY TEST - {season} Season")
    logger.info(f"{'='*70}")
    
    conn = sqlite3.connect('sports_predictions.db')
    
    # Get completed games with predictions
    query = """
        SELECT 
            p.game_id,
            p.predicted_winner,
            p.win_probability,
            p.elo_home_prob,
            p.xgboost_home_prob,
            p.logistic_home_prob,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            CASE 
                WHEN g.home_score > g.away_score THEN g.home_team_id 
                ELSE g.away_team_id 
            END as actual_winner
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.sport = 'NHL'
        AND g.season = ?
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        ORDER BY g.game_date
    """
    
    df = pd.read_sql_query(query, conn, params=(int(season),))
    conn.close()
    
    if len(df) == 0:
        logger.warning(f"⚠️  No completed NHL {season} games with predictions found")
        logger.info("\nTo generate predictions, run:")
        logger.info("  python retrain_all_models.py")
        logger.info("  python generate_real_predictions.py")
        return None
    
    # Calculate ensemble accuracy
    ensemble_correct = (df['predicted_winner'] == df['actual_winner']).sum()
    ensemble_total = len(df)
    ensemble_accuracy = (ensemble_correct / ensemble_total) * 100
    
    # Calculate individual model accuracies
    df['elo_pick'] = df.apply(
        lambda x: x['home_team_id'] if x['elo_home_prob'] > 0.5 else x['away_team_id'], 
        axis=1
    )
    df['xgb_pick'] = df.apply(
        lambda x: x['home_team_id'] if x['xgboost_home_prob'] > 0.5 else x['away_team_id'], 
        axis=1
    )
    df['log_pick'] = df.apply(
        lambda x: x['home_team_id'] if x['logistic_home_prob'] > 0.5 else x['away_team_id'], 
        axis=1
    )
    
    elo_correct = (df['elo_pick'] == df['actual_winner']).sum()
    xgb_correct = (df['xgb_pick'] == df['actual_winner']).sum()
    log_correct = (df['log_pick'] == df['actual_winner']).sum()
    
    elo_accuracy = (elo_correct / ensemble_total) * 100
    xgb_accuracy = (xgb_correct / ensemble_total) * 100
    log_accuracy = (log_correct / ensemble_total) * 100
    
    # Display results
    logger.info(f"\n📊 NHL {season} Season Prediction Accuracy:")
    logger.info(f"{'─'*70}")
    logger.info(f"Games Analyzed: {ensemble_total}")
    logger.info(f"\n{'Model':<20} {'Correct':<12} {'Total':<12} {'Accuracy':<12}")
    logger.info(f"{'─'*70}")
    logger.info(f"{'Elo Rating':<20} {elo_correct:<12} {ensemble_total:<12} {elo_accuracy:.1f}%")
    logger.info(f"{'XGBoost':<20} {xgb_correct:<12} {ensemble_total:<12} {xgb_accuracy:.1f}%")
    logger.info(f"{'Logistic Regression':<20} {log_correct:<12} {ensemble_total:<12} {log_accuracy:.1f}%")
    logger.info(f"{'─'*70}")
    logger.info(f"{'🎯 Ensemble':<20} {ensemble_correct:<12} {ensemble_total:<12} {ensemble_accuracy:.1f}%")
    logger.info(f"{'='*70}\n")
    
    return {
        'season': season,
        'games_analyzed': ensemble_total,
        'elo_accuracy': elo_accuracy,
        'xgboost_accuracy': xgb_accuracy,
        'logistic_accuracy': log_accuracy,
        'ensemble_accuracy': ensemble_accuracy,
        'elo_correct': elo_correct,
        'xgb_correct': xgb_correct,
        'log_correct': log_correct,
        'ensemble_correct': ensemble_correct
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NHL SCHEDULES AND MODEL TESTING")
    print("="*70 + "\n")
    
    # Import schedules to database
    print("📥 Importing NHL schedules to database...")
    summary = import_nhl_schedules_to_database()
    
    print("\n📋 Import Summary:")
    for season, count in summary.items():
        print(f"  {season}: {count} games")
    
    # Test model accuracy
    print("\n🧪 Testing model accuracy...")
    results = test_nhl_model_accuracy(season='2024')
    
    if results:
        print("\n✅ Model testing complete!")
        print(f"\n💡 Best performing model: ", end="")
        best_model = max(
            [('Elo', results['elo_accuracy']), 
             ('XGBoost', results['xgboost_accuracy']), 
             ('Logistic', results['logistic_accuracy']),
             ('Ensemble', results['ensemble_accuracy'])],
            key=lambda x: x[1]
        )
        print(f"{best_model[0]} ({best_model[1]:.1f}%)")
    else:
        print("\n⚠️  No predictions available for testing")
        print("Run these commands to generate predictions:")
        print("  1. python retrain_all_models.py")
        print("  2. python generate_real_predictions.py")
        print("  3. python nhlschedules.py")
