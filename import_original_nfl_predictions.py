#!/usr/bin/env python3
"""
Import Original NFL Predictions from CSV Files
Restores the verified 72% Elo / 70.5% XGBoost predictions
"""
import sqlite3
import csv
import sys

DATABASE = 'sports_predictions_original.db'

def import_from_csv(csv_file):
    """Import predictions from CSV file"""
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Delete existing NFL predictions
    cursor.execute("DELETE FROM predictions WHERE sport = 'NFL'")
    print(f"✅ Cleared existing NFL predictions")
    
    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        predictions = list(reader)
    
    print(f"📊 Found {len(predictions)} predictions in {csv_file}")
    
    inserted = 0
    for pred in predictions:
        try:
            # Extract probabilities (they're stored as percentages in some files)
            elo_prob = float(pred['elo_home_prob']) if 'elo_home_prob' in pred else float(pred.get('elo_prob', '0.5').replace('%', '')) / 100
            
            # Handle different column names
            if 'xgboost_home_prob' in pred:
                xgb_prob = float(pred['xgboost_home_prob'])
            elif 'xgb_prob' in pred:
                xgb_prob = float(pred['xgb_prob'].replace('%', '')) / 100
            else:
                xgb_prob = 0.5
            
            if 'logistic_home_prob' in pred:
                cat_prob = float(pred['logistic_home_prob'])
            elif 'glmnet_home_prob' in pred:
                cat_prob = float(pred['glmnet_home_prob'])
            elif 'cat_prob' in pred:
                cat_prob = float(pred['cat_prob'].replace('%', '')) / 100
            else:
                cat_prob = 0.5
            
            if 'blended_home_prob' in pred:
                ensemble_prob = float(pred['blended_home_prob'])
            elif 'ensemble_prob' in pred:
                ensemble_prob = float(pred['ensemble_prob'].replace('%', '')) / 100
            else:
                ensemble_prob = (0.5 * elo_prob + 0.3 * xgb_prob + 0.2 * cat_prob)
            
            # Get game info
            game_date = pred['date']
            home_team = pred['home_team']
            away_team = pred['away_team']
            
            # Find matching game in database
            cursor.execute('''
                SELECT game_id FROM games
                WHERE sport = 'NFL' 
                AND game_date = ?
                AND home_team_id = ?
                AND away_team_id = ?
            ''', (game_date, home_team, away_team))
            
            game = cursor.fetchone()
            if game:
                game_id = game[0]
                
                # Insert prediction
                cursor.execute('''
                    INSERT INTO predictions 
                    (sport, league, game_id, game_date, home_team_id, away_team_id, 
                     elo_home_prob, xgboost_home_prob, logistic_home_prob, win_probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'NFL',
                    'NFL',
                    game_id,
                    game_date,
                    home_team,
                    away_team,
                    float(elo_prob),
                    float(xgb_prob),
                    float(cat_prob),
                    float(ensemble_prob)
                ))
                inserted += 1
            else:
                print(f"⚠️  Game not found: {game_date} {away_team} @ {home_team}")
        
        except Exception as e:
            print(f"❌ Error importing {pred.get('date', 'unknown')}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ SUCCESS!")
    print(f"📊 Imported {inserted} NFL predictions")
    
    return inserted

if __name__ == '__main__':
    print("\n" + "="*60)
    print("IMPORTING ORIGINAL NFL PREDICTIONS")
    print("="*60 + "\n")
    
    # Try nfl_predictions.csv first (has all 272 games)
    try:
        count = import_from_csv('nfl_predictions.csv')
        print(f"\n🎯 Restored {count} predictions from nfl_predictions.csv")
    except Exception as e:
        print(f"Error with nfl_predictions.csv: {e}")
        print("Trying nfl_predictions_output.csv...")
        count = import_from_csv('nfl_predictions_output.csv')
        print(f"\n🎯 Restored {count} predictions from nfl_predictions_output.csv")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60 + "\n")
