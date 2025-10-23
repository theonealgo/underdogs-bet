"""
Process NFL games data and run models to generate predictions
"""
import pandas as pd
import sqlite3
from datetime import datetime
import sys

def parse_nfl_games(file_path):
    """Parse the NFL games data file - format: Date Visitor VisScore Home HomeScore Box"""
    games = []
    
    # Team abbreviation mapping
    team_map = {
        'DAL': 'Dallas Cowboys', 'PHI': 'Philadelphia Eagles', 'KC': 'Kansas City Chiefs',
        'LAC': 'Los Angeles Chargers', 'ARI': 'Arizona Cardinals', 'NO': 'New Orleans Saints',
        'PIT': 'Pittsburgh Steelers', 'NYJ': 'New York Jets', 'MIA': 'Miami Dolphins',
        'IND': 'Indianapolis Colts', 'TB': 'Tampa Bay Buccaneers', 'ATL': 'Atlanta Falcons',
        'NYG': 'New York Giants', 'WAS': 'Washington Commanders', 'CAR': 'Carolina Panthers',
        'JAX': 'Jacksonville Jaguars', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
        'LV': 'Las Vegas Raiders', 'NE': 'New England Patriots', 'SF': 'San Francisco 49ers',
        'SEA': 'Seattle Seahawks', 'TEN': 'Tennessee Titans', 'DEN': 'Denver Broncos',
        'HOU': 'Houston Texans', 'LA': 'Los Angeles Rams', 'DET': 'Detroit Lions',
        'GB': 'Green Bay Packers', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
        'MIN': 'Minnesota Vikings', 'CHI': 'Chicago Bears'
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip header lines
            if 'Week' in line or 'Date' in line or 'Visitor' in line:
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            
            try:
                # Format: Date Visitor VisScore Home HomeScore [OT] Box
                date_str = parts[0].strip()
                visitor_abbr = parts[1].strip()
                visitor_score = int(parts[2].strip()) if parts[2].strip().isdigit() else None
                home_abbr = parts[3].strip()
                home_score = int(parts[4].strip()) if parts[4].strip().isdigit() else None
                
                # Skip if we don't have scores
                if visitor_score is None or home_score is None:
                    continue
                
                # Convert abbreviations to full names
                away_team = team_map.get(visitor_abbr, visitor_abbr)
                home_team = team_map.get(home_abbr, home_abbr)
                
                # Parse date - format is MM/DD
                date_parts = date_str.split('/')
                if len(date_parts) == 2:
                    month, day = date_parts
                    # Assume 2025 season
                    game_date = datetime(2025, int(month), int(day))
                    
                    # Determine winner
                    if home_score > visitor_score:
                        winner = home_team
                    elif visitor_score > home_score:
                        winner = away_team
                    else:
                        winner = 'Tie'
                    
                    games.append({
                        'date': game_date,
                        'date_str': game_date.strftime('%d/%m/%Y'),
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_score': visitor_score,
                        'home_score': home_score,
                        'winner': winner
                    })
            except Exception as e:
                continue
    
    df = pd.DataFrame(games)
    print(f"Parsed {len(df)} NFL games")
    if len(df) > 0:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df

def update_database_scores(games_df):
    """Update the database with actual scores"""
    conn = sqlite3.connect('sports_predictions.db')
    cursor = conn.cursor()
    
    updated = 0
    for _, game in games_df.iterrows():
        # Find the game in database
        cursor.execute("""
            SELECT id FROM games 
            WHERE sport='NFL' 
            AND game_date = date(?)
            AND home_team_id LIKE ?
            AND away_team_id LIKE ?
        """, (game['date'].strftime('%Y-%m-%d'), f"%{game['home_team'][-10:]}%", f"%{game['away_team'][-10:]}%"))
        
        result = cursor.fetchone()
        if result:
            game_id = result[0]
            cursor.execute("""
                UPDATE games 
                SET home_score = ?, away_score = ?, status = 'completed'
                WHERE id = ?
            """, (game['home_score'], game['away_score'], game_id))
            updated += 1
    
    conn.commit()
    conn.close()
    print(f"Updated {updated} games in database")
    return updated

def run_nfl_models(games_df):
    """Run NFL models on the games"""
    # Import the NFL predictor
    sys.path.insert(0, 'src/models')
    from nfl_ensemble_predictor import NFLEnsemblePredictor
    
    # Load all completed games for training
    conn = sqlite3.connect('sports_predictions.db')
    
    # Get historical games up to the test period
    historical_df = pd.read_sql("""
        SELECT 
            game_date,
            home_team_id as home_team,
            away_team_id as away_team,
            home_score,
            away_score,
            CASE 
                WHEN home_score > away_score THEN 1
                WHEN away_score > home_score THEN 0
                ELSE NULL
            END as result
        FROM games
        WHERE sport='NFL' 
        AND status='completed'
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
        ORDER BY game_date
    """, conn)
    
    # Parse dates - database uses DD/MM/YYYY format
    historical_df['game_date'] = pd.to_datetime(historical_df['game_date'], format='%d/%m/%Y')
    
    conn.close()
    
    print(f"Loaded {len(historical_df)} historical NFL games for training")
    
    # Initialize and train predictor
    predictor = NFLEnsemblePredictor()
    
    # Split: use games before September 4, 2025 for training
    train_cutoff = datetime(2025, 9, 4)
    train_df = historical_df[historical_df['game_date'] < train_cutoff].copy()
    
    if len(train_df) > 0:
        print(f"Training on {len(train_df)} games before {train_cutoff}")
        train_results = predictor.train(train_df)
        print(f"Training results: {train_results}")
    else:
        print("No training data available - using Elo only")
    
    # Make predictions on test games
    predictions = []
    for _, game in games_df.iterrows():
        pred = predictor.predict_game(game['home_team'], game['away_team'])
        
        # Determine actual winner
        if game['home_score'] > game['away_score']:
            actual_winner = game['home_team']
        elif game['away_score'] > game['home_score']:
            actual_winner = game['away_team']
        else:
            actual_winner = 'Tie'
        
        # Check if predictions are correct
        elo_correct = (pred['elo_home_prob'] > 0.5 and actual_winner == game['home_team']) or \
                     (pred['elo_home_prob'] <= 0.5 and actual_winner == game['away_team'])
        
        glmnet_correct = (pred['glmnet_home_prob'] > 0.5 and actual_winner == game['home_team']) or \
                        (pred['glmnet_home_prob'] <= 0.5 and actual_winner == game['away_team'])
        
        xgb_correct = (pred['xgboost_home_prob'] > 0.5 and actual_winner == game['home_team']) or \
                     (pred['xgboost_home_prob'] <= 0.5 and actual_winner == game['away_team'])
        
        ensemble_correct = (pred['blended_home_prob'] > 0.5 and actual_winner == game['home_team']) or \
                          (pred['blended_home_prob'] <= 0.5 and actual_winner == game['away_team'])
        
        predictions.append({
            'date': game['date_str'],
            'away_team': game['away_team'],
            'home_team': game['home_team'],
            'away_score': game['away_score'],
            'home_score': game['home_score'],
            'actual_winner': actual_winner,
            'elo_prob': f"{pred['elo_home_prob']*100:.1f}%",
            'glmnet_prob': f"{pred['glmnet_home_prob']*100:.1f}%",
            'xgb_prob': f"{pred['xgboost_home_prob']*100:.1f}%",
            'ensemble_prob': f"{pred['blended_home_prob']*100:.1f}%",
            'predicted_winner': pred['predicted_winner'],
            'elo_correct': 'Correct' if elo_correct else 'Incorrect',
            'glmnet_correct': 'Correct' if glmnet_correct else 'Incorrect',
            'xgb_correct': 'Correct' if xgb_correct else 'Incorrect',
            'ensemble_correct': 'Correct' if ensemble_correct else 'Incorrect',
        })
    
    pred_df = pd.DataFrame(predictions)
    
    # Calculate accuracies (excluding ties)
    non_tie_df = pred_df[pred_df['actual_winner'] != 'Tie'].copy()
    
    elo_acc = (non_tie_df['elo_correct'] == 'Correct').sum() / len(non_tie_df) * 100
    glmnet_acc = (non_tie_df['glmnet_correct'] == 'Correct').sum() / len(non_tie_df) * 100
    xgb_acc = (non_tie_df['xgb_correct'] == 'Correct').sum() / len(non_tie_df) * 100
    ensemble_acc = (non_tie_df['ensemble_correct'] == 'Correct').sum() / len(non_tie_df) * 100
    
    print("\n" + "="*80)
    print("NFL MODEL PERFORMANCE")
    print("="*80)
    print(f"Test Period: {games_df['date'].min().strftime('%Y-%m-%d')} to {games_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Games: {len(games_df)} ({len(non_tie_df)} non-tie)")
    print(f"\nElo:           {elo_acc:.1f}% ({(non_tie_df['elo_correct'] == 'Correct').sum()}/{len(non_tie_df)})")
    print(f"GLMNet:        {glmnet_acc:.1f}% ({(non_tie_df['glmnet_correct'] == 'Correct').sum()}/{len(non_tie_df)})")
    print(f"XGBoost:       {xgb_acc:.1f}% ({(non_tie_df['xgb_correct'] == 'Correct').sum()}/{len(non_tie_df)})")
    print(f"Meta Ensemble: {ensemble_acc:.1f}% ({(non_tie_df['ensemble_correct'] == 'Correct').sum()}/{len(non_tie_df)})")
    print("="*80)
    
    # Save results to CSV
    pred_df.to_csv('nfl_predictions_results.csv', index=False)
    print(f"\nResults saved to nfl_predictions_results.csv")
    
    return pred_df

if __name__ == '__main__':
    # Parse the NFL games data - use the new clearer format file
    games_df = parse_nfl_games('attached_assets/Pasted-Week-1-Date-Visitor-Home-Box-09-04-DAL-20-PHI-24-Box-09-05-KC-21-LAC-27-Box-09-07-ARI-20-NO-13--1761182874333_1761182874333.txt')
    
    # Update database with scores
    # update_database_scores(games_df)
    
    # Run models and get predictions
    results_df = run_nfl_models(games_df)
