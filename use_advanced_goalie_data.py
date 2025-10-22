#!/usr/bin/env python3
"""
Enable advanced goalie data for NHL predictions
Switches from API data to user-provided comprehensive goalie stats
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE = 'sports_predictions.db'

def check_advanced_data_available():
    """Check if advanced goalie data exists"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='advanced_goalie_stats'
    """)
    
    has_table = cursor.fetchone() is not None
    
    if has_table:
        cursor.execute("SELECT COUNT(*) FROM advanced_goalie_stats")
        count = cursor.fetchone()[0]
        logger.info(f"✓ Advanced goalie data available: {count} goalies")
    else:
        count = 0
        logger.warning("✗ Advanced goalie data not available")
    
    conn.close()
    return has_table and count > 0

def get_goalie_features(home_goalie_name, away_goalie_name, use_advanced=True):
    """
    Get goalie features for prediction
    Returns: dict with goalie differential and advanced metrics
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    features = {
        'goalie_sv_pct_diff': 0.0,
        'goalie_gsax_diff': 0.0,
        'goalie_high_danger_diff': 0.0,
        'goalie_rebound_control_diff': 0.0
    }
    
    if use_advanced:
        # Get home goalie advanced stats
        cursor.execute("""
            SELECT save_pct, goals_saved_above_expected, 
                   high_danger_sv_pct, rebound_control_pct
            FROM advanced_goalie_stats
            WHERE goalie_name = ?
        """, (home_goalie_name,))
        home_stats = cursor.fetchone()
        
        # Get away goalie advanced stats
        cursor.execute("""
            SELECT save_pct, goals_saved_above_expected,
                   high_danger_sv_pct, rebound_control_pct
            FROM advanced_goalie_stats
            WHERE goalie_name = ?
        """, (away_goalie_name,))
        away_stats = cursor.fetchone()
        
        if home_stats and away_stats:
            features['goalie_sv_pct_diff'] = home_stats[0] - away_stats[0]
            features['goalie_gsax_diff'] = (home_stats[1] - away_stats[1]) / 100  # Normalize
            features['goalie_high_danger_diff'] = home_stats[2] - away_stats[2]
            features['goalie_rebound_control_diff'] = home_stats[3] - away_stats[3]
        else:
            logger.warning(f"Advanced stats not found for {home_goalie_name} or {away_goalie_name}")
    
    conn.close()
    return features

def show_goalie_comparison(team1, team2):
    """Show detailed goalie comparison between two teams"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Get team 1 goalie
    cursor.execute("""
        SELECT tg.goalie_name, ags.save_pct, ags.gaa, 
               ags.goals_saved_above_expected, ags.high_danger_sv_pct,
               ags.rebound_control_pct, ags.games_played
        FROM team_goalies tg
        JOIN advanced_goalie_stats ags ON tg.goalie_name = ags.goalie_name
        WHERE tg.team_abbr = ? OR tg.team_name = ?
    """, (team1, team1))
    team1_goalie = cursor.fetchone()
    
    # Get team 2 goalie
    cursor.execute("""
        SELECT tg.goalie_name, ags.save_pct, ags.gaa,
               ags.goals_saved_above_expected, ags.high_danger_sv_pct,
               ags.rebound_control_pct, ags.games_played
        FROM team_goalies tg
        JOIN advanced_goalie_stats ags ON tg.goalie_name = ags.goalie_name
        WHERE tg.team_abbr = ? OR tg.team_name = ?
    """, (team2, team2))
    team2_goalie = cursor.fetchone()
    
    conn.close()
    
    if team1_goalie and team2_goalie:
        print(f"\n🥅 Goalie Matchup: {team1} vs {team2}")
        print("=" * 70)
        print(f"{'Metric':<25} {team1_goalie[0]:<22} {team2_goalie[0]:<22}")
        print("-" * 70)
        print(f"{'Games Played':<25} {team1_goalie[6]:<22} {team2_goalie[6]:<22}")
        print(f"{'Save %':<25} {team1_goalie[1]:.3f}{' '*18} {team2_goalie[1]:.3f}")
        print(f"{'GAA':<25} {team1_goalie[2]:.2f}{' '*19} {team2_goalie[2]:.2f}")
        print(f"{'GSAX (Total)':<25} {team1_goalie[3]:>5.1f}{' '*16} {team2_goalie[3]:>5.1f}")
        print(f"{'High Danger SV%':<25} {team1_goalie[4]:.3f}{' '*18} {team2_goalie[4]:.3f}")
        print(f"{'Rebound Control %':<25} {team1_goalie[5]:.3f}{' '*18} {team2_goalie[5]:.3f}")
        
        # Calculate advantages
        sv_diff = team1_goalie[1] - team2_goalie[1]
        gsax_diff = team1_goalie[3] - team2_goalie[3]
        
        print("\n📊 Advantage:")
        if sv_diff > 0.005:
            print(f"   {team1} has +{sv_diff:.3f} save % advantage")
        elif sv_diff < -0.005:
            print(f"   {team2} has +{abs(sv_diff):.3f} save % advantage")
        else:
            print(f"   Even matchup (save % within 0.005)")
        
        return True
    else:
        print(f"\n⚠️ Could not find goalie data for {team1} or {team2}")
        return False

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ADVANCED GOALIE DATA STATUS")
    print("="*70)
    
    available = check_advanced_data_available()
    
    if available:
        print("\n✅ Advanced goalie data is ACTIVE")
        print("\n💡 Available advanced metrics:")
        print("   1. Save % Differential (existing)")
        print("   2. Goals Saved Above Expected (GSAX)")
        print("   3. High Danger Save % Differential")
        print("   4. Rebound Control Differential")
        
        # Show example comparison
        print("\n" + "="*70)
        show_goalie_comparison("TOR", "BOS")
        show_goalie_comparison("COL", "VGK")
        
    else:
        print("\n❌ Advanced goalie data NOT available")
        print("\nTo import advanced data, run:")
        print("   python import_advanced_goalie_data.py")
