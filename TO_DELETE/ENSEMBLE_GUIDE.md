# Universal Ensemble Prediction System Guide

## 🎯 What You Have

A **universal sports prediction system** that works with **ANY sport** using a powerful ensemble approach:

- **Elo Ratings**: Dynamic team strength ratings that update after each game
- **GLMNet**: Regularized logistic regression with elastic net penalty
- **XGBoost**: Gradient boosting classifier optimized for predictions
- **Blended Ensemble**: Weighted combination (30% Elo + 35% GLMNet + 35% XGBoost)

## 📋 Supported Sports

- **MLB** (Major League Baseball)
- **NFL** (National Football League)
- **NBA** (National Basketball Association)
- **NHL** (National Hockey League)
- **NCAAF** (NCAA Football)
- **NCAAB** (NCAA Basketball)
- **WNBA** (Women's National Basketball Association)

## 🚀 Quick Start

### 1. Prepare Your CSV

Your CSV needs at least these columns (auto-detected):
- **Home Team** (or "Home", "HomeTeam", etc.)
- **Away Team** (or "Away", "AwayTeam", "Visitor", etc.)
- **Result** (optional - for training)
  - Scores: "24-20", "3-2"
  - Text: "H", "A", "Home", "Away"
  - Numbers: "1", "2" (1=Home, 2=Away)

Optional columns:
- **Date** (any format)
- **Game ID** (Match Number, ID, etc.)
- **Venue** (Location, Stadium, etc.)

### 2. Run the Predictor

```bash
python universal_sports_predictor.py <SPORT> <csv_file>
```

**Examples:**
```bash
# NFL predictions
python universal_sports_predictor.py NFL nfl_schedule.csv

# MLB predictions
python universal_sports_predictor.py MLB mlb_schedule.csv

# NBA predictions
python universal_sports_predictor.py NBA nba_schedule.csv

# NHL predictions
python universal_sports_predictor.py NHL nhl_schedule.csv
```

### 3. View Results

After running, you get:

1. **Console output** with sample predictions
2. **CSV file**: `{sport}_predictions.csv` with all predictions
3. **Trained model**: `models/{sport}_ensemble.pkl` (saved for reuse)
4. **Database records**: Games and predictions stored automatically

## 📊 Output Format

The predictions CSV includes:

- `away_team` / `home_team`: Team names
- `elo_home_prob`: Elo prediction (0-1)
- `glmnet_home_prob`: GLMNet prediction (0-1)
- `xgboost_home_prob`: XGBoost prediction (0-1)
- `blended_home_prob`: **Final prediction** (0-1)
- `predicted_winner`: Team predicted to win
- `confidence`: Prediction confidence (0-1)

## 🎓 How It Works

### Training Phase
1. **Elo System**: Builds team ratings from historical results
2. **Feature Creation**: Creates features from Elo ratings and home advantage
3. **GLMNet Training**: Trains regularized logistic regression
4. **XGBoost Training**: Trains gradient boosting classifier
5. **Model Saving**: Saves all models for future predictions

### Prediction Phase
1. **Feature Extraction**: Gets current Elo ratings for both teams
2. **Individual Predictions**: Gets prediction from each model
3. **Ensemble Blending**: Combines using weighted average
4. **Winner Selection**: Picks team with >50% probability

## 📈 Performance

**NFL Example** (272 games):
- Training: 64 games with results
- GLMNet: 61% accuracy
- XGBoost: 78% accuracy
- Teams: 32 with Elo ratings

Your results will vary based on:
- Number of training games
- Data quality
- Sport characteristics

## 🔄 Updating Models

To retrain with new data:

1. Add new games with results to your CSV
2. Run the predictor again
3. Models automatically retrain on all historical data
4. New predictions incorporate updated team strengths

## 💡 Tips

1. **More training data = better predictions**
   - Include as many completed games as possible
   - Models improve with more historical results

2. **CSV flexibility**
   - Column names are auto-detected
   - Multiple result formats supported
   - Don't worry about exact formatting

3. **Model reuse**
   - Trained models are saved automatically
   - Can be loaded later for faster predictions
   - Useful for production deployments

4. **Database integration**
   - All predictions stored in SQLite database
   - Accessible via Streamlit web interface
   - Historical tracking included

## 🐛 Troubleshooting

**No predictions generated?**
- Check if your CSV has the required columns
- Ensure team names are consistent
- Verify result format is recognized

**Low accuracy?**
- Need more training data
- Check for data quality issues
- Consider sport-specific tuning

**Column not detected?**
- CSV column detection is flexible but not perfect
- Manually rename columns if needed
- Common names: "Home Team", "Away Team", "Result"

## 📞 Next Steps

1. **Upload your MLB/NBA/NHL CSVs** - Same format as NFL
2. **Run predictions** - See results for each sport
3. **View in web interface** - Check Streamlit app
4. **Compare models** - See which performs best

---

**Questions?** Just ask! The system is designed to be simple and flexible.
