import pandas as pd
from datetime import datetime

# Predictions data from user's file
predictions_text = """04/09/2025        Dallas Cowboys  @       Philadelphia Eagles     Philadelphia Eagles     55.5%   86.4%   62.8%
05/09/2025      Kansas City Chiefs      @       Los Angeles Chargers    Kansas City Chiefs      42.1%   83.9%   56.4%
07/09/2025      Baltimore Ravens        @       Buffalo Bills   Buffalo Bills   51.6%   59.2%   51.1%
07/09/2025      New York Giants @       Washington Commanders   Washington Commanders   52.0%   89.2%   57.6%
07/09/2025      Tennessee Titans        @       Denver Broncos  Denver Broncos  52.4%   79.9%   54.5%
07/09/2025      San Francisco 49ers     @       Seattle Seahawks        San Francisco 49ers     48.2%   32.6%   50.1%
07/09/2025      Detroit Lions   @       Green Bay Packers       Detroit Lions   49.8%   67.1%   62.8%
07/09/2025      Houston Texans  @       Los Angeles Rams        Houston Texans  48.2%   34.0%   50.6%
07/09/2025      Tampa Bay Buccaneers    @       Atlanta Falcons Tampa Bay Buccaneers    50.3%   59.6%   60.5%
07/09/2025      Cincinnati Bengals      @       Cleveland Browns        Cincinnati Bengals      50.6%   84.7%   69.4%
07/09/2025      Miami Dolphins  @       Indianapolis Colts      Miami Dolphins  51.5%   48.9%   57.4%
07/09/2025      Carolina Panthers       @       Jacksonville Jaguars    Carolina Panthers       51.1%   55.6%   59.6%
07/09/2025      Las Vegas Raiders       @       New England Patriots    Las Vegas Raiders       50.0%   65.2%   62.4%
07/09/2025      Arizona Cardinals       @       New Orleans Saints      Arizona Cardinals       50.0%   64.5%   62.1%
07/09/2025      Pittsburgh Steelers     @       New York Jets   Pittsburgh Steelers     50.3%   66.4%   62.9%
08/09/2025      Minnesota Vikings       @       Chicago Bears   Minnesota Vikings       41.1%   80.6%   55.7%
09/09/2025      Minnesota Vikings       @       Chicago Bears   Minnesota Vikings       51.3%   86.5%   70.2%
11/09/2025      Washington Commanders   @       Green Bay Packers       Washington Commanders   51.2%   48.9%   52.9%
14/09/2025      Atlanta Falcons @       Minnesota Vikings       Minnesota Vikings       55.5%   71.9%   57.3%
14/09/2025      Cleveland Browns        @       Baltimore Ravens        Baltimore Ravens        52.0%   92.2%   58.6%
14/09/2025      Jacksonville Jaguars    @       Cincinnati Bengals      Cincinnati Bengals      52.4%   82.1%   55.3%
14/09/2025      New York Giants @       Dallas Cowboys  Dallas Cowboys  52.4%   71.8%   51.7%
14/09/2025      Chicago Bears   @       Detroit Lions   Detroit Lions   51.8%   87.4%   56.9%
14/09/2025      New England Patriots    @       Miami Dolphins  Miami Dolphins  51.2%   68.6%   50.0%
14/09/2025      San Francisco 49ers     @       New Orleans Saints      San Francisco 49ers     50.3%   65.7%   62.7%
14/09/2025      Buffalo Bills   @       New York Jets   Buffalo Bills   51.3%   85.7%   70.0%
14/09/2025      Seattle Seahawks        @       Pittsburgh Steelers     Seattle Seahawks        50.3%   59.1%   60.3%
14/09/2025      Los Angeles Rams        @       Tennessee Titans        Los Angeles Rams        50.6%   82.7%   68.7%
14/09/2025      Carolina Panthers       @       Arizona Cardinals       Carolina Panthers       49.2%   37.7%   52.4%
14/09/2025      Denver Broncos  @       Indianapolis Colts      Denver Broncos  50.7%   67.1%   63.3%
14/09/2025      Philadelphia Eagles     @       Kansas City Chiefs      Philadelphia Eagles     49.6%   38.5%   52.7%
15/09/2025      Atlanta Falcons @       Minnesota Vikings       Minnesota Vikings       51.8%   79.8%   54.3%
15/09/2025      Tampa Bay Buccaneers    @       Houston Texans  Tampa Bay Buccaneers    49.4%   63.8%   61.5%
15/09/2025      Los Angeles Chargers    @       Las Vegas Raiders       Los Angeles Chargers    50.3%   71.5%   64.7%
18/09/2025      Miami Dolphins  @       Buffalo Bills   Buffalo Bills   55.9%   83.1%   61.5%
21/09/2025      Arizona Cardinals       @       San Francisco 49ers     San Francisco 49ers     55.9%   81.6%   60.9%
21/09/2025      Kansas City Chiefs      @       New York Giants Kansas City Chiefs      41.2%   94.6%   55.8%
21/09/2025      Atlanta Falcons @       Carolina Panthers       Atlanta Falcons 50.3%   70.5%   64.3%
21/09/2025      Green Bay Packers       @       Cleveland Browns        Green Bay Packers       50.6%   88.3%   70.5%
21/09/2025      Houston Texans  @       Jacksonville Jaguars    Houston Texans  50.3%   71.4%   64.7%
21/09/2025      Cincinnati Bengals      @       Minnesota Vikings       Minnesota Vikings       51.8%   71.9%   51.5%
21/09/2025      Pittsburgh Steelers     @       New England Patriots    Pittsburgh Steelers     50.6%   78.3%   67.2%
21/09/2025      Los Angeles Rams        @       Philadelphia Eagles     Philadelphia Eagles     51.8%   72.4%   51.7%
21/09/2025      New York Jets   @       Tampa Bay Buccaneers    Tampa Bay Buccaneers    52.0%   73.6%   52.2%
21/09/2025      Indianapolis Colts      @       Tennessee Titans        Indianapolis Colts      50.3%   66.1%   62.8%
21/09/2025      Las Vegas Raiders       @       Washington Commanders   Washington Commanders   51.8%   83.8%   55.6%
21/09/2025      Denver Broncos  @       Los Angeles Chargers    Denver Broncos  51.0%   49.8%   57.4%
21/09/2025      New Orleans Saints      @       Seattle Seahawks        Seattle Seahawks        52.4%   79.8%   54.5%
21/09/2025      Dallas Cowboys  @       Chicago Bears   Dallas Cowboys  51.1%   58.4%   60.5%
22/09/2025      Detroit Lions   @       Baltimore Ravens        Detroit Lions   42.7%   71.7%   54.7%
25/09/2025      Seattle Seahawks        @       Arizona Cardinals       Seattle Seahawks        41.9%   76.1%   55.5%
28/09/2025      Green Bay Packers       @       Dallas Cowboys  Green Bay Packers       41.7%   75.9%   55.4%
28/09/2025      Jacksonville Jaguars    @       San Francisco 49ers     Jacksonville Jaguars    48.8%   31.4%   50.0%
28/09/2025      Minnesota Vikings       @       Pittsburgh Steelers     Minnesota Vikings       50.1%   79.1%   67.1%
28/09/2025      Washington Commanders   @       Atlanta Falcons Washington Commanders   50.1%   73.7%   65.2%
28/09/2025      New Orleans Saints      @       Buffalo Bills   Buffalo Bills   51.8%   89.3%   57.5%
28/09/2025      Cleveland Browns        @       Detroit Lions   Detroit Lions   51.8%   93.9%   59.1%
28/09/2025      Tennessee Titans        @       Houston Texans  Houston Texans  52.4%   71.1%   51.5%
28/09/2025      Carolina Panthers       @       New England Patriots    Carolina Panthers       50.8%   59.1%   60.7%
28/09/2025      Los Angeles Chargers    @       New York Giants Los Angeles Chargers    50.6%   80.0%   67.7%
28/09/2025      Philadelphia Eagles     @       Tampa Bay Buccaneers    Philadelphia Eagles     50.1%   74.3%   65.4%
28/09/2025      Indianapolis Colts      @       Los Angeles Rams        Los Angeles Rams        51.8%   71.1%   51.2%
28/09/2025      Baltimore Ravens        @       Kansas City Chiefs      Kansas City Chiefs      51.8%   67.4%   50.0%
28/09/2025      Chicago Bears   @       Las Vegas Raiders       Chicago Bears   51.5%   53.3%   59.0%
29/09/2025      Cincinnati Bengals      @       Denver Broncos  Denver Broncos  52.7%   62.4%   52.6%
29/09/2025      New York Jets   @       Miami Dolphins  New York Jets   51.5%   45.5%   56.3%
02/10/2025      San Francisco 49ers     @       Los Angeles Rams        Los Angeles Rams        51.8%   69.0%   50.4%
05/10/2025      New England Patriots    @       Buffalo Bills   Buffalo Bills   55.9%   88.4%   63.8%
05/10/2025      Minnesota Vikings       @       Cleveland Browns        Minnesota Vikings       51.0%   93.4%   72.4%
05/10/2025      Houston Texans  @       Baltimore Ravens        Baltimore Ravens        51.8%   79.8%   54.2%
05/10/2025      Miami Dolphins  @       Carolina Panthers       Miami Dolphins  50.8%   60.2%   61.0%
05/10/2025      Las Vegas Raiders       @       Indianapolis Colts      Las Vegas Raiders       51.5%   45.1%   56.1%
05/10/2025      New York Giants @       New Orleans Saints      New York Giants 51.5%   47.2%   56.9%
05/10/2025      Dallas Cowboys  @       New York Jets   Dallas Cowboys  50.0%   62.2%   61.3%
05/10/2025      Denver Broncos  @       Philadelphia Eagles     Philadelphia Eagles     51.8%   76.0%   52.9%
05/10/2025      Tennessee Titans        @       Arizona Cardinals       Tennessee Titans        48.8%   32.9%   50.5%
05/10/2025      Tampa Bay Buccaneers    @       Seattle Seahawks        Tampa Bay Buccaneers    51.0%   49.4%   57.3%
05/10/2025      Detroit Lions   @       Cincinnati Bengals      Detroit Lions   50.1%   73.6%   65.1%
05/10/2025      Washington Commanders   @       Los Angeles Chargers    Washington Commanders   49.8%   67.3%   62.9%
06/10/2025      Kansas City Chiefs      @       Jacksonville Jaguars    Kansas City Chiefs      41.2%   94.8%   55.8%
09/10/2025      Philadelphia Eagles     @       New York Giants Philadelphia Eagles     41.2%   92.3%   56.2%
12/10/2025      Detroit Lions   @       Kansas City Chiefs      Kansas City Chiefs      51.9%   57.0%   50.9%
12/10/2025      Denver Broncos  @       New York Jets   Denver Broncos  50.3%   71.9%   64.8%
12/10/2025      Los Angeles Rams        @       Baltimore Ravens        Los Angeles Rams        48.2%   33.0%   50.2%
12/10/2025      Dallas Cowboys  @       Carolina Panthers       Dallas Cowboys  50.3%   67.4%   63.3%
12/10/2025      Arizona Cardinals       @       Indianapolis Colts      Arizona Cardinals       51.5%   51.2%   58.2%
12/10/2025      Seattle Seahawks        @       Jacksonville Jaguars    Seattle Seahawks        50.6%   81.8%   68.4%
12/10/2025      Los Angeles Chargers    @       Miami Dolphins  Los Angeles Chargers    50.7%   68.3%   63.7%
12/10/2025      Cleveland Browns        @       Pittsburgh Steelers     Pittsburgh Steelers     52.4%   79.0%   54.2%
12/10/2025      San Francisco 49ers     @       Tampa Bay Buccaneers    San Francisco 49ers     48.2%   33.1%   50.3%
12/10/2025      Tennessee Titans        @       Las Vegas Raiders       Tennessee Titans        49.2%   38.5%   52.7%
12/10/2025      Cincinnati Bengals      @       Green Bay Packers       Cincinnati Bengals      50.6%   42.3%   54.6%
12/10/2025      New England Patriots    @       New Orleans Saints      New England Patriots    51.1%   43.2%   55.3%
13/10/2025      Chicago Bears   @       Washington Commanders   Washington Commanders   55.9%   73.0%   57.7%
13/10/2025      Buffalo Bills   @       Atlanta Falcons Buffalo Bills   50.1%   76.0%   66.0%
16/10/2025      Pittsburgh Steelers     @       Cincinnati Bengals      Pittsburgh Steelers     50.7%   51.8%   53.7%"""

# Actual results from user's file
results_text = """09/04/2025    Dallas Cowboys  20      Philadelphia Eagles     24
09/05/2025      Kansas City Chiefs      21      Los Angeles Chargers    27
09/07/2025      Arizona Cardinals       20      New Orleans Saints      13
09/07/2025      Pittsburgh Steelers     34      New York Jets   32
09/07/2025      Miami Dolphins  8       Indianapolis Colts      33
09/07/2025      Tampa Bay Buccaneers    23      Atlanta Falcons 20
09/07/2025      New York Giants 6       Washington Commanders   21
09/07/2025      Carolina Panthers       10      Jacksonville Jaguars    26
09/07/2025      Cincinnati Bengals      17      Cleveland Browns        16
09/07/2025      Las Vegas Raiders       20      New England Patriots    13
09/07/2025      San Francisco 49ers     17      Seattle Seahawks        13
09/07/2025      Tennessee Titans        12      Denver Broncos  20
09/07/2025      Houston Texans  9       Los Angeles Rams        14
09/07/2025      Detroit Lions   13      Green Bay Packers       27
09/07/2025      Baltimore Ravens        40      Buffalo Bills   41
09/08/2025      Minnesota Vikings       27      Chicago Bears   24
09/11/2025      Washington Commanders   18      Green Bay Packers       27
09/14/2025      Los Angeles Rams        33      Tennessee Titans        19
09/14/2025      New York Giants 37      Dallas Cowboys  40
09/14/2025      San Francisco 49ers     26      New Orleans Saints      21
09/14/2025      Buffalo Bills   30      New York Jets   10
09/14/2025      New England Patriots    33      Miami Dolphins  27
09/14/2025      Seattle Seahawks        31      Pittsburgh Steelers     17
09/14/2025      Chicago Bears   21      Detroit Lions   52
09/14/2025      Jacksonville Jaguars    27      Cincinnati Bengals      31
09/14/2025      Cleveland Browns        17      Baltimore Ravens        41
09/14/2025      Carolina Panthers       22      Arizona Cardinals       27
09/14/2025      Denver Broncos  28      Indianapolis Colts      29
09/14/2025      Philadelphia Eagles     20      Kansas City Chiefs      17
09/14/2025      Atlanta Falcons 22      Minnesota Vikings       6
09/15/2025      Tampa Bay Buccaneers    20      Houston Texans  19
09/15/2025      Los Angeles Chargers    20      Las Vegas Raiders       9
09/18/2025      Miami Dolphins  21      Buffalo Bills   31
09/21/2025      Indianapolis Colts      41      Tennessee Titans        20
09/21/2025      Cincinnati Bengals      10      Minnesota Vikings       48
09/21/2025      Pittsburgh Steelers     21      New England Patriots    14
09/21/2025      Los Angeles Rams        26      Philadelphia Eagles     33
09/21/2025      New York Jets   27      Tampa Bay Buccaneers    29
09/21/2025      Las Vegas Raiders       24      Washington Commanders   41
09/21/2025      Atlanta Falcons 0       Carolina Panthers       30
09/21/2025      Houston Texans  10      Jacksonville Jaguars    17
09/21/2025      Green Bay Packers       10      Cleveland Browns        13
09/21/2025      New Orleans Saints      13      Seattle Seahawks        44
09/21/2025      Denver Broncos  20      Los Angeles Chargers    23
09/21/2025      Arizona Cardinals       15      San Francisco 49ers     16
09/21/2025      Dallas Cowboys  14      Chicago Bears   31
09/22/2025      Detroit Lions   38      Baltimore Ravens        30
09/25/2025      Seattle Seahawks        23      Arizona Cardinals       20
09/28/2025      Minnesota Vikings       21      Pittsburgh Steelers     24
09/28/2025      Tennessee Titans        0       Houston Texans  26
09/28/2025      Philadelphia Eagles     31      Tampa Bay Buccaneers    25
09/28/2025      Carolina Panthers       13      New England Patriots    42
09/28/2025      Los Angeles Chargers    18      New York Giants 21
09/28/2025      Washington Commanders   27      Atlanta Falcons 34
09/28/2025      New Orleans Saints      19      Buffalo Bills   31
09/28/2025      Cleveland Browns        10      Detroit Lions   34
09/28/2025      Indianapolis Colts      20      Los Angeles Rams        27
09/28/2025      Jacksonville Jaguars    26      San Francisco 49ers     21
09/28/2025      Chicago Bears   25      Las Vegas Raiders       24
09/28/2025      Baltimore Ravens        20      Kansas City Chiefs      37
09/28/2025      Green Bay Packers       40      Dallas Cowboys  40
09/29/2025      New York Jets   21      Miami Dolphins  27
09/29/2025      Cincinnati Bengals      3       Denver Broncos  28
10/02/2025      San Francisco 49ers     26      Los Angeles Rams        23
10/05/2025      Minnesota Vikings       21      Cleveland Browns        17
10/05/2025      New York Giants 14      New Orleans Saints      26
10/05/2025      Dallas Cowboys  37      New York Jets   22
10/05/2025      Denver Broncos  21      Philadelphia Eagles     17
10/05/2025      Houston Texans  44      Baltimore Ravens        10
10/05/2025      Las Vegas Raiders       6       Indianapolis Colts      40
10/05/2025      Miami Dolphins  24      Carolina Panthers       27
10/05/2025      Tampa Bay Buccaneers    38      Seattle Seahawks        35
10/05/2025      Tennessee Titans        22      Arizona Cardinals       21
10/05/2025      Washington Commanders   27      Los Angeles Chargers    10
10/05/2025      Detroit Lions   37      Cincinnati Bengals      24
10/05/2025      New England Patriots    23      Buffalo Bills   20
10/06/2025      Kansas City Chiefs      28      Jacksonville Jaguars    31
10/09/2025      Philadelphia Eagles     17      New York Giants 34
10/12/2025      Denver Broncos  13      New York Jets   11
10/12/2025      Cleveland Browns        9       Pittsburgh Steelers     23
10/12/2025      Dallas Cowboys  27      Carolina Panthers       30
10/12/2025      Seattle Seahawks        20      Jacksonville Jaguars    12
10/12/2025      Los Angeles Rams        17      Baltimore Ravens        3
10/12/2025      Arizona Cardinals       27      Indianapolis Colts      31
10/12/2025      Los Angeles Chargers    29      Miami Dolphins  27
10/12/2025      New England Patriots    25      New Orleans Saints      19
10/12/2025      Tennessee Titans        10      Las Vegas Raiders       20
10/12/2025      San Francisco 49ers     19      Tampa Bay Buccaneers    30
10/12/2025      Cincinnati Bengals      18      Green Bay Packers       27
10/12/2025      Detroit Lions   17      Kansas City Chiefs      30
10/13/2025      Buffalo Bills   14      Atlanta Falcons 24
10/13/2025      Chicago Bears   25      Washington Commanders   24
10/16/2025      Pittsburgh Steelers     31      Cincinnati Bengals      33"""

def parse_predictions(text):
    preds = []
    for line in text.strip().split('\n'):
        parts = line.split('\t')
        date = parts[0]
        away = parts[1]
        home = parts[3]
        pick = parts[4]
        xgb_pct = float(parts[5].replace('%', ''))
        elo_pct = float(parts[6].replace('%', ''))
        
        preds.append({
            'date': date,
            'away': away,
            'home': home,
            'pick': pick,
            'xgb_home_prob': xgb_pct,
            'elo_home_prob': elo_pct
        })
    return preds

def parse_results(text):
    results = []
    for line in text.strip().split('\n'):
        parts = line.split('\t')
        date = parts[0]
        away = parts[1]
        away_score = int(parts[2])
        home = parts[3]
        home_score = int(parts[4])
        
        results.append({
            'date': date,
            'away': away,
            'home': home,
            'away_score': away_score,
            'home_score': home_score,
            'winner': home if home_score > away_score else away
        })
    return results

# Parse data
predictions = parse_predictions(predictions_text)
results = parse_results(results_text)

# Match predictions to results
matched = []
unmatched = []

for pred in predictions:
    found = False
    for result in results:
        # Try exact match first
        if pred['date'] == result['date'] and pred['away'] == result['away'] and pred['home'] == result['home']:
            matched.append({**pred, **result})
            found = True
            break
    if not found:
        unmatched.append(pred)

print(f"📊 NFL 2025 SEASON ACCURACY CALCULATION")
print("="*80)
print(f"Total predictions: {len(predictions)}")
print(f"Total results: {len(results)}")
print(f"Matched games: {len(matched)}")
print(f"Unmatched: {len(unmatched)}")

if len(unmatched) > 0:
    print("\nFirst 3 unmatched predictions:")
    for p in unmatched[:3]:
        print(f"  {p['date']} {p['away']} @ {p['home']}")
    print("\nFirst 3 results:")
    for r in results[:3]:
        print(f"  {r['date']} {r['away']} @ {r['home']}")

print()

# Calculate accuracy
xgb_correct = 0
elo_correct = 0
main_correct = 0

for game in matched:
    # XGB prediction
    xgb_pick = game['home'] if game['xgb_home_prob'] > 50 else game['away']
    if xgb_pick == game['winner']:
        xgb_correct += 1
    
    # Elo prediction
    elo_pick = game['home'] if game['elo_home_prob'] > 50 else game['away']
    if elo_pick == game['winner']:
        elo_correct += 1
    
    # Main pick
    if game['pick'] == game['winner']:
        main_correct += 1

total = len(matched)

print("Model Performance:")
print(f"XGB Pick:   {xgb_correct}/{total} = {xgb_correct/total*100:.1f}%")
print(f"Elo Pick:   {elo_correct}/{total} = {elo_correct/total*100:.1f}%")
print(f"Main Pick:  {main_correct}/{total} = {main_correct/total*100:.1f}%")
print()

# Show some examples where Main Pick was wrong but XGB was right
print("Examples where XGB was RIGHT but Main Pick was WRONG:")
print("-"*80)
count = 0
for game in matched:
    xgb_pick = game['home'] if game['xgb_home_prob'] > 50 else game['away']
    xgb_right = (xgb_pick == game['winner'])
    main_right = (game['pick'] == game['winner'])
    
    if xgb_right and not main_right and count < 5:
        print(f"{game['date']} {game['away']} @ {game['home']}")
        print(f"  XGB: {game['xgb_home_prob']:.1f}% (picked {xgb_pick}) ✓")
        print(f"  Elo: {game['elo_home_prob']:.1f}%")
        print(f"  Main picked: {game['pick']} ✗")
        print(f"  Winner: {game['winner']}")
        print()
        count += 1
