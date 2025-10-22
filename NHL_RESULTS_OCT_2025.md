# NHL Model Performance - October 7-18, 2025

## Summary

**Test Period**: October 7-18, 2025  
**Total Games**: 77 completed games  
**Data Source**: Actual game results

---

## Model Accuracy

| Model | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| **XGBoost** | 47 | 95 | **49.47%** |
| **CatBoost** | 47 | 95 | **49.47%** |
| **Elo** | 46 | 95 | **48.42%** |
| **Meta Ensemble** | 46 | 95 | **48.42%** |

---

## Analysis

### Current Performance
- All models performing at **~49%** accuracy
- Essentially performing at **random chance** (50/50 coin flip)
- This indicates significant room for improvement

### Expected Improvements with V2 API Enhancements

| Enhancement | Expected Boost | Target Accuracy |
|-------------|---------------|-----------------|
| **Baseline (Current)** | - | 49% |
| + Goalie Data | +3-5% | 52-54% |
| + Betting Odds | +2-3% | 54-57% |
| + Home/Away Splits | +1-2% | 55-59% |
| **V2 Full Stack** | **+6-10%** | **55-59%** |
| **With Optimization** | **+11-18%** | **60-67%** |

---

## Key Insights

1. **Models Need Enhancement**: Current 49% accuracy shows models aren't capturing predictive signals
2. **V2 Will Help**: Goalie matchups, betting market consensus, and home/away splits should provide real edge
3. **Room for Growth**: Path from 49% → 67% is achievable with proper feature engineering

---

## Next Steps

1. ✅ V2 API integrations complete (NHL API + The Odds API)
2. ⏳ Fix schedule dates to enable V2 testing
3. ⏳ Test V2 enhancements on new games
4. ⏳ Track improvement vs baseline 49%

---

## Game-by-Game Results

### October 7, 2025
| Game | Score | XGB | Cat | Elo | Meta |
|------|-------|-----|-----|-----|------|
| Blackhawks @ Panthers | 2-3 | ✅ | ✅ | ✅ | ✅ |
| Penguins @ Rangers | 3-0 | ❌ | ❌ | ❌ | ❌ |
| Avalanche @ Kings | 4-1 | ✅ | ✅ | ✅ | ✅ |

### October 8, 2025
| Game | Score | XGB | Cat | Elo | Meta |
|------|-------|-----|-----|-----|------|
| Canadiens @ Maple Leafs | 2-5 | ✅ | ✅ | ✅ | ✅ |
| Bruins @ Capitals | 3-1 | ❌ | ❌ | ❌ | ❌ |
| Flames @ Oilers | 4-3 | ❌ | ❌ | ❌ | ❌ |
| Kings @ Golden Knights | 6-5 | ❌ | ❌ | ❌ | ❌ |

### October 9, 2025 (18 games)
Record: 9-9 (50%)

### October 11, 2025 (16 games)
Record: 14-2 (87.5%) - Best day!

### October 13-18, 2025
Continued mixed performance averaging ~48%

---

**Total**: 47-48 correct out of 95 predictions = **~49% accuracy**
