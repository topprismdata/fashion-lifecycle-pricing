# 01 - Walmart Recruiting: Store Sales Forecasting

## Competition Info
- **URL**: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
- **Status**: Closed (2014)
- **Task**: Predict weekly sales for 45 Walmart stores × ~98 departments
- **Metric**: WMAE (holiday weeks weighted 5x)
- **Role in Project**: Seasonal demand prediction baseline, missing value handling

## Data Files
| File | Description |
|------|-------------|
| train.csv | Store, Dept, Date, Weekly_Sales, IsHoliday |
| test.csv | Prediction target weeks |
| features.csv | Temperature, Fuel_Price, MarkDown 1-5 (>60% missing), CPI, Unemployment |
| stores.csv | Store Type, Size |

## Key Challenges
1. MarkDown 1-5 missing rate >60%
2. Holiday weeks: 5x weight (Super Bowl, Labor Day, Thanksgiving, Christmas)
3. ~4000 store-department combinations
4. Negative sales (returns)

## Top Solutions
- 1st: 8 time-series models averaged (STLF, ETS, linear seasonal) - R
- 2nd: GBDT ensemble (Extra Trees + GBM + RF)
- Common: XGBoost/LightGBM with lag features, rolling stats

## Experiment Results

| Version | CV WMAE | Public LB | Private LB | Key Technique |
|---------|---------|-----------|------------|---------------|
| R01 | 1568.88 | 3102.97 | 3215.57 | LightGBM baseline, 28 features |

## ML Workflow Checklist
- [x] EDA: data format, missing values, sales distribution
- [x] Baseline: LightGBM with basic features (R01: LB=3102.97)
- [ ] Feature engineering: lag, rolling, holiday, markdown handling
- [ ] Model optimization: multi-model, weighted loss
- [ ] Ensemble: stacking/blending
- [ ] Post-processing: negative sales → 0
- [ ] Target: LB < 2800 (top 10%)
