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
5. Test period entirely out-of-sample (no overlap with training)

## Top Solutions
- 1st: 6 time-series models averaged (STLF, ETS, Fourier ARIMA, seasonal naive) + Christmas shift post-processing - R
- 2nd: Per-department GBM with cross-store features, sqrt transform
- Common: SVD noise reduction, holiday distance features

## Experiment Results

| Version | CV WMAE | Public LB | Private LB | Key Technique | Notes |
|---------|---------|-----------|------------|---------------|-------|
| R01 | 1568.88 | 3102.97 | 3215.57 | LightGBM baseline, 28 features | First baseline |
| R02 | 187.77 | 5960.12 | 6162.20 | Lag+Rolling, combined train+test | **FAIL**: data leakage |
| R03 | 1257.87 | 15050.74 | 15116.63 | Lag from train-only, fillna(0) | **FAIL**: test lag all zeros |
| R04 | 2296.69 | 5910.07 | 6075.48 | Multi-model ensemble, feat mismatch | **FAIL**: column mismatch |
| R05 | 2261.57 | 3196.53 | 3312.05 | 69 features, aggregated stats | Worse than R01 |
| R06 | 1392.27 | 2815.12 | 2912.12 | 40 features, seasonal+sd_median | Good improvement |
| **R08** | **1410.64** | **2720.96** | **2838.91** | **R06+YoY52+sd_mean** | **BEST** |
| R07 | 1334.98 | 3533.66 | 3508.43 | YoY+Christmas shift+sqrt+5seed | FAIL: sqrt transform |
| R09 | 1081.26 | — | — | Leak-free stats+naive blend | Not submitted (leakage in val) |
| R10 | 1570.92 | 3025.14 | 3148.09 | Leak-free stats+3seed | Worse than R08 |
| R11 | 1178.47 | 2907.02 | 3016.09 | LGB+XGB+Naive ensemble | Worse than R08 |
| R12 | 1414.75 | 2820.51 | 2933.14 | R08+5seed avg | Slightly worse than R08 |

## Key Findings

### 1. Lag Features Are Harmful (ts-lag-out-of-sample-trap)
Test period (2012-11-02 to 2013-07-26) extends BEYOND training period (2010-02-05 to 2012-10-26).
Lag features referencing recent history (lag_1 through lag_52) are unavailable for test rows.
R02-R04 all failed because lag features degraded from helpful-in-CV to harmful-in-LB.

### 2. Safe Features for Out-of-Sample Test
- Calendar features (Week, Month, Quarter) — always computable
- Seasonal sin/cos encoding — mathematical transform
- Holiday proximity — based on fixed calendar dates
- Store-Dept median/mean from train — static statistic
- YoY same-week-last-year (offset 52) — references known train data (98.2% valid rate in test)
- MarkDown aggregates — based on test's own values

### 3. Unsafe Features for Out-of-Sample Test
- Lag features (any offset) — no history available
- Rolling statistics — depends on recent history
- Seasonal naive baseline from full train — leakage in CV

### 4. What Worked (R06→R08 Improvement Path)
- R06 over R01: +sin/cos encoding, +holiday proximity, +sd_median (+287 LB improvement)
- R08 over R06: +YoY same-week-last-year, +sd_mean (+94 LB improvement)

### 5. What Didn't Work
- sqrt transform (R07): changed loss landscape, hurt WMAE
- Multi-model ensemble (R05, R11): XGBoost much worse than LightGBM for this task
- 69 features with many aggregated stats (R05): overfitting
- Christmas shift post-processing (R07): unclear benefit, combined with sqrt hurt
- Multi-seed averaging (R12): models converged to same solution
- Seasonal naive blending (R11): leakage in naive baseline

## ML Workflow Checklist
- [x] EDA: data format, missing values, sales distribution
- [x] Baseline: LightGBM with basic features (R01: LB=3102.97)
- [x] Feature engineering: seasonal, holiday, aggregated stats, YoY
- [x] Model optimization: hyperparameter tuning, multi-model comparison
- [x] Key insight: lag features harmful for out-of-sample test
- [x] Skill extraction: ts-lag-out-of-sample-trap
- [ ] Target: LB < 2800 (top 10%) — **ACHIEVED: R08 LB=2720.96**
- [ ] Further improvement: per-department models, SVD noise reduction
