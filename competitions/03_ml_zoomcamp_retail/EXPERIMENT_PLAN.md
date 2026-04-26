# ML Zoomcamp 2024 Retail Demand Forecast -- Experiment Plan

## Competition Summary

- **Goal**: Predict daily `quantity` for 13,636 items across 4 stores, 1 month ahead (2024-09-27 to 2024-10-26)
- **Metric**: RMSE (unweighted, simpler than WMAE)
- **Data**: 25 months sales history (~362MB), 9 data files total
- **Top LB**: 8.9580 (1st), 9.2348 (2nd AutoGluon), 9.4650 (4th CatBoost)
- **Target**: mean=5.6, median=2.0, max=4952, highly right-skewed
- **Cold items**: 211 / 13,636 (1.5%) -- need fallback strategy

## Validation Strategy

**Primary: Time-based holdout (simulating the competition split)**

```
Train: 2023-08-04 ~ 2024-08-26 (last 30 days as validation)
Val:   2024-08-27 ~ 2024-09-26 (30 days immediately before test)
Test:  2024-09-27 ~ 2024-10-26 (30 days to predict)
```

- This mirrors the actual test split structure (predict 1 month ahead)
- RMSE on val serves as primary model selection criterion
- Submit to LB only for milestone rounds (R01, R04, R07, R10)

**Secondary: Multi-fold temporal CV (for R06+)**

```
Fold 1: Train through 2024-06-26, Val 2024-06-27 ~ 2024-07-26
Fold 2: Train through 2024-07-26, Val 2024-07-27 ~ 2024-08-26
Fold 3: Train through 2024-08-26, Val 2024-08-27 ~ 2024-09-26
```

- 3-fold temporal CV for robust hyperparameter tuning
- Only used in later rounds when single-fold is unstable

**Critical: Avoid the ts-lag-out-of-sample-trap**

- Test period starts AFTER train ends (no date overlap)
- Rolling/lag features must be computed with proper lookback from last training date
- See skill: `ts-lag-out-of-sample-trap` for the full methodology
- For a 30-day test horizon, only use features computed from data BEFORE the test start date
- Rolling features should be shifted appropriately (e.g., rolling_mean_7 computed as of date - 1)

---

## R01: Baseline (CatBoost + Minimal Features)

**Goal**: Get on the leaderboard with minimal effort, establish a benchmark.

**Target**: RMSE < 12.0 on LB

### Features

| Feature Group | Features | Notes |
|---|---|---|
| Date features | day_of_week, day_of_month, month, is_weekend, is_month_start/end | Basic temporal |
| Item identity | item_id (categorical), store_id (categorical) | Let CatBoost handle high cardinality |
| Catalog | dept_name, class_name (categorical) | Join from catalog.csv |
| Lag (last known) | quantity_lag_7, quantity_lag_14, quantity_lag_30 | Rolling window from train, aligned to last known date |
| Price | price_base (from sales, latest known) | Simple last known price |

### Implementation

1. Load sales.csv, filter out negative quantities
2. Left-join test with train using (item_id, store_id, date - lag) for lag features
3. For each test date, compute rolling features from the last known training data
4. Train CatBoost with RMSE objective
5. Post-process: clip predictions to [0, max_train_quantity_per_item]

### Reusable Skills

- `tabular-feature-engineering-patterns` -- Pattern 3 (binary flags for is_weekend etc.)
- `ts-lag-out-of-sample-trap` -- Compute lag features correctly for out-of-sample test
- `catboost-multicore-config` -- CatBoost training configuration

### Success Criteria

- CV RMSE < 11.0
- LB RMSE < 12.0
- Baseline file: `scripts/r01_baseline.py`

---

## R02: Data Cleaning + Rolling Feature Engineering

**Goal**: Clean data quality issues and add comprehensive rolling window features. Expected: 15-20% improvement.

**Target**: RMSE < 10.5 on LB

### Data Cleaning

| Issue | Action | Rationale |
|---|---|---|
| Negative quantity | Remove rows with quantity < 0 | Returns/errors, not demand |
| Inf price_base | Replace inf with NaN, then fill with item-level median | sum_total/0 artifact |
| Negative sum_total | Remove rows | Data error |
| Zero quantity | Keep (genuine zero-demand days) | Important signal for RMSE |

### New Features: Rolling Statistics

| Feature | Windows | Aggregations | Group By |
|---|---|---|---|
| Rolling quantity stats | 7, 14, 30, 60 days | mean, std, min, max, median | (item_id, store_id) |
| Rolling quantity stats | 7, 14, 30 days | mean, std | (item_id) -- across all stores |
| Rolling nonzero ratio | 7, 30 days | mean (proportion of days with quantity > 0) | (item_id, store_id) |
| Rolling trend | 7/30 ratio | mean_7 / (mean_30 + 1) | (item_id, store_id) |
| Exponential moving average | span 7, 14, 30 | ewm mean | (item_id, store_id) |
| Quantity quantile ratio | 7d | q90/q50 ratio (burstiness) | (item_id, store_id) |

### Implementation Details

- Compute rolling features on the FULL training set, then extract values at the cutoff date
- For test inference: use rolling features computed as of 2024-09-26 (last training date)
- For validation: use rolling features computed as of 2024-08-26 (val cutoff date)
- Use `pandas.groupby().rolling()` with proper date ordering
- Downcast to float32 to manage memory

### Reusable Skills

- `ts-forecasting-stale-lag-methodology` -- How to properly compute rolling features for multi-step forecast
- `tabular-feature-engineering-patterns` -- Pattern 4 (log transforms for skewed quantity)

### Success Criteria

- CV RMSE improvement of 15%+ vs R01
- Feature importance: rolling features in top 10

---

## R03: Multi-Source Data Integration + Calendar Features

**Goal**: Leverage all 9 data files and add calendar/promotion features. Expected: 5-10% improvement.

**Target**: RMSE < 10.0 on LB

### New Features from External Data

| Source | Features | Notes |
|---|---|---|
| discounts_history.csv | is_on_promo (binary), promo_type_code (cat), discount_pct, discount_duration_days | Per (item, store, date) |
| price_history.csv | price_change_pct, n_price_changes_30d, price_vs_base_ratio | Price dynamics |
| markdowns.csv | is_marked_down (binary), markdown_depth (normal - sale price) | Clearance items |
| online.csv | online_quantity_7d, online_quantity_30d, online_nonzero_ratio | Store 1 only |
| actual_matrix.csv | is_in_assortment (binary) | Was item available? |
| catalog.csv | item_type (cat), weight_volume, weight_netto | Product attributes |
| stores.csv | division, format, city, area | Store attributes (only 4 rows) |

### Calendar Features

| Feature | Description | Source |
|---|---|---|
| Russian holidays | is_holiday, days_to_next_holiday, days_since_last_holiday | `holidays` library, country='RU' |
| Pay day effect | is_pay_day (1st, 15th of month, common in Russia) | Domain knowledge |
| Month-end effect | is_month_end_3d (within 3 days of month end) | Date arithmetic |
| Week number | week_of_year (1-52) | Captures annual seasonality |
| Quarter | quarter (1-4) | Seasonal patterns |
| Day-of-week encoded | sin/cos encoding for cyclical pattern | Better than raw int |

### Implementation Details

- Merge promotions as left-join (most days have no promo)
- For price history: use last known price before forecast date
- Online data only relevant for store_id=1
- Actual matrix indicates item availability -- helps with zero-demand prediction

### Reusable Skills

- `domain-knowledge-constraints-trap` -- Don't over-constrain with domain assumptions
- `external-data-fusion-redundancy` -- Verify new features add signal beyond existing ones

### Success Criteria

- CV RMSE improvement of 5%+ vs R02
- No feature leakage from future data
- Memory usage stays under 16GB

---

## R04: Advanced Time Series Features + Target Transforms

**Goal**: Add advanced temporal features and experiment with target transformations. Expected: 3-5% improvement.

**Target**: RMSE < 9.8 on LB

### Advanced Time Series Features

| Feature | Description | Notes |
|---|---|---|
| YoY 364-day features | quantity from same day-of-week last year | Requires data from 2023-09, which we have |
| YoY 7d average | 7-day average centered on same week last year | Smooths YoY |
| Seasonal decomposition | STL trend + seasonal residual as features | Per (item, store) if enough data |
| Fourier terms | sin/cos pairs for weekly and annual cycles | 2-3 pairs each |
| Autocorrelation features | acf_lag_1, acf_lag_7 from last 60 days | Captures item-level persistence |
| Zero-run length | Current consecutive zero streak (if applicable) | Intermittent demand pattern |
| Item age | Days since first appearance in sales data | New vs mature items |

### Target Transformations (experiment with each)

| Transform | Formula | When to Use |
|---|---|---|
| Log1p | log(1 + quantity) | Reduces skew, natural for count data |
| Square root | sqrt(quantity) | Milder than log, preserves zeros better |
| None (raw) | quantity as-is | Baseline for comparison |

**Important**: For RMSE, target transforms may not help as much as for RMSLE. But for CatBoost with highly skewed targets, log1p often improves performance by reducing the influence of extreme values.

### Validation of Feature Utility

- Train model with and without each feature group
- Use CatBoost feature importance (PredictionValuesChange)
- Remove features with importance < 0.01% to reduce noise

### Reusable Skills

- `yoy-364day-features` -- YoY feature implementation with 364-day alignment
- `ts-day-specific-forecasting` -- Consider if unified model struggles (unlikely needed for 30-day horizon)
- `unified-vs-day-specific-forecasting` -- Unified model likely wins here (same conclusion as Favorita)
- `ts-forecasting-stale-lag-methodology` -- Reference for multi-step approach

### Success Criteria

- CV RMSE improvement of 3%+ vs R03
- Feature importance analysis completed
- Best target transform identified

---

## R05: CatBoost Hyperparameter Optimization

**Goal**: Systematically tune CatBoost hyperparameters. Expected: 2-3% improvement.

**Target**: RMSE < 9.5 on LB

### Hyperparameter Search Space

| Parameter | Search Range | Notes |
|---|---|---|
| learning_rate | [0.01, 0.1] | Log-uniform |
| depth | [6, 10] | CatBoost handles deeper trees well |
| l2_leaf_reg | [1, 10] | Regularization |
| random_strength | [0.5, 3.0] | Randomization for prediction diversity |
| bagging_temperature | [0, 1] | Bootstrap randomization |
| border_count | [128, 254] | Quantization bins |
| grow_policy | Depthwise vs Lossguide | Different tree growing strategies |
| iterations | 2000-5000 | With early stopping |
| auto_class_weights | None vs Balanced | If using two-stage approach |

### Tuning Strategy

1. **Phase 1**: Optuna with 50 trials, 3-fold temporal CV
2. **Phase 2**: Refine top-5 configurations with full training
3. **Phase 3**: Ensemble top-3 models (seeds, slight param variations)

### Categorical Feature Handling

CatBoost natively handles categoricals. Key columns to mark as `cat_features`:
- item_id (high cardinality: ~13K)
- store_id (4 values)
- dept_name, class_name, subclass_name
- promo_type_code
- item_type

### Reusable Skills

- `catboost-multicore-config` -- Optimal thread/core settings
- `cv-lb-divergence-xgboost` -- Monitor CV-LB correlation during tuning

### Success Criteria

- Best CV RMSE configuration identified
- LB RMSE within 5% of CV RMSE (no severe CV-LB divergence)

---

## R06: Alternative Models + Feature Alignment

**Goal**: Train LightGBM and XGBoost as diverse models for ensemble. Expected: marginal single-model improvement, but sets up ensemble.

**Target**: RMSE < 9.5 per model, ensemble-ready

### Models to Train

| Model | Why | Configuration |
|---|---|---|
| LightGBM | Fast, different inductive bias than CatBoost | Similar hyperparam space |
| XGBoost | Third opinion for ensemble diversity | histogram tree method |
| CatBoost (variant) | Different seed + slight param variation | From R05 best params |

### Feature Handling Differences

| Aspect | CatBoost | LightGBM | XGBoost |
|---|---|---|---|
| Categoricals | Native handling | Label encode + enable_categorical | One-hot or target encode |
| Missing values | Native handling | Native handling | Native handling |
| Feature alignment | **CRITICAL** | Verify same feature order | Verify same feature order |

### Key Skill Application

- `lightgbm-prediction-feature-alignment` -- MUST verify features align between train/inference
- `lightgbm-unified-hyperparameter-tuning` -- Tuning approach
- `feature-combination-synergy` -- Feature subsets per model can improve ensemble diversity

### Success Criteria

- All 3 models trained with same feature set
- Pairwise prediction correlation < 0.95 (sufficient diversity)
- Each model CV RMSE < 9.8

---

## R07: Zero-Inflated Two-Stage Modeling

**Goal**: Address the zero-heavy distribution with a two-stage approach. Expected: 2-4% improvement on items with many zeros.

**Target**: RMSE < 9.3 on LB

### Two-Stage Architecture

```
Stage 1: Binary Classifier (is quantity > 0?)
  -> P(nonzero) for each (item, store, date)

Stage 2: Regression on non-zero samples
  -> Predict log1p(quantity) for positive samples only

Combine: final_pred = P(nonzero) * expm1(regression_pred)
```

### Classifier Details

| Aspect | Choice | Rationale |
|---|---|---|
| Model | CatBoost classifier | Same categorical handling |
| Features | Same as regression + zero-specific features | is_in_assortment, zero_run_length |
| Threshold | Optimize on validation set | Default 0.5 may not be optimal |
| Metric | AUC-ROC for monitoring | But threshold tuned for RMSE |

### Zero-Specific Features

- `zero_ratio_30d`: proportion of zero days in last 30
- `zero_run_current`: consecutive zeros up to cutoff date
- `is_in_assortment`: from actual_matrix.csv
- `days_since_last_sale`: time gap to last positive quantity
- `n_sales_days_30d`: count of days with sales in last 30

### When Two-Stage Helps

- Items with >50% zero days benefit most
- High-volume items (quantity > 10 daily) may not benefit
- Consider two-stage only for items with zero_ratio > 30%

### Reusable Skills

- `zero-inflated-two-stage-forecasting` -- Full two-stage methodology
- `rmsle-zero-threshold-asymmetry` -- Understanding threshold sensitivity (applies to RMSE too)

### Success Criteria

- Stage 1 AUC > 0.85
- Combined two-stage RMSE better than single-stage for zero-heavy items
- Overall CV RMSE improvement of 2%+ vs R05 best

---

## R08: Model Ensemble

**Goal**: Combine multiple models for robust predictions. Expected: 1-3% improvement.

**Target**: RMSE < 9.1 on LB

### Ensemble Strategies

| Strategy | Weight | Method |
|---|---|---|
| Simple average | Equal | CatBoost + LightGBM + XGBoost |
| Weighted average | Optimize on val | Grid search over weights |
| Stacking | Meta-learner | Ridge regression on OOF predictions |
| Two-stage blend | 80/20 | Two-stage model vs single-stage |

### Weighted Average Implementation

```python
# Grid search for optimal weights
best_rmse = float('inf')
for w1 in np.arange(0.3, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.4, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * pred_cb + w2 * pred_lgb + w3 * pred_xgb
        rmse = np.sqrt(mean_squared_error(y_val, blend))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = (w1, w2, w3)
```

### Stacking Implementation (if weighted average plateaus)

```python
# Out-of-fold predictions for meta-learner
# Use 3-fold temporal CV to generate OOF preds
oof_cb = get_oof_predictions(catboost_model, folds)
oof_lgb = get_oof_predictions(lightgbm_model, folds)
oof_xgb = get_oof_predictions(xgboost_model, folds)

# Meta features
meta_X = np.column_stack([oof_cb, oof_lgb, oof_xgb])
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X, y_train)
```

### Per-Store and Per-Category Blending

- Experiment with different model weights per store_id
- Experiment with different weights for high-volume vs low-volume items
- May help since online sales only exist for store 1

### Reusable Skills

- `kaggle-optimal-blending` -- 80/20 blending rule
- `ensemble-model-correlation-trap` -- Check model diversity before blending
- `model-ensemble` -- General ensemble methodology

### Success Criteria

- Ensemble CV RMSE better than best single model
- LB RMSE < 9.1
- Model prediction correlation matrix documented

---

## R09: Post-Processing + Cold Item Handling

**Goal**: Refine predictions with post-processing and special handling for edge cases. Expected: 0.5-1% improvement.

**Target**: RMSE < 9.0 on LB

### Post-Processing Steps

| Step | Method | Rationale |
|---|---|---|
| Clipping | Clip to [0, item_historical_max * 1.5] | Remove impossible predictions |
| Zero forcing | Force prediction to 0 if P(nonzero) < threshold | Reduces RMSE for zero days |
| Smooth rounding | Round predictions to nearest 0.5 | Quantity is count data |
| Store-level scaling | Scale predictions to match store's avg daily total | Preserve store-level patterns |

### Cold Item Handling (211 items)

These items appear in test but NOT in training:

| Strategy | Implementation |
|---|---|
| Class-level average | Use average quantity for items in same (dept, class) |
| Store-class average | Average for same store + class combination |
| Global median | Fallback: median quantity across all items |
| Zero prediction | If no similar items found, predict 0 |

```python
# Cold item strategy
cold_items = test_items - train_items
for item in cold_items:
    # Find similar items from same subclass
    subclass = catalog.loc[item, 'subclass_name']
    similar_items = catalog[catalog['subclass_name'] == subclass].index
    similar_items_in_train = similar_items & train_items
    if len(similar_items_in_train) > 0:
        cold_pred[item] = sales[sales['item_id'].isin(similar_items_in_train)].groupby('store_id')['quantity'].median()
    else:
        cold_pred[item] = 0
```

### Day-of-Week Adjustment

From EDA: Friday is highest (6.23), Sunday is lowest (5.27). This ~18% range matters.

- Compute per-item DoW ratios from training data
- Apply as multiplicative adjustment to predictions
- Verify this doesn't hurt high-volume items where the model already learned DoW

### Reusable Skills

- `ts-day-specific-forecasting` -- Day-of-week specific adjustments
- `small-dataset-optimization-limits` -- Cold items have minimal data, keep it simple

### Success Criteria

- Cold item RMSE < global RMSE (reasonable fallback)
- Post-processed RMSE improves by 0.5%+ vs raw ensemble
- No negative predictions in final submission

---

## R10: Final Submission + Robustness Checks

**Goal**: Produce the best possible submission with thorough validation.

**Target**: RMSE as low as possible, top-5% on LB

### Final Model Configuration

| Component | Configuration |
|---|---|
| Best single model | CatBoost with R05 optimized hyperparams |
| Two-stage model | R07 zero-inflated approach |
| Ensemble | R08 weighted blend of 3-5 models |
| Post-processing | R09 clipping + cold item handling |

### Robustness Checks

| Check | Method | Pass Criteria |
|---|---|---|
| Feature leakage | Train with last 30 days excluded, verify no future data | CV and LB consistent |
| Submission format | Match sample_submission.csv exactly | row_id, quantity columns, 883,680 rows |
| No NaN predictions | Check all predictions are finite | 0 NaN/Inf values |
| Prediction range | All predictions in [0, 5000] | Reasonable bounds |
| Seed stability | Train with 3 different seeds | RMSE std < 0.05 |
| Time consistency | Predictions follow temporal patterns | No sudden jumps between days |

### Submission File Validation

```python
sub = pd.read_csv("submission_r10.csv")
assert len(sub) == 883680, f"Expected 883680 rows, got {len(sub)}"
assert sub.columns.tolist() == ['row_id', 'quantity'], f"Wrong columns: {sub.columns.tolist()}"
assert sub['quantity'].isna().sum() == 0, "NaN in predictions"
assert (sub['quantity'] < 0).sum() == 0, "Negative predictions"
assert sub['quantity'].max() <= 5000, "Unreasonable max prediction"
```

### Final Deliverables

| File | Description |
|---|---|
| `scripts/r10_final_pipeline.py` | End-to-end pipeline (train + predict) |
| `outputs/submissions/submission_r10.csv` | Final submission |
| `outputs/models/r10_catboost.cbm` | Best CatBoost model |
| `outputs/models/r10_lgb.txt` | Best LightGBM model |
| `EXPERIMENT_LOG.md` | Results from all rounds |

### Reusable Skills to Extract After Completion

- `retail-demand-forecasting-pipeline` -- Full pipeline for retail demand forecasting
- `multi-source-feature-merge` -- Efficient merging of 9 data files
- `russian-holiday-features` -- Russian holiday calendar feature generation
- `cold-item-forecasting` -- Strategies for items with no training history

---

## Experiment Tracking Template

| Round | CV RMSE | LB RMSE | Key Technique | Time | Notes |
|-------|---------|---------|---------------|------|-------|
| R01 | _ | _ | CatBoost baseline + rolling | _ | _ |
| R02 | _ | _ | Data cleaning + rolling stats | _ | _ |
| R03 | _ | _ | Multi-source + calendar | _ | _ |
| R04 | _ | _ | YoY + target transforms | _ | _ |
| R05 | _ | _ | HPO CatBoost | _ | _ |
| R06 | _ | _ | LightGBM + XGBoost | _ | _ |
| R07 | _ | _ | Two-stage zero-inflated | _ | _ |
| R08 | _ | _ | Ensemble blending | _ | _ |
| R09 | _ | _ | Post-processing + cold items | _ | _ |
| R10 | _ | _ | Final submission | _ | _ |

---

## Skills Reuse Map

| Skill | Rounds Applied | How |
|-------|---------------|-----|
| `ts-lag-out-of-sample-trap` | R01-R04 | Proper lag feature computation for non-overlapping test |
| `zero-inflated-two-stage-forecasting` | R07 | Two-stage model for zero-heavy items |
| `yoy-364day-features` | R04 | Year-over-year seasonal features |
| `tabular-feature-engineering-patterns` | R01-R04 | Systematic feature creation patterns |
| `ts-forecasting-stale-lag-methodology` | R02-R04 | Rolling feature methodology for multi-step |
| `ts-day-specific-forecasting` | R09 | Day-of-week adjustments |
| `unified-vs-day-specific-forecasting` | R04 | Stick with unified model (proven better) |
| `kaggle-optimal-blending` | R08 | 80/20 blending rule |
| `catboost-multicore-config` | R01, R05 | Training configuration |
| `lightgbm-prediction-feature-alignment` | R06 | Verify feature alignment |
| `cv-lb-divergence-xgboost` | R05-R06 | Monitor CV-LB correlation |
| `feature-combination-synergy` | R06 | Feature subsets for diversity |
| `ensemble-model-correlation-trap` | R08 | Check diversity before blending |
| `rmsle-zero-threshold-asymmetry` | R07, R09 | Threshold sensitivity analysis |
| `domain-knowledge-constraints-trap` | R03 | Don't over-constrain features |
| `external-data-fusion-redundancy` | R03 | Verify new data sources add value |
| `small-dataset-optimization-limits` | R09 | Cold items -- keep fallback simple |
| `pandas-groupby-replaces-per-row-loop` | R02 | Efficient rolling feature computation |
| `ml-pipeline-unit-testing` | R10 | Data quality validation |
| `pairwise-target-encoding-strategy` | R06 | Interaction encoding for non-CatBoost models |
| `sigmoid-smoothing-target-encoding` | R06 | Smoothed target encoding for LightGBM/XGBoost |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Memory overflow (362MB sales.csv) | Medium | High | Process in chunks, float32 downcasting, use DuckDB |
| CV-LB divergence | Medium | High | Use temporal holdout mirroring competition split |
| Feature leakage | Low | Critical | Strict cutoff: never use data from test period |
| Overfitting to validation set | Medium | Medium | Final model selection on LB, use 3-fold CV |
| Cold items drag down score | Low | Low | Simple class-level fallback, don't overthink |
| Ensemble hurts (negative diversity) | Low | Medium | Check prediction correlation, fallback to best single model |

---

## Hardware & Time Estimates

| Round | Est. Time | Peak Memory | Notes |
|-------|-----------|-------------|-------|
| R01 | 30 min | 8 GB | Simple features, fast training |
| R02 | 2 hours | 12 GB | Rolling features computation heavy |
| R03 | 3 hours | 14 GB | Multi-file joins |
| R04 | 2 hours | 12 GB | YoY features, transform experiments |
| R05 | 4 hours | 8 GB | HPO with Optuna |
| R06 | 3 hours | 12 GB | Multiple models |
| R07 | 3 hours | 10 GB | Two separate models |
| R08 | 2 hours | 8 GB | Ensembling is fast |
| R09 | 1 hour | 4 GB | Post-processing only |
| R10 | 2 hours | 14 GB | Full pipeline retrain |
| **Total** | **~22 hours** | **14 GB peak** | |

All estimates assume Apple Silicon Mac with 16GB+ RAM.
