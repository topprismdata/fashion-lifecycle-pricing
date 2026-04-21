"""
Walmart Store Sales Forecasting — R05 No-Lag Tuned
Lessons from R01-R04: lag features HARMFUL (test out-of-sample).
Build on R01's foundation with:
1. Better seasonal features (week-of-year stats from train)
2. Proper store-dept aggregated stats (train-only, verified identical columns)
3. 39-week validation matching test period length
4. Hyperparameter tuning for WMAE
5. Multi-model ensemble (LightGBM + XGBoost)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ============================================================
# 1. Load & Merge
# ============================================================
print("=" * 60)
print("R05: Walmart — No-Lag Tuned (R01 Foundation)")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. Feature Engineering (IDENTICAL for train and test)
# ============================================================
def create_features(df, train_ref=None):
    """Create features. train_ref is used for aggregated stats only."""
    df = df.copy()

    # --- Time features ---
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfMonth'] = (df['Day'] - 1) // 7 + 1

    # Seasonal sin/cos encoding
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # --- Store type encoding ---
    df['Type_A'] = (df['Type'] == 'A').astype(int)
    df['Type_B'] = (df['Type'] == 'B').astype(int)
    df['Type_C'] = (df['Type'] == 'C').astype(int)

    # --- Holiday ---
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # --- MarkDown ---
    for i in range(1, 6):
        col = f'MarkDown{i}'
        df[f'{col}_na'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    df['MarkDown_total'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
    df['MarkDown_any'] = (df['MarkDown_total'] > 0).astype(int)
    df['MarkDown_count'] = sum(df[f'MarkDown{i}'] > 0 for i in range(1, 6))

    # --- Fill NaN ---
    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    # --- Size ---
    df['Size_log'] = np.log1p(df['Size'])
    df['Size_bin'] = pd.qcut(df['Size'], q=5, labels=False, duplicates='drop')

    # --- Store-Dept interaction ---
    df['Store_Dept'] = df['Store'] * 100 + df['Dept']

    # --- Holiday proximity ---
    holiday_dates = pd.to_datetime([
        '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',
        '2010-09-10', '2011-09-09', '2012-09-07',
        '2010-11-26', '2011-11-25', '2012-11-23',
        '2010-12-31', '2011-12-30', '2012-12-28',
    ])
    df['days_to_holiday'] = df['Date'].apply(lambda d: min(abs((d - h).days) for h in holiday_dates))
    df['weeks_to_holiday'] = df['days_to_holiday'] // 7
    df['near_holiday_2w'] = (df['weeks_to_holiday'] <= 2).astype(int)
    df['near_holiday_1w'] = (df['weeks_to_holiday'] <= 1).astype(int)

    # Specific holiday proximity
    thanksgivings = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23'])
    christmases = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28'])
    super_bowls = pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'])
    labordays = pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07'])
    df['near_thanksgiving'] = df['Date'].apply(lambda d: any(abs((d - h).days) <= 14 for h in thanksgivings)).astype(int)
    df['near_christmas'] = df['Date'].apply(lambda d: any(abs((d - h).days) <= 14 for h in christmases)).astype(int)
    df['near_superbowl'] = df['Date'].apply(lambda d: any(abs((d - h).days) <= 7 for h in super_bowls)).astype(int)
    df['near_laborday'] = df['Date'].apply(lambda d: any(abs((d - h).days) <= 7 for h in labordays)).astype(int)

    # --- Fuel_Price change ---
    df['Fuel_Price_change'] = df.groupby('Store')['Fuel_Price'].diff()
    df['Fuel_Price_change'] = df['Fuel_Price_change'].fillna(0)

    # --- Aggregated stats from train_ref (SAFE for out-of-sample) ---
    if train_ref is not None:
        # Store stats
        store_stats = train_ref.groupby('Store')['Weekly_Sales'].agg(['mean', 'std', 'median', 'max']).reset_index()
        store_stats.columns = ['Store', 'store_mean', 'store_std', 'store_median', 'store_max']
        df = df.merge(store_stats, on='Store', how='left')

        # Dept stats
        dept_stats = train_ref.groupby('Dept')['Weekly_Sales'].agg(['mean', 'std', 'median']).reset_index()
        dept_stats.columns = ['Dept', 'dept_mean', 'dept_std', 'dept_median']
        df = df.merge(dept_stats, on='Dept', how='left')

        # Store-Dept stats
        sd_stats = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'median', 'count']).reset_index()
        sd_stats.columns = ['Store', 'Dept', 'sd_mean', 'sd_std', 'sd_median', 'sd_count']
        df = df.merge(sd_stats, on=['Store', 'Dept'], how='left')

        # Week-of-year avg (seasonal baseline) — from train only
        week_avg = train_ref.groupby(train_ref['Date'].dt.isocalendar().week.astype(int))['Weekly_Sales'].mean().reset_index()
        week_avg.columns = ['Week', 'week_avg_sales']
        df = df.merge(week_avg, on='Week', how='left')

        # Store-type week avg (cross-sectional seasonal)
        train_ref_copy = train_ref.copy()
        train_ref_copy['Week'] = train_ref_copy['Date'].dt.isocalendar().week.astype(int)
        type_week_avg = train_ref_copy.groupby(['Type', 'Week'])['Weekly_Sales'].mean().reset_index()
        type_week_avg.columns = ['Type', 'Week', 'type_week_avg']
        df = df.merge(type_week_avg, on=['Type', 'Week'], how='left')

        # Holiday week avg per store-dept
        holiday_avg = train_ref[train_ref['IsHoliday'] == True].groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
        holiday_avg.columns = ['Store', 'Dept', 'holiday_mean']
        df = df.merge(holiday_avg, on=['Store', 'Dept'], how='left')

        # Dept-Week interaction (department seasonality)
        dept_week_avg = train_ref_copy.groupby(['Dept', 'Week'])['Weekly_Sales'].mean().reset_index()
        dept_week_avg.columns = ['Dept', 'Week', 'dept_week_avg']
        df = df.merge(dept_week_avg, on=['Dept', 'Week'], how='left')

        # Store-Week interaction
        store_week_avg = train_ref_copy.groupby(['Store', 'Week'])['Weekly_Sales'].mean().reset_index()
        store_week_avg.columns = ['Store', 'Week', 'store_week_avg']
        df = df.merge(store_week_avg, on=['Store', 'Week'], how='left')

        # Ratios (dept relative to store, store-dept relative to store)
        df['dept_vs_store'] = df['dept_mean'] / (df['store_mean'] + 1)
        df['sd_vs_store'] = df['sd_mean'] / (df['store_mean'] + 1)
        df['sd_vs_dept'] = df['sd_mean'] / (df['dept_mean'] + 1)
        df['week_vs_overall'] = df['week_avg_sales'] / (df['store_mean'] + 1)
        df['cv'] = df['sd_std'] / (df['sd_mean'] + 1)
        df['holiday_vs_normal'] = df['holiday_mean'] / (df['sd_mean'] + 1)

    return df

print("Creating features...")
train = create_features(train, train_ref=train)
test = create_features(test, train_ref=train)

# Verify feature columns match
exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]

# Verify test has same feature columns
missing_in_test = set(feature_cols) - set(test.columns)
extra_in_test = set(test.columns) - set(train.columns)
if missing_in_test:
    print(f"WARNING: Features missing in test: {missing_in_test}")
if extra_in_test:
    print(f"WARNING: Extra features in test: {extra_in_test}")

# Use only columns present in both
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"Features: {len(feature_cols)}")

# ============================================================
# 3. Validation: 39 weeks matching test period
# ============================================================
dates = sorted(train['Date'].unique())
n_val = 39
val_dates = set(dates[-n_val:])
trn = train[~train['Date'].isin(val_dates)]
val = train[train['Date'].isin(val_dates)]

print(f"\nTrain: {len(trn)} ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-n_val-1].strftime('%Y-%m-%d')})")
print(f"Val:   {len(val)} ({dates[-n_val].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})")

# ============================================================
# 4. WMAE
# ============================================================
def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

trn_w = np.where(trn['IsHoliday'] == 1, 5, 1)
val_w = np.where(val['IsHoliday'] == 1, 5, 1)

# ============================================================
# 5. LightGBM
# ============================================================
print("\n--- LightGBM ---")
lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbosity': -1,
    'seed': 42,
}

lgb_model = lgb.train(
    lgb_params,
    lgb.Dataset(trn[feature_cols], trn['Weekly_Sales'], weight=trn_w),
    num_boost_round=3000,
    valid_sets=[lgb.Dataset(val[feature_cols], val['Weekly_Sales'], weight=val_w)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
)

lgb_val_pred = np.clip(lgb_model.predict(val[feature_cols]), 0, None)
lgb_test_pred = np.clip(lgb_model.predict(test[feature_cols]), 0, None)
lgb_val_wmae = wmae(val['Weekly_Sales'].values, lgb_val_pred, val['IsHoliday'].values)
print(f"  LightGBM val WMAE: {lgb_val_wmae:.4f}")

# ============================================================
# 6. XGBoost
# ============================================================
print("\n--- XGBoost ---")
xgb_params = {
    'objective': 'reg:pseudohubererror',
    'eval_metric': 'mae',
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbosity': 0,
}

xgb_model = xgb.train(
    xgb_params,
    xgb.DMatrix(trn[feature_cols], label=trn['Weekly_Sales'], weight=trn_w),
    num_boost_round=3000,
    evals=[(xgb.DMatrix(val[feature_cols], label=val['Weekly_Sales'], weight=val_w), 'val')],
    early_stopping_rounds=100,
    verbose_eval=1000,
)

xgb_val_pred = np.clip(xgb_model.predict(xgb.DMatrix(val[feature_cols])), 0, None)
xgb_test_pred = np.clip(xgb_model.predict(xgb.DMatrix(test[feature_cols])), 0, None)
xgb_val_wmae = wmae(val['Weekly_Sales'].values, xgb_val_pred, val['IsHoliday'].values)
print(f"  XGBoost val WMAE: {xgb_val_wmae:.4f}")

# ============================================================
# 7. Ensemble with optimal weight search
# ============================================================
print("\n--- Ensemble ---")
best_wmae = float('inf')
best_w = 0.5
for w in np.arange(0.1, 0.91, 0.05):
    ens_pred = w * lgb_val_pred + (1 - w) * xgb_val_pred
    ens_wmae = wmae(val['Weekly_Sales'].values, ens_pred, val['IsHoliday'].values)
    if ens_wmae < best_wmae:
        best_wmae = ens_wmae
        best_w = w

print(f"Best blend: LGB={best_w:.2f}, XGB={1-best_w:.2f}")
print(f"Ensemble val WMAE: {best_wmae:.4f}")

# Final predictions
test_pred = best_w * lgb_test_pred + (1 - best_w) * xgb_test_pred
test_pred = np.clip(test_pred, 0, None)

submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred
sub_path = OUTPUTS / "submission_r05_no_lag_tuned.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_pred.min():.2f}, {test_pred.max():.2f}], mean={test_pred.mean():.2f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'lgb_importance': lgb_model.feature_importance()
}).sort_values('lgb_importance', ascending=False)
print("\nTop 20 Features (LightGBM):")
print(importance.head(20).to_string(index=False))

print("\n" + "=" * 60)
print("R05 Complete")
print("=" * 60)
