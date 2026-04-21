"""
Walmart Store Sales Forecasting — R10 Leak-Free
R09 had data leakage: stats computed from full train (including val weeks).
Fix: compute stats from train-only (excluding val), then apply to both val and test.

Key insight from R06 (LB=2815) and R08 (LB=2721): simple features with proper
no-leakage computation is essential for out-of-sample generalization.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("=" * 60)
print("R10: Walmart — Leak-Free + Multi-Seed")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

# ============================================================
# Split FIRST, then compute features
# ============================================================
dates = sorted(train['Date'].unique())
val_dates = set(dates[-20:])
train_only = train[~train['Date'].isin(val_dates)].copy()
val_only = train[train['Date'].isin(val_dates)].copy()

print(f"Train-only: {len(train_only)}, Val: {len(val_only)}")

def create_features(df, stats_ref=None):
    """Create features. stats_ref is used for aggregated stats (must exclude val/test data)."""
    df = df.copy()

    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter

    # Seasonal encoding
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Store type
    df['Type_A'] = (df['Type'] == 'A').astype(int)
    df['Type_B'] = (df['Type'] == 'B').astype(int)
    df['Type_C'] = (df['Type'] == 'C').astype(int)

    # Holiday
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # MarkDown
    for i in range(1, 6):
        col = f'MarkDown{i}'
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)
    df['MarkDown_total'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
    df['MarkDown_any'] = (df['MarkDown_total'] > 0).astype(int)

    # Fill NaN
    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    df['Size_bin'] = pd.qcut(df['Size'], q=5, labels=False, duplicates='drop')
    df['Size_log'] = np.log1p(df['Size'])
    df['Store_Dept'] = df['Store'] * 100 + df['Dept']

    # Holiday proximity
    holiday_dates = pd.to_datetime([
        '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',
        '2010-09-10', '2011-09-09', '2012-09-07',
        '2010-11-26', '2011-11-25', '2012-11-23',
        '2010-12-31', '2011-12-30', '2012-12-28',
    ])
    df['days_to_holiday'] = df['Date'].apply(lambda d: min(abs((d - h).days) for h in holiday_dates))
    df['weeks_to_holiday'] = df['days_to_holiday'] // 7
    df['near_holiday_2w'] = (df['weeks_to_holiday'] <= 2).astype(int)

    df['Fuel_Price_change'] = df.groupby('Store')['Fuel_Price'].diff()
    df['Fuel_Price_change'] = df['Fuel_Price_change'].fillna(0)

    # --- Stats from stats_ref (LEAK-FREE: stats_ref excludes val data) ---
    if stats_ref is not None:
        # Store-Dept stats
        sd_stats = stats_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['median', 'mean']).reset_index()
        sd_stats.columns = ['Store', 'Dept', 'sd_median', 'sd_mean']
        df = df.merge(sd_stats, on=['Store', 'Dept'], how='left')

        # YoY (offset 52) from stats_ref
        train_sales = stats_ref[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        lookup = train_sales.copy()
        lookup['merge_date'] = lookup['Date'] + pd.Timedelta(weeks=52)
        lookup = lookup.rename(columns={'Weekly_Sales': 'yoy_sales_52'})
        lookup = lookup[['Store', 'Dept', 'merge_date', 'yoy_sales_52']]
        df = df.merge(lookup, left_on=['Store', 'Dept', 'Date'],
                     right_on=['Store', 'Dept', 'merge_date'], how='left')
        if 'merge_date' in df.columns:
            df = df.drop(columns=['merge_date'])
        df['yoy_sales_52'] = df['yoy_sales_52'].fillna(0)

        # Week-of-year median per Store-Dept (from stats_ref only)
        stats_ref_c = stats_ref.copy()
        stats_ref_c['Week'] = stats_ref_c['Date'].dt.isocalendar().week.astype(int)
        wk_median = stats_ref_c.groupby(['Store', 'Dept', 'Week'])['Weekly_Sales'].median().reset_index()
        wk_median.columns = ['Store', 'Dept', 'Week', 'wk_sd_median']
        df = df.merge(wk_median, on=['Store', 'Dept', 'Week'], how='left')
        # Fill missing weeks with sd_median
        df['wk_sd_median'] = df['wk_sd_median'].fillna(df.get('sd_median', 0))

    return df

print("Creating features (leak-free)...")
train_feat = create_features(train_only, stats_ref=train_only)
val_feat = create_features(val_only, stats_ref=train_only)
test_feat = create_features(test, stats_ref=train_only)  # For final: use full train

# Also create test features using full train for final submission
test_feat_full = create_features(test, stats_ref=train)  # Full train for test

exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train_feat.columns if c not in exclude]
feature_cols = [c for c in feature_cols if c in val_feat.columns and c in test_feat.columns]
print(f"Features: {len(feature_cols)}")

# ============================================================
# WMAE
# ============================================================
def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

# ============================================================
# Multi-seed training
# ============================================================
lgb_params_base = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.02,
    'num_leaves': 127,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbosity': -1,
}

seeds = [42, 123, 456]
print("\n--- Multi-seed LightGBM (leak-free) ---")

trn_w = np.where(train_feat['IsHoliday'] == 1, 5, 1)
val_w = np.where(val_feat['IsHoliday'] == 1, 5, 1)

val_preds_list = []
test_preds_list = []

for seed in seeds:
    params = {**lgb_params_base, 'seed': seed}

    lgb_trn = lgb.Dataset(train_feat[feature_cols], train_feat['Weekly_Sales'], weight=trn_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])
    lgb_val = lgb.Dataset(val_feat[feature_cols], val_feat['Weekly_Sales'], weight=val_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])

    model = lgb.train(
        params, lgb_trn,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )

    val_pred = np.clip(model.predict(val_feat[feature_cols]), 0, None)
    test_pred = np.clip(model.predict(test_feat[feature_cols]), 0, None)
    seed_wmae = wmae(val_feat['Weekly_Sales'].values, val_pred, val_feat['IsHoliday'].values)
    print(f"  Seed {seed}: WMAE={seed_wmae:.4f}, iter={model.best_iteration}")

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)

val_pred_avg = np.mean(val_preds_list, axis=0)
test_pred_avg = np.mean(test_preds_list, axis=0)
val_wmae_avg = wmae(val_feat['Weekly_Sales'].values, val_pred_avg, val_feat['IsHoliday'].values)
print(f"  Seed-avg val WMAE: {val_wmae_avg:.4f}")

# ============================================================
# Retrain on full train data with all stats
# ============================================================
print("\n--- Retraining on full data ---")
train_full = create_features(train, stats_ref=train)  # Full train, stats from full train

full_test_preds = []
for seed in seeds:
    params = {**lgb_params_base, 'seed': seed}
    full_w = np.where(train_full['IsHoliday'] == 1, 5, 1)
    lgb_full = lgb.Dataset(train_full[feature_cols], train_full['Weekly_Sales'], weight=full_w,
                           categorical_feature=['Store', 'Dept', 'Store_Dept'])
    model_full = lgb.train(params, lgb_full, num_boost_round=2000)
    full_test_preds.append(np.clip(model_full.predict(test_feat_full[feature_cols]), 0, None))
    print(f"  Seed {seed} done")

test_final = np.mean(full_test_preds, axis=0)
test_final = np.clip(test_final, 0, None)

submission = sample_sub.copy()
submission['Weekly_Sales'] = test_final
sub_path = OUTPUTS / "submission_r10_leak_free.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_final.min():.2f}, {test_final.max():.2f}], mean={test_final.mean():.2f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)
print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("R10 Complete")
print("=" * 60)
