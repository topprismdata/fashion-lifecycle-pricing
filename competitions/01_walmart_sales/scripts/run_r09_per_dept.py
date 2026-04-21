"""
Walmart Store Sales Forecasting — R09 Per-Department + Enhanced YoY
R08 best so far (LB=2721). Try:
1. Per-department models (top solutions' key technique)
2. YoY offsets 51+53 in addition to 52
3. Week-of-year store-dept median (seasonal baseline)
4. Multi-seed averaging (3 seeds)
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
print("R09: Walmart — Per-Dept + Multi-Seed")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

# ============================================================
# Feature Engineering (same as R08)
# ============================================================
def create_features(df, train_ref=None):
    df = df.copy()

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter

    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df['Type_A'] = (df['Type'] == 'A').astype(int)
    df['Type_B'] = (df['Type'] == 'B').astype(int)
    df['Type_C'] = (df['Type'] == 'C').astype(int)

    df['IsHoliday'] = df['IsHoliday'].astype(int)

    for i in range(1, 6):
        col = f'MarkDown{i}'
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    df['MarkDown_total'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
    df['MarkDown_any'] = (df['MarkDown_total'] > 0).astype(int)

    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    df['Size_bin'] = pd.qcut(df['Size'], q=5, labels=False, duplicates='drop')
    df['Size_log'] = np.log1p(df['Size'])
    df['Store_Dept'] = df['Store'] * 100 + df['Dept']

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

    if train_ref is not None:
        # Store-Dept stats
        sd_stats = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['median', 'mean', 'std']).reset_index()
        sd_stats.columns = ['Store', 'Dept', 'sd_median', 'sd_mean', 'sd_std']
        df = df.merge(sd_stats, on=['Store', 'Dept'], how='left')

        # YoY features (offset 52 — proven in R08)
        train_sales = train_ref[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        lookup = train_sales.copy()
        lookup['merge_date'] = lookup['Date'] + pd.Timedelta(weeks=52)
        lookup = lookup.rename(columns={'Weekly_Sales': 'yoy_sales_52'})
        lookup = lookup[['Store', 'Dept', 'merge_date', 'yoy_sales_52']]
        df = df.merge(lookup, left_on=['Store', 'Dept', 'Date'],
                     right_on=['Store', 'Dept', 'merge_date'], how='left')
        if 'merge_date' in df.columns:
            df = df.drop(columns=['merge_date'])
        df['yoy_sales_52'] = df['yoy_sales_52'].fillna(0)

        # Week-of-year median per Store-Dept (seasonal baseline)
        train_ref_c = train_ref.copy()
        train_ref_c['Week'] = train_ref_c['Date'].dt.isocalendar().week.astype(int)
        wk_median = train_ref_c.groupby(['Store', 'Dept', 'Week'])['Weekly_Sales'].median().reset_index()
        wk_median.columns = ['Store', 'Dept', 'Week', 'wk_sd_median']
        df = df.merge(wk_median, on=['Store', 'Dept', 'Week'], how='left')
        df['wk_sd_median'] = df['wk_sd_median'].fillna(df['sd_median'])

        # CV (coefficient of variation)
        df['cv'] = df['sd_std'] / (df['sd_mean'] + 1)

    return df

print("Creating features...")
train = create_features(train, train_ref=train)
test = create_features(test, train_ref=train)

exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"Features: {len(feature_cols)}")

# ============================================================
# Approach A: Global model with multi-seed (like R08 but 3 seeds)
# ============================================================
dates = sorted(train['Date'].unique())
val_dates = dates[-20:]
train_dates = dates[:-20]
trn = train[train['Date'].isin(train_dates)]
val = train[train['Date'].isin(val_dates)]
print(f"Train: {len(trn)}, Val: {len(val)}")

def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

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
print("\n--- Multi-seed global model ---")
val_preds_list = []
test_preds_list = []

for seed in seeds:
    params = {**lgb_params_base, 'seed': seed}
    trn_w = np.where(trn['IsHoliday'] == 1, 5, 1)
    val_w = np.where(val['IsHoliday'] == 1, 5, 1)

    lgb_trn = lgb.Dataset(trn[feature_cols], trn['Weekly_Sales'], weight=trn_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])
    lgb_val = lgb.Dataset(val[feature_cols], val['Weekly_Sales'], weight=val_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])

    model = lgb.train(
        params, lgb_trn,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )

    val_pred = np.clip(model.predict(val[feature_cols]), 0, None)
    test_pred = np.clip(model.predict(test[feature_cols]), 0, None)
    seed_wmae = wmae(val['Weekly_Sales'].values, val_pred, val['IsHoliday'].values)
    print(f"  Seed {seed}: WMAE={seed_wmae:.4f}, iter={model.best_iteration}")

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)

val_pred_avg = np.mean(val_preds_list, axis=0)
test_pred_avg = np.mean(test_preds_list, axis=0)
val_wmae_avg = wmae(val['Weekly_Sales'].values, val_pred_avg, val['IsHoliday'].values)
print(f"  Seed-avg val WMAE: {val_wmae_avg:.4f}")

# ============================================================
# Approach B: Per-department seasonal naive + blend
# ============================================================
print("\n--- Per-dept seasonal naive ---")
# For each (Store, Dept, Week), predict last year's value for that week
train_ref_c = train.copy()
train_ref_c['Week'] = train_ref_c['Date'].dt.isocalendar().week.astype(int)

# Get the last year's sales for each Store-Dept-Week
last_year = train_ref_c[train_ref_c['Date'] >= '2011-11-01'].groupby(['Store', 'Dept', 'Week'])['Weekly_Sales'].mean().reset_index()
last_year.columns = ['Store', 'Dept', 'Week', 'naive_pred']

# Apply to validation
val_with_week = val.copy()
val_with_week['Week'] = val_with_week['Date'].dt.isocalendar().week.astype(int)
val_naive = val_with_week.merge(last_year, on=['Store', 'Dept', 'Week'], how='left')
val_naive['naive_pred'] = val_naive['naive_pred'].fillna(val_naive['sd_median'])
naive_wmae = wmae(val['Weekly_Sales'].values, np.clip(val_naive['naive_pred'].values, 0, None), val['IsHoliday'].values)
print(f"  Naive val WMAE: {naive_wmae:.4f}")

# Blend: LGB + seasonal naive
print("\n--- Blending ---")
best_wmae = float('inf')
best_w = 0
for w in np.arange(0.7, 1.0, 0.02):
    blend = w * val_pred_avg + (1 - w) * np.clip(val_naive['naive_pred'].values, 0, None)
    blend_wmae = wmae(val['Weekly_Sales'].values, blend, val['IsHoliday'].values)
    if blend_wmae < best_wmae:
        best_wmae = blend_wmae
        best_w = w

print(f"Best blend: LGB={best_w:.2f}, Naive={1-best_w:.2f}, WMAE={best_wmae:.4f}")

# ============================================================
# Retrain on full data + submit
# ============================================================
print("\n--- Retraining on full data (3 seeds) ---")
full_test_preds = []

for seed in seeds:
    params = {**lgb_params_base, 'seed': seed}
    full_w = np.where(train['IsHoliday'] == 1, 5, 1)
    lgb_full = lgb.Dataset(train[feature_cols], train['Weekly_Sales'], weight=full_w,
                           categorical_feature=['Store', 'Dept', 'Store_Dept'])
    model_full = lgb.train(params, lgb_full, num_boost_round=2000)
    full_test_preds.append(np.clip(model_full.predict(test[feature_cols]), 0, None))
    print(f"  Seed {seed} done")

lgb_test_final = np.mean(full_test_preds, axis=0)

# Get naive predictions for test
test_with_week = test.copy()
test_with_week['Week'] = test_with_week['Date'].dt.isocalendar().week.astype(int)
test_naive = test_with_week.merge(last_year, on=['Store', 'Dept', 'Week'], how='left')
test_naive['naive_pred'] = test_naive['naive_pred'].fillna(0)

# Final blend
test_final = best_w * lgb_test_final + (1 - best_w) * np.clip(test_naive['naive_pred'].values, 0, None)
test_final = np.clip(test_final, 0, None)

submission = sample_sub.copy()
submission['Weekly_Sales'] = test_final
sub_path = OUTPUTS / "submission_r09_per_dept_blend.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_final.min():.2f}, {test_final.max():.2f}], mean={test_final.mean():.2f}")

print("\n" + "=" * 60)
print("R09 Complete")
print("=" * 60)
