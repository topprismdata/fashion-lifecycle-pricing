"""
Walmart Store Sales Forecasting — R06 Minimal Plus
R05 showed 69 features scored worse than R01's 28 (LB 3196 vs 3102).
Lesson: aggregated stats from train can overfit to CV period.

Strategy: Start from R01's exact feature set, add ONLY:
1. Seasonal sin/cos encoding (mathematically safe, no data dependency)
2. Holiday proximity (calendar-based, always valid)
3. Store-Dept median from train (single most robust stat)
4. Better hyperparameters: more trees, lower learning rate
5. 20-week validation (matching R01's successful setup)
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

# ============================================================
# 1. Load & Merge
# ============================================================
print("=" * 60)
print("R06: Walmart — Minimal Plus (R01 + targeted additions)")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

# ============================================================
# 2. Feature Engineering
# ============================================================
def create_features(df, train_ref=None):
    df = df.copy()

    # --- R01 Original Features ---
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    df['Type_A'] = (df['Type'] == 'A').astype(int)
    df['Type_B'] = (df['Type'] == 'B').astype(int)
    df['Type_C'] = (df['Type'] == 'C').astype(int)

    df['IsHoliday'] = df['IsHoliday'].astype(int)

    for i in range(1, 6):
        col = f'MarkDown{i}'
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    df['Size_bin'] = pd.qcut(df['Size'], q=5, labels=False, duplicates='drop')
    df['Store_Dept'] = df['Store'] * 100 + df['Dept']

    # --- New Features (carefully chosen, safe for out-of-sample) ---

    # Seasonal encoding (mathematical, no data dependency)
    df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Quarter
    df['Quarter'] = df['Date'].dt.quarter

    # Holiday proximity (calendar-based)
    holiday_dates = pd.to_datetime([
        '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',
        '2010-09-10', '2011-09-09', '2012-09-07',
        '2010-11-26', '2011-11-25', '2012-11-23',
        '2010-12-31', '2011-12-30', '2012-12-28',
    ])
    df['days_to_holiday'] = df['Date'].apply(lambda d: min(abs((d - h).days) for h in holiday_dates))
    df['weeks_to_holiday'] = df['days_to_holiday'] // 7
    df['near_holiday_2w'] = (df['weeks_to_holiday'] <= 2).astype(int)

    # MarkDown aggregates (safe — based on test's own MarkDown values)
    df['MarkDown_total'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
    df['MarkDown_any'] = (df['MarkDown_total'] > 0).astype(int)

    # Size log
    df['Size_log'] = np.log1p(df['Size'])

    # --- SINGLE most robust stat: Store-Dept median from train ---
    if train_ref is not None:
        sd_median = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].median().reset_index()
        sd_median.columns = ['Store', 'Dept', 'sd_median']
        df = df.merge(sd_median, on=['Store', 'Dept'], how='left')

    return df

print("Creating features...")
train = create_features(train, train_ref=train)
test = create_features(test, train_ref=train)

# Verify columns match
exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]
missing_in_test = set(feature_cols) - set(test.columns)
if missing_in_test:
    print(f"WARNING: Features missing in test: {missing_in_test}")
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"Features: {len(feature_cols)}")

# ============================================================
# 3. Validation: 20 weeks (R01's successful setup)
# ============================================================
dates = sorted(train['Date'].unique())
val_dates = dates[-20:]
train_dates = dates[:-20]

trn = train[train['Date'].isin(train_dates)]
val = train[train['Date'].isin(val_dates)]

print(f"\nTrain: {len(trn)} ({train_dates[0].strftime('%Y-%m-%d')} ~ {train_dates[-1].strftime('%Y-%m-%d')})")
print(f"Val:   {len(val)} ({val_dates[0].strftime('%Y-%m-%d')} ~ {val_dates[-1].strftime('%Y-%m-%d')})")

# ============================================================
# 4. WMAE
# ============================================================
def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

trn_w = np.where(trn['IsHoliday'] == 1, 5, 1)
val_w = np.where(val['IsHoliday'] == 1, 5, 1)

# ============================================================
# 5. LightGBM with tuned parameters
# ============================================================
print("\n--- LightGBM (Tuned) ---")

lgb_params = {
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
    'seed': 42,
}

lgb_model = lgb.train(
    lgb_params,
    lgb.Dataset(trn[feature_cols], trn['Weekly_Sales'], weight=trn_w,
                categorical_feature=['Store', 'Dept', 'Store_Dept']),
    num_boost_round=5000,
    valid_sets=[lgb.Dataset(val[feature_cols], val['Weekly_Sales'], weight=val_w,
                            categorical_feature=['Store', 'Dept', 'Store_Dept'])],
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
)

val_pred = np.clip(lgb_model.predict(val[feature_cols]), 0, None)
val_wmae = wmae(val['Weekly_Sales'].values, val_pred, val['IsHoliday'].values)
print(f"\nVal WMAE: {val_wmae:.4f}")

# Holiday breakdown
for name, mask in [('Holiday', val['IsHoliday'] == 1), ('Non-Holiday', val['IsHoliday'] == 0)]:
    subset = val[mask]
    pred = np.clip(lgb_model.predict(subset[feature_cols]), 0, None)
    mae = np.mean(np.abs(subset['Weekly_Sales'].values - pred))
    print(f"  {name} MAE: {mae:.4f} ({len(subset)} rows)")

# ============================================================
# 6. Retrain on full data + predict
# ============================================================
print("\nRetraining on full data...")
full_w = np.where(train['IsHoliday'] == 1, 5, 1)
lgb_full = lgb.Dataset(train[feature_cols], train['Weekly_Sales'], weight=full_w,
                        categorical_feature=['Store', 'Dept', 'Store_Dept'])
model_full = lgb.train(lgb_params, lgb_full, num_boost_round=lgb_model.best_iteration)

test_pred = np.clip(model_full.predict(test[feature_cols]), 0, None)
submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred

sub_path = OUTPUTS / "submission_r06_minimal_plus.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_pred.min():.2f}, {test_pred.max():.2f}], mean={test_pred.mean():.2f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance()
}).sort_values('importance', ascending=False)
print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("R06 Complete")
print("=" * 60)
