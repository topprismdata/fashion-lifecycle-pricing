"""
Walmart Store Sales Forecasting — R02 Lag + Rolling Features
Add time-series features: lag, rolling stats, holiday proximity

Key improvements over R01:
1. Lag features (1/2/3/4/52 weeks)
2. Rolling statistics (4/8/13/52 weeks)
3. Holiday proximity features
4. Store-Dept level aggregated stats
5. Better validation: use same length as test period
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
# 1. Load Data
# ============================================================
print("=" * 60)
print("R02: Walmart — Lag + Rolling Features")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

# ============================================================
# 2. Merge Data
# ============================================================
def merge_all(df, features, stores):
    df = df.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')
    return df

train = merge_all(train, features, stores)
test = merge_all(test, features, stores)

# Combine for lag feature computation (avoid data leakage carefully)
test['Weekly_Sales'] = np.nan
combined = pd.concat([train, test], ignore_index=True)
combined = combined.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

print(f"Combined: {combined.shape}")

# ============================================================
# 3. Feature Engineering
# ============================================================
def create_features(df):
    df = df.copy()

    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter

    # Store type encoding
    df['Type_enc'] = df['Type'].map({'A': 0, 'B': 1, 'C': 2})

    # Holiday flag
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # MarkDown: fill NA + missing indicators
    for i in range(1, 6):
        col = f'MarkDown{i}'
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    # Total markdown
    df['MarkDown_total'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
    df['MarkDown_any'] = (df['MarkDown_total'] > 0).astype(int)

    # Fill other NaN
    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    # Size features
    df['Size_log'] = np.log1p(df['Size'])

    # Store-Dept interaction
    df['Store_Dept'] = df['Store'] * 100 + df['Dept']

    return df

# ============================================================
# 4. Lag + Rolling Features (computed on combined data)
# ============================================================
print("Computing lag features...")

# Sort for proper lag computation
combined = combined.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

# Lag features
for lag in [1, 2, 3, 4, 5, 10, 26, 52]:
    combined[f'sales_lag_{lag}'] = combined.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)

# Rolling features (only using past data)
for window in [4, 8, 13, 26, 52]:
    grp = combined.groupby(['Store', 'Dept'])['Weekly_Sales']
    combined[f'sales_rolling_mean_{window}'] = grp.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    combined[f'sales_rolling_std_{window}'] = grp.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )
    combined[f'sales_rolling_min_{window}'] = grp.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).min()
    )
    combined[f'sales_rolling_max_{window}'] = grp.transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).max()
    )

# Diff features
combined['sales_diff_1'] = combined.groupby(['Store', 'Dept'])['Weekly_Sales'].diff(1)
combined['sales_diff_4'] = combined.groupby(['Store', 'Dept'])['Weekly_Sales'].diff(4)
combined['sales_diff_52'] = combined.groupby(['Store', 'Dept'])['Weekly_Sales'].diff(52)

# YoY (year-over-year) feature
combined['sales_yoy_ratio'] = combined['sales_lag_52'] / (combined['sales_lag_1'] + 1)

# Expanding mean (cumulative average up to that point)
combined['sales_expanding_mean'] = combined.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
)

print(f"Lag features computed. Shape: {combined.shape}")

# ============================================================
# 5. Holiday Proximity Features
# ============================================================
print("Computing holiday features...")

# Define all holiday dates
holiday_dates = [
    # Super Bowl
    '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',
    # Labor Day
    '2010-09-10', '2011-09-09', '2012-09-07',
    # Thanksgiving
    '2010-11-26', '2011-11-25', '2012-11-23',
    # Christmas
    '2010-12-31', '2011-12-30', '2012-12-28',
]
holiday_dates = pd.to_datetime(holiday_dates)

def compute_holiday_proximity(df, holiday_dates):
    df = df.copy()
    df['days_to_nearest_holiday'] = df['Date'].apply(
        lambda d: min(abs((d - h).days) for h in holiday_dates)
    )
    df['weeks_to_nearest_holiday'] = df['days_to_nearest_holiday'] // 7

    # Specific holiday proximity
    super_bowl = pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'])
    thanksgiving = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23'])
    christmas = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28'])

    df['days_to_thanksgiving'] = df['Date'].apply(
        lambda d: min(abs((d - h).days) for h in thanksgiving) if d < thanksgiving.max() + pd.Timedelta(days=60) else 999
    )
    df['days_to_christmas'] = df['Date'].apply(
        lambda d: min(abs((d - h).days) for h in christmas) if d < christmas.max() + pd.Timedelta(days=60) else 999
    )
    return df

combined = compute_holiday_proximity(combined, holiday_dates)

# ============================================================
# 6. Pre-compute time columns for aggregation stats
# ============================================================
combined['Week'] = combined['Date'].dt.isocalendar().week.astype(int)
combined['Type_enc'] = combined['Type'].map({'A': 0, 'B': 1, 'C': 2})

# ============================================================
# 7. Store-Dept Aggregated Stats
# ============================================================
print("Computing store-dept aggregated stats...")

# These are computed on train data only to avoid leakage
train_only = combined[combined['Weekly_Sales'].notna()].copy()

# Store-level stats
store_stats = train_only.groupby('Store')['Weekly_Sales'].agg(['mean', 'std', 'median']).reset_index()
store_stats.columns = ['Store', 'store_mean', 'store_std', 'store_median']
combined = combined.merge(store_stats, on='Store', how='left')

# Dept-level stats
dept_stats = train_only.groupby('Dept')['Weekly_Sales'].agg(['mean', 'std', 'median']).reset_index()
dept_stats.columns = ['Dept', 'dept_mean', 'dept_std', 'dept_median']
combined = combined.merge(dept_stats, on='Dept', how='left')

# Store-Dept level stats
sd_stats = train_only.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'median']).reset_index()
sd_stats.columns = ['Store', 'Dept', 'sd_mean', 'sd_std', 'sd_median']
combined = combined.merge(sd_stats, on=['Store', 'Dept'], how='left')

# Week-of-year avg sales (seasonal baseline)
week_avg = train_only.groupby('Week')['Weekly_Sales'].mean().reset_index()
week_avg.columns = ['Week', 'week_avg_sales']
combined = combined.merge(week_avg, on='Week', how='left')

# Store-Type-Week avg (store type seasonality)
type_week_avg = train_only.groupby(['Type_enc', 'Week'])['Weekly_Sales'].mean().reset_index()
type_week_avg.columns = ['Type_enc', 'Week', 'type_week_avg']
combined = combined.merge(type_week_avg, on=['Type_enc', 'Week'], how='left')

# ============================================================
# 8. Apply create_features and split
# ============================================================
combined = create_features(combined)

# Split back
train_df = combined[combined['Weekly_Sales'].notna()].copy()
test_df = combined[combined['Weekly_Sales'].isna()].copy()

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ============================================================
# 8. Define Feature Columns
# ============================================================
exclude_cols = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

print(f"Total features: {len(feature_cols)}")

# ============================================================
# 9. Validation Strategy
# ============================================================
# Use last ~39 weeks (same as test period: 2012-11-02 to 2013-07-26)
# Train period ends 2012-10-26, so use last 39 weeks as val
train_sorted = train_df.sort_values('Date')
dates = sorted(train_sorted['Date'].unique())
n_val_weeks = 39
val_dates = set(dates[-n_val_weeks:])

trn = train_sorted[~train_sorted['Date'].isin(val_dates)]
val = train_sorted[train_sorted['Date'].isin(val_dates)]

print(f"\nTrain: {len(trn)} rows ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-n_val_weeks-1].strftime('%Y-%m-%d')})")
print(f"Val:   {len(val)} rows ({dates[-n_val_weeks].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})")

# Weights
trn_weights = np.where(trn['IsHoliday'] == 1, 5, 1)
val_weights = np.where(val['IsHoliday'] == 1, 5, 1)

# ============================================================
# 10. WMAE Metric
# ============================================================
def wmae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# ============================================================
# 11. LightGBM Training
# ============================================================
params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.02,
    'num_leaves': 127,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbosity': -1,
    'seed': 42,
}

cat_features = ['Store', 'Dept', 'Store_Dept', 'Type_enc', 'Quarter']

lgb_train = lgb.Dataset(
    trn[feature_cols], trn['Weekly_Sales'],
    weight=trn_weights
)
lgb_val = lgb.Dataset(
    val[feature_cols], val['Weekly_Sales'],
    weight=val_weights, reference=lgb_train
)

print("\nTraining LightGBM...")
model = lgb.train(
    params, lgb_train,
    num_boost_round=5000,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[
        lgb.early_stopping(200),
        lgb.log_evaluation(500),
    ]
)

# ============================================================
# 12. Evaluate
# ============================================================
val_pred = model.predict(val[feature_cols])
val_pred = np.clip(val_pred, 0, None)

val_wmae = wmae(val['Weekly_Sales'].values, val_pred, val_weights)
val_mae = np.mean(np.abs(val['Weekly_Sales'].values - val_pred))

print(f"\nValidation WMAE: {val_wmae:.4f}")
print(f"Validation MAE:  {val_mae:.4f}")

# Holiday-specific
for name, mask in [('Holiday', val['IsHoliday'] == 1), ('Non-Holiday', val['IsHoliday'] == 0)]:
    subset = val[mask]
    pred = np.clip(model.predict(subset[feature_cols]), 0, None)
    mae = np.mean(np.abs(subset['Weekly_Sales'].values - pred))
    print(f"  {name} MAE: {mae:.4f} ({len(subset)} rows)")

# ============================================================
# 13. Feature Importance
# ============================================================
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 20 Features:")
print(importance.head(20).to_string(index=False))

# ============================================================
# 14. Retrain on full data and predict
# ============================================================
print("\nRetraining on full data...")
full_weights = np.where(train_df['IsHoliday'] == 1, 5, 1)
lgb_full = lgb.Dataset(train_df[feature_cols], train_df['Weekly_Sales'], weight=full_weights)

best_iter = model.best_iteration
model_full = lgb.train(params, lgb_full, num_boost_round=best_iter)

test_pred = model_full.predict(test_df[feature_cols])
test_pred = np.clip(test_pred, 0, None)

submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred

sub_path = OUTPUTS / "submission_r02_lag_rolling.csv"
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"Predictions range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
print(f"Predictions mean: {test_pred.mean():.2f}")

print("\n" + "=" * 60)
print("R02 Complete")
print("=" * 60)
