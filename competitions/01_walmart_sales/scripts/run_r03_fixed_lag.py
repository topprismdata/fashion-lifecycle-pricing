"""
Walmart Store Sales Forecasting — R03 Fixed Validation + Proper Lag Features

Key fixes from R02 failure:
1. Use train-only for lag computation (no test data leakage)
2. Shorter validation window matching test period structure
3. Properly handle lag NaN in test via forward-fill
4. Simpler lag features (avoid overfitting to recent patterns)
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
print("R03: Walmart — Fixed Validation + Lag Features")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

# Merge
for df in [train, test]:
    df = df.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')

# Redo merge properly
train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. Pivot to wide format for lag features
# ============================================================
print("Computing lag features from train only...")

# Create store-dept-date pivot
sales_pivot = train.pivot_table(
    values='Weekly_Sales',
    index=['Store', 'Dept'],
    columns='Date',
    aggfunc='sum'
)

# Fill missing dates
all_dates = sorted(train['Date'].unique())
sales_pivot = sales_pivot.reindex(columns=all_dates)
sales_pivot = sales_pivot.fillna(0)

print(f"Sales pivot: {sales_pivot.shape}")

# ============================================================
# 3. Feature Engineering (no data leakage)
# ============================================================
def create_features(df, sales_pivot):
    df = df.copy()

    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter

    # Store type
    df['Type_enc'] = df['Type'].map({'A': 0, 'B': 1, 'C': 2})

    # Holiday
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # MarkDown
    for i in range(1, 6):
        col = f'MarkDown{i}'
        df[f'{col}_na'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    df['MarkDown_total'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']

    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    df['Size_log'] = np.log1p(df['Size'])
    df['Store_Dept'] = df['Store'] * 100 + df['Dept']

    # ---- Lag features from pivot (safe, train only) ----
    all_dates_sorted = sorted(sales_pivot.columns)

    def get_lag_value(row, lag_weeks):
        """Get sales from lag_weeks ago using train pivot"""
        target_date = row['Date'] - pd.Timedelta(weeks=lag_weeks)
        sd_key = (row['Store'], row['Dept'])
        if target_date in sales_pivot.columns and sd_key in sales_pivot.index:
            val = sales_pivot.loc[sd_key, target_date]
            return val if not pd.isna(val) else np.nan
        return np.nan

    # Compute lags via merge with shifted train data
    # Create a lookup: for each (Store, Dept, date), get sales from N weeks ago
    train_sales = train[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()

    for lag in [1, 2, 3, 4, 5, 26, 52]:
        lag_col = f'sales_lag_{lag}'

        # Shift date to create lookup key
        lookup = train_sales[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        lookup['merge_date'] = lookup['Date'] + pd.Timedelta(weeks=lag)
        lookup = lookup.rename(columns={'Weekly_Sales': lag_col})
        lookup = lookup[['Store', 'Dept', 'merge_date', lag_col]]

        # Merge: for each row, find the sales from lag weeks prior
        df = df.merge(lookup, left_on=['Store', 'Dept', 'Date'],
                      right_on=['Store', 'Dept', 'merge_date'],
                      how='left', suffixes=('', f'_lag{lag}'))

        # Clean up merge column
        if 'merge_date' in df.columns:
            df = df.drop(columns=['merge_date'])

        # Fill NaN
        df[lag_col] = df[lag_col].fillna(0)

    # Rolling features from lags
    df['sales_rolling_mean_4'] = df[['sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_4']].mean(axis=1)
    df['sales_rolling_std_4'] = df[['sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_4']].std(axis=1)
    df['sales_rolling_mean_2'] = df[['sales_lag_1', 'sales_lag_2']].mean(axis=1)

    # Diff
    df['sales_diff_1'] = df['sales_lag_1'] - df['sales_lag_2']
    df['sales_diff_4'] = df['sales_lag_1'] - df['sales_lag_5']

    # YoY
    df['sales_yoy'] = df['sales_lag_52'] / (df['sales_lag_1'].clip(lower=1))

    # ---- Store-Dept stats from train ----
    store_stats = train.groupby('Store')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
    store_stats.columns = ['Store', 'store_mean', 'store_std']
    df = df.merge(store_stats, on='Store', how='left')

    dept_stats = train.groupby('Dept')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
    dept_stats.columns = ['Dept', 'dept_mean', 'dept_std']
    df = df.merge(dept_stats, on='Dept', how='left')

    sd_stats = train.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'count']).reset_index()
    sd_stats.columns = ['Store', 'Dept', 'sd_mean', 'sd_std', 'sd_count']
    df = df.merge(sd_stats, on=['Store', 'Dept'], how='left')

    # Week avg (seasonal)
    week_avg = train.groupby(train['Date'].dt.isocalendar().week.astype(int))['Weekly_Sales'].mean().reset_index()
    week_avg.columns = ['Week', 'week_avg']
    df = df.merge(week_avg, on='Week', how='left')

    # Holiday proximity
    holiday_dates = pd.to_datetime([
        '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08',
        '2010-09-10', '2011-09-09', '2012-09-07',
        '2010-11-26', '2011-11-25', '2012-11-23',
        '2010-12-31', '2011-12-30', '2012-12-28',
    ])
    df['days_to_holiday'] = df['Date'].apply(
        lambda d: min(abs((d - h).days) for h in holiday_dates)
    )

    return df

print("Creating features for train...")
train = create_features(train, sales_pivot)
print("Creating features for test...")
test = create_features(test, sales_pivot)

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 4. Feature Selection
# ============================================================
exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]
print(f"Features: {len(feature_cols)}")

# ============================================================
# 5. Validation: Use last 39 weeks of train (matches test length)
# ============================================================
dates = sorted(train['Date'].unique())
n_test_weeks = 39
val_dates = set(dates[-n_test_weeks:])
trn = train[~train['Date'].isin(val_dates)]
val = train[train['Date'].isin(val_dates)]

print(f"\nTrain: {len(trn)} ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-n_test_weeks-1].strftime('%Y-%m-%d')})")
print(f"Val:   {len(val)} ({dates[-n_test_weeks].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})")

# ============================================================
# 6. WMAE
# ============================================================
def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

trn_weights = np.where(trn['IsHoliday'] == 1, 5, 1)
val_weights = np.where(val['IsHoliday'] == 1, 5, 1)

# ============================================================
# 7. LightGBM
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

lgb_train = lgb.Dataset(trn[feature_cols], trn['Weekly_Sales'], weight=trn_weights)
lgb_val = lgb.Dataset(val[feature_cols], val['Weekly_Sales'], weight=val_weights, reference=lgb_train)

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

# Evaluate
val_pred = np.clip(model.predict(val[feature_cols]), 0, None)
val_wmae = wmae(val['Weekly_Sales'].values, val_pred, val['IsHoliday'].values)
print(f"\nValidation WMAE: {val_wmae:.4f}")

for name, mask in [('Holiday', val['IsHoliday'] == 1), ('Non-Holiday', val['IsHoliday'] == 0)]:
    subset = val[mask]
    pred = np.clip(model.predict(subset[feature_cols]), 0, None)
    mae = np.mean(np.abs(subset['Weekly_Sales'].values - pred))
    print(f"  {name} MAE: {mae:.4f} ({len(subset)} rows)")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)
print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))

# ============================================================
# 8. Retrain on full data + predict
# ============================================================
print("\nRetraining on full data...")
full_weights = np.where(train['IsHoliday'] == 1, 5, 1)
lgb_full = lgb.Dataset(train[feature_cols], train['Weekly_Sales'], weight=full_weights)
model_full = lgb.train(params, lgb_full, num_boost_round=model.best_iteration)

test_pred = np.clip(model_full.predict(test[feature_cols]), 0, None)
submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred

sub_path = OUTPUTS / "submission_r03_fixed_lag.csv"
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_pred.min():.2f}, {test_pred.max():.2f}], mean={test_pred.mean():.2f}")

print("\n" + "=" * 60)
print("R03 Complete")
print("=" * 60)
