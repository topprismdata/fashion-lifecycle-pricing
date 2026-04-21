"""
Walmart Store Sales Forecasting — R01 Baseline
LightGBM with basic feature engineering

WMAE metric: holiday weeks weighted 5x, non-holiday 1x
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
print("R01: Walmart Baseline — LightGBM")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. Merge Data
# ============================================================
def merge_data(df, features, stores):
    df = df.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = df.merge(stores, on='Store', how='left')
    return df

train = merge_data(train, features, stores)
test = merge_data(test, features, stores)

print(f"Train after merge: {train.shape}")

# ============================================================
# 3. Feature Engineering
# ============================================================
def create_features(df):
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Store type encoding
    df['Type_A'] = (df['Type'] == 'A').astype(int)
    df['Type_B'] = (df['Type'] == 'B').astype(int)
    df['Type_C'] = (df['Type'] == 'C').astype(int)

    # Holiday flag as int
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # MarkDown missing indicators
    for i in range(1, 6):
        col = f'MarkDown{i}'
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(0)

    # Fill other NaN
    df['CPI'] = df['CPI'].fillna(df['CPI'].median())
    df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].median())

    # Size bins
    df['Size_bin'] = pd.qcut(df['Size'], q=5, labels=False, duplicates='drop')

    # Store-Dept interaction
    df['Store_Dept'] = df['Store'] * 100 + df['Dept']

    return df

train = create_features(train)
test = create_features(test)

# ============================================================
# 4. Define Features
# ============================================================
feature_cols = [
    'Store', 'Dept', 'IsHoliday',
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'MarkDown1_missing', 'MarkDown2_missing', 'MarkDown3_missing',
    'MarkDown4_missing', 'MarkDown5_missing',
    'Type_A', 'Type_B', 'Type_C', 'Size', 'Size_bin',
    'Year', 'Month', 'Week', 'Day', 'DayOfWeek',
    'Store_Dept',
]

print(f"Features: {len(feature_cols)}")

# ============================================================
# 5. Validation: Time-based split
# ============================================================
# Use last ~20 weeks as validation (similar to test period length)
train_sorted = train.sort_values('Date').reset_index(drop=True)
dates = train_sorted['Date'].unique()
dates_sorted = sorted(dates)

# Split: last 20 weeks for validation
val_dates = dates_sorted[-20:]
train_dates = dates_sorted[:-20]

trn = train_sorted[train_sorted['Date'].isin(train_dates)]
val = train_sorted[train_sorted['Date'].isin(val_dates)]

print(f"Train: {len(trn)} rows ({train_dates[0].strftime('%Y-%m-%d')} ~ {train_dates[-1].strftime('%Y-%m-%d')})")
print(f"Val:   {len(val)} rows ({val_dates[0].strftime('%Y-%m-%d')} ~ {val_dates[-1].strftime('%Y-%m-%d')})")

# ============================================================
# 6. WMAE Metric
# ============================================================
def wmae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# Sample weights: 5x for holiday
trn_weights = np.where(trn['IsHoliday'] == 1, 5, 1)
val_weights = np.where(val['IsHoliday'] == 1, 5, 1)

# ============================================================
# 7. LightGBM Training
# ============================================================
params = {
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

lgb_train = lgb.Dataset(
    trn[feature_cols], trn['Weekly_Sales'],
    weight=trn_weights, categorical_feature=['Store', 'Dept', 'Store_Dept']
)
lgb_val = lgb.Dataset(
    val[feature_cols], val['Weekly_Sales'],
    weight=val_weights, reference=lgb_train,
    categorical_feature=['Store', 'Dept', 'Store_Dept']
)

print("\nTraining LightGBM...")
model = lgb.train(
    params, lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(200),
    ]
)

# ============================================================
# 8. Evaluate
# ============================================================
val_pred = model.predict(val[feature_cols])
val_pred = np.clip(val_pred, 0, None)  # clip negative predictions

val_wmae = wmae(val['Weekly_Sales'].values, val_pred, val_weights)
val_mae = np.mean(np.abs(val['Weekly_Sales'].values - val_pred))

print(f"\nValidation WMAE: {val_wmae:.4f}")
print(f"Validation MAE:  {val_mae:.4f}")

# Holiday-specific
val_holiday = val[val['IsHoliday'] == 1]
val_nonholiday = val[val['IsHoliday'] == 0]
holiday_pred = np.clip(model.predict(val_holiday[feature_cols]), 0, None)
nonholiday_pred = np.clip(model.predict(val_nonholiday[feature_cols]), 0, None)

if len(val_holiday) > 0:
    holiday_mae = np.mean(np.abs(val_holiday['Weekly_Sales'].values - holiday_pred))
    print(f"Holiday MAE: {holiday_mae:.4f} ({len(val_holiday)} rows)")
if len(val_nonholiday) > 0:
    nonholiday_mae = np.mean(np.abs(val_nonholiday['Weekly_Sales'].values - nonholiday_pred))
    print(f"Non-Holiday MAE: {nonholiday_mae:.4f} ({len(val_nonholiday)} rows)")

# ============================================================
# 9. Feature Importance
# ============================================================
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))

# ============================================================
# 10. Generate Submission
# ============================================================
test_pred = model.predict(test[feature_cols])
test_pred = np.clip(test_pred, 0, None)

submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred

sub_path = OUTPUTS / "submission_r01_baseline.csv"
submission.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")
print(f"Predictions range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
print(f"Predictions mean: {test_pred.mean():.2f}")

print("\n" + "=" * 60)
print("R01 Baseline Complete")
print("=" * 60)
