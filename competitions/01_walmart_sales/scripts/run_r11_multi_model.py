"""
Walmart Store Sales Forecasting — R11 Multi-Model Ensemble
R08 is best (LB=2721). Try to improve with:
1. R08's features as base (proven to work)
2. LightGBM + XGBoost + CatBoost ensemble
3. Proper weighted blending based on val scores
4. Seasonal naive as additional ensemble member
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

print("=" * 60)
print("R11: Walmart — Multi-Model Ensemble")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

# R08 features
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
        sd_stats = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['median', 'mean']).reset_index()
        sd_stats.columns = ['Store', 'Dept', 'sd_median', 'sd_mean']
        df = df.merge(sd_stats, on=['Store', 'Dept'], how='left')
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
    return df

print("Creating features...")
train = create_features(train, train_ref=train)
test = create_features(test, train_ref=train)

exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"Features: {len(feature_cols)}")

# Validation
dates = sorted(train['Date'].unique())
val_dates = dates[-20:]
train_dates = dates[:-20]
trn = train[train['Date'].isin(train_dates)]
val = train[train['Date'].isin(val_dates)]
print(f"Train: {len(trn)}, Val: {len(val)}")

def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

trn_w = np.where(trn['IsHoliday'] == 1, 5, 1)
val_w = np.where(val['IsHoliday'] == 1, 5, 1)

# ============================================================
# Model 1: LightGBM
# ============================================================
print("\n--- LightGBM ---")
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
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
)

lgb_val_pred = np.clip(lgb_model.predict(val[feature_cols]), 0, None)
lgb_test_pred = np.clip(lgb_model.predict(test[feature_cols]), 0, None)
lgb_wmae = wmae(val['Weekly_Sales'].values, lgb_val_pred, val['IsHoliday'].values)
print(f"  LGB WMAE: {lgb_wmae:.4f}")

# ============================================================
# Model 2: XGBoost with MAE objective
# ============================================================
print("\n--- XGBoost ---")
xgb_params = {
    'objective': 'reg:absoluteerror',
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
    verbose_eval=False,
)

xgb_val_pred = np.clip(xgb_model.predict(xgb.DMatrix(val[feature_cols])), 0, None)
xgb_test_pred = np.clip(xgb_model.predict(xgb.DMatrix(test[feature_cols])), 0, None)
xgb_wmae = wmae(val['Weekly_Sales'].values, xgb_val_pred, val['IsHoliday'].values)
print(f"  XGB WMAE: {xgb_wmae:.4f}")

# ============================================================
# Model 3: Seasonal Naive (last year's same week sales)
# ============================================================
print("\n--- Seasonal Naive ---")
# For val/test, predict the same week's sales from the previous year
train_c = train.copy()
train_c['Week'] = train_c['Date'].dt.isocalendar().week.astype(int)
train_c['Year'] = train_c['Date'].dt.year

# Get last year's per Store-Dept-Week average
last_year_sales = train_c.groupby(['Store', 'Dept', 'Week'])['Weekly_Sales'].mean().reset_index()
last_year_sales.columns = ['Store', 'Dept', 'Week', 'naive_pred']

val_c = val.copy()
val_c['Week'] = val_c['Date'].dt.isocalendar().week.astype(int)
val_naive = val_c.merge(last_year_sales, on=['Store', 'Dept', 'Week'], how='left')
val_naive['naive_pred'] = val_naive['naive_pred'].fillna(val_naive['sd_median']).fillna(0)
naive_val_pred = np.clip(val_naive['naive_pred'].values, 0, None)
naive_wmae = wmae(val['Weekly_Sales'].values, naive_val_pred, val['IsHoliday'].values)
print(f"  Naive WMAE: {naive_wmae:.4f}")

test_c = test.copy()
test_c['Week'] = test_c['Date'].dt.isocalendar().week.astype(int)
test_naive = test_c.merge(last_year_sales, on=['Store', 'Dept', 'Week'], how='left')
test_naive['naive_pred'] = test_naive['naive_pred'].fillna(test_naive.get('sd_median', 0)).fillna(0)
naive_test_pred = np.clip(test_naive['naive_pred'].values, 0, None)

# ============================================================
# Blending: Optimize weights
# ============================================================
print("\n--- Blending ---")
best_wmae = float('inf')
best_weights = None

for w1 in np.arange(0.5, 0.96, 0.05):
    for w2 in np.arange(0.0, 1.0 - w1 + 0.01, 0.05):
        w3 = 1 - w1 - w2
        if w3 < -0.01:
            continue
        blend = w1 * lgb_val_pred + w2 * xgb_val_pred + max(w3, 0) * naive_val_pred
        blend_wmae = wmae(val['Weekly_Sales'].values, blend, val['IsHoliday'].values)
        if blend_wmae < best_wmae:
            best_wmae = blend_wmae
            best_weights = (w1, w2, max(w3, 0))

w1, w2, w3 = best_weights
print(f"Best: LGB={w1:.2f}, XGB={w2:.2f}, Naive={w3:.2f}")
print(f"Ensemble val WMAE: {best_wmae:.4f}")

# Final predictions
test_final = w1 * lgb_test_pred + w2 * xgb_test_pred + w3 * naive_test_pred
test_final = np.clip(test_final, 0, None)

submission = sample_sub.copy()
submission['Weekly_Sales'] = test_final
sub_path = OUTPUTS / "submission_r11_multi_model.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_final.min():.2f}, {test_final.max():.2f}], mean={test_final.mean():.2f}")

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance()
}).sort_values('importance', ascending=False)
print("\nTop 10 Features:")
print(importance.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("R11 Complete")
print("=" * 60)
