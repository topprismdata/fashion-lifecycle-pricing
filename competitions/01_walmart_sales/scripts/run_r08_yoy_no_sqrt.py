"""
Walmart Store Sales Forecasting — R08 YoY No-Sqrt
R07 failed (LB=3534) due to sqrt transform changing loss landscape.
This version: R06 foundation + YoY features WITHOUT sqrt transform.

Key question: Do YoY features (same-week-last-year) actually help?
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
print("R08: Walmart — R06 + YoY (No sqrt)")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

# ============================================================
# Feature Engineering — R06 features + YoY
# ============================================================
def create_features(df, train_ref=None):
    df = df.copy()

    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Seasonal encoding
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
    df['Quarter'] = df['Date'].dt.quarter

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

    # Fuel_Price change
    df['Fuel_Price_change'] = df.groupby('Store')['Fuel_Price'].diff()
    df['Fuel_Price_change'] = df['Fuel_Price_change'].fillna(0)

    if train_ref is not None:
        # Store-Dept median (R06's best stat)
        sd_median = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].median().reset_index()
        sd_median.columns = ['Store', 'Dept', 'sd_median']
        df = df.merge(sd_median, on=['Store', 'Dept'], how='left')

        # Store-Dept mean
        sd_mean = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
        sd_mean.columns = ['Store', 'Dept', 'sd_mean']
        df = df.merge(sd_mean, on=['Store', 'Dept'], how='left')

        # Same-week-last-year (offset 52 only — simplest, most direct)
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

        # Check what fraction of rows got valid YoY values
        if 'Date' in df.columns:
            test_min = pd.Timestamp('2012-11-02')
            yoy_test = df[df['Date'] >= test_min]
            if len(yoy_test) > 0:
                yoy_valid = (yoy_test['yoy_sales_52'] != 0).mean()
                print(f"  YoY52 valid rate in test period: {yoy_valid:.1%}")

    return df

print("Creating features...")
train = create_features(train, train_ref=train)
test = create_features(test, train_ref=train)

exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"Features: {len(feature_cols)}")

# Validation: 20 weeks
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
# LightGBM (same as R06, no sqrt, no multi-seed)
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
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
)

val_pred = np.clip(lgb_model.predict(val[feature_cols]), 0, None)
val_wmae_score = wmae(val['Weekly_Sales'].values, val_pred, val['IsHoliday'].values)
print(f"\nVal WMAE: {val_wmae_score:.4f}")

# Retrain on full data
print("Retraining on full data...")
full_w = np.where(train['IsHoliday'] == 1, 5, 1)
lgb_full = lgb.Dataset(train[feature_cols], train['Weekly_Sales'], weight=full_w,
                        categorical_feature=['Store', 'Dept', 'Store_Dept'])
model_full = lgb.train(lgb_params, lgb_full, num_boost_round=lgb_model.best_iteration)

test_pred = np.clip(model_full.predict(test[feature_cols]), 0, None)
submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred
sub_path = OUTPUTS / "submission_r08_yoy_no_sqrt.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_pred.min():.2f}, {test_pred.max():.2f}], mean={test_pred.mean():.2f}")

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance()
}).sort_values('importance', ascending=False)
print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("R08 Complete")
print("=" * 60)
