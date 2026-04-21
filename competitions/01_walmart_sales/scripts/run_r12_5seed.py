"""
Walmart Store Sales Forecasting — R12 R08 + 5-seed
R08 is best (LB=2721). Exact same features, just more seeds for averaging.
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
print("R12: R08 exact + 5-seed average")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

train = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')
test = test.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left').merge(stores, on='Store', how='left')

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

train = create_features(train, train_ref=train)
test = create_features(test, train_ref=train)

exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"Features: {len(feature_cols)}")

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

seeds = [42, 123, 456, 789, 2024]
print(f"\n--- {len(seeds)}-seed LightGBM ---")

trn_w = np.where(trn['IsHoliday'] == 1, 5, 1)
val_w = np.where(val['IsHoliday'] == 1, 5, 1)

test_preds = []
for seed in seeds:
    params = {**lgb_params_base, 'seed': seed}
    lgb_trn = lgb.Dataset(trn[feature_cols], trn['Weekly_Sales'], weight=trn_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])
    lgb_val = lgb.Dataset(val[feature_cols], val['Weekly_Sales'], weight=val_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])
    model = lgb.train(params, lgb_trn, num_boost_round=5000,
                      valid_sets=[lgb_val],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    val_pred = np.clip(model.predict(val[feature_cols]), 0, None)
    seed_wmae = wmae(val['Weekly_Sales'].values, val_pred, val['IsHoliday'].values)
    print(f"  Seed {seed}: WMAE={seed_wmae:.4f}, iter={model.best_iteration}")
    test_preds.append(np.clip(model.predict(test[feature_cols]), 0, None))

test_pred = np.mean(test_preds, axis=0)
submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred
sub_path = OUTPUTS / "submission_r12_5seed.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions mean: {test_pred.mean():.2f}")
print("\n" + "=" * 60)
print("R12 Complete")
print("=" * 60)
