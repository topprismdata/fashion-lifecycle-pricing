"""
Walmart Store Sales Forecasting — R07 YoY + Christmas Shift
Based on top solutions research, key improvements over R06 (LB=2815):
1. Same-week-last-year features (offset 51, 52, 53 weeks — SAFE, references train)
2. Christmas shift post-processing (critical for weeks 48-52)
3. sqrt target transform (stabilize variance)
4. Multiple seed averaging (5 seeds)
5. R06's proven features as foundation
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
print("R07: Walmart — YoY + Christmas Shift + Seed Avg")
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

    # Specific holiday proximity
    thanksgivings = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23'])
    christmases = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28'])
    df['near_thanksgiving'] = df['Date'].apply(lambda d: any(abs((d - h).days) <= 14 for h in thanksgivings)).astype(int)
    df['near_christmas'] = df['Date'].apply(lambda d: any(abs((d - h).days) <= 14 for h in christmases)).astype(int)

    # Fuel_Price change
    df['Fuel_Price_change'] = df.groupby('Store')['Fuel_Price'].diff()
    df['Fuel_Price_change'] = df['Fuel_Price_change'].fillna(0)

    # --- Aggregated stats from train (SAFE for out-of-sample) ---
    if train_ref is not None:
        # Store-Dept median
        sd_median = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].median().reset_index()
        sd_median.columns = ['Store', 'Dept', 'sd_median']
        df = df.merge(sd_median, on=['Store', 'Dept'], how='left')

        # Store-Dept mean
        sd_mean = train_ref.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
        sd_mean.columns = ['Store', 'Dept', 'sd_mean']
        df = df.merge(sd_mean, on=['Store', 'Dept'], how='left')

        # --- Same-week-last-year features (SAFE: references train data at offset 51/52/53) ---
        train_sales = train_ref[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()

        for offset in [51, 52, 53]:
            lookup = train_sales[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
            lookup['merge_date'] = lookup['Date'] + pd.Timedelta(weeks=offset)
            lookup = lookup.rename(columns={'Weekly_Sales': f'yoy_sales_{offset}'})
            lookup = lookup[['Store', 'Dept', 'merge_date', f'yoy_sales_{offset}']]
            df = df.merge(lookup, left_on=['Store', 'Dept', 'Date'],
                         right_on=['Store', 'Dept', 'merge_date'], how='left')
            if 'merge_date' in df.columns:
                df = df.drop(columns=['merge_date'])
            df[f'yoy_sales_{offset}'] = df[f'yoy_sales_{offset}'].fillna(0)

        # YoY aggregation
        df['yoy_mean'] = df[['yoy_sales_51', 'yoy_sales_52', 'yoy_sales_53']].mean(axis=1)
        df['yoy_max'] = df[['yoy_sales_51', 'yoy_sales_52', 'yoy_sales_53']].max(axis=1)

        # Week-of-year average from train (seasonal baseline)
        week_avg = train_ref.groupby(train_ref['Date'].dt.isocalendar().week.astype(int))['Weekly_Sales'].mean().reset_index()
        week_avg.columns = ['Week', 'week_avg_sales']
        df = df.merge(week_avg, on='Week', how='left')

    return df

print("Creating features...")
train = create_features(train, train_ref=train)
test = create_features(test, train_ref=train)

exclude = ['Date', 'Weekly_Sales', 'Type']
feature_cols = [c for c in train.columns if c not in exclude]
missing_in_test = set(feature_cols) - set(test.columns)
if missing_in_test:
    print(f"WARNING: missing in test: {missing_in_test}")
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"Features: {len(feature_cols)}")

# ============================================================
# 3. Validation: 20 weeks
# ============================================================
dates = sorted(train['Date'].unique())
val_dates = dates[-20:]
train_dates = dates[:-20]

trn = train[train['Date'].isin(train_dates)]
val = train[train['Date'].isin(val_dates)]

print(f"\nTrain: {len(trn)}, Val: {len(val)}")

# ============================================================
# 4. WMAE
# ============================================================
def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

# ============================================================
# 5. Train with sqrt transform + multiple seeds
# ============================================================
print("\n--- Multi-seed LightGBM with sqrt transform ---")

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
val_preds_list = []
test_preds_list = []

for seed in seeds:
    params = {**lgb_params_base, 'seed': seed}

    # sqrt transform
    trn_y = np.sqrt(np.maximum(trn['Weekly_Sales'].values, 0))
    val_y = np.sqrt(np.maximum(val['Weekly_Sales'].values, 0))

    trn_w = np.where(trn['IsHoliday'] == 1, 5, 1)
    val_w = np.where(val['IsHoliday'] == 1, 5, 1)

    lgb_trn = lgb.Dataset(trn[feature_cols], trn_y, weight=trn_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])
    lgb_val = lgb.Dataset(val[feature_cols], val_y, weight=val_w,
                          categorical_feature=['Store', 'Dept', 'Store_Dept'])

    model = lgb.train(
        params, lgb_trn,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )

    # Inverse transform (square back)
    val_pred = np.clip(model.predict(val[feature_cols]), 0, None) ** 2
    test_pred = np.clip(model.predict(test[feature_cols]), 0, None) ** 2

    val_wmae = wmae(val['Weekly_Sales'].values, val_pred, val['IsHoliday'].values)
    print(f"  Seed {seed}: val WMAE={val_wmae:.4f}, best_iter={model.best_iteration}")

    val_preds_list.append(val_pred)
    test_preds_list.append(test_pred)

# Average predictions across seeds
val_pred_avg = np.mean(val_preds_list, axis=0)
test_pred_avg = np.mean(test_preds_list, axis=0)

val_wmae_avg = wmae(val['Weekly_Sales'].values, val_pred_avg, val['IsHoliday'].values)
print(f"\nSeed-averaged val WMAE: {val_wmae_avg:.4f}")

# ============================================================
# 6. Christmas shift post-processing
# ============================================================
print("\n--- Christmas Shift Post-Processing ---")

# The Christmas sales spike shifts between reporting weeks because
# Christmas falls on different days of the week each year.
# We redistribute sales in weeks 48-52.

test_with_pred = test[['Store', 'Dept', 'Date', 'Week']].copy()
test_with_pred['pred'] = test_pred_avg

# Apply shift for weeks around Christmas (48-52)
# The shift moves a fraction of each week's prediction to the next week
shift = 2.5 / 7  # 2.5 days shift

christmas_mask = (test_with_pred['Week'] >= 48) & (test_with_pred['Week'] <= 52)
n_shifted = christmas_mask.sum()

if n_shifted > 0:
    # Simple shift: move (shift) fraction to next week
    # Sort by store-dept-date to ensure correct ordering
    test_with_pred = test_with_pred.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    shifted_pred = test_with_pred['pred'].values.copy()

    for i in range(len(shifted_pred) - 1):
        if (test_with_pred.iloc[i]['Week'] >= 48 and
            test_with_pred.iloc[i]['Week'] <= 51 and
            test_with_pred.iloc[i]['Store'] == test_with_pred.iloc[i+1]['Store'] and
            test_with_pred.iloc[i]['Dept'] == test_with_pred.iloc[i+1]['Dept']):
            # Move shift fraction to next week
            moved = shifted_pred[i] * shift
            shifted_pred[i] -= moved
            shifted_pred[i+1] += moved

    test_with_pred['pred_shifted'] = shifted_pred
    print(f"  Applied Christmas shift to {n_shifted} rows")
else:
    test_with_pred['pred_shifted'] = test_with_pred['pred']
    print("  No Christmas weeks found in test")

# Use shifted predictions
test_final = test_with_pred['pred_shifted'].values
test_final = np.clip(test_final, 0, None)

# ============================================================
# 7. Retrain on full data for final submission
# ============================================================
print("\n--- Retraining on full data (seed average) ---")
full_test_preds = []

for seed in seeds:
    params = {**lgb_params_base, 'seed': seed}
    full_y = np.sqrt(np.maximum(train['Weekly_Sales'].values, 0))
    full_w = np.where(train['IsHoliday'] == 1, 5, 1)

    lgb_full = lgb.Dataset(train[feature_cols], full_y, weight=full_w,
                           categorical_feature=['Store', 'Dept', 'Store_Dept'])
    model_full = lgb.train(params, lgb_full, num_boost_round=2000)  # fixed rounds

    pred = np.clip(model_full.predict(test[feature_cols]), 0, None) ** 2
    full_test_preds.append(pred)
    print(f"  Seed {seed} done")

test_pred_final = np.mean(full_test_preds, axis=0)

# Apply Christmas shift to final predictions
test_final_df = test[['Store', 'Dept', 'Date', 'Week']].copy()
test_final_df['pred'] = test_pred_final

christmas_mask = (test_final_df['Week'] >= 48) & (test_final_df['Week'] <= 52)
if christmas_mask.sum() > 0:
    test_final_df = test_final_df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    shifted = test_final_df['pred'].values.copy()
    shift_frac = 2.5 / 7
    for i in range(len(shifted) - 1):
        if (test_final_df.iloc[i]['Week'] >= 48 and
            test_final_df.iloc[i]['Week'] <= 51 and
            test_final_df.iloc[i]['Store'] == test_final_df.iloc[i+1]['Store'] and
            test_final_df.iloc[i]['Dept'] == test_final_df.iloc[i+1]['Dept']):
            moved = shifted[i] * shift_frac
            shifted[i] -= moved
            shifted[i+1] += moved
    test_final_df['pred'] = np.clip(shifted, 0, None)

test_pred_final = test_final_df['pred'].values

submission = sample_sub.copy()
submission['Weekly_Sales'] = test_pred_final
sub_path = OUTPUTS / "submission_r07_yoy_christmas.csv"
submission.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Predictions: [{test_pred_final.min():.2f}, {test_pred_final.max():.2f}], mean={test_pred_final.mean():.2f}")

# Feature importance from first seed model
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)
print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("R07 Complete")
print("=" * 60)
