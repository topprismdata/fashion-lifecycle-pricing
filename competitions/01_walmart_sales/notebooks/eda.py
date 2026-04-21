"""
Walmart Store Sales Forecasting — EDA
Stage 1, Competition 01
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data_raw"

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("Walmart Store Sales Forecasting — EDA")
print("=" * 60)

train = pd.read_csv(DATA / "train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["Date"])
features = pd.read_csv(DATA / "features.csv", parse_dates=["Date"])
stores = pd.read_csv(DATA / "stores.csv")
sample_sub = pd.read_csv(DATA / "sampleSubmission.csv")

print(f"\nTrain: {train.shape}")
print(f"Test:  {test.shape}")
print(f"Features: {features.shape}")
print(f"Stores: {stores.shape}")

# ============================================================
# 2. Train Data Overview
# ============================================================
print("\n" + "=" * 60)
print("Train Data Overview")
print("=" * 60)

print(f"\nDate range: {train['Date'].min()} to {train['Date'].max()}")
print(f"Test date range: {test['Date'].min()} to {test['Date'].max()}")
print(f"Stores: {train['Store'].nunique()}")
print(f"Departments: {train['Dept'].nunique()}")
print(f"Store-Dept combos: {train.groupby(['Store','Dept']).ngroups}")

print(f"\nWeekly_Sales stats:")
print(train['Weekly_Sales'].describe())
print(f"\nNegative sales: {(train['Weekly_Sales'] < 0).sum()} ({(train['Weekly_Sales'] < 0).mean()*100:.2f}%)")
print(f"Zero sales: {(train['Weekly_Sales'] == 0).sum()} ({(train['Weekly_Sales'] == 0).mean()*100:.2f}%)")

print(f"\nIsHoliday distribution:")
print(train['IsHoliday'].value_counts())

# ============================================================
# 3. Store Analysis
# ============================================================
print("\n" + "=" * 60)
print("Store Analysis")
print("=" * 60)

print(f"\nStore types: {stores['Type'].value_counts().to_dict()}")
print(f"Store size range: {stores['Size'].min()} - {stores['Size'].max()}")
print(f"Store size by type:")
print(stores.groupby('Type')['Size'].describe())

# ============================================================
# 4. Features Analysis
# ============================================================
print("\n" + "=" * 60)
print("Features Analysis (Missing Values)")
print("=" * 60)

missing = features.isnull().sum()
missing_pct = (features.isnull().sum() / len(features) * 100).round(2)
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
print(missing_df[missing_df["missing"] > 0].sort_values("pct", ascending=False))

print(f"\nFeature date range: {features['Date'].min()} to {features['Date'].max()}")

# ============================================================
# 5. Holiday Analysis
# ============================================================
print("\n" + "=" * 60)
print("Holiday Analysis")
print("=" * 60)

holiday_dates = features[features['IsHoliday'] == True]['Date'].unique()
holiday_dates = sorted(holiday_dates)
print(f"Holiday weeks in features: {len(holiday_dates)}")
for d in holiday_dates:
    print(f"  {d.strftime('%Y-%m-%d')}")

# Check test holidays
test_holidays = test[test['IsHoliday'] == True]['Date'].unique()
print(f"\nTest holiday weeks: {len(test_holidays)}")
for d in sorted(test_holidays):
    print(f"  {pd.Timestamp(d).strftime('%Y-%m-%d')}")

# ============================================================
# 6. Sales by Store Type and Holiday
# ============================================================
print("\n" + "=" * 60)
print("Sales Distribution by Store Type and Holiday")
print("=" * 60)

train_stores = train.merge(stores, on='Store', how='left')
print("\nMean Weekly_Sales by Store Type:")
print(train_stores.groupby('Type')['Weekly_Sales'].mean())

print("\nMean Weekly_Sales: Holiday vs Non-Holiday:")
print(train_stores.groupby('IsHoliday')['Weekly_Sales'].mean())

# ============================================================
# 7. Department Analysis
# ============================================================
print("\n" + "=" * 60)
print("Top/Bottom Departments by Sales")
print("=" * 60)

dept_sales = train.groupby('Dept')['Weekly_Sales'].agg(['mean', 'sum', 'count']).sort_values('sum', ascending=False)
print("\nTop 10 Depts by total sales:")
print(dept_sales.head(10))
print(f"\nBottom 10 Depts by total sales:")
print(dept_sales.tail(10))

# Check if test has new depts/stores
train_depts = set(train['Dept'].unique())
test_depts = set(test['Dept'].unique())
train_stores_set = set(train['Store'].unique())
test_stores_set = set(test['Store'].unique())

print(f"\nTrain depts: {len(train_depts)}, Test depts: {len(test_depts)}")
print(f"New depts in test: {test_depts - train_depts}")
print(f"Train stores: {len(train_stores_set)}, Test stores: {len(test_stores_set)}")
print(f"New stores in test: {test_stores_set - train_stores_set}")

# ============================================================
# 8. Time Series Patterns
# ============================================================
print("\n" + "=" * 60)
print("Time Series Patterns")
print("=" * 60)

# Aggregate weekly sales
weekly_total = train.groupby('Date')['Weekly_Sales'].sum().sort_index()
print(f"\nTotal weekly sales stats:")
print(weekly_total.describe())

# Year-over-year
train_stores['Year'] = train_stores['Date'].dt.year
train_stores['Week'] = train_stores['Date'].dt.isocalendar().week.astype(int)

yearly_weekly = train_stores.groupby(['Year', 'Week'])['Weekly_Sales'].mean()
print(f"\nPeak weeks (avg sales):")
print(yearly_weekly.groupby('Week').mean().nlargest(10))

print("\n" + "=" * 60)
print("EDA Complete")
print("=" * 60)
