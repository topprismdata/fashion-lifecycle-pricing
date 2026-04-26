#!/usr/bin/env python3
"""
R01 EDA: ML Zoomcamp 2024 Retail Demand Forecast
Quick data exploration and baseline pipeline.
"""
import sys
from pathlib import Path

DATA_DIR = Path("/Users/guohongbin/projects/fashion-lifecycle-pricing/competitions/03_ml_zoomcamp_retail/data/raw")

import numpy as np
import pandas as pd

print("=" * 70)
print("  R01 EDA: ML Zoomcamp 2024 Retail Demand Forecast")
print("=" * 70)

# ============================================================
# 1. Load Data
# ============================================================
print("\n--- 1. Load Data ---")

print("Loading sales.csv...")
sales = pd.read_csv(DATA_DIR / "sales.csv", index_col=0)
print(f"  Shape: {sales.shape}")
print(f"  Columns: {list(sales.columns)}")
print(f"  Dtypes:\n{sales.dtypes}")
print(f"  Head:\n{sales.head(3)}")
print(f"  Describe:\n{sales.describe()}")

print("\nLoading test.csv (semicolon-separated)...")
test = pd.read_csv(DATA_DIR / "test.csv", sep=";")
print(f"  Shape: {test.shape}")
print(f"  Columns: {list(test.columns)}")
print(f"  Date range: {test['date'].min()} ~ {test['date'].max()}")
print(f"  Unique items: {test['item_id'].nunique()}")
print(f"  Unique stores: {test['store_id'].nunique()}")

print("\nLoading catalog.csv...")
catalog = pd.read_csv(DATA_DIR / "catalog.csv", index_col=0)
print(f"  Shape: {catalog.shape}")
print(f"  Columns: {list(catalog.columns)}")
print(f"  Unique depts: {catalog['dept_name'].nunique()}")
print(f"  Unique classes: {catalog['class_name'].nunique()}")
print(f"  Unique subclasses: {catalog['subclass_name'].nunique()}")

print("\nLoading stores.csv...")
stores = pd.read_csv(DATA_DIR / "stores.csv", index_col=0)
print(f"  Shape: {stores.shape}")
print(f"  {stores}")

# ============================================================
# 2. Data Quality
# ============================================================
print("\n--- 2. Data Quality ---")

print(f"Negative quantities: {(sales['quantity'] < 0).sum():,} ({(sales['quantity'] < 0).mean()*100:.2f}%)")
print(f"Zero quantities: {(sales['quantity'] == 0).sum():,} ({(sales['quantity'] == 0).mean()*100:.2f}%)")
print(f"NaN quantities: {sales['quantity'].isna().sum():,}")

print(f"\nNaN price_base: {sales['price_base'].isna().sum():,}")
print(f"Inf price_base: {np.isinf(sales['price_base']).sum():,}")
print(f"NaN sum_total: {sales['sum_total'].isna().sum():,}")
print(f"Negative sum_total: {(sales['sum_total'] < 0).sum():,}")

print(f"\nDate range: {sales['date'].min()} ~ {sales['date'].max()}")
print(f"Unique dates: {sales['date'].nunique()}")
print(f"Unique items: {sales['item_id'].nunique()}")
print(f"Unique stores: {sales['store_id'].nunique()}")

# ============================================================
# 3. Target Distribution
# ============================================================
print("\n--- 3. Target Distribution ---")

qty = sales['quantity']
print(f"Mean: {qty.mean():.2f}")
print(f"Median: {qty.median():.2f}")
print(f"Std: {qty.std():.2f}")
print(f"Min: {qty.min()}, Max: {qty.max()}")
print(f"Quantiles:")
for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  {q*100:.0f}%: {qty.quantile(q):.1f}")

# ============================================================
# 4. Missing Values (all files)
# ============================================================
print("\n--- 4. Missing Values ---")

for fname in ["sales.csv", "catalog.csv", "stores.csv", "online.csv",
              "discounts_history.csv", "price_history.csv", "markdowns.csv",
              "actual_matrix.csv"]:
    sep = ";" if fname == "test.csv" else ","
    try:
        df = pd.read_csv(DATA_DIR / fname, index_col=0, sep=sep, nrows=100)
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n{fname}:")
            for col, cnt in missing[missing > 0].items():
                print(f"  {col}: {cnt}/{len(df)} ({cnt/len(df)*100:.1f}%)")
        else:
            print(f"{fname}: no missing values (first 100 rows)")
    except Exception as e:
        print(f"{fname}: error - {e}")

# ============================================================
# 5. Test Set Analysis
# ============================================================
print("\n--- 5. Test Set Analysis ---")

test_items = set(test['item_id'].unique())
train_items = set(sales['item_id'].unique())
cold_items = test_items - train_items
print(f"Test items: {len(test_items):,}")
print(f"Train items: {len(train_items):,}")
print(f"Cold items (in test but not train): {len(cold_items):,}")

test_stores = set(test['store_id'].unique())
print(f"Test stores: {test_stores}")

test_dates = pd.to_datetime(test['date'])
print(f"Test date range: {test_dates.min()} ~ {test_dates.max()}")
print(f"Test rows per store:")
print(test['store_id'].value_counts().to_string())

# ============================================================
# 6. Sales Over Time (aggregated daily)
# ============================================================
print("\n--- 6. Sales Over Time ---")

sales['date'] = pd.to_datetime(sales['date'])
daily = sales.groupby('date').agg(
    total_qty=('quantity', 'sum'),
    n_items=('item_id', 'nunique'),
    n_transactions=('quantity', 'count')
).reset_index()

print(f"Daily stats:")
print(f"  Mean daily qty: {daily['total_qty'].mean():,.0f}")
print(f"  Mean daily items: {daily['n_items'].mean():,.0f}")
print(f"  Mean daily transactions: {daily['n_transactions'].mean():,.0f}")

# Day of week pattern
sales['dow'] = sales['date'].dt.dayofweek
dow_qty = sales.groupby('dow')['quantity'].mean()
print(f"\nAvg quantity by day of week:")
for d, q in dow_qty.items():
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"  {days[d]}: {q:.2f}")

# ============================================================
# 7. Online vs Offline
# ============================================================
print("\n--- 7. Online Sales ---")

online = pd.read_csv(DATA_DIR / "online.csv", index_col=0, nrows=10000)
print(f"Online columns: {list(online.columns)}")
print(f"Online shape (sample): {online.shape}")
print(f"Online stores: {sorted(online['store_id'].unique())}")

# ============================================================
# 8. Sample Submission
# ============================================================
print("\n--- 8. Sample Submission ---")

sub = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col=0)
print(f"Shape: {sub.shape}")
print(f"Columns: {list(sub.columns)}")
print(f"Head:\n{sub.head(3)}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("  EDA Summary")
print("=" * 70)
print(f"""
Key findings:
1. Target (quantity): mean={qty.mean():.1f}, highly skewed (max={qty.max()})
2. Negative quantities: {(sales['quantity'] < 0).mean()*100:.1f}% — need to handle
3. Inf prices: {np.isinf(sales['price_base']).sum()} rows — from sum_total/0
4. Test period: 2024-09 to 2024-10 (1 month forecast)
5. Cold items: {len(cold_items):,} / {len(test_items):,} — need popular default
6. 4 stores, ~28K items, 25 months of data
7. Strong day-of-week pattern
8. Semicolon-separated test.csv
9. Need rolling features (7/14/30 day) per top solutions
10. Catalog has hierarchical categories (dept→class→subclass)
""")
