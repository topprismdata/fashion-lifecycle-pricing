#!/usr/bin/env python3
"""
R08 AutoGluon Multi-Model Ensemble — Final Submission.

Strategy: Combine our R06 dense grid pipeline with the 2nd place solution's
AutoGluon approach. Two model tracks:

Track A: AutoGluon multi-model stacking (LightGBM + XGBoost + RF + CatBoost)
         using the 2nd place's minimal 8-feature set on the dense grid.

Track B: Aggressive CatBoost (depth=12, lr=0.5, like 4th place) with our
         full 56-feature R06 feature set.

Final: Blend Track A + Track B predictions.

Key improvements over R06 (LB 15.42):
  1. Multi-model ensemble via AutoGluon (2nd place used this)
  2. Aggressive CatBoost config (4th place used depth=12)
  3. Rolling averages computed via SQL window functions (2nd place approach)
  4. Clean minimal feature set to reduce noise

Validation: time-based holdout (same as R06)
  Train: 2024-03-01 ~ 2024-08-26 (dense grid)
  Val:   2024-08-27 ~ 2024-09-26 (dense grid)
  Test:  2024-09-27 ~ 2024-10-26 (already dense)

Usage:
    python scripts/run_r08_autogluon.py
"""

import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_PATH = DATA_DIR / "sales.csv"
TEST_PATH = DATA_DIR / "test.csv"
CATALOG_PATH = DATA_DIR / "catalog.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
DISCOUNTS_PATH = DATA_DIR / "discounts_history.csv"
PRICE_HISTORY_PATH = DATA_DIR / "price_history.csv"
MARKDOWNS_PATH = DATA_DIR / "markdowns.csv"
ACTUAL_MATRIX_PATH = DATA_DIR / "actual_matrix.csv"
STORES_PATH = DATA_DIR / "stores.csv"
ONLINE_PATH = DATA_DIR / "online.csv"

SUBMISSION_PATH = OUTPUT_DIR / "submission_r08_autogluon.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_START = "2024-03-01"
VAL_START = "2024-08-27"
VAL_END = "2024-09-26"

# Track B CatBoost (aggressive, like 4th place)
CAT_FEATURES_R06 = [
    "item_id", "store_id", "dept_name", "class_name",
    "promo_type", "store_division", "store_format", "store_city",
]

DATE_FEATURES = [
    "day_of_week", "day_of_month", "month",
    "is_weekend", "is_month_start", "is_month_end",
]

LAG_DAYS = [7, 14, 30]
ROLL_MEAN_WINDOWS = [7, 14, 30, 60]
ROLL_STD_WINDOWS = [7, 14, 30]
ROLL_MIN_WINDOWS = [7, 30]
ROLL_MAX_WINDOWS = [7, 30]
EWM_SPANS = [7, 30]

RUSSIAN_HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
    "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08",
    "2023-02-23", "2023-03-08", "2023-05-01", "2023-05-09",
    "2023-06-12", "2023-11-04",
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
    "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
    "2024-02-23", "2024-03-08", "2024-05-01", "2024-05-09",
    "2024-06-12", "2024-11-04",
    "2024-02-24", "2024-03-09",
    "2024-05-02", "2024-05-03", "2024-05-10",
    "2024-06-13", "2024-11-05",
])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Step 1: Build Dense Grid (SQL-based, like 2nd place)
# ===========================================================================
def build_dense_grid_sql() -> pd.DataFrame:
    """
    Build dense grid + rolling features entirely in DuckDB SQL.
    Inspired by 2nd place solution's SQL approach.
    """
    log("Building dense grid + rolling features via DuckDB SQL...")

    con = duckdb.connect()

    # Load sales
    log("  Loading sales...")
    con.execute(f"""
        CREATE OR REPLACE TABLE sales AS
        SELECT
            CAST(date AS DATE) AS date,
            item_id,
            quantity,
            price_base,
            CAST(store_id AS VARCHAR) AS store_id
        FROM read_csv('{SALES_PATH}', header=true, columns={{
            'Unnamed: 0': 'VARCHAR',
            'date': 'VARCHAR',
            'item_id': 'VARCHAR',
            'quantity': 'DOUBLE',
            'price_base': 'DOUBLE',
            'sum_total': 'DOUBLE',
            'store_id': 'VARCHAR'
        }})
        WHERE quantity >= 0
    """)

    # Load online
    log("  Loading online...")
    con.execute(f"""
        CREATE OR REPLACE TABLE online AS
        SELECT
            CAST(date AS DATE) AS date,
            item_id,
            quantity,
            CAST(store_id AS VARCHAR) AS store_id
        FROM read_csv('{ONLINE_PATH}', header=true, columns={{
            'Unnamed: 0': 'VARCHAR',
            'date': 'VARCHAR',
            'item_id': 'VARCHAR',
            'quantity': 'DOUBLE',
            'price_base': 'DOUBLE',
            'sum_total': 'DOUBLE',
            'store_id': 'VARCHAR'
        }})
        WHERE quantity >= 0
    """)

    # Load actual_matrix
    log("  Loading actual_matrix...")
    con.execute(f"""
        CREATE OR REPLACE TABLE actual_matrix AS
        SELECT
            item_id,
            CAST(date AS DATE) AS date,
            CAST(store_id AS VARCHAR) AS store_id
        FROM read_csv('{ACTUAL_MATRIX_PATH}', header=true, columns={{
            'Unnamed: 0': 'VARCHAR',
            'item_id': 'VARCHAR',
            'date': 'VARCHAR',
            'store_id': 'VARCHAR'
        }})
    """)

    # Step 1: Merge offline + online sales
    log("  Aggregating offline + online sales...")
    con.execute("""
        CREATE OR REPLACE TABLE daily_sales AS
        WITH offline_agg AS (
            SELECT date, item_id, store_id,
                   SUM(quantity) AS offline_qty,
                   AVG(price_base) AS price_base
            FROM sales
            GROUP BY date, item_id, store_id
        ),
        online_agg AS (
            SELECT date, item_id, store_id,
                   SUM(quantity) AS online_qty
            FROM online
            GROUP BY date, item_id, store_id
        )
        SELECT
            COALESCE(o.date, n.date) AS date,
            COALESCE(o.item_id, n.item_id) AS item_id,
            COALESCE(o.store_id, n.store_id) AS store_id,
            COALESCE(o.offline_qty, 0) AS offline_qty,
            COALESCE(o.price_base, 0) AS price_base,
            COALESCE(n.online_qty, 0) AS online_qty,
            COALESCE(o.offline_qty, 0) + COALESCE(n.online_qty, 0) AS total_qty
        FROM offline_agg o
        FULL OUTER JOIN online_agg n
            ON o.date = n.date
            AND o.item_id = n.item_id
            AND o.store_id = n.store_id
    """)

    # Step 2: Get unique (item_id, store_id) combos from grid period
    log("  Building dense grid...")
    dense_df = con.execute(f"""
        WITH combos AS (
            SELECT DISTINCT item_id, store_id FROM (
                SELECT item_id, store_id FROM sales
                    WHERE CAST(date AS DATE) >= '{GRID_START}'
                UNION
                SELECT item_id, store_id FROM online
                    WHERE CAST(date AS DATE) >= '{GRID_START}'
                UNION
                SELECT item_id, store_id FROM actual_matrix
                    WHERE CAST(date AS DATE) >= '{GRID_START}'
            )
        ),
        date_range AS (
            SELECT unnest(generate_series(
                CAST('{GRID_START}' AS DATE),
                CAST('{VAL_END}' AS DATE),
                INTERVAL 1 DAY
            )) AS date
        ),
        grid AS (
            SELECT d.date, c.item_id, c.store_id
            FROM date_range d
            CROSS JOIN combos c
        )
        SELECT
            g.date,
            g.item_id,
            g.store_id,
            COALESCE(s.offline_qty, 0) AS offline_qty,
            COALESCE(s.price_base, 0) AS price_base,
            COALESCE(s.online_qty, 0) AS online_qty,
            COALESCE(s.total_qty, 0) AS total_qty
        FROM grid g
        LEFT JOIN daily_sales s
            ON g.date = s.date
            AND g.item_id = s.item_id
            AND g.store_id = s.store_id
        ORDER BY g.date, g.item_id, g.store_id
    """).fetchdf()

    log(f"  Dense grid shape: {dense_df.shape}")
    log(f"  total_qty zeros: {(dense_df['total_qty'] == 0).mean()*100:.1f}%")
    con.close()
    return dense_df


def build_rolling_features_sql(train_df: pd.DataFrame, target_dates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages using DuckDB SQL window functions.
    Uses the 2nd place approach: rolling means over the training data.
    """
    log("Computing rolling features via DuckDB SQL...")

    con = duckdb.connect()

    # Register the training data (only positive quantities + zeros from grid)
    # Build daily quantity from train_df
    daily_qty = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(total_qty=("total_qty", "sum"))
    )
    daily_qty = daily_qty.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    con.register("daily_qty", daily_qty)

    # Compute rolling averages using SQL window functions
    log("  Computing 7/14/30-day rolling averages...")
    rolling_df = con.execute("""
        SELECT
            item_id,
            store_id,
            date,
            total_qty,
            COALESCE(
                AVG(total_qty) OVER (
                    PARTITION BY item_id, store_id
                    ORDER BY date
                    ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                ), 0
            ) AS rolling_7d_avg,
            COALESCE(
                AVG(total_qty) OVER (
                    PARTITION BY item_id, store_id
                    ORDER BY date
                    ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
                ), 0
            ) AS rolling_14d_avg,
            COALESCE(
                AVG(total_qty) OVER (
                    PARTITION BY item_id, store_id
                    ORDER BY date
                    ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                ), 0
            ) AS rolling_30d_avg,
            COALESCE(
                AVG(total_qty) OVER (
                    PARTITION BY item_id, store_id
                    ORDER BY date
                    ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
                ), 0
            ) AS rolling_60d_avg,
            COALESCE(
                STDDEV(total_qty) OVER (
                    PARTITION BY item_id, store_id
                    ORDER BY date
                    ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                ), 0
            ) AS rolling_7d_std,
            COALESCE(
                STDDEV(total_qty) OVER (
                    PARTITION BY item_id, store_id
                    ORDER BY date
                    ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
                ), 0
            ) AS rolling_30d_std
        FROM daily_qty
        ORDER BY item_id, store_id, date
    """).fetchdf()

    con.close()

    # For target dates (val + test), forward-fill the last known rolling values
    log("  Forward-filling rolling features for target dates...")
    target_groups = target_dates_df[["item_id", "store_id"]].drop_duplicates()

    # Get the last available rolling values per (item_id, store_id) from training
    last_rolling = rolling_df.groupby(["item_id", "store_id"]).last().reset_index()
    last_rolling = last_rolling[["item_id", "store_id", "date",
                                  "rolling_7d_avg", "rolling_14d_avg",
                                  "rolling_30d_avg", "rolling_60d_avg",
                                  "rolling_7d_std", "rolling_30d_std"]]

    # For each target date, use the most recent rolling values
    # Simple approach: merge the last rolling values (they're constant for all test dates)
    result = target_dates_df.merge(
        last_rolling[["item_id", "store_id",
                       "rolling_7d_avg", "rolling_14d_avg",
                       "rolling_30d_avg", "rolling_60d_avg",
                       "rolling_7d_std", "rolling_30d_std"]],
        on=["item_id", "store_id"],
        how="left",
    )

    # Fill NaN with 0
    for col in ["rolling_7d_avg", "rolling_14d_avg", "rolling_30d_avg",
                "rolling_60d_avg", "rolling_7d_std", "rolling_30d_std"]:
        result[col] = result[col].fillna(0)

    log(f"  Rolling features for {len(result):,} target rows")
    return result


# ===========================================================================
# Quantile Outlier Removal (same as R06)
# ===========================================================================
def remove_quantile_outliers(df: pd.DataFrame) -> pd.DataFrame:
    log("Applying quantile outlier removal (P1-P99 per item x store x dow)...")
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    pos_mask = df["total_qty"] > 0
    log(f"  Positive quantity rows: {pos_mask.sum():,} / {len(df):,}")

    grouped = df.loc[pos_mask].groupby(["item_id", "store_id", "day_of_week"])["total_qty"]
    lower = grouped.quantile(0.01).rename("q01")
    upper = grouped.quantile(0.99).rename("q99")
    bounds = pd.concat([lower, upper], axis=1).reset_index()

    before = len(df)
    df = df.merge(bounds, on=["item_id", "store_id", "day_of_week"], how="left")
    keep_mask = (df["total_qty"] == 0) | (
        (df["total_qty"] >= df["q01"]) & (df["total_qty"] <= df["q99"])
    )
    df = df[keep_mask].drop(columns=["q01", "q99"]).reset_index(drop=True)

    removed = before - len(df)
    log(f"  Removed {removed:,} outliers ({removed/before*100:.2f}%)")
    return df


# ===========================================================================
# Data Loading
# ===========================================================================
def load_test():
    log("Loading test data...")
    test = pd.read_csv(TEST_PATH, sep=";")
    test["date"] = pd.to_datetime(test["date"], format="%d.%m.%Y")
    test["store_id"] = test["store_id"].astype(str)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    log(f"  test: {test.shape}, date range: {test['date'].min()} ~ {test['date'].max()}")
    return test, sample_sub


def load_catalog():
    catalog = pd.read_csv(CATALOG_PATH, index_col=0)
    log(f"  catalog: {catalog.shape}")
    return catalog


def load_stores():
    stores = pd.read_csv(STORES_PATH)
    stores["store_id"] = stores["store_id"].astype(str)
    log(f"  stores: {stores.shape}")
    return stores


# ===========================================================================
# Feature Engineering — Minimal Track (2nd place style)
# ===========================================================================
def create_minimal_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Create the minimal feature set inspired by 2nd place.
    Only 8 core features + rolling averages.
    """
    log("Creating minimal feature set (2nd place style)...")

    def add_date_features(df):
        df = df.copy()
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day"] = df["date"].dt.day
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        return df

    train = add_date_features(train_df)
    val = add_date_features(val_df)
    test = add_date_features(test_df)

    # Rolling features from SQL
    # For train: compute rolling from full_train up to but not including val
    # For val/test: use last known rolling values
    log("  Computing rolling for train...")
    train_rolling = build_rolling_features_sql(
        full_train[full_train["date"] < VAL_START], train
    )

    log("  Computing rolling for val...")
    val_rolling = build_rolling_features_sql(
        full_train[full_train["date"] <= VAL_END], val
    )

    log("  Computing rolling for test...")
    test_rolling = build_rolling_features_sql(full_train, test)

    # Merge rolling features
    rolling_cols = ["rolling_7d_avg", "rolling_14d_avg", "rolling_30d_avg",
                    "rolling_60d_avg", "rolling_7d_std", "rolling_30d_std"]

    train = train.merge(
        train_rolling[["item_id", "store_id", "date"] + rolling_cols],
        on=["item_id", "store_id", "date"], how="left"
    )
    val = val.merge(
        val_rolling[["item_id", "store_id", "date"] + rolling_cols],
        on=["item_id", "store_id", "date"], how="left"
    )
    test = test.merge(
        test_rolling[["item_id", "store_id", "date"] + rolling_cols],
        on=["item_id", "store_id", "date"], how="left"
    )

    for col in rolling_cols:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)
        test[col] = test[col].fillna(0)

    # Feature columns — minimal set
    feature_cols = [
        "item_id", "store_id", "day", "week", "day_of_week",
        "rolling_7d_avg", "rolling_14d_avg", "rolling_30d_avg",
        "rolling_60d_avg", "rolling_7d_std", "rolling_30d_std",
    ]
    cat_features = ["item_id", "store_id"]

    log(f"  Minimal features: {len(feature_cols)} ({len(cat_features)} categorical)")
    return train, val, test, feature_cols, cat_features


# ===========================================================================
# Feature Engineering — Full Track (R06 style)
# ===========================================================================
def create_full_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_train: pd.DataFrame,
    catalog: pd.DataFrame,
    stores: pd.DataFrame,
    discounts: pd.DataFrame,
    price_history: pd.DataFrame,
    markdowns: pd.DataFrame,
    actual_matrix: pd.DataFrame,
    online: pd.DataFrame,
    sales_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Create the full R06 feature set (56 features).
    This is Track B — aggressive CatBoost with rich features.
    """
    log("Creating full feature set (R06 style)...")

    # Date features
    def add_date_features(df):
        df = df.copy()
        dt = df["date"]
        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["month"] = dt.dt.month
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = dt.dt.is_month_start.astype(int)
        df["is_month_end"] = dt.dt.is_month_end.astype(int)
        # Cyclical
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        week = dt.dt.isocalendar().week.astype(int)
        df["week_sin"] = np.sin(2 * np.pi * week / 52)
        df["week_cos"] = np.cos(2 * np.pi * week / 52)
        return df

    train = add_date_features(train_df)
    val = add_date_features(val_df)
    test = add_date_features(test_df)

    # Lag features
    train = create_lag_features(sales_full, full_train, train, VAL_START)
    val = create_lag_features(sales_full, full_train, val, VAL_END)
    test = create_lag_features(sales_full, full_train, test, None)

    # Price
    train = create_price_feature(sales_full, train)
    val = create_price_feature(sales_full, val)
    test = create_price_feature(sales_full, test)

    # Catalog
    train = merge_catalog(train, catalog)
    val = merge_catalog(val, catalog)
    test = merge_catalog(test, catalog)

    # Rolling features (simplified — use SQL approach)
    train_rolling = build_rolling_features_sql(
        full_train[full_train["date"] < VAL_START], train
    )
    val_rolling = build_rolling_features_sql(
        full_train[full_train["date"] <= VAL_END], val
    )
    test_rolling = build_rolling_features_sql(full_train, test)

    rolling_cols = ["rolling_7d_avg", "rolling_14d_avg", "rolling_30d_avg",
                    "rolling_60d_avg", "rolling_7d_std", "rolling_30d_std"]

    train = train.merge(
        train_rolling[["item_id", "store_id", "date"] + rolling_cols],
        on=["item_id", "store_id", "date"], how="left"
    )
    val = val.merge(
        val_rolling[["item_id", "store_id", "date"] + rolling_cols],
        on=["item_id", "store_id", "date"], how="left"
    )
    test = test.merge(
        test_rolling[["item_id", "store_id", "date"] + rolling_cols],
        on=["item_id", "store_id", "date"], how="left"
    )
    for col in rolling_cols:
        train[col] = train[col].fillna(0)
        val[col] = val[col].fillna(0)
        test[col] = test[col].fillna(0)

    # Store features
    train = create_store_features(stores, train)
    val = create_store_features(stores, val)
    test = create_store_features(stores, test)

    # Store aggregation
    train = create_store_aggregation_features(full_train[full_train["date"] < VAL_START], train)
    val = create_store_aggregation_features(full_train[full_train["date"] <= VAL_END], val)
    test = create_store_aggregation_features(full_train, test)

    # DOW statistics
    train = create_dow_statistics(full_train[full_train["date"] < VAL_START], train)
    val = create_dow_statistics(full_train[full_train["date"] <= VAL_END], val)
    test = create_dow_statistics(full_train, test)

    # Discounts
    disc = discounts.copy()
    disc["store_id"] = disc["store_id"].astype(str)
    train = create_discount_features(disc, train)
    val = create_discount_features(disc, val)
    test = create_discount_features(disc, test)

    # Price history
    ph = price_history.copy()
    ph["store_id"] = ph["store_id"].astype(str)
    train = create_price_history_features(ph, train, sales_full)
    val = create_price_history_features(ph, val, sales_full)
    test = create_price_history_features(ph, test, sales_full)

    # Markdowns
    md = markdowns.copy()
    md["store_id"] = md["store_id"].astype(str)
    train = create_markdown_features(md, train)
    val = create_markdown_features(md, val)
    test = create_markdown_features(md, test)

    # Availability
    am = actual_matrix.copy()
    am["store_id"] = am["store_id"].astype(str)
    train = create_availability_features(am, train)
    val = create_availability_features(am, val)
    test = create_availability_features(am, test)

    # Holidays
    train = create_holiday_features(train)
    val = create_holiday_features(val)
    test = create_holiday_features(test)

    # Online
    on = online.copy()
    on["store_id"] = on["store_id"].astype(str)
    train = create_online_features(on, train)
    val = create_online_features(on, val)
    test = create_online_features(on, test)

    # Feature columns
    r01_features = DATE_FEATURES + [
        f"quantity_lag_{lag}" for lag in LAG_DAYS
    ] + ["price_base_latest"] + CAT_FEATURES_R06[:4]

    rolling_features = rolling_cols

    store_agg_features = ["store_daily_qty", "item_store_qty_ratio"]
    dow_features = ["item_dow_mean", "item_dow_std"]
    discount_features = [
        "is_promo", "promo_type", "promo_discount_pct", "number_disc_day",
        "item_promo_freq_30d",
    ]
    price_hist_features = [
        "price_change_flag", "price_vs_base_ratio", "item_price_volatility_30d",
    ]
    markdown_features = ["is_markdown", "markdown_discount_pct"]
    store_meta_features = ["store_division", "store_format", "store_city", "store_area"]
    availability_features = ["item_available"]
    holiday_features = ["is_holiday", "days_to_next_holiday"]
    online_features = ["online_qty_7d", "has_online_sales"]
    cyclical_features = ["dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos"]

    feature_cols = (
        r01_features + rolling_features + store_agg_features + dow_features
        + discount_features + price_hist_features + markdown_features
        + store_meta_features + availability_features
        + holiday_features + online_features + cyclical_features
    )
    cat_features = CAT_FEATURES_R06

    log(f"  Full features: {len(feature_cols)} ({len(cat_features)} categorical)")
    return train, val, test, feature_cols, cat_features


# ===========================================================================
# Shared feature engineering functions
# ===========================================================================
def create_lag_features(sales_full, full_train, target_df, cutoff_date):
    """Create lag features."""
    qty_col = "total_qty"

    # Build history: original sales before grid + grid data
    pre_grid = sales_full[["date", "item_id", "store_id", "quantity"]].copy()
    pre_grid = pre_grid.rename(columns={"quantity": "total_qty"})
    pre_grid["store_id"] = pre_grid["store_id"].astype(str)

    if cutoff_date:
        post_grid = full_train[full_train["date"] <= cutoff_date][
            ["date", "item_id", "store_id", "total_qty"]
        ].copy()
    else:
        post_grid = full_train[["date", "item_id", "store_id", "total_qty"]].copy()

    history = pd.concat([pre_grid, post_grid], ignore_index=True)
    history = history.sort_values(["item_id", "store_id", "date"])

    if cutoff_date:
        history = history[history["date"] <= cutoff_date]

    daily_qty = (
        history.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(quantity=("total_qty", "sum"))
    )

    result = target_df.copy()
    for lag in LAG_DAYS:
        lag_col = f"quantity_lag_{lag}"
        shifted = daily_qty[["item_id", "store_id", "date", "quantity"]].copy()
        shifted["lag_date"] = shifted["date"] + pd.Timedelta(days=lag)
        shifted = shifted.rename(columns={"quantity": lag_col})
        shifted = shifted[["item_id", "store_id", "lag_date", lag_col]]
        result = result.merge(
            shifted,
            left_on=["item_id", "store_id", "date"],
            right_on=["item_id", "store_id", "lag_date"],
            how="left",
        )
        result = result.drop(columns=["lag_date"])
        result[lag_col] = result[lag_col].fillna(0)

    return result


def create_price_feature(sales_full, target_df):
    price_sales = sales_full[["date", "item_id", "price_base", "store_id"]].copy()
    price_sales["store_id"] = price_sales["store_id"].astype(str)
    price_sales["date"] = pd.to_datetime(price_sales["date"])
    price_sales = price_sales[price_sales["price_base"] > 0]
    latest_price = (
        price_sales.sort_values("date")
        .groupby(["item_id", "store_id"])
        .agg(price_base_latest=("price_base", "last"))
        .reset_index()
    )
    result = target_df.merge(latest_price, on=["item_id", "store_id"], how="left")
    median_price = latest_price["price_base_latest"].median()
    result["price_base_latest"] = result["price_base_latest"].fillna(median_price)
    return result


def merge_catalog(df, catalog):
    cat_subset = catalog[["item_id", "dept_name", "class_name"]].copy()
    cat_subset = cat_subset.drop_duplicates(subset=["item_id"])
    result = df.merge(cat_subset, on="item_id", how="left")
    result["dept_name"] = result["dept_name"].fillna("UNKNOWN")
    result["class_name"] = result["class_name"].fillna("UNKNOWN")
    return result


def create_store_features(stores, target_df):
    stores_r = stores.rename(columns={
        "division": "store_division", "format": "store_format",
        "city": "store_city", "area": "store_area",
    })
    result = target_df.merge(
        stores_r[["store_id", "store_division", "store_format", "store_city", "store_area"]],
        on="store_id", how="left"
    )
    for col in ["store_division", "store_format", "store_city"]:
        result[col] = result[col].fillna("UNKNOWN")
    result["store_area"] = result["store_area"].fillna(0)
    return result


def create_store_aggregation_features(train_df, target_df):
    daily = (
        train_df.groupby(["store_id", "date"], as_index=False)
        .agg(store_qty=("total_qty", "sum"))
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek
    store_dow = daily.groupby(["store_id", "day_of_week"], as_index=False).agg(
        store_daily_qty=("store_qty", "mean")
    )
    item_daily = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(item_qty=("total_qty", "sum"))
    )
    item_daily["day_of_week"] = item_daily["date"].dt.dayofweek
    item_dow = item_daily.groupby(["item_id", "store_id", "day_of_week"], as_index=False).agg(
        item_dow_mean_qty=("item_qty", "mean")
    )
    ratio = item_dow.merge(store_dow, on=["store_id", "day_of_week"], how="left")
    ratio["item_store_qty_ratio"] = np.where(
        ratio["store_daily_qty"] > 0,
        ratio["item_dow_mean_qty"] / ratio["store_daily_qty"], 0
    )
    ratio = ratio[["item_id", "store_id", "day_of_week", "store_daily_qty", "item_store_qty_ratio"]]
    result = target_df.copy()
    result = result.merge(ratio, on=["item_id", "store_id", "day_of_week"], how="left")
    result["store_daily_qty"] = result["store_daily_qty"].fillna(0)
    result["item_store_qty_ratio"] = result["item_store_qty_ratio"].fillna(0)
    return result


def create_dow_statistics(train_df, target_df):
    train_copy = train_df.copy()
    train_copy["day_of_week"] = train_copy["date"].dt.dayofweek
    dow_stats = (
        train_copy.groupby(["item_id", "day_of_week"])
        .agg(item_dow_mean=("total_qty", "mean"), item_dow_std=("total_qty", "std"))
        .reset_index()
    )
    result = target_df.copy()
    result = result.merge(dow_stats, on=["item_id", "day_of_week"], how="left")
    result["item_dow_mean"] = result["item_dow_mean"].fillna(0)
    result["item_dow_std"] = result["item_dow_std"].fillna(0)
    return result


def create_discount_features(discounts, target_df):
    disc = discounts.copy()
    disc["promo_type_code"] = disc["promo_type_code"].fillna(0).astype(str)
    disc_agg = (
        disc.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(
            sale_price_before=("sale_price_before_promo", "first"),
            sale_price_time=("sale_price_time_promo", "first"),
            promo_type_code=("promo_type_code", "first"),
            number_disc_day=("number_disc_day", "max"),
        )
    )
    disc_agg["promo_discount_pct"] = np.where(
        disc_agg["sale_price_before"] > 0,
        (disc_agg["sale_price_before"] - disc_agg["sale_price_time"])
        / disc_agg["sale_price_before"] * 100, 0
    )
    disc_agg["promo_discount_pct"] = disc_agg["promo_discount_pct"].clip(0, 100)
    disc_agg["is_promo"] = 1
    disc_lookup = disc_agg[["item_id", "store_id", "date", "is_promo",
                             "promo_discount_pct", "number_disc_day", "promo_type_code"]]
    result = target_df.merge(disc_lookup, on=["item_id", "store_id", "date"], how="left")
    result = result.rename(columns={"promo_type_code": "promo_type"})
    result["is_promo"] = result["is_promo"].fillna(0).astype(int)
    result["promo_discount_pct"] = result["promo_discount_pct"].fillna(0)
    result["number_disc_day"] = result["number_disc_day"].fillna(0)
    result["promo_type"] = result["promo_type"].fillna("NONE")
    result["item_promo_freq_30d"] = 0  # Simplified
    return result


def create_price_history_features(price_history, target_df, sales_full):
    ph = price_history.copy()
    ph = ph.sort_values(["item_id", "store_id", "date"])
    ph["prev_price"] = ph.groupby(["item_id", "store_id"])["price"].shift(1)
    ph["price_change_flag"] = (
        (ph["price"] != ph["prev_price"]) & ph["prev_price"].notna()
    ).astype(int)
    ph_lookup = ph[["item_id", "store_id", "date", "price", "price_change_flag"]].copy()
    result = target_df.merge(ph_lookup, on=["item_id", "store_id", "date"], how="left", suffixes=("", "_ph"))
    if "price_base_latest" in result.columns:
        result["price_vs_base_ratio"] = np.where(
            result["price_base_latest"] > 0,
            result["price"] / result["price_base_latest"], 1.0
        )
    else:
        result["price_vs_base_ratio"] = 1.0
    result["price_change_flag"] = result["price_change_flag"].fillna(0).astype(int)
    result["price_vs_base_ratio"] = result["price_vs_base_ratio"].fillna(1.0)
    result["item_price_volatility_30d"] = 0  # Simplified
    return result


def create_markdown_features(markdowns, target_df):
    md = markdowns.copy()
    md_agg = (
        md.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(normal_price=("normal_price", "mean"), md_price=("price", "mean"))
    )
    md_agg["markdown_discount_pct"] = np.where(
        md_agg["normal_price"] > 0,
        (md_agg["normal_price"] - md_agg["md_price"]) / md_agg["normal_price"] * 100, 0
    )
    md_agg["markdown_discount_pct"] = md_agg["markdown_discount_pct"].clip(0, 100)
    md_agg["is_markdown"] = 1
    result = target_df.merge(
        md_agg[["item_id", "store_id", "date", "is_markdown", "markdown_discount_pct"]],
        on=["item_id", "store_id", "date"], how="left"
    )
    result["is_markdown"] = result["is_markdown"].fillna(0).astype(int)
    result["markdown_discount_pct"] = result["markdown_discount_pct"].fillna(0)
    return result


def create_availability_features(actual_matrix, target_df):
    am = actual_matrix.copy()
    am["item_available"] = 1
    am = am.drop_duplicates(subset=["item_id", "store_id", "date"])
    result = target_df.merge(
        am[["item_id", "store_id", "date", "item_available"]],
        on=["item_id", "store_id", "date"], how="left"
    )
    result["item_available"] = result["item_available"].fillna(0).astype(int)
    return result


def create_holiday_features(target_df):
    result = target_df.copy()
    holidays = RUSSIAN_HOLIDAYS.sort_values().values
    result["is_holiday"] = result["date"].isin(holidays).astype(int)
    holiday_series = pd.Series(holidays).sort_values().reset_index(drop=True)
    all_dates = result["date"].unique()
    date_to_next = {}
    for d in sorted(all_dates):
        future = holiday_series[holiday_series >= pd.Timestamp(d)]
        date_to_next[d] = (future.iloc[0] - pd.Timestamp(d)).days if len(future) > 0 else 365
    result["days_to_next_holiday"] = result["date"].map(date_to_next)
    result["days_to_next_holiday"] = result["days_to_next_holiday"].fillna(365).astype(int)
    return result


def create_online_features(online, target_df):
    on = online[online["store_id"] == "1"].copy()
    online_daily = (
        on.groupby(["item_id", "date"], as_index=False)
        .agg(online_qty=("quantity", "sum"))
    )
    online_daily = online_daily.sort_values(["item_id", "date"])
    online_daily["online_qty_7d"] = (
        online_daily.groupby("item_id")["online_qty"]
        .shift(1).rolling(window=7, min_periods=1).sum().values
    )
    items_with_online = set(on["item_id"].unique())
    result = target_df.merge(
        online_daily[["item_id", "date", "online_qty_7d"]],
        on=["item_id", "date"], how="left"
    )
    result["online_qty_7d"] = result["online_qty_7d"].fillna(0)
    result["has_online_sales"] = result["item_id"].isin(items_with_online).astype(int)
    return result


# ===========================================================================
# Track A: AutoGluon
# ===========================================================================
def train_autogluon(X_train, y_train, X_val, y_val, cat_features):
    """Train AutoGluon multi-model ensemble."""
    from autogluon.tabular import TabularPredictor

    log("=" * 50)
    log("Track A: AutoGluon Multi-Model Ensemble")
    log("=" * 50)

    # Prepare data
    train_data = X_train.copy()
    train_data["target"] = y_train

    val_data = X_val.copy()
    val_data["target"] = y_val

    # Ensure categorical columns are string
    for col in cat_features:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype(str)
            val_data[col] = val_data[col].astype(str)

    ag_path = OUTPUT_DIR / "r08_autogluon"
    if ag_path.exists():
        import shutil
        shutil.rmtree(ag_path)

    log("Training AutoGluon (presets=medium_quality, time_limit=1800s)...")
    log(f"  Train: {train_data.shape}, Val: {val_data.shape}")

    predictor = TabularPredictor(
        label="target",
        eval_metric="rmse",
        path=str(ag_path),
    )

    predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        presets="medium_quality",
        time_limit=1800,  # 30 minutes
        excluded_model_types=["KNN", "NN_TORCH"],
        ag_args_fit={"drop_unique": False},
    )

    log("AutoGluon leaderboard:")
    lb = predictor.leaderboard(silent=True)
    for _, row in lb.head(10).iterrows():
        log(f"  {row['model']:40s}  RMSE={row.get('score_val', 0):.4f}")

    return predictor


# ===========================================================================
# Track B: Aggressive CatBoost (4th place style)
# ===========================================================================
def train_catboost_aggressive(X_train, y_train, X_val, y_val, cat_features):
    """Train aggressive CatBoost (depth=12, like 4th place)."""
    from catboost import CatBoostRegressor, Pool

    log("=" * 50)
    log("Track B: Aggressive CatBoost (depth=12, lr=0.3)")
    log("=" * 50)

    # Ensure categorical columns are string
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)

    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_features)

    params = {
        "iterations": 3000,
        "learning_rate": 0.3,
        "depth": 12,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
        "task_type": "CPU",
        "thread_count": -1,
        "random_seed": 42,
        "verbose": 200,
        "l2_leaf_reg": 2,
    }

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    log(f"  Best iteration: {model.get_best_iteration()}")
    log(f"  Best val RMSE: {model.get_best_score()['validation']['RMSE']:.4f}")

    # Feature importance
    importances = model.get_feature_importance()
    sorted_idx = np.argsort(importances)[::-1]
    log("Feature importance (top 20):")
    feature_names = X_train.columns.tolist()
    for i in sorted_idx[:20]:
        log(f"  {feature_names[i]:40s} {importances[i]:8.2f}")

    return model


# ===========================================================================
# Post-processing
# ===========================================================================
def postprocess_predictions(preds, train_df, test_df, catalog):
    max_qty = train_df["total_qty"].max()
    log(f"Clipping predictions to [0, {max_qty}]")
    preds = np.clip(preds, 0, max_qty)
    preds = np.nan_to_num(preds, nan=0.0)
    return preds


# ===========================================================================
# Main Pipeline
# ===========================================================================
def main():
    log("=" * 70)
    log("R08 AutoGluon Multi-Model Ensemble — FINAL SUBMISSION")
    log("=" * 70)

    t0 = time.time()

    # ----------------------------------------------------------------
    # 1. Build dense grid
    # ----------------------------------------------------------------
    log("--- Step 1: Build Dense Grid ---")
    dense_df = build_dense_grid_sql()

    # Ensure store_id is string everywhere
    dense_df["store_id"] = dense_df["store_id"].astype(str)
    dense_df["item_id"] = dense_df["item_id"].astype(str)

    log(f"  Dense grid: {dense_df.shape[0]:,} rows")
    log(f"  Zeros: {(dense_df['total_qty'] == 0).mean()*100:.1f}%")

    # ----------------------------------------------------------------
    # 2. Load data
    # ----------------------------------------------------------------
    log("--- Step 2: Load Data ---")
    test, sample_sub = load_test()
    catalog = load_catalog()
    stores = load_stores()
    discounts = pd.read_csv(DISCOUNTS_PATH, usecols=[
        "date", "item_id", "sale_price_before_promo",
        "sale_price_time_promo", "promo_type_code",
        "number_disc_day", "store_id"
    ])
    discounts["date"] = pd.to_datetime(discounts["date"])
    discounts["store_id"] = discounts["store_id"].astype(str)

    price_history = pd.read_csv(PRICE_HISTORY_PATH, usecols=["date", "item_id", "price", "store_id"])
    price_history["date"] = pd.to_datetime(price_history["date"])
    price_history["store_id"] = price_history["store_id"].astype(str)

    markdowns = pd.read_csv(MARKDOWNS_PATH, usecols=["date", "item_id", "normal_price", "price", "store_id"])
    markdowns["date"] = pd.to_datetime(markdowns["date"])
    markdowns["store_id"] = markdowns["store_id"].astype(str)

    actual_matrix = pd.read_csv(ACTUAL_MATRIX_PATH, usecols=["item_id", "date", "store_id"])
    actual_matrix["date"] = pd.to_datetime(actual_matrix["date"])
    actual_matrix["store_id"] = actual_matrix["store_id"].astype(str)

    online = pd.read_csv(ONLINE_PATH, usecols=["date", "item_id", "quantity", "store_id"])
    online["date"] = pd.to_datetime(online["date"])
    online["store_id"] = online["store_id"].astype(str)

    sales_full = pd.read_csv(SALES_PATH, usecols=["date", "item_id", "quantity", "price_base", "store_id"])
    sales_full["date"] = pd.to_datetime(sales_full["date"])
    sales_full["store_id"] = sales_full["store_id"].astype(str)
    sales_full = sales_full[sales_full["quantity"] >= 0].copy()
    log(f"  Full sales: {sales_full.shape}")

    # ----------------------------------------------------------------
    # 3. Outlier removal
    # ----------------------------------------------------------------
    log("--- Step 3: Outlier Removal ---")
    dense_df = remove_quantile_outliers(dense_df)

    # ----------------------------------------------------------------
    # 4. Split train/val
    # ----------------------------------------------------------------
    log("--- Step 4: Split Train/Val ---")
    train_mask = dense_df["date"] < VAL_START
    val_mask = (dense_df["date"] >= VAL_START) & (dense_df["date"] <= VAL_END)

    train_df = dense_df[train_mask].copy()
    val_df = dense_df[val_mask].copy()
    full_train = dense_df.copy()

    log(f"Train: {len(train_df):,}, Val: {len(val_df):,}")

    # ================================================================
    # Track A: AutoGluon with minimal features
    # ================================================================
    log("--- Track A: Minimal Features ---")
    train_a, val_a, test_a, feat_a, cat_a = create_minimal_features(
        train_df.copy(), val_df.copy(), test.copy(), full_train
    )

    X_train_a = train_a[feat_a].copy()
    y_train_a = train_a["total_qty"].values
    X_val_a = val_a[feat_a].copy()
    y_val_a = val_a["total_qty"].values
    X_test_a = test_a[feat_a].copy()

    # Ensure categorical columns are string for AutoGluon
    for col in cat_a:
        X_train_a[col] = X_train_a[col].astype(str)
        X_val_a[col] = X_val_a[col].astype(str)
        X_test_a[col] = X_test_a[col].astype(str)

    ag_predictor = train_autogluon(X_train_a, y_train_a, X_val_a, y_val_a, cat_a)

    # Predict with AutoGluon
    log("Predicting with AutoGluon...")
    preds_a = ag_predictor.predict(X_test_a)
    preds_a = np.clip(preds_a, 0, None)

    # AutoGluon val score
    val_preds_a = ag_predictor.predict(X_val_a)
    val_rmse_a = np.sqrt(np.mean((val_preds_a - y_val_a) ** 2))
    log(f"Track A val RMSE: {val_rmse_a:.4f}")

    # ================================================================
    # Track B: Aggressive CatBoost with full features
    # ================================================================
    log("--- Track B: Full Features ---")
    train_b, val_b, test_b, feat_b, cat_b = create_full_features(
        train_df.copy(), val_df.copy(), test.copy(), full_train,
        catalog, stores, discounts, price_history, markdowns,
        actual_matrix, online, sales_full
    )

    # Fill missing columns
    for col in feat_b:
        if col not in train_b.columns:
            train_b[col] = 0
            val_b[col] = 0
            test_b[col] = 0

    X_train_b = train_b[feat_b].copy()
    y_train_b = train_b["total_qty"].values
    X_val_b = val_b[feat_b].copy()
    y_val_b = val_b["total_qty"].values
    X_test_b = test_b[feat_b].copy()

    cb_model = train_catboost_aggressive(X_train_b, y_train_b, X_val_b, y_val_b, cat_b)

    # Predict with CatBoost
    from catboost import Pool
    log("Predicting with CatBoost...")
    test_pool_b = Pool(X_test_b, cat_features=cat_b)
    preds_b = cb_model.predict(test_pool_b)
    preds_b = np.clip(preds_b, 0, None)

    # CatBoost val score
    val_pool_b = Pool(X_val_b, cat_features=cat_b)
    val_preds_b = cb_model.predict(val_pool_b)
    val_rmse_b = np.sqrt(np.mean((val_preds_b - y_val_b) ** 2))
    log(f"Track B val RMSE: {val_rmse_b:.4f}")

    # ================================================================
    # Ensemble: Weighted average of Track A + Track B
    # ================================================================
    log("--- Ensemble ---")

    # Find optimal blend weight on validation set
    best_weight = 0.5
    best_rmse = float("inf")

    for w in np.arange(0.0, 1.01, 0.05):
        blend_val = w * val_preds_a + (1 - w) * val_preds_b
        rmse = np.sqrt(np.mean((blend_val - y_val_a) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = w

    log(f"  Optimal blend weight: Track A={best_weight:.2f}, Track B={1-best_weight:.2f}")
    log(f"  Ensemble val RMSE: {best_rmse:.4f}")

    # Final predictions
    preds = best_weight * preds_a + (1 - best_weight) * preds_b
    preds = postprocess_predictions(preds, dense_df, test, catalog)

    # ================================================================
    # Summary
    # ================================================================
    log("Prediction distribution:")
    log(f"  mean={preds.mean():.4f}, median={np.median(preds):.4f}")
    log(f"  min={preds.min():.4f}, max={preds.max():.4f}")
    log(f"  std={preds.std():.4f}")
    log(f"  pct zeros: {(preds == 0).mean()*100:.2f}%")

    # Generate submission
    submission = pd.DataFrame({
        "row_id": test["row_id"].values,
        "quantity": preds,
    })

    assert len(submission) == len(sample_sub)
    assert list(submission.columns) == list(sample_sub.columns)

    submission.to_csv(SUBMISSION_PATH, index=False)
    log(f"Submission saved to {SUBMISSION_PATH}")

    elapsed = time.time() - t0
    log("=" * 70)
    log("R08 Summary")
    log("=" * 70)
    log(f"Track A (AutoGluon minimal) val RMSE: {val_rmse_a:.4f}")
    log(f"Track B (CatBoost full) val RMSE:     {val_rmse_b:.4f}")
    log(f"Ensemble val RMSE:                     {best_rmse:.4f}")
    log(f"Blend weight: A={best_weight:.2f}, B={1-best_weight:.2f}")
    log(f"Total time: {elapsed:.1f}s")
    log(f"Submission: {SUBMISSION_PATH}")
    log("=" * 70)

    return best_rmse


if __name__ == "__main__":
    score = main()
    sys.exit(0)
