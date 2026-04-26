#!/usr/bin/env python3
"""
R06 Dense Cross-Join Grid: The #1 Missing Technique for ML Zoomcamp 2024.

Key insight from 2nd place solution (RMSE 9.23): create a COMPLETE
date x store x item grid where every combination gets a row. Missing
sales are explicitly 0, not NaN. This teaches the model that
"no sales" = 0.

Changes from R03 (our best, CV 21.55):
  1. Dense date x (item_id, store_id) cross-join grid (training data
     only; test is already dense). NaN quantities filled with 0.
  2. Offline + online quantity summed as target.
  3. Quantile outlier removal (P1-P99 per item x store x day_of_week).
  4. Cyclical time encoding: sin/cos for dow, month, week_of_year.
  5. Training window: 2024-03-01 onwards (Option C memory compromise).

All R03 features (50 features) are retained plus 6 new cyclical features.

Validation: time-based holdout (same as R01-R03)
  Train: 2024-03-01 ~ 2024-08-26 (dense grid)
  Val:   2024-08-27 ~ 2024-09-26 (dense grid)
  Test:  2024-09-27 ~ 2024-10-26 (already dense)

Usage:
    python scripts/run_r06_crossjoin.py
"""

import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

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

SUBMISSION_PATH = OUTPUT_DIR / "submission_r06_crossjoin.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID_START = "2024-03-01"  # Dense grid start date (Option C: last ~6 months)
VAL_START = "2024-08-27"
VAL_END = "2024-09-26"

CAT_FEATURES = [
    "item_id", "store_id", "dept_name", "class_name",
    "promo_type", "store_division", "store_format", "store_city",
]
DATE_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]
LAG_DAYS = [7, 14, 30]

# Rolling feature configuration
ROLL_MEAN_WINDOWS = [7, 14, 30, 60]
ROLL_STD_WINDOWS = [7, 14, 30]
ROLL_MIN_WINDOWS = [7, 30]
ROLL_MAX_WINDOWS = [7, 30]
EWM_SPANS = [7, 30]

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 50,
    "task_type": "CPU",
    "thread_count": -1,
    "random_seed": 42,
    "verbose": 100,
}

# ---------------------------------------------------------------------------
# Russian Holidays 2023-2024 (hardcoded to avoid dependency issues)
# ---------------------------------------------------------------------------
RUSSIAN_HOLIDAYS = pd.to_datetime([
    # 2023
    "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
    "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08",
    "2023-02-23", "2023-03-08", "2023-05-01", "2023-05-09",
    "2023-06-12", "2023-11-04",
    # 2024
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
    "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
    "2024-02-23", "2024-03-08", "2024-05-01", "2024-05-09",
    "2024-06-12", "2024-11-04",
    # 2024 extended holiday periods
    "2024-02-24", "2024-03-09",
    "2024-05-02", "2024-05-03", "2024-05-10",
    "2024-06-13", "2024-11-05",
])


def log(msg: str) -> None:
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Step 1: Build Dense Grid with DuckDB
# ===========================================================================
def build_dense_grid() -> pd.DataFrame:
    """
    Build the dense date x (item_id, store_id) grid using DuckDB.

    1. Collect all unique (item_id, store_id) combinations from
       sales + online + actual_matrix (for the grid period).
    2. Generate all dates from GRID_START to VAL_END.
    3. Cross-join dates x (item_id, store_id).
    4. Left join offline sales quantity -> fill NaN with 0.
    5. Left join online quantity -> fill NaN with 0.
    6. total_qty = offline_qty + online_qty.
    """
    log("Building dense grid with DuckDB...")

    con = duckdb.connect()

    # Load the relevant data into DuckDB
    log("  Loading sales into DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TABLE sales AS
        SELECT
            CAST(date AS DATE) AS date,
            item_id,
            quantity,
            price_base,
            store_id
        FROM read_csv('{SALES_PATH}', header=true, columns={{
            'Unnamed: 0': 'VARCHAR',
            'date': 'VARCHAR',
            'item_id': 'VARCHAR',
            'quantity': 'DOUBLE',
            'price_base': 'DOUBLE',
            'sum_total': 'DOUBLE',
            'store_id': 'VARCHAR'
        }})
        WHERE CAST(date AS DATE) >= '{GRID_START}'
          AND quantity >= 0
    """)

    log("  Loading online into DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TABLE online AS
        SELECT
            CAST(date AS DATE) AS date,
            item_id,
            quantity,
            store_id
        FROM read_csv('{ONLINE_PATH}', header=true, columns={{
            'Unnamed: 0': 'VARCHAR',
            'date': 'VARCHAR',
            'item_id': 'VARCHAR',
            'quantity': 'DOUBLE',
            'price_base': 'DOUBLE',
            'sum_total': 'DOUBLE',
            'store_id': 'VARCHAR'
        }})
        WHERE CAST(date AS DATE) >= '{GRID_START}'
          AND quantity >= 0
    """)

    log("  Loading actual_matrix into DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TABLE actual_matrix AS
        SELECT
            item_id,
            CAST(date AS DATE) AS date,
            store_id
        FROM read_csv('{ACTUAL_MATRIX_PATH}', header=true, columns={{
            'Unnamed: 0': 'VARCHAR',
            'item_id': 'VARCHAR',
            'date': 'VARCHAR',
            'store_id': 'VARCHAR'
        }})
        WHERE CAST(date AS DATE) >= '{GRID_START}'
    """)

    # Get all unique (item_id, store_id) combos from the grid period
    log("  Collecting unique (item_id, store_id) combos...")
    combo_count = con.execute("""
        WITH all_combos AS (
            SELECT DISTINCT item_id, store_id FROM sales
            UNION
            SELECT DISTINCT item_id, store_id FROM online
            UNION
            SELECT DISTINCT item_id, store_id FROM actual_matrix
        )
        SELECT COUNT(*) FROM all_combos
    """).fetchone()[0]
    log(f"  Unique (item_id, store_id) combos: {combo_count}")

    # Build dense grid
    log("  Creating dense cross-join grid...")
    dense_df = con.execute(f"""
        WITH all_combos AS (
            SELECT DISTINCT item_id, store_id FROM sales
            UNION
            SELECT DISTINCT item_id, store_id FROM online
            UNION
            SELECT DISTINCT item_id, store_id FROM actual_matrix
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
            CROSS JOIN all_combos c
        ),
        -- Aggregate offline sales per (date, item_id, store_id)
        sales_agg AS (
            SELECT date, item_id, store_id,
                   SUM(quantity) AS offline_qty,
                   AVG(price_base) AS price_base
            FROM sales
            GROUP BY date, item_id, store_id
        ),
        -- Aggregate online sales per (date, item_id, store_id)
        online_agg AS (
            SELECT date, item_id, store_id,
                   SUM(quantity) AS online_qty
            FROM online
            GROUP BY date, item_id, store_id
        )
        SELECT
            g.date,
            g.item_id,
            g.store_id,
            COALESCE(s.offline_qty, 0) AS offline_qty,
            COALESCE(s.price_base, 0) AS price_base,
            COALESCE(o.online_qty, 0) AS online_qty,
            COALESCE(s.offline_qty, 0) + COALESCE(o.online_qty, 0) AS total_qty
        FROM grid g
        LEFT JOIN sales_agg s ON g.date = s.date
            AND g.item_id = s.item_id AND g.store_id = s.store_id
        LEFT JOIN online_agg o ON g.date = o.date
            AND g.item_id = o.item_id AND g.store_id = o.store_id
        ORDER BY g.date, g.item_id, g.store_id
    """).fetchdf()

    con.close()

    # Normalize types: DuckDB returns VARCHAR, convert back to int64
    # to match CSV-loaded external data (catalog, discounts, etc.)
    dense_df["store_id"] = dense_df["store_id"].astype(int)
    dense_df["item_id"] = dense_df["item_id"].astype(int)

    log(f"  Dense grid shape: {dense_df.shape}")
    log(f"  Date range: {dense_df['date'].min()} ~ {dense_df['date'].max()}")
    log(f"  total_qty zeros: {(dense_df['total_qty'] == 0).mean()*100:.1f}%")
    log(f"  total_qty > 0:   {(dense_df['total_qty'] > 0).mean()*100:.1f}%")

    return dense_df


def build_dense_grid_fallback() -> pd.DataFrame:
    """
    Fallback: build dense grid with pandas if DuckDB fails.
    Memory-optimized: process in chunks.
    """
    log("Building dense grid with pandas (fallback)...")

    # Load data from GRID_START onwards
    sales = pd.read_csv(SALES_PATH, usecols=["date", "item_id", "quantity", "price_base", "store_id"])
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales[(sales["date"] >= GRID_START) & (sales["quantity"] >= 0)].copy()

    online = pd.read_csv(ONLINE_PATH, usecols=["date", "item_id", "quantity", "store_id"])
    online["date"] = pd.to_datetime(online["date"])
    online = online[(online["date"] >= GRID_START) & (online["quantity"] >= 0)].copy()

    am = pd.read_csv(ACTUAL_MATRIX_PATH, usecols=["item_id", "date", "store_id"])
    am["date"] = pd.to_datetime(am["date"])
    am = am[am["date"] >= GRID_START].copy()

    # All unique combos
    combos_sales = sales[["item_id", "store_id"]].drop_duplicates()
    combos_online = online[["item_id", "store_id"]].drop_duplicates()
    combos_am = am[["item_id", "store_id"]].drop_duplicates()
    all_combos = pd.concat([combos_sales, combos_online, combos_am]).drop_duplicates()
    log(f"  Unique (item_id, store_id) combos: {len(all_combos)}")

    # Date range
    dates = pd.date_range(GRID_START, VAL_END, freq="D")
    log(f"  Dates: {len(dates)} days")

    # Cross join
    log("  Creating cross-join grid...")
    dates_df = pd.DataFrame({"date": dates})
    all_combos["_tmp_key"] = 1
    dates_df["_tmp_key"] = 1
    grid = dates_df.merge(all_combos, on="_tmp_key").drop("_tmp_key", axis=1)
    log(f"  Grid shape: {grid.shape}")

    # Aggregate offline sales
    sales_agg = (
        sales.groupby(["date", "item_id", "store_id"])
        .agg(offline_qty=("quantity", "sum"), price_base=("price_base", "mean"))
        .reset_index()
    )

    # Aggregate online sales
    online_agg = (
        online.groupby(["date", "item_id", "store_id"])
        .agg(online_qty=("quantity", "sum"))
        .reset_index()
    )

    # Left join
    grid = grid.merge(sales_agg, on=["date", "item_id", "store_id"], how="left")
    grid = grid.merge(online_agg, on=["date", "item_id", "store_id"], how="left")

    grid["offline_qty"] = grid["offline_qty"].fillna(0)
    grid["online_qty"] = grid["online_qty"].fillna(0)
    grid["price_base"] = grid["price_base"].fillna(0)
    grid["total_qty"] = grid["offline_qty"] + grid["online_qty"]

    grid = grid.sort_values(["date", "item_id", "store_id"]).reset_index(drop=True)

    log(f"  Dense grid shape: {grid.shape}")
    log(f"  total_qty zeros: {(grid['total_qty'] == 0).mean()*100:.1f}%")

    return grid


# ===========================================================================
# Step 3: Quantile Outlier Removal
# ===========================================================================
def remove_quantile_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where total_qty falls outside P1-P99 per
    (item_id, store_id, day_of_week). Only applied to rows with qty > 0.
    Zero rows are never removed.
    """
    log("Applying quantile outlier removal (P1-P99 per item x store x dow)...")

    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek

    # Only compute bounds for positive-quantity rows
    pos_mask = df["total_qty"] > 0
    log(f"  Positive quantity rows: {pos_mask.sum():,} / {len(df):,} "
        f"({pos_mask.mean()*100:.1f}%)")

    grouped = df.loc[pos_mask].groupby(["item_id", "store_id", "day_of_week"])["total_qty"]
    lower = grouped.quantile(0.01).rename("q01")
    upper = grouped.quantile(0.99).rename("q99")
    bounds = pd.concat([lower, upper], axis=1).reset_index()

    log(f"  Outlier bounds computed for {len(bounds)} groups")

    # Merge bounds back
    before = len(df)
    df = df.merge(bounds, on=["item_id", "store_id", "day_of_week"], how="left")

    # Keep zero-quantity rows always; for positive rows, keep only if within bounds
    keep_mask = (df["total_qty"] == 0) | (
        (df["total_qty"] >= df["q01"]) & (df["total_qty"] <= df["q99"])
    )
    df = df[keep_mask].drop(columns=["q01", "q99"]).reset_index(drop=True)

    removed = before - len(df)
    log(f"  Removed {removed:,} outliers ({removed/before*100:.2f}%)")
    log(f"  After removal: {len(df):,} rows")

    return df


# ===========================================================================
# Data Loading (external sources)
# ===========================================================================
def load_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load test and sample submission."""
    log("Loading test data...")
    test = pd.read_csv(TEST_PATH, sep=";")
    test["date"] = pd.to_datetime(test["date"], format="%d.%m.%Y")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    log(f"  test: {test.shape}, date range: {test['date'].min()} ~ {test['date'].max()}")
    log(f"  sample_sub: {sample_sub.shape}")
    return test, sample_sub


def load_catalog() -> pd.DataFrame:
    """Load catalog.csv."""
    log("Loading catalog...")
    catalog = pd.read_csv(CATALOG_PATH, index_col=0)
    log(f"  catalog: {catalog.shape}")
    return catalog


def load_discounts() -> pd.DataFrame:
    """Load discounts_history.csv with only needed columns."""
    log("Loading discounts_history...")
    disc = pd.read_csv(
        DISCOUNTS_PATH,
        usecols=["date", "item_id", "sale_price_before_promo",
                  "sale_price_time_promo", "promo_type_code",
                  "number_disc_day", "store_id"],
    )
    disc["date"] = pd.to_datetime(disc["date"])
    log(f"  discounts: {disc.shape}")
    return disc


def load_price_history() -> pd.DataFrame:
    """Load price_history.csv."""
    log("Loading price_history...")
    ph = pd.read_csv(PRICE_HISTORY_PATH, usecols=["date", "item_id", "price", "store_id"])
    ph["date"] = pd.to_datetime(ph["date"])
    log(f"  price_history: {ph.shape}")
    return ph


def load_markdowns() -> pd.DataFrame:
    """Load markdowns.csv."""
    log("Loading markdowns...")
    md = pd.read_csv(MARKDOWNS_PATH, usecols=["date", "item_id", "normal_price", "price", "store_id"])
    md["date"] = pd.to_datetime(md["date"])
    log(f"  markdowns: {md.shape}")
    return md


def load_actual_matrix() -> pd.DataFrame:
    """Load actual_matrix.csv."""
    log("Loading actual_matrix...")
    am = pd.read_csv(ACTUAL_MATRIX_PATH, usecols=["item_id", "date", "store_id"])
    am["date"] = pd.to_datetime(am["date"])
    log(f"  actual_matrix: {am.shape}")
    return am


def load_stores() -> pd.DataFrame:
    """Load stores.csv."""
    log("Loading stores...")
    stores = pd.read_csv(STORES_PATH)
    log(f"  stores: {stores.shape}")
    return stores


def load_online_raw() -> pd.DataFrame:
    """Load online.csv for feature engineering (full date range)."""
    log("Loading online sales for features...")
    on = pd.read_csv(ONLINE_PATH, usecols=["date", "item_id", "quantity", "store_id"])
    on["date"] = pd.to_datetime(on["date"])
    log(f"  online: {on.shape}")
    return on


def load_sales_for_lags() -> pd.DataFrame:
    """
    Load the full sales data (for lag/rolling computations that need
    historical data before GRID_START).
    """
    log("Loading full sales data for lag features...")
    sales = pd.read_csv(SALES_PATH, usecols=["date", "item_id", "quantity", "price_base", "store_id"])
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales[sales["quantity"] >= 0].copy()
    log(f"  Full sales: {sales.shape}, date range: {sales['date'].min()} ~ {sales['date'].max()}")
    return sales


# ===========================================================================
# Feature Engineering (from R03, adapted for dense grid)
# ===========================================================================
def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date-based features from the date column."""
    df = df.copy()
    dt = df["date"]
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin/cos encoding for day_of_week, month, week_of_year.
    From 4th/5th place solutions.
    """
    df = df.copy()
    # Day of week: cycle of 7
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Month: cycle of 12
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Week of year: cycle of 52
    week = df["date"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)
    return df


def create_lag_features(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create lag features for target_df using data from history_df.

    history_df: full sales data (all dates) with 'total_qty' as the
                quantity column (offline + online merged).
    target_df: the dataframe to add lag features to.
    """
    log("Creating lag features...")

    # Build daily quantity lookup from history
    # Use total_qty if available, otherwise quantity
    qty_col = "total_qty" if "total_qty" in history_df.columns else "quantity"
    daily_qty = (
        history_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(quantity=(qty_col, "sum"))
    )
    log(f"  Daily quantity lookup: {len(daily_qty)} rows, "
        f"{daily_qty['item_id'].nunique()} items, "
        f"{daily_qty['store_id'].nunique()} stores")

    result = target_df.copy()

    for lag in LAG_DAYS:
        lag_col = f"quantity_lag_{lag}"
        log(f"  Computing {lag_col}...")

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

        fill_rate = result[lag_col].notna().mean()
        log(f"    {lag_col} fill rate: {fill_rate:.4f}")

    for lag in LAG_DAYS:
        lag_col = f"quantity_lag_{lag}"
        result[lag_col] = result[lag_col].fillna(0)

    return result


def create_price_feature(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add latest known price_base per (item_id, store_id)."""
    log("Creating price feature...")

    qty_col = "price_base" if "price_base" in history_df.columns else None
    if qty_col is None or history_df["price_base"].sum() == 0:
        # If price_base is not in history (e.g., dense grid with 0 prices),
        # load original sales for prices
        log("  Loading price data from original sales...")
        price_sales = pd.read_csv(
            SALES_PATH,
            usecols=["date", "item_id", "price_base", "store_id"],
        )
        price_sales["date"] = pd.to_datetime(price_sales["date"])
        price_sales = price_sales[price_sales["price_base"] > 0]
        latest_price = (
            price_sales.sort_values("date")
            .groupby(["item_id", "store_id"])
            .agg(price_base_latest=("price_base", "last"))
            .reset_index()
        )
    else:
        latest_price = (
            history_df[history_df["price_base"] > 0]
            .sort_values("date")
            .groupby(["item_id", "store_id"])
            .agg(price_base_latest=("price_base", "last"))
            .reset_index()
        )

    result = target_df.merge(latest_price, on=["item_id", "store_id"], how="left")
    fill_rate = result["price_base_latest"].notna().mean()
    log(f"  price_base_latest fill rate: {fill_rate:.4f}")

    # Fill missing with global median
    median_price = latest_price["price_base_latest"].median()
    result["price_base_latest"] = result["price_base_latest"].fillna(median_price)
    return result


def merge_catalog(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Merge catalog features: dept_name, class_name."""
    cat_subset = catalog[["item_id", "dept_name", "class_name"]].copy()
    cat_subset = cat_subset.drop_duplicates(subset=["item_id"])
    result = df.merge(cat_subset, on="item_id", how="left")
    missing = result["dept_name"].isna().sum()
    log(f"Merged catalog features. Rows with missing dept_name: {missing}")
    result["dept_name"] = result["dept_name"].fillna("UNKNOWN")
    result["class_name"] = result["class_name"].fillna("UNKNOWN")
    return result


def _build_daily_series(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build daily time series per (item_id, store_id) from training data."""
    qty_col = "total_qty" if "total_qty" in train_df.columns else "quantity"
    daily = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(qty=(qty_col, "sum"))
    )
    daily = daily.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    return daily


def create_rolling_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create rolling window statistics for target_df using training data only."""
    log("Creating rolling features...")

    daily = _build_daily_series(train_df)
    log(f"  Daily series: {len(daily)} rows, "
        f"{daily['item_id'].nunique()} items, "
        f"{daily['store_id'].nunique()} stores")

    grouped = daily.groupby(["item_id", "store_id"])

    for w in ROLL_MEAN_WINDOWS:
        col = f"qty_roll_mean_{w}"
        daily[col] = grouped["qty"].shift(1).rolling(window=w, min_periods=1).mean().values

    for w in ROLL_STD_WINDOWS:
        col = f"qty_roll_std_{w}"
        daily[col] = grouped["qty"].shift(1).rolling(window=w, min_periods=1).std().values

    for w in ROLL_MIN_WINDOWS:
        col = f"qty_roll_min_{w}"
        daily[col] = grouped["qty"].shift(1).rolling(window=w, min_periods=1).min().values

    for w in ROLL_MAX_WINDOWS:
        col = f"qty_roll_max_{w}"
        daily[col] = grouped["qty"].shift(1).rolling(window=w, min_periods=1).max().values

    for span in EWM_SPANS:
        col = f"qty_ewm_{span}"
        daily[col] = grouped["qty"].shift(1).ewm(span=span, min_periods=1).mean().values

    rolling_cols = (
        [f"qty_roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
        + [f"qty_roll_std_{w}" for w in ROLL_STD_WINDOWS]
        + [f"qty_roll_min_{w}" for w in ROLL_MIN_WINDOWS]
        + [f"qty_roll_max_{w}" for w in ROLL_MAX_WINDOWS]
        + [f"qty_ewm_{span}" for span in EWM_SPANS]
    )

    target_dates = target_df["date"].unique()
    target_groups = target_df[["item_id", "store_id"]].drop_duplicates()

    log("  Extending daily series to cover target dates...")
    target_date_rows = []
    for date in target_dates:
        frame = target_groups.copy()
        frame["date"] = date
        frame["qty"] = 0
        target_date_rows.append(frame)

    target_date_df = pd.concat(target_date_rows, ignore_index=True)

    daily["_is_original"] = True
    target_date_df["_is_original"] = False
    for col in rolling_cols:
        target_date_df[col] = np.nan

    combined = pd.concat([daily, target_date_df], ignore_index=True)
    combined = combined.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    log("  Forward-filling rolling stats...")
    for col in rolling_cols:
        combined[col] = combined.groupby(["item_id", "store_id"])[col].ffill()

    lookup = combined[~combined["_is_original"]].copy()
    lookup = lookup.drop(columns=["_is_original", "qty"])

    log(f"  Lookup table for merge: {len(lookup)} rows")

    result = target_df.merge(lookup, on=["item_id", "store_id", "date"], how="left")

    for col in rolling_cols:
        fill_rate = result[col].notna().mean()
        result[col] = result[col].fillna(0)
        log(f"  {col}: fill rate={fill_rate:.4f}")

    return result


def create_store_aggregation_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create store-level aggregation features."""
    log("Creating store aggregation features...")

    qty_col = "total_qty" if "total_qty" in train_df.columns else "quantity"

    train_daily = (
        train_df.groupby(["store_id", "date"], as_index=False)
        .agg(store_qty=(qty_col, "sum"))
    )
    train_daily["day_of_week"] = train_daily["date"].dt.dayofweek

    store_dow_qty = (
        train_daily.groupby(["store_id", "day_of_week"], as_index=False)
        .agg(store_daily_qty=("store_qty", "mean"))
    )

    train_item_daily = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(item_qty=(qty_col, "sum"))
    )
    train_item_daily["day_of_week"] = train_item_daily["date"].dt.dayofweek

    item_dow_qty = (
        train_item_daily.groupby(["item_id", "store_id", "day_of_week"], as_index=False)
        .agg(item_dow_mean_qty=("item_qty", "mean"))
    )

    item_store_ratio = item_dow_qty.merge(store_dow_qty, on=["store_id", "day_of_week"], how="left")
    item_store_ratio["item_store_qty_ratio"] = np.where(
        item_store_ratio["store_daily_qty"] > 0,
        item_store_ratio["item_dow_mean_qty"] / item_store_ratio["store_daily_qty"],
        0,
    )
    item_store_ratio = item_store_ratio[["item_id", "store_id", "day_of_week",
                                         "store_daily_qty", "item_store_qty_ratio"]]

    result = target_df.copy()
    result = result.merge(item_store_ratio, on=["item_id", "store_id", "day_of_week"], how="left")

    result["store_daily_qty"] = result["store_daily_qty"].fillna(0)
    result["item_store_qty_ratio"] = result["item_store_qty_ratio"].fillna(0)

    log(f"  store_daily_qty fill rate: {result['store_daily_qty'].notna().mean():.4f}")
    log(f"  item_store_qty_ratio fill rate: {result['item_store_qty_ratio'].notna().mean():.4f}")

    return result


def create_dow_statistics(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create day-of-week statistics per item."""
    log("Creating day-of-week statistics...")

    qty_col = "total_qty" if "total_qty" in train_df.columns else "quantity"

    train_copy = train_df.copy()
    train_copy["day_of_week"] = train_copy["date"].dt.dayofweek

    dow_stats = (
        train_copy.groupby(["item_id", "day_of_week"])
        .agg(
            item_dow_mean=(qty_col, "mean"),
            item_dow_std=(qty_col, "std"),
        )
        .reset_index()
    )

    result = target_df.copy()
    result = result.merge(dow_stats, on=["item_id", "day_of_week"], how="left")

    result["item_dow_mean"] = result["item_dow_mean"].fillna(0)
    result["item_dow_std"] = result["item_dow_std"].fillna(0)

    log(f"  item_dow_mean fill rate: {result['item_dow_mean'].notna().mean():.4f}")
    log(f"  item_dow_std fill rate:  {result['item_dow_std'].notna().mean():.4f}")

    return result


# ===========================================================================
# R03 Multi-Source Features
# ===========================================================================
def create_discount_features(
    discounts: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create discount/promotion features from discounts_history."""
    log("Creating discount/promotion features...")

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
        / disc_agg["sale_price_before"] * 100,
        0,
    )
    disc_agg["promo_discount_pct"] = disc_agg["promo_discount_pct"].clip(0, 100)
    disc_agg["is_promo"] = 1

    promo_daily = disc_agg[["item_id", "store_id", "date", "is_promo",
                             "promo_discount_pct", "number_disc_day",
                             "promo_type_code"]].copy()
    promo_daily = promo_daily.sort_values(["item_id", "store_id", "date"])

    promo_daily["item_promo_freq_30d"] = (
        promo_daily.groupby(["item_id", "store_id"])["is_promo"]
        .shift(1)
        .rolling(window=30, min_periods=1)
        .sum()
        .values
    )

    promo_lookup_cols = [
        "item_id", "store_id", "date", "is_promo",
        "promo_discount_pct", "number_disc_day",
        "promo_type_code", "item_promo_freq_30d",
    ]
    promo_lookup = promo_daily[promo_lookup_cols].copy()

    target_dates = target_df["date"].unique()
    target_groups = target_df[["item_id", "store_id"]].drop_duplicates()

    target_date_rows = []
    for date in target_dates:
        frame = target_groups.copy()
        frame["date"] = date
        target_date_rows.append(frame)
    target_date_df = pd.concat(target_date_rows, ignore_index=True)
    target_date_df["is_promo"] = np.nan
    target_date_df["promo_discount_pct"] = np.nan
    target_date_df["number_disc_day"] = np.nan
    target_date_df["promo_type_code"] = np.nan
    target_date_df["item_promo_freq_30d"] = np.nan

    promo_lookup["_is_original"] = True
    target_date_df["_is_original"] = False

    ffill_cols = ["is_promo", "promo_discount_pct", "number_disc_day", "item_promo_freq_30d"]
    combined = pd.concat([promo_lookup, target_date_df], ignore_index=True)
    combined = combined.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    for col in ffill_cols:
        combined[col] = combined.groupby(["item_id", "store_id"])[col].ffill()

    combined["promo_type_code"] = combined.groupby(["item_id", "store_id"])["promo_type_code"].ffill()

    lookup = combined[~combined["_is_original"]].copy()
    lookup = lookup.drop(columns=["_is_original"])
    log(f"  Discount lookup: {len(lookup)} rows")

    result = target_df.merge(lookup, on=["item_id", "store_id", "date"], how="left")

    result = result.rename(columns={"promo_type_code": "promo_type"})

    result["is_promo"] = result["is_promo"].fillna(0).astype(int)
    result["promo_discount_pct"] = result["promo_discount_pct"].fillna(0)
    result["number_disc_day"] = result["number_disc_day"].fillna(0)
    result["promo_type"] = result["promo_type"].fillna("NONE")
    result["item_promo_freq_30d"] = result["item_promo_freq_30d"].fillna(0)

    log(f"  is_promo fill rate: {(result['is_promo'] == 1).mean():.4f}")
    log(f"  promo_discount_pct > 0: {(result['promo_discount_pct'] > 0).mean():.4f}")

    return result


def create_price_history_features(
    price_history: pd.DataFrame,
    target_df: pd.DataFrame,
    sales_train: pd.DataFrame,
) -> pd.DataFrame:
    """Create price history features from price_history.csv."""
    log("Creating price history features...")

    ph = price_history.copy()
    ph = ph.sort_values(["item_id", "store_id", "date"])

    ph["prev_price"] = ph.groupby(["item_id", "store_id"])["price"].shift(1)
    ph["price_change_flag"] = (
        (ph["price"] != ph["prev_price"]) & ph["prev_price"].notna()
    ).astype(int)

    ph["price_diff"] = ph.groupby(["item_id", "store_id"])["price"].diff()
    ph["item_price_volatility_30d"] = (
        ph.groupby(["item_id", "store_id"])["price_diff"]
        .shift(1)
        .rolling(window=30, min_periods=1)
        .std()
        .values
    )

    ph_lookup = ph[["item_id", "store_id", "date", "price", "price_change_flag",
                     "item_price_volatility_30d"]].copy()

    target_dates = target_df["date"].unique()
    target_groups = target_df[["item_id", "store_id"]].drop_duplicates()

    target_date_rows = []
    for date in target_dates:
        frame = target_groups.copy()
        frame["date"] = date
        target_date_rows.append(frame)
    target_date_df = pd.concat(target_date_rows, ignore_index=True)
    target_date_df["price"] = np.nan
    target_date_df["price_change_flag"] = np.nan
    target_date_df["item_price_volatility_30d"] = np.nan

    ph_lookup["_is_original"] = True
    target_date_df["_is_original"] = False

    combined = pd.concat([ph_lookup, target_date_df], ignore_index=True)
    combined = combined.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    ffill_cols = ["price", "item_price_volatility_30d"]
    for col in ffill_cols:
        combined[col] = combined.groupby(["item_id", "store_id"])[col].ffill()

    lookup = combined[~combined["_is_original"]].copy()
    lookup = lookup.drop(columns=["_is_original"])
    log(f"  Price history lookup: {len(lookup)} rows")

    result = target_df.merge(lookup, on=["item_id", "store_id", "date"], how="left", suffixes=("", "_ph"))

    if "price_base_latest" in result.columns:
        result["price_vs_base_ratio"] = np.where(
            result["price_base_latest"] > 0,
            result["price"] / result["price_base_latest"],
            1.0,
        )
    else:
        result["price_vs_base_ratio"] = 1.0

    result["price_change_flag"] = result["price_change_flag"].fillna(0).astype(int)
    result["item_price_volatility_30d"] = result["item_price_volatility_30d"].fillna(0)
    result["price_vs_base_ratio"] = result["price_vs_base_ratio"].fillna(1.0)
    result["price"] = result["price"].fillna(result.get("price_base_latest", 0))

    log(f"  price_change_flag rate: {result['price_change_flag'].mean():.4f}")
    log(f"  price_vs_base_ratio mean: {result['price_vs_base_ratio'].mean():.4f}")

    return result


def create_markdown_features(
    markdowns: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create markdown features from markdowns.csv."""
    log("Creating markdown features...")

    md = markdowns.copy()

    md_agg = (
        md.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(
            normal_price=("normal_price", "mean"),
            md_price=("price", "mean"),
        )
    )
    md_agg["markdown_discount_pct"] = np.where(
        md_agg["normal_price"] > 0,
        (md_agg["normal_price"] - md_agg["md_price"]) / md_agg["normal_price"] * 100,
        0,
    )
    md_agg["markdown_discount_pct"] = md_agg["markdown_discount_pct"].clip(0, 100)
    md_agg["is_markdown"] = 1

    md_lookup = md_agg[["item_id", "store_id", "date", "is_markdown", "markdown_discount_pct"]].copy()

    result = target_df.merge(md_lookup, on=["item_id", "store_id", "date"], how="left")

    result["is_markdown"] = result["is_markdown"].fillna(0).astype(int)
    result["markdown_discount_pct"] = result["markdown_discount_pct"].fillna(0)

    log(f"  is_markdown rate: {result['is_markdown'].mean():.4f}")

    return result


def create_store_features(
    stores: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge store metadata from stores.csv."""
    log("Creating store metadata features...")

    stores_renamed = stores.rename(columns={
        "division": "store_division",
        "format": "store_format",
        "city": "store_city",
        "area": "store_area",
    })

    result = target_df.merge(
        stores_renamed[["store_id", "store_division", "store_format", "store_city", "store_area"]],
        on="store_id",
        how="left",
    )

    result["store_division"] = result["store_division"].fillna("UNKNOWN")
    result["store_format"] = result["store_format"].fillna("UNKNOWN")
    result["store_city"] = result["store_city"].fillna("UNKNOWN")
    result["store_area"] = result["store_area"].fillna(0)

    log(f"  store_division values: {result['store_division'].unique().tolist()}")
    log(f"  store_format values: {result['store_format'].unique().tolist()}")

    return result


def create_availability_features(
    actual_matrix: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create availability feature from actual_matrix.csv."""
    log("Creating availability features...")

    am = actual_matrix.copy()
    am["item_available"] = 1
    am = am.drop_duplicates(subset=["item_id", "store_id", "date"])

    result = target_df.merge(
        am[["item_id", "store_id", "date", "item_available"]],
        on=["item_id", "store_id", "date"],
        how="left",
    )

    result["item_available"] = result["item_available"].fillna(0).astype(int)

    log(f"  item_available rate: {result['item_available'].mean():.4f}")

    return result


def create_holiday_features(target_df: pd.DataFrame) -> pd.DataFrame:
    """Create Russian holiday features."""
    log("Creating Russian holiday features...")

    result = target_df.copy()
    holidays = RUSSIAN_HOLIDAYS.sort_values().values

    result["is_holiday"] = result["date"].isin(holidays).astype(int)

    holiday_series = pd.Series(holidays).sort_values().reset_index(drop=True)

    all_dates = result["date"].unique()
    date_to_next = {}
    for d in sorted(all_dates):
        future = holiday_series[holiday_series >= pd.Timestamp(d)]
        if len(future) > 0:
            date_to_next[d] = (future.iloc[0] - pd.Timestamp(d)).days
        else:
            date_to_next[d] = 365

    result["days_to_next_holiday"] = result["date"].map(date_to_next)
    result["days_to_next_holiday"] = result["days_to_next_holiday"].fillna(365).astype(int)

    log(f"  is_holiday rate: {result['is_holiday'].mean():.4f}")
    log(f"  days_to_next_holiday range: {result['days_to_next_holiday'].min()} ~ {result['days_to_next_holiday'].max()}")

    return result


def create_online_features(
    online: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create online sales features from online.csv."""
    log("Creating online sales features...")

    on = online.copy()
    on = on[on["store_id"] == 1].copy()

    online_daily = (
        on.groupby(["item_id", "date"], as_index=False)
        .agg(online_qty=("quantity", "sum"))
    )
    online_daily = online_daily.sort_values(["item_id", "date"])

    online_daily["online_qty_7d"] = (
        online_daily.groupby("item_id")["online_qty"]
        .shift(1)
        .rolling(window=7, min_periods=1)
        .sum()
        .values
    )

    items_with_online = set(on["item_id"].unique())
    log(f"  Items with online sales: {len(items_with_online)}")

    online_lookup = online_daily[["item_id", "date", "online_qty_7d"]].copy()

    target_dates = target_df["date"].unique()
    target_items = target_df["item_id"].unique()

    target_date_rows = []
    for item_id in target_items:
        if item_id in items_with_online:
            for date in target_dates:
                target_date_rows.append({"item_id": item_id, "date": date, "online_qty_7d": np.nan})

    if target_date_rows:
        target_date_df = pd.DataFrame(target_date_rows)
        online_lookup["_is_original"] = True
        target_date_df["_is_original"] = False

        combined = pd.concat([online_lookup, target_date_df], ignore_index=True)
        combined = combined.sort_values(["item_id", "date"]).reset_index(drop=True)
        combined["online_qty_7d"] = combined.groupby("item_id")["online_qty_7d"].ffill()

        lookup = combined[~combined["_is_original"]].copy()
        lookup = lookup.drop(columns=["_is_original"])
    else:
        lookup = pd.DataFrame(columns=["item_id", "date", "online_qty_7d"])

    log(f"  Online lookup: {len(lookup)} rows")

    result = target_df.merge(lookup, on=["item_id", "date"], how="left")

    result["online_qty_7d"] = result["online_qty_7d"].fillna(0)
    result["has_online_sales"] = (result["item_id"].isin(items_with_online)).astype(int)

    log(f"  has_online_sales rate: {result['has_online_sales'].mean():.4f}")
    log(f"  online_qty_7d mean: {result['online_qty_7d'].mean():.4f}")

    return result


# ===========================================================================
# Training
# ===========================================================================
def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: list[str],
) -> CatBoostRegressor:
    """Train CatBoost regressor with early stopping."""
    log("Training CatBoost...")
    log(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    log(f"  Cat features: {cat_features}")

    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_features)

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    log(f"  Best iteration: {model.get_best_iteration()}")
    log(f"  Best val RMSE: {model.get_best_score()['validation']['RMSE']:.4f}")

    return model


def show_feature_importance(model: CatBoostRegressor, feature_names: list[str]) -> None:
    """Print feature importance."""
    importances = model.get_feature_importance()
    sorted_idx = np.argsort(importances)[::-1]
    log("Feature importance (top 30):")
    for i in sorted_idx[:30]:
        log(f"  {feature_names[i]:40s} {importances[i]:8.2f}")


# ===========================================================================
# Post-processing
# ===========================================================================
def postprocess_predictions(
    preds: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog: pd.DataFrame,
) -> np.ndarray:
    """
    Clip predictions to [0, max observed quantity].
    For cold items (not in train), predict class-level mean.
    Fill any remaining NaN with 0.
    """
    qty_col = "total_qty" if "total_qty" in train_df.columns else "quantity"
    max_qty = train_df[qty_col].max()
    log(f"Clipping predictions to [0, {max_qty}]")
    preds = np.clip(preds, 0, max_qty)

    nan_count = np.isnan(preds).sum()
    if nan_count > 0:
        log(f"  WARNING: {nan_count} NaN predictions found, filling with 0")
        preds = np.nan_to_num(preds, nan=0.0)

    train_items = set(train_df["item_id"].unique())
    cold_mask = ~test_df["item_id"].isin(train_items)
    n_cold = cold_mask.sum()
    n_cold_items = test_df.loc[cold_mask, "item_id"].nunique()
    log(f"Cold items: {n_cold_items} items ({n_cold} rows, {cold_mask.mean()*100:.2f}%)")

    if n_cold > 0:
        train_item_mean = train_df.groupby("item_id")[qty_col].mean().reset_index()
        train_item_mean.columns = ["item_id", "item_mean_qty"]

        catalog_subset = catalog[["item_id", "dept_name"]].drop_duplicates(subset=["item_id"])
        train_item_mean = train_item_mean.merge(catalog_subset, on="item_id", how="left")

        dept_mean = train_item_mean.groupby("dept_name")["item_mean_qty"].mean()
        overall_mean = train_item_mean["item_mean_qty"].mean()

        cold_items = test_df.loc[cold_mask, "item_id"].unique()
        cold_catalog = catalog_subset[catalog_subset["item_id"].isin(cold_items)]
        cold_item_dept_mean = cold_catalog.set_index("item_id")["dept_name"].map(dept_mean)
        cold_item_dept_mean = cold_item_dept_mean.fillna(overall_mean)

        cold_dept_means = test_df.loc[cold_mask, "item_id"].map(cold_item_dept_mean)
        preds[cold_mask] = cold_dept_means.values
        log(f"  Set cold item predictions to dept-level mean (overall mean={overall_mean:.4f})")

    preds = np.nan_to_num(preds, nan=0.0)
    return preds


# ===========================================================================
# Main Pipeline
# ===========================================================================
def main() -> float:
    log("=" * 70)
    log("R06 Dense Cross-Join Grid: The #1 Missing Technique")
    log("=" * 70)

    t0 = time.time()

    # ----------------------------------------------------------------
    # 1. Build dense grid (THE KEY CHANGE)
    # ----------------------------------------------------------------
    log("--- Step 1: Build Dense Grid ---")
    try:
        dense_df = build_dense_grid()
    except Exception as e:
        log(f"  DuckDB failed ({e}), falling back to pandas...")
        dense_df = build_dense_grid_fallback()

    log(f"  Dense grid: {dense_df.shape[0]:,} rows x {dense_df.shape[1]} cols")
    log(f"  Zeros: {(dense_df['total_qty'] == 0).sum():,} "
        f"({(dense_df['total_qty'] == 0).mean()*100:.1f}%)")
    log(f"  Positives: {(dense_df['total_qty'] > 0).sum():,}")

    # ----------------------------------------------------------------
    # 2. Load external data sources
    # ----------------------------------------------------------------
    log("--- Step 2: Load External Data ---")
    test, sample_sub = load_test()
    catalog = load_catalog()
    discounts = load_discounts()
    price_history = load_price_history()
    markdowns = load_markdowns()
    actual_matrix = load_actual_matrix()
    stores = load_stores()
    online = load_online_raw()

    # Load full sales for lag/rolling computations (needs pre-GRID_START data)
    sales_full = load_sales_for_lags()

    # ----------------------------------------------------------------
    # 3. Quantile outlier removal
    # ----------------------------------------------------------------
    log("--- Step 3: Quantile Outlier Removal ---")
    dense_df = remove_quantile_outliers(dense_df)

    # ----------------------------------------------------------------
    # 4. Split train/val
    # ----------------------------------------------------------------
    log("--- Step 4: Split Train/Val ---")
    train_mask = dense_df["date"] < VAL_START
    val_mask = (dense_df["date"] >= VAL_START) & (dense_df["date"] <= VAL_END)

    train_df = dense_df[train_mask].copy()
    val_df = dense_df[val_mask].copy()

    log(f"Train split: {len(train_df):,} rows, "
        f"{train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    log(f"Val split:   {len(val_df):,} rows, "
        f"{val_df['date'].min().date()} ~ {val_df['date'].max().date()}")

    # Full data for val/test feature engineering (train + val)
    full_train = dense_df.copy()

    # Build history for lag features: full sales (pre-GRID_START) + dense grid
    # The lag features need the complete historical quantity series
    # Combine original sales (for pre-GRID_START history) with dense grid
    log("  Building combined history for lag features...")
    sales_for_lags = sales_full[["date", "item_id", "store_id", "quantity"]].copy()
    sales_for_lags = sales_for_lags.rename(columns={"quantity": "total_qty"})
    # Combine: use original sales before GRID_START + dense grid from GRID_START
    pre_grid = sales_for_lags[sales_for_lags["date"] < GRID_START]
    post_grid = full_train[["date", "item_id", "store_id", "total_qty"]].copy()
    history_for_lags = pd.concat([pre_grid, post_grid], ignore_index=True)
    history_for_lags = history_for_lags.sort_values(["item_id", "store_id", "date"])
    log(f"  Combined history: {len(history_for_lags):,} rows, "
        f"{history_for_lags['date'].min().date()} ~ {history_for_lags['date'].max().date()}")

    # ================================================================
    # Feature Engineering
    # ================================================================

    # ----------------------------------------------------------------
    # 5. Date features
    # ----------------------------------------------------------------
    log("--- Step 5: Date Features ---")
    train_df = create_date_features(train_df)
    val_df = create_date_features(val_df)
    test = create_date_features(test)

    # ----------------------------------------------------------------
    # 6. Cyclical time encoding (NEW in R06)
    # ----------------------------------------------------------------
    log("--- Step 6: Cyclical Time Encoding (NEW) ---")
    train_df = create_cyclical_features(train_df)
    val_df = create_cyclical_features(val_df)
    test = create_cyclical_features(test)
    log("  Added dow_sin/cos, month_sin/cos, week_sin/cos (6 features)")

    # ----------------------------------------------------------------
    # 7. Lag features
    # ----------------------------------------------------------------
    log("--- Step 7: Lag Features ---")
    train_df = create_lag_features(history_for_lags[history_for_lags["date"] < VAL_START], train_df)
    val_df = create_lag_features(history_for_lags[history_for_lags["date"] <= VAL_END], val_df)
    test = create_lag_features(history_for_lags, test)

    # ----------------------------------------------------------------
    # 8. Price feature
    # ----------------------------------------------------------------
    log("--- Step 8: Price Feature ---")
    train_df = create_price_feature(sales_full, train_df)
    val_df = create_price_feature(sales_full, val_df)
    test = create_price_feature(sales_full, test)

    # ----------------------------------------------------------------
    # 9. Merge catalog
    # ----------------------------------------------------------------
    log("--- Step 9: Catalog Features ---")
    train_df = merge_catalog(train_df, catalog)
    val_df = merge_catalog(val_df, catalog)
    test = merge_catalog(test, catalog)

    # ----------------------------------------------------------------
    # 10. Rolling window features
    # ----------------------------------------------------------------
    log("--- Step 10: Rolling Features ---")
    train_df = create_rolling_features(full_train[full_train["date"] < VAL_START], train_df)
    val_df = create_rolling_features(full_train[full_train["date"] <= VAL_END], val_df)
    test = create_rolling_features(full_train, test)

    # ----------------------------------------------------------------
    # 11. Store aggregation features
    # ----------------------------------------------------------------
    log("--- Step 11: Store Aggregation Features ---")
    train_df = create_store_aggregation_features(full_train[full_train["date"] < VAL_START], train_df)
    val_df = create_store_aggregation_features(full_train[full_train["date"] <= VAL_END], val_df)
    test = create_store_aggregation_features(full_train, test)

    # ----------------------------------------------------------------
    # 12. Day-of-week statistics
    # ----------------------------------------------------------------
    log("--- Step 12: DOW Statistics ---")
    train_df = create_dow_statistics(full_train[full_train["date"] < VAL_START], train_df)
    val_df = create_dow_statistics(full_train[full_train["date"] <= VAL_END], val_df)
    test = create_dow_statistics(full_train, test)

    # ----------------------------------------------------------------
    # 13. Discount/Promotion features
    # ----------------------------------------------------------------
    log("--- Step 13: Discount Features ---")
    train_df = create_discount_features(discounts, train_df)
    val_df = create_discount_features(discounts, val_df)
    test = create_discount_features(discounts, test)

    # ----------------------------------------------------------------
    # 14. Price history features
    # ----------------------------------------------------------------
    log("--- Step 14: Price History Features ---")
    train_df = create_price_history_features(price_history, train_df, sales_full)
    val_df = create_price_history_features(price_history, val_df, sales_full)
    test = create_price_history_features(price_history, test, sales_full)

    # ----------------------------------------------------------------
    # 15. Markdown features
    # ----------------------------------------------------------------
    log("--- Step 15: Markdown Features ---")
    train_df = create_markdown_features(markdowns, train_df)
    val_df = create_markdown_features(markdowns, val_df)
    test = create_markdown_features(markdowns, test)

    # ----------------------------------------------------------------
    # 16. Store metadata
    # ----------------------------------------------------------------
    log("--- Step 16: Store Metadata ---")
    train_df = create_store_features(stores, train_df)
    val_df = create_store_features(stores, val_df)
    test = create_store_features(stores, test)

    # ----------------------------------------------------------------
    # 17. Availability features
    # ----------------------------------------------------------------
    log("--- Step 17: Availability Features ---")
    train_df = create_availability_features(actual_matrix, train_df)
    val_df = create_availability_features(actual_matrix, val_df)
    test = create_availability_features(actual_matrix, test)

    # ----------------------------------------------------------------
    # 18. Holiday features
    # ----------------------------------------------------------------
    log("--- Step 18: Holiday Features ---")
    train_df = create_holiday_features(train_df)
    val_df = create_holiday_features(val_df)
    test = create_holiday_features(test)

    # ----------------------------------------------------------------
    # 19. Online sales features
    # ----------------------------------------------------------------
    log("--- Step 19: Online Sales Features ---")
    train_df = create_online_features(online, train_df)
    val_df = create_online_features(online, val_df)
    test = create_online_features(online, test)

    # ================================================================
    # Feature columns
    # ================================================================
    # R01 features
    r01_features = DATE_FEATURES + [
        f"quantity_lag_{lag}" for lag in LAG_DAYS
    ] + ["price_base_latest"] + CAT_FEATURES[:4]

    # R02 rolling features
    rolling_features = (
        [f"qty_roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
        + [f"qty_roll_std_{w}" for w in ROLL_STD_WINDOWS]
        + [f"qty_roll_min_{w}" for w in ROLL_MIN_WINDOWS]
        + [f"qty_roll_max_{w}" for w in ROLL_MAX_WINDOWS]
        + [f"qty_ewm_{span}" for span in EWM_SPANS]
    )

    # R02 store aggregation features
    store_agg_features = ["store_daily_qty", "item_store_qty_ratio"]

    # R02 day-of-week statistics
    dow_features = ["item_dow_mean", "item_dow_std"]

    # R03 discount features
    discount_features = [
        "is_promo", "promo_type", "promo_discount_pct", "number_disc_day",
        "item_promo_freq_30d",
    ]

    # R03 price history features
    price_hist_features = [
        "price_change_flag", "price_vs_base_ratio", "item_price_volatility_30d",
    ]

    # R03 markdown features
    markdown_features = ["is_markdown", "markdown_discount_pct"]

    # R03 store metadata (categorical)
    store_meta_features = ["store_division", "store_format", "store_city", "store_area"]

    # R03 availability features
    availability_features = ["item_available"]

    # R03 holiday features
    holiday_features = ["is_holiday", "days_to_next_holiday"]

    # R03 online features
    online_features = ["online_qty_7d", "has_online_sales"]

    # R06 NEW: cyclical time encoding
    cyclical_features = ["dow_sin", "dow_cos", "month_sin", "month_cos", "week_sin", "week_cos"]

    feature_cols = (
        r01_features + rolling_features + store_agg_features + dow_features
        + discount_features + price_hist_features + markdown_features
        + store_meta_features + availability_features
        + holiday_features + online_features + cyclical_features
    )

    log(f"Feature columns ({len(feature_cols)}):")
    log(f"  R01 features:       {len(r01_features)}")
    log(f"  Rolling features:   {len(rolling_features)}")
    log(f"  Store agg features: {len(store_agg_features)}")
    log(f"  DOW features:       {len(dow_features)}")
    log(f"  Discount features:  {len(discount_features)}")
    log(f"  Price hist features:{len(price_hist_features)}")
    log(f"  Markdown features:  {len(markdown_features)}")
    log(f"  Store meta features:{len(store_meta_features)}")
    log(f"  Availability:       {len(availability_features)}")
    log(f"  Holiday features:   {len(holiday_features)}")
    log(f"  Online features:    {len(online_features)}")
    log(f"  Cyclical (NEW):     {len(cyclical_features)}")
    log(f"  Total:              {len(feature_cols)}")

    # ================================================================
    # Prepare training data
    # ================================================================
    for col in feature_cols:
        if col not in train_df.columns:
            log(f"  WARNING: Missing column '{col}' in train_df, adding as 0")
            train_df[col] = 0
            val_df[col] = 0
            test[col] = 0

    X_train = train_df[feature_cols].copy()
    y_train = train_df["total_qty"].values
    X_val = val_df[feature_cols].copy()
    y_val = val_df["total_qty"].values
    X_test = test[feature_cols].copy()

    log(f"Training data shape: {X_train.shape}")
    log(f"Validation data shape: {X_val.shape}")
    log(f"Test data shape: {X_test.shape}")
    log(f"Target (train) mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    log(f"Target (val) mean:   {y_val.mean():.4f}, std: {y_val.std():.4f}")
    log(f"Target zeros (train): {(y_train == 0).mean()*100:.1f}%")
    log(f"Target zeros (val):   {(y_val == 0).mean()*100:.1f}%")

    # Ensure categorical columns are string type for CatBoost
    for col in CAT_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)

    # ================================================================
    # Train model
    # ================================================================
    model = train_catboost(X_train, y_train, X_val, y_val, CAT_FEATURES)

    # ================================================================
    # Feature importance
    # ================================================================
    show_feature_importance(model, feature_cols)

    # ================================================================
    # Predict on test
    # ================================================================
    log("Predicting on test set...")
    test_pool = Pool(X_test, cat_features=CAT_FEATURES)
    preds = model.predict(test_pool)

    # ================================================================
    # Post-process
    # ================================================================
    preds = postprocess_predictions(preds, dense_df, test, catalog)

    # ================================================================
    # Prediction distribution stats
    # ================================================================
    log("Prediction distribution:")
    log(f"  mean={preds.mean():.4f}, median={np.median(preds):.4f}")
    log(f"  min={preds.min():.4f}, max={preds.max():.4f}")
    log(f"  std={preds.std():.4f}")
    log(f"  pct zeros: {(preds == 0).mean()*100:.2f}%")
    log(f"  NaN count: {np.isnan(preds).sum()}")

    # ================================================================
    # Generate submission
    # ================================================================
    log("Generating submission...")
    submission = pd.DataFrame({
        "row_id": test["row_id"].values,
        "quantity": preds,
    })

    assert len(submission) == len(sample_sub), (
        f"Submission length mismatch: {len(submission)} vs {len(sample_sub)}"
    )
    assert list(submission.columns) == list(sample_sub.columns), (
        f"Column mismatch: {submission.columns.tolist()} vs {sample_sub.columns.tolist()}"
    )

    submission.to_csv(SUBMISSION_PATH, index=False)
    log(f"Submission saved to {SUBMISSION_PATH}")
    log(f"  Rows: {len(submission)}")
    log(f"  Columns: {submission.columns.tolist()}")

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - t0
    best_score = model.get_best_score()["validation"]["RMSE"]
    log("=" * 70)
    log("R06 Dense Cross-Join Grid Summary")
    log("=" * 70)
    log(f"Val RMSE:       {best_score:.4f}")
    log(f"Best iter:      {model.get_best_iteration()}")
    log(f"Train time:     {elapsed:.1f}s")
    log(f"Grid rows:      {dense_df.shape[0]:,}")
    log(f"Train rows:     {len(train_df):,}")
    log(f"Val rows:       {len(val_df):,}")
    log(f"Total features: {len(feature_cols)}")
    log(f"  R01:          {len(r01_features)}")
    log(f"  Rolling:      {len(rolling_features)}")
    log(f"  Store agg:    {len(store_agg_features)}")
    log(f"  DOW stats:    {len(dow_features)}")
    log(f"  Discounts:    {len(discount_features)}")
    log(f"  Price hist:   {len(price_hist_features)}")
    log(f"  Markdowns:    {len(markdown_features)}")
    log(f"  Store meta:   {len(store_meta_features)}")
    log(f"  Availability: {len(availability_features)}")
    log(f"  Holidays:     {len(holiday_features)}")
    log(f"  Online:       {len(online_features)}")
    log(f"  Cyclical:     {len(cyclical_features)}")
    log(f"Pred mean:      {preds.mean():.4f}")
    log(f"Pred median:    {np.median(preds):.4f}")
    log(f"Submission:     {SUBMISSION_PATH}")
    log("=" * 70)

    return best_score


if __name__ == "__main__":
    score = main()
    sys.exit(0)
