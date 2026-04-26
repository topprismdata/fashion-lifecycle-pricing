#!/usr/bin/env python3
"""
R03 Multi-Source Features: CatBoost + Rolling Stats + External Data for ML Zoomcamp 2024.

Builds on R02 Rolling Features (31 features) by adding:
  1. Discount/Promotion features (from discounts_history.csv):
     - is_promo, promo_type, promo_discount_pct, promo_days, item_promo_freq_30d
  2. Price features (from price_history.csv):
     - price_change_flag, price_vs_base_ratio, item_price_volatility_30d
  3. Markdown features (from markdowns.csv):
     - is_markdown, markdown_discount_pct
  4. Store metadata (from stores.csv):
     - store_division, store_format, store_city, store_area
  5. Availability features (from actual_matrix.csv):
     - item_available
  6. Russian holiday features:
     - is_holiday, days_to_next_holiday
  7. Online sales features (from online.csv):
     - online_qty_7d, has_online_sales

All R02 features (31) are retained.

Validation: time-based holdout (same as R01/R02)
  Train: before 2024-08-27
  Val:   2024-08-27 ~ 2024-09-26 (30 days)
  Test:  2024-09-27 ~ 2024-10-26 (30 days)

Usage:
    python scripts/run_r03_multisource.py
"""

import sys
import time
import warnings
from pathlib import Path

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

SUBMISSION_PATH = OUTPUT_DIR / "submission_r03_multisource.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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
    "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08",  # New Year
    "2023-02-23",  # Defender of the Fatherland Day
    "2023-03-08",  # International Women's Day
    "2023-05-01",  # Spring and Labor Day
    "2023-05-09",  # Victory Day
    "2023-06-12",  # Russia Day
    "2023-11-04",  # Unity Day
    # 2024
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
    "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",  # New Year
    "2024-02-23",  # Defender of the Fatherland Day
    "2024-03-08",  # International Women's Day
    "2024-05-01",  # Spring and Labor Day
    "2024-05-09",  # Victory Day
    "2024-06-12",  # Russia Day
    "2024-11-04",  # Unity Day
    # 2024 extended holiday periods (typical bridge days / official non-working)
    "2024-02-24",  # Bridge day after 23 Feb
    "2024-03-09",  # Day after Women's Day (Saturday, often off)
    "2024-05-02", "2024-05-03",  # Bridge days after May 1
    "2024-05-10",  # Bridge day after Victory Day
    "2024-06-13",  # Day after Russia Day
    "2024-11-05",  # Day after Unity Day
])


def log(msg: str) -> None:
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Data Loading
# ===========================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load sales, test, catalog, and sample submission."""
    log("Loading core data...")
    sales = pd.read_csv(SALES_PATH, index_col=0)
    sales["date"] = pd.to_datetime(sales["date"])

    test = pd.read_csv(TEST_PATH, sep=";")
    test["date"] = pd.to_datetime(test["date"], format="%d.%m.%Y")

    catalog = pd.read_csv(CATALOG_PATH, index_col=0)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    log(f"  sales: {sales.shape}, date range: {sales['date'].min()} ~ {sales['date'].max()}")
    log(f"  test:  {test.shape}, date range: {test['date'].min()} ~ {test['date'].max()}")
    log(f"  catalog: {catalog.shape}")
    log(f"  sample_sub: {sample_sub.shape}")
    return sales, test, catalog, sample_sub


def load_discounts() -> pd.DataFrame:
    """Load discounts_history.csv with only needed columns."""
    log("Loading discounts_history (selected columns)...")
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


def load_online() -> pd.DataFrame:
    """Load online.csv."""
    log("Loading online sales...")
    on = pd.read_csv(ONLINE_PATH, usecols=["date", "item_id", "quantity", "store_id"])
    on["date"] = pd.to_datetime(on["date"])
    log(f"  online: {on.shape}")
    return on


# ===========================================================================
# Preprocessing
# ===========================================================================
def filter_negative_quantities(sales: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with negative quantity."""
    before = len(sales)
    sales = sales[sales["quantity"] >= 0].copy()
    after = len(sales)
    log(f"Filtered negative quantities: {before} -> {after} (removed {before - after})")
    return sales


# ===========================================================================
# R01 Features (unchanged)
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


def create_lag_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create lag features for target_df using data from train_df only."""
    log("Creating lag features...")

    daily_qty = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(quantity=("quantity", "sum"))
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
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add latest known price_base per (item_id, store_id)."""
    log("Creating price feature...")
    latest_price = (
        train_df.groupby(["item_id", "store_id"])
        .agg(price_base_latest=("price_base", "last"))
        .reset_index()
    )
    result = target_df.merge(latest_price, on=["item_id", "store_id"], how="left")
    fill_rate = result["price_base_latest"].notna().mean()
    log(f"  price_base_latest fill rate: {fill_rate:.4f}")
    median_price = train_df["price_base"].median()
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


# ===========================================================================
# R02 Features (unchanged)
# ===========================================================================
def _build_daily_series(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build a complete daily time series per (item_id, store_id) from training data."""
    daily = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(qty=("quantity", "sum"))
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

    train_daily = (
        train_df.groupby(["store_id", "date"], as_index=False)
        .agg(store_qty=("quantity", "sum"))
    )
    train_daily["day_of_week"] = train_daily["date"].dt.dayofweek

    store_dow_qty = (
        train_daily.groupby(["store_id", "day_of_week"], as_index=False)
        .agg(store_daily_qty=("store_qty", "mean"))
    )

    train_item_daily = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(item_qty=("quantity", "sum"))
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

    train_copy = train_df.copy()
    train_copy["day_of_week"] = train_copy["date"].dt.dayofweek

    dow_stats = (
        train_copy.groupby(["item_id", "day_of_week"])
        .agg(
            item_dow_mean=("quantity", "mean"),
            item_dow_std=("quantity", "std"),
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
# R03 NEW: Multi-Source Features
# ===========================================================================

# --- 1. Discount/Promotion Features ---
def create_discount_features(
    discounts: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create discount/promotion features from discounts_history.

    Features:
      - is_promo: whether item was on promo on that date
      - promo_type: promo_type_code (categorical)
      - promo_discount_pct: discount percentage
      - promo_days: number of discount days
      - item_promo_freq_30d: promo frequency in last 30 days
    """
    log("Creating discount/promotion features...")

    disc = discounts.copy()
    disc["promo_type_code"] = disc["promo_type_code"].fillna(0).astype(str)

    # Aggregate to (item_id, store_id, date) level in case of multiple promos
    disc_agg = (
        disc.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(
            sale_price_before=("sale_price_before_promo", "first"),
            sale_price_time=("sale_price_time_promo", "first"),
            promo_type_code=("promo_type_code", "first"),
            number_disc_day=("number_disc_day", "max"),
        )
    )

    # Compute discount percentage
    disc_agg["promo_discount_pct"] = np.where(
        disc_agg["sale_price_before"] > 0,
        (disc_agg["sale_price_before"] - disc_agg["sale_price_time"])
        / disc_agg["sale_price_before"] * 100,
        0,
    )
    disc_agg["promo_discount_pct"] = disc_agg["promo_discount_pct"].clip(0, 100)

    # is_promo flag
    disc_agg["is_promo"] = 1

    # For promo frequency: count promo days per (item_id, store_id) in rolling 30-day windows
    # Build a daily promo indicator series
    promo_daily = disc_agg[["item_id", "store_id", "date", "is_promo",
                             "promo_discount_pct", "number_disc_day",
                             "promo_type_code"]].copy()
    promo_daily = promo_daily.sort_values(["item_id", "store_id", "date"])

    # Rolling 30-day count of promo days
    promo_daily["item_promo_freq_30d"] = (
        promo_daily.groupby(["item_id", "store_id"])["is_promo"]
        .shift(1)
        .rolling(window=30, min_periods=1)
        .sum()
        .values
    )

    # Build lookup with forward-fill for target dates
    promo_lookup_cols = [
        "item_id", "store_id", "date", "is_promo",
        "promo_discount_pct", "number_disc_day",
        "promo_type_code", "item_promo_freq_30d",
    ]
    promo_lookup = promo_daily[promo_lookup_cols].copy()

    # Extend to cover target dates
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

    # promo_type_code: use forward fill (last known promo type)
    combined["promo_type_code"] = combined.groupby(["item_id", "store_id"])["promo_type_code"].ffill()

    lookup = combined[~combined["_is_original"]].copy()
    lookup = lookup.drop(columns=["_is_original"])
    log(f"  Discount lookup: {len(lookup)} rows")

    result = target_df.merge(lookup, on=["item_id", "store_id", "date"], how="left")

    # Rename promo_type_code -> promo_type for consistency with CAT_FEATURES
    result = result.rename(columns={"promo_type_code": "promo_type"})

    # Fill defaults for no-promo cases
    result["is_promo"] = result["is_promo"].fillna(0).astype(int)
    result["promo_discount_pct"] = result["promo_discount_pct"].fillna(0)
    result["number_disc_day"] = result["number_disc_day"].fillna(0)
    result["promo_type"] = result["promo_type"].fillna("NONE")
    result["item_promo_freq_30d"] = result["item_promo_freq_30d"].fillna(0)

    log(f"  is_promo fill rate: {(result['is_promo'] == 1).mean():.4f}")
    log(f"  promo_discount_pct > 0: {(result['promo_discount_pct'] > 0).mean():.4f}")

    return result


# --- 2. Price History Features ---
def create_price_history_features(
    price_history: pd.DataFrame,
    target_df: pd.DataFrame,
    sales_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create price history features from price_history.csv.

    Features:
      - price_change_flag: whether price changed from previous known price
      - price_vs_base_ratio: current price / base price (from sales)
      - item_price_volatility_30d: std of price over 30 days
    """
    log("Creating price history features...")

    ph = price_history.copy()
    ph = ph.sort_values(["item_id", "store_id", "date"])

    # Previous price per (item_id, store_id)
    ph["prev_price"] = ph.groupby(["item_id", "store_id"])["price"].shift(1)
    ph["price_change_flag"] = (
        (ph["price"] != ph["prev_price"]) & ph["prev_price"].notna()
    ).astype(int)

    # 30-day price volatility (std of price changes)
    ph["price_diff"] = ph.groupby(["item_id", "store_id"])["price"].diff()
    ph["item_price_volatility_30d"] = (
        ph.groupby(["item_id", "store_id"])["price_diff"]
        .shift(1)
        .rolling(window=30, min_periods=1)
        .std()
        .values
    )

    # Build lookup with the latest info per (item_id, store_id, date)
    ph_lookup = ph[["item_id", "store_id", "date", "price", "price_change_flag",
                     "item_price_volatility_30d"]].copy()

    # Extend to cover target dates
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

    # Compute price_vs_base_ratio: price_history price / sales price_base
    # The target_df should already have price_base_latest from create_price_feature
    if "price_base_latest" in result.columns:
        result["price_vs_base_ratio"] = np.where(
            result["price_base_latest"] > 0,
            result["price"] / result["price_base_latest"],
            1.0,
        )
    else:
        result["price_vs_base_ratio"] = 1.0

    # Fill NaN
    result["price_change_flag"] = result["price_change_flag"].fillna(0).astype(int)
    result["item_price_volatility_30d"] = result["item_price_volatility_30d"].fillna(0)
    result["price_vs_base_ratio"] = result["price_vs_base_ratio"].fillna(1.0)
    result["price"] = result["price"].fillna(result.get("price_base_latest", 0))

    log(f"  price_change_flag rate: {result['price_change_flag'].mean():.4f}")
    log(f"  price_vs_base_ratio mean: {result['price_vs_base_ratio'].mean():.4f}")

    return result


# --- 3. Markdown Features ---
def create_markdown_features(
    markdowns: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create markdown features from markdowns.csv.

    Features:
      - is_markdown: whether item had markdown on that date
      - markdown_discount_pct: discount percentage
    """
    log("Creating markdown features...")

    md = markdowns.copy()

    # Aggregate per (item_id, store_id, date)
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

    # Direct merge (markdowns are sparse, no forward fill needed -- exact date match)
    result = target_df.merge(md_lookup, on=["item_id", "store_id", "date"], how="left")

    result["is_markdown"] = result["is_markdown"].fillna(0).astype(int)
    result["markdown_discount_pct"] = result["markdown_discount_pct"].fillna(0)

    log(f"  is_markdown rate: {result['is_markdown'].mean():.4f}")

    return result


# --- 4. Store Metadata Features ---
def create_store_features(
    stores: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge store metadata from stores.csv.

    Features:
      - store_division, store_format, store_city, store_area
    """
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


# --- 5. Availability Features ---
def create_availability_features(
    actual_matrix: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create availability feature from actual_matrix.csv.

    Features:
      - item_available: whether item was listed as available on that date
    """
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


# --- 6. Russian Holiday Features ---
def create_holiday_features(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Russian holiday features.

    Features:
      - is_holiday: whether date is a Russian holiday
      - days_to_next_holiday: days until the next holiday
    """
    log("Creating Russian holiday features...")

    result = target_df.copy()
    holidays = RUSSIAN_HOLIDAYS.sort_values().values

    result["is_holiday"] = result["date"].isin(holidays).astype(int)

    # days_to_next_holiday: for each date, find the next holiday
    holiday_series = pd.Series(holidays).sort_values().reset_index(drop=True)

    def _days_to_next(d):
        future = holiday_series[holiday_series >= d]
        if len(future) > 0:
            return (future.iloc[0] - d).days
        return 365  # fallback

    # Vectorized approach: merge with a sorted holiday list
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


# --- 7. Online Sales Features ---
def create_online_features(
    online: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create online sales features from online.csv.

    Online sales are store_id=1 only. For each item, compute:
      - online_qty_7d: total online quantity in last 7 days
      - has_online_sales: binary flag (1 if item ever had online sales)

    No data leakage: only use data from dates < target date.
    """
    log("Creating online sales features...")

    on = online.copy()
    # Filter to store_id=1 (only store with online data)
    on = on[on["store_id"] == 1].copy()

    # Aggregate daily online quantity per item
    online_daily = (
        on.groupby(["item_id", "date"], as_index=False)
        .agg(online_qty=("quantity", "sum"))
    )
    online_daily = online_daily.sort_values(["item_id", "date"])

    # Rolling 7-day sum of online quantity (shifted by 1 to avoid leakage)
    online_daily["online_qty_7d"] = (
        online_daily.groupby("item_id")["online_qty"]
        .shift(1)
        .rolling(window=7, min_periods=1)
        .sum()
        .values
    )

    # Items that ever had online sales
    items_with_online = set(on["item_id"].unique())
    log(f"  Items with online sales: {len(items_with_online)}")

    # Build lookup with forward-fill for target dates
    online_lookup = online_daily[["item_id", "date", "online_qty_7d"]].copy()

    # For target rows: forward-fill the online_qty_7d
    target_dates = target_df["date"].unique()
    # Get items that appear in target_df
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
    max_qty = train_df["quantity"].max()
    log(f"Clipping predictions to [0, {max_qty}]")
    preds = np.clip(preds, 0, max_qty)

    # Fix NaN predictions
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
        train_item_mean = train_df.groupby("item_id")["quantity"].mean().reset_index()
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

    # Final NaN safety check
    preds = np.nan_to_num(preds, nan=0.0)

    return preds


# ===========================================================================
# Main Pipeline
# ===========================================================================
def main() -> float:
    log("=" * 70)
    log("R03 Multi-Source Features: CatBoost + External Data")
    log("=" * 70)

    t0 = time.time()

    # ----------------------------------------------------------------
    # 1. Load core data
    # ----------------------------------------------------------------
    sales, test, catalog, sample_sub = load_data()

    # ----------------------------------------------------------------
    # 2. Load external data sources
    # ----------------------------------------------------------------
    discounts = load_discounts()
    price_history = load_price_history()
    markdowns = load_markdowns()
    actual_matrix = load_actual_matrix()
    stores = load_stores()
    online = load_online()

    # ----------------------------------------------------------------
    # 3. Filter negative quantities
    # ----------------------------------------------------------------
    sales = filter_negative_quantities(sales)

    # ----------------------------------------------------------------
    # 4. Split train/val
    # ----------------------------------------------------------------
    train_mask = sales["date"] < VAL_START
    val_mask = (sales["date"] >= VAL_START) & (sales["date"] <= VAL_END)

    train_sales = sales[train_mask].copy()
    val_sales = sales[val_mask].copy()

    log(f"Train split: {len(train_sales)} rows, "
        f"{train_sales['date'].min().date()} ~ {train_sales['date'].max().date()}")
    log(f"Val split:   {len(val_sales)} rows, "
        f"{val_sales['date'].min().date()} ~ {val_sales['date'].max().date()}")

    # Full training data for val/test lag features
    full_train = sales[sales["date"] <= VAL_END].copy()

    # ================================================================
    # R01 Features
    # ================================================================

    # ----------------------------------------------------------------
    # 5. Date features
    # ----------------------------------------------------------------
    log("--- R01 Features ---")
    log("Creating date features...")
    train_sales = create_date_features(train_sales)
    val_sales = create_date_features(val_sales)
    test = create_date_features(test)

    # ----------------------------------------------------------------
    # 6. Lag features
    # ----------------------------------------------------------------
    train_sales = create_lag_features(train_sales, train_sales)
    val_sales = create_lag_features(full_train, val_sales)
    test = create_lag_features(full_train, test)

    # ----------------------------------------------------------------
    # 7. Price feature
    # ----------------------------------------------------------------
    train_sales = create_price_feature(train_sales, train_sales)
    val_sales = create_price_feature(full_train, val_sales)
    test = create_price_feature(full_train, test)

    # ----------------------------------------------------------------
    # 8. Merge catalog
    # ----------------------------------------------------------------
    train_sales = merge_catalog(train_sales, catalog)
    val_sales = merge_catalog(val_sales, catalog)
    test = merge_catalog(test, catalog)

    # ================================================================
    # R02 Features
    # ================================================================
    log("--- R02 Features ---")

    # ----------------------------------------------------------------
    # 9. Rolling window features
    # ----------------------------------------------------------------
    train_sales = create_rolling_features(train_sales, train_sales)
    val_sales = create_rolling_features(full_train, val_sales)
    test = create_rolling_features(full_train, test)

    # ----------------------------------------------------------------
    # 10. Store aggregation features
    # ----------------------------------------------------------------
    train_sales = create_store_aggregation_features(train_sales, train_sales)
    val_sales = create_store_aggregation_features(full_train, val_sales)
    test = create_store_aggregation_features(full_train, test)

    # ----------------------------------------------------------------
    # 11. Day-of-week statistics
    # ----------------------------------------------------------------
    train_sales = create_dow_statistics(train_sales, train_sales)
    val_sales = create_dow_statistics(full_train, val_sales)
    test = create_dow_statistics(full_train, test)

    # ================================================================
    # R03 NEW Features
    # ================================================================
    log("--- R03 New Features ---")

    # ----------------------------------------------------------------
    # 12. Discount/Promotion features
    # ----------------------------------------------------------------
    train_sales = create_discount_features(discounts, train_sales)
    val_sales = create_discount_features(discounts, val_sales)
    test = create_discount_features(discounts, test)

    # ----------------------------------------------------------------
    # 13. Price history features
    # ----------------------------------------------------------------
    train_sales = create_price_history_features(price_history, train_sales, train_sales)
    val_sales = create_price_history_features(price_history, val_sales, full_train)
    test = create_price_history_features(price_history, test, full_train)

    # ----------------------------------------------------------------
    # 14. Markdown features
    # ----------------------------------------------------------------
    train_sales = create_markdown_features(markdowns, train_sales)
    val_sales = create_markdown_features(markdowns, val_sales)
    test = create_markdown_features(markdowns, test)

    # ----------------------------------------------------------------
    # 15. Store metadata
    # ----------------------------------------------------------------
    train_sales = create_store_features(stores, train_sales)
    val_sales = create_store_features(stores, val_sales)
    test = create_store_features(stores, test)

    # ----------------------------------------------------------------
    # 16. Availability features
    # ----------------------------------------------------------------
    train_sales = create_availability_features(actual_matrix, train_sales)
    val_sales = create_availability_features(actual_matrix, val_sales)
    test = create_availability_features(actual_matrix, test)

    # ----------------------------------------------------------------
    # 17. Holiday features
    # ----------------------------------------------------------------
    train_sales = create_holiday_features(train_sales)
    val_sales = create_holiday_features(val_sales)
    test = create_holiday_features(test)

    # ----------------------------------------------------------------
    # 18. Online sales features
    # ----------------------------------------------------------------
    train_sales = create_online_features(online, train_sales)
    val_sales = create_online_features(online, val_sales)
    test = create_online_features(online, test)

    # ================================================================
    # Feature columns
    # ================================================================

    # R01 features
    r01_features = DATE_FEATURES + [
        f"quantity_lag_{lag}" for lag in LAG_DAYS
    ] + ["price_base_latest"] + CAT_FEATURES[:4]  # only item_id, store_id, dept_name, class_name

    # R02 rolling features
    rolling_features = (
        [f"qty_roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
        + [f"qty_roll_std_{w}" for w in ROLL_STD_WINDOWS]
        + [f"qty_roll_min_{w}" for w in ROLL_MIN_WINDOWS]
        + [f"qty_roll_max_{w}" for w in ROLL_MAX_WINDOWS]
        + [f"qty_ewm_{span}" for span in EWM_SPANS]
    )

    # R02 store aggregation features
    store_features = ["store_daily_qty", "item_store_qty_ratio"]

    # R02 day-of-week statistics
    dow_features = ["item_dow_mean", "item_dow_std"]

    # R03 discount/promotion features (promo_type is categorical, in CAT_FEATURES)
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

    # R03 store metadata features (categorical)
    store_meta_features = ["store_division", "store_format", "store_city", "store_area"]

    # R03 availability features
    availability_features = ["item_available"]

    # R03 holiday features
    holiday_features = ["is_holiday", "days_to_next_holiday"]

    # R03 online features
    online_features = ["online_qty_7d", "has_online_sales"]

    feature_cols = (
        r01_features + rolling_features + store_features + dow_features
        + discount_features + price_hist_features + markdown_features
        + store_meta_features + availability_features
        + holiday_features + online_features
    )

    log(f"Feature columns ({len(feature_cols)}):")
    log(f"  R01 features:       {len(r01_features)}")
    log(f"  Rolling features:   {len(rolling_features)}")
    log(f"  Store agg features: {len(store_features)}")
    log(f"  DOW features:       {len(dow_features)}")
    log(f"  Discount features:  {len(discount_features)}")
    log(f"  Price hist features:{len(price_hist_features)}")
    log(f"  Markdown features:  {len(markdown_features)}")
    log(f"  Store meta features:{len(store_meta_features)}")
    log(f"  Availability:       {len(availability_features)}")
    log(f"  Holiday features:   {len(holiday_features)}")
    log(f"  Online features:    {len(online_features)}")
    log(f"  Total:              {len(feature_cols)}")

    # ================================================================
    # Prepare training data
    # ================================================================
    # Verify all feature columns exist
    for col in feature_cols:
        if col not in train_sales.columns:
            log(f"  WARNING: Missing column '{col}' in train_sales, adding as 0")
            train_sales[col] = 0
            val_sales[col] = 0
            test[col] = 0

    X_train = train_sales[feature_cols].copy()
    y_train = train_sales["quantity"].values
    X_val = val_sales[feature_cols].copy()
    y_val = val_sales["quantity"].values
    X_test = test[feature_cols].copy()

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
    preds = postprocess_predictions(preds, sales, test, catalog)

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
    log("R03 Multi-Source Features Summary")
    log("=" * 70)
    log(f"Val RMSE:       {best_score:.4f}")
    log(f"Best iter:      {model.get_best_iteration()}")
    log(f"Train time:     {elapsed:.1f}s")
    log(f"Total features: {len(feature_cols)}")
    log(f"  R01:          {len(r01_features)}")
    log(f"  Rolling:      {len(rolling_features)}")
    log(f"  Store agg:    {len(store_features)}")
    log(f"  DOW stats:    {len(dow_features)}")
    log(f"  Discounts:    {len(discount_features)}")
    log(f"  Price hist:   {len(price_hist_features)}")
    log(f"  Markdowns:    {len(markdown_features)}")
    log(f"  Store meta:   {len(store_meta_features)}")
    log(f"  Availability: {len(availability_features)}")
    log(f"  Holidays:     {len(holiday_features)}")
    log(f"  Online:       {len(online_features)}")
    log(f"Pred mean:      {preds.mean():.4f}")
    log(f"Pred median:    {np.median(preds):.4f}")
    log(f"Submission:     {SUBMISSION_PATH}")
    log("=" * 70)

    return best_score


if __name__ == "__main__":
    score = main()
    sys.exit(0)
