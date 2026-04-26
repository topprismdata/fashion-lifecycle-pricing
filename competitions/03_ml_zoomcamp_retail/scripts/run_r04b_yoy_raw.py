#!/usr/bin/env python3
"""
R04b YoY + Raw Target: Extends R03 Multi-Source Features.

Builds on R03 (50 features) by adding:
  1. YoY (Year-over-Year) 364-day features:
     - qty_yoy_364: qty[D] - qty[D-364] per (item_id, store_id)
     - qty_yoy_364_ratio: qty[D] / (qty[D-364] + 1)
     - qty_yoy_diff_7d: (7d sum at D) - (7d sum at D-364)
     - item_year_mean: mean daily qty per item in same month last year
     - item_store_year_trend: growth/decline vs last year
  2. Target Transform:
     - Train with log1p(quantity) as target
     - Predict and expm1() to convert back
  3. Additional Time Features:
     - week_of_year, quarter, day_of_year, is_pay_day, days_since_first_sale
  4. Item Lifecycle Features:
     - item_age_days, item_total_sales, item_n_stores, is_new_item

All R03 features (50) are retained. Same validation split and CatBoost config.

Validation: time-based holdout (same as R01/R02/R03)
  Train: before 2024-08-27
  Val:   2024-08-27 ~ 2024-09-26 (30 days)
  Test:  2024-09-27 ~ 2024-10-26 (30 days)

Usage:
    python scripts/run_r04_yoy_transform.py
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

SUBMISSION_PATH = OUTPUT_DIR / "submission_r04b_yoy_raw.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAL_START = "2024-08-27"
VAL_END = "2024-09-26"
YOY_LAG = 364  # 364 = 52 * 7, aligns day-of-week exactly

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
# R03 Features (unchanged from R03)
# ===========================================================================

# --- 1. Discount/Promotion Features ---
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


# --- 2. Price History Features ---
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


# --- 3. Markdown Features ---
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


# --- 4. Store Metadata Features ---
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


# --- 5. Availability Features ---
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


# --- 6. Russian Holiday Features ---
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


# --- 7. Online Sales Features ---
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
# R04 NEW Features
# ===========================================================================

# --- R04-1: YoY (Year-over-Year) 364-day Features ---
def create_yoy_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create Year-over-Year features using 364-day lag (= 52 weeks, preserves day-of-week).

    Features:
      - qty_yoy_364: quantity at date D minus quantity at date D-364 per (item_id, store_id)
      - qty_yoy_364_ratio: qty[D] / (qty[D-364] + 1) -- ratio to last year
      - qty_yoy_diff_7d: (7-day sum ending at D) - (7-day sum ending at D-364) -- weekly YoY change
      - item_year_mean: mean daily quantity per item in the same month last year
      - item_store_year_trend: how much the item-store combination grew/declined vs last year

    No data leakage: all lookups use only data from dates < D.
    """
    log("Creating YoY 364-day features...")

    # Build daily quantity lookup from training data
    daily_qty = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(qty=("quantity", "sum"))
    )
    daily_qty = daily_qty.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    log(f"  Daily qty for YoY: {len(daily_qty)} rows")

    # --- qty_yoy_364 and qty_yoy_364_ratio ---
    # For each target row, look up qty at D-364
    shifted_364 = daily_qty[["item_id", "store_id", "date", "qty"]].copy()
    shifted_364["lag_date"] = shifted_364["date"] + pd.Timedelta(days=YOY_LAG)
    shifted_364 = shifted_364.rename(columns={"qty": "qty_d364"})
    shifted_364 = shifted_364[["item_id", "store_id", "lag_date", "qty_d364"]]

    result = target_df.copy()
    result = result.merge(
        shifted_364,
        left_on=["item_id", "store_id", "date"],
        right_on=["item_id", "store_id", "lag_date"],
        how="left",
    )
    result = result.drop(columns=["lag_date"], errors="ignore")

    # Also merge current-day qty for computing YoY diff (for train rows only)
    # For val/test rows, qty is not available -- we use the 7-day rolling sum instead
    current_qty = daily_qty[["item_id", "store_id", "date", "qty"]].copy()
    current_qty = current_qty.rename(columns={"qty": "qty_current"})
    result = result.merge(current_qty, on=["item_id", "store_id", "date"], how="left")

    # qty_yoy_364: difference vs last year
    result["qty_yoy_364"] = result["qty_current"].fillna(0) - result["qty_d364"].fillna(0)

    # qty_yoy_364_ratio: ratio vs last year
    result["qty_yoy_364_ratio"] = np.where(
        result["qty_d364"].notna(),
        result["qty_current"].fillna(0) / (result["qty_d364"] + 1),
        0,
    )

    # Fill NaN for target rows (val/test don't have qty_current)
    result["qty_yoy_364"] = result["qty_yoy_364"].fillna(0)
    result["qty_yoy_364_ratio"] = result["qty_yoy_364_ratio"].fillna(0)

    # Clean up temp columns
    result = result.drop(columns=["qty_current", "qty_d364"], errors="ignore")

    yoy_fill_rate_364 = result["qty_yoy_364"].notna().mean()
    yoy_fill_rate_ratio = result["qty_yoy_364_ratio"].notna().mean()
    log(f"  qty_yoy_364 fill rate: {yoy_fill_rate_364:.4f}")
    log(f"  qty_yoy_364_ratio fill rate: {yoy_fill_rate_ratio:.4f}")

    # --- qty_yoy_diff_7d: weekly YoY change ---
    # Compute 7-day rolling sum for each (item_id, store_id)
    daily_qty_sorted = daily_qty.sort_values(["item_id", "store_id", "date"]).copy()
    grouped = daily_qty_sorted.groupby(["item_id", "store_id"])
    daily_qty_sorted["qty_7d_sum"] = grouped["qty"].shift(1).rolling(window=7, min_periods=1).sum().values

    # 7-day sum at D-364
    shifted_7d = daily_qty_sorted[["item_id", "store_id", "date", "qty_7d_sum"]].copy()
    shifted_7d["lag_date"] = shifted_7d["date"] + pd.Timedelta(days=YOY_LAG)
    shifted_7d = shifted_7d.rename(columns={"qty_7d_sum": "qty_7d_sum_d364"})
    shifted_7d = shifted_7d[["item_id", "store_id", "lag_date", "qty_7d_sum_d364"]]

    # Merge 7d sum (current) and 7d sum (D-364) into result
    seven_d_lookup = daily_qty_sorted[["item_id", "store_id", "date", "qty_7d_sum"]].copy()

    # Extend with forward fill for target dates
    target_dates = target_df["date"].unique()
    target_groups = target_df[["item_id", "store_id"]].drop_duplicates()

    target_date_rows = []
    for date in target_dates:
        frame = target_groups.copy()
        frame["date"] = date
        target_date_rows.append(frame)
    target_date_df = pd.concat(target_date_rows, ignore_index=True)
    target_date_df["qty_7d_sum"] = np.nan

    seven_d_lookup["_is_original"] = True
    target_date_df["_is_original"] = False

    combined_7d = pd.concat([seven_d_lookup, target_date_df], ignore_index=True)
    combined_7d = combined_7d.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)
    combined_7d["qty_7d_sum"] = combined_7d.groupby(["item_id", "store_id"])["qty_7d_sum"].ffill()

    lookup_7d = combined_7d[~combined_7d["_is_original"]].copy()
    lookup_7d = lookup_7d.drop(columns=["_is_original"])

    result = result.merge(lookup_7d, on=["item_id", "store_id", "date"], how="left", suffixes=("", "_curr"))

    result = result.merge(
        shifted_7d,
        left_on=["item_id", "store_id", "date"],
        right_on=["item_id", "store_id", "lag_date"],
        how="left",
    )
    result = result.drop(columns=["lag_date"], errors="ignore")

    result["qty_yoy_diff_7d"] = result["qty_7d_sum"].fillna(0) - result["qty_7d_sum_d364"].fillna(0)
    result["qty_yoy_diff_7d"] = result["qty_yoy_diff_7d"].fillna(0)

    result = result.drop(columns=["qty_7d_sum", "qty_7d_sum_d364"], errors="ignore")

    log(f"  qty_yoy_diff_7d fill rate: {result['qty_yoy_diff_7d'].notna().mean():.4f}")

    # --- item_year_mean: mean daily qty per item in the same month last year ---
    daily_qty_item = (
        train_df.groupby(["item_id", "date"], as_index=False)
        .agg(qty=("quantity", "sum"))
    )
    daily_qty_item["month"] = daily_qty_item["date"].dt.month
    daily_qty_item["year"] = daily_qty_item["date"].dt.year

    # Mean daily qty per item per month (across all years in training)
    item_month_mean = (
        daily_qty_item.groupby(["item_id", "month"], as_index=False)
        .agg(item_month_mean_qty=("qty", "mean"))
    )

    result["month"] = result["date"].dt.month
    result = result.merge(item_month_mean, on=["item_id", "month"], how="left")
    result = result.rename(columns={"item_month_mean_qty": "item_year_mean"})

    result["item_year_mean"] = result["item_year_mean"].fillna(0)
    log(f"  item_year_mean fill rate: {result['item_year_mean'].notna().mean():.4f}")

    # --- item_store_year_trend: growth/decline vs last year ---
    # For each (item_id, store_id), compute total qty in last 364 days vs the 364 days before that
    daily_qty_is = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(qty=("quantity", "sum"))
    )

    # Compute totals per (item_id, store_id) for recent year and prior year
    # Use the training data's date range to split
    train_max_date = train_df["date"].max()
    year_ago = train_max_date - pd.Timedelta(days=YOY_LAG)
    two_years_ago = train_max_date - pd.Timedelta(days=2 * YOY_LAG)

    recent_mask = daily_qty_is["date"] > year_ago
    prior_mask = (daily_qty_is["date"] > two_years_ago) & (daily_qty_is["date"] <= year_ago)

    recent_totals = (
        daily_qty_is[recent_mask]
        .groupby(["item_id", "store_id"], as_index=False)
        .agg(recent_qty=("qty", "sum"))
    )
    prior_totals = (
        daily_qty_is[prior_mask]
        .groupby(["item_id", "store_id"], as_index=False)
        .agg(prior_qty=("qty", "sum"))
    )

    trend = recent_totals.merge(prior_totals, on=["item_id", "store_id"], how="outer")
    trend["recent_qty"] = trend["recent_qty"].fillna(0)
    trend["prior_qty"] = trend["prior_qty"].fillna(0)
    trend["item_store_year_trend"] = np.where(
        trend["prior_qty"] > 0,
        (trend["recent_qty"] - trend["prior_qty"]) / trend["prior_qty"],
        0,
    )
    trend = trend[["item_id", "store_id", "item_store_year_trend"]]

    result = result.merge(trend, on=["item_id", "store_id"], how="left")
    result["item_store_year_trend"] = result["item_store_year_trend"].fillna(0)

    # Clip extreme trends to [-10, 10] for stability
    result["item_store_year_trend"] = result["item_store_year_trend"].clip(-10, 10)

    log(f"  item_store_year_trend fill rate: {result['item_store_year_trend'].notna().mean():.4f}")
    log(f"  item_store_year_trend mean: {result['item_store_year_trend'].mean():.4f}")

    # Drop helper month column if it was added
    if "month" in result.columns and "month" not in target_df.columns:
        result = result.drop(columns=["month"])

    return result


# --- R04-2: Additional Time Features ---
def create_extended_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional date-based features on top of R01's date features.

    Features:
      - week_of_year: ISO week number
      - quarter: quarter of year
      - day_of_year: day of year (1-366)
      - is_pay_day: Russian pay days (10th and 25th of each month)
    """
    log("Creating extended date features...")

    result = df.copy()
    dt = result["date"]

    result["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    result["quarter"] = dt.dt.quarter
    result["day_of_year"] = dt.dt.dayofyear
    result["is_pay_day"] = (
        (dt.dt.day == 10) | (dt.dt.day == 25)
    ).astype(int)

    log(f"  week_of_year range: {result['week_of_year'].min()} ~ {result['week_of_year'].max()}")
    log(f"  quarter values: {sorted(result['quarter'].unique())}")
    log(f"  is_pay_day rate: {result['is_pay_day'].mean():.4f}")

    return result


# --- R04-3: Item Lifecycle Features ---
def create_lifecycle_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create item lifecycle features based on training data.

    Features:
      - item_age_days: number of days since item first appeared in training data
      - item_total_sales: total quantity sold for this item in training data
      - item_n_stores: number of stores this item is sold in
      - is_new_item: binary flag for items that appeared recently (within last 90 days)
      - days_since_first_sale: days since item was first sold in training data
    """
    log("Creating item lifecycle features...")

    # Compute per-item statistics from training data
    train_max_date = train_df["date"].max()

    item_first_sale = (
        train_df.groupby("item_id", as_index=False)
        .agg(first_sale_date=("date", "min"))
    )

    item_stats = (
        train_df.groupby("item_id", as_index=False)
        .agg(
            item_total_sales=("quantity", "sum"),
            item_n_stores=("store_id", "nunique"),
        )
    )

    # Merge first sale date into item_stats
    item_stats = item_stats.merge(item_first_sale, on="item_id", how="left")
    item_stats["item_first_sale_ts"] = item_stats["first_sale_date"]

    # is_new_item: items that first appeared within the last 90 days of training data
    new_item_threshold = train_max_date - pd.Timedelta(days=90)
    item_stats["is_new_item"] = (item_stats["first_sale_date"] >= new_item_threshold).astype(int)

    log(f"  Total items: {len(item_stats)}")
    log(f"  New items (< 90 days): {item_stats['is_new_item'].sum()}")
    log(f"  item_total_sales range: {item_stats['item_total_sales'].min():.0f} ~ {item_stats['item_total_sales'].max():.0f}")
    log(f"  item_n_stores range: {item_stats['item_n_stores'].min()} ~ {item_stats['item_n_stores'].max()}")

    # Merge into target_df
    result = target_df.copy()
    result = result.merge(
        item_stats[["item_id", "item_first_sale_ts", "item_total_sales",
                     "item_n_stores", "is_new_item"]],
        on="item_id",
        how="left",
    )

    # Compute item_age_days and days_since_first_sale
    result["item_age_days"] = (result["date"] - result["item_first_sale_ts"]).dt.days
    result["days_since_first_sale"] = result["item_age_days"]

    # Fill NaN for items not in training data (cold items)
    result["item_age_days"] = result["item_age_days"].fillna(-1)
    result["days_since_first_sale"] = result["days_since_first_sale"].fillna(-1)
    result["item_total_sales"] = result["item_total_sales"].fillna(0)
    result["item_n_stores"] = result["item_n_stores"].fillna(0)
    result["is_new_item"] = result["is_new_item"].fillna(0).astype(int)

    # Drop the helper timestamp column
    result = result.drop(columns=["item_first_sale_ts"], errors="ignore")

    log(f"  item_age_days fill rate: {(result['item_age_days'] >= 0).mean():.4f}")
    log(f"  is_new_item rate: {result['is_new_item'].mean():.4f}")

    return result


# ===========================================================================
# Training
# ===========================================================================
def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_features: list[str],
    use_log_target: bool = True,
) -> tuple[CatBoostRegressor, float]:
    """
    Train CatBoost regressor with early stopping.

    If use_log_target is True, train with log1p(quantity) and return
    the model + the RMSE computed in the original (expm1) scale.
    """
    log("Training CatBoost...")
    log(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    log(f"  Cat features: {cat_features}")
    log(f"  Log1p target transform: {use_log_target}")

    if use_log_target:
        y_train_transformed = np.log1p(y_train)
        y_val_transformed = np.log1p(y_val)
    else:
        y_train_transformed = y_train
        y_val_transformed = y_val

    log(f"  y_train: mean={y_train.mean():.4f}, max={y_train.max():.4f}, "
        f"log1p mean={y_train_transformed.mean():.4f}, log1p max={y_train_transformed.max():.4f}")

    train_pool = Pool(X_train, label=y_train_transformed, cat_features=cat_features)
    val_pool = Pool(X_val, label=y_val_transformed, cat_features=cat_features)

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    log(f"  Best iteration: {model.get_best_iteration()}")
    log(f"  Best val RMSE (log scale): {model.get_best_score()['validation']['RMSE']:.4f}")

    # Compute validation RMSE in original scale
    val_preds_log = model.predict(val_pool)
    if use_log_target:
        val_preds = np.expm1(val_preds_log)
    else:
        val_preds = val_preds_log

    val_preds = np.clip(val_preds, 0, None)
    val_preds = np.nan_to_num(val_preds, nan=0.0)
    val_rmse_original = np.sqrt(np.mean((val_preds - y_val) ** 2))
    log(f"  Val RMSE (original scale): {val_rmse_original:.4f}")

    return model, val_rmse_original


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
    use_log_target: bool = True,
) -> np.ndarray:
    """
    Post-process predictions:
    1. If use_log_target, apply expm1 to convert back from log scale
    2. Clip to [0, max observed quantity]
    3. For cold items, predict class-level mean
    4. Fill any remaining NaN with 0
    """
    # Inverse transform
    if use_log_target:
        log(f"Applying expm1 inverse transform...")
        preds = np.expm1(preds)

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
    log("R04b YoY + Raw Target: CatBoost + YoY Features + log1p")
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
    log("--- R01 Features ---")
    log("Creating date features...")
    train_sales = create_date_features(train_sales)
    val_sales = create_date_features(val_sales)
    test = create_date_features(test)

    train_sales = create_lag_features(train_sales, train_sales)
    val_sales = create_lag_features(full_train, val_sales)
    test = create_lag_features(full_train, test)

    train_sales = create_price_feature(train_sales, train_sales)
    val_sales = create_price_feature(full_train, val_sales)
    test = create_price_feature(full_train, test)

    train_sales = merge_catalog(train_sales, catalog)
    val_sales = merge_catalog(val_sales, catalog)
    test = merge_catalog(test, catalog)

    # ================================================================
    # R02 Features
    # ================================================================
    log("--- R02 Features ---")

    train_sales = create_rolling_features(train_sales, train_sales)
    val_sales = create_rolling_features(full_train, val_sales)
    test = create_rolling_features(full_train, test)

    train_sales = create_store_aggregation_features(train_sales, train_sales)
    val_sales = create_store_aggregation_features(full_train, val_sales)
    test = create_store_aggregation_features(full_train, test)

    train_sales = create_dow_statistics(train_sales, train_sales)
    val_sales = create_dow_statistics(full_train, val_sales)
    test = create_dow_statistics(full_train, test)

    # ================================================================
    # R03 Features
    # ================================================================
    log("--- R03 Features ---")

    train_sales = create_discount_features(discounts, train_sales)
    val_sales = create_discount_features(discounts, val_sales)
    test = create_discount_features(discounts, test)

    train_sales = create_price_history_features(price_history, train_sales, train_sales)
    val_sales = create_price_history_features(price_history, val_sales, full_train)
    test = create_price_history_features(price_history, test, full_train)

    train_sales = create_markdown_features(markdowns, train_sales)
    val_sales = create_markdown_features(markdowns, val_sales)
    test = create_markdown_features(markdowns, test)

    train_sales = create_store_features(stores, train_sales)
    val_sales = create_store_features(stores, val_sales)
    test = create_store_features(stores, test)

    train_sales = create_availability_features(actual_matrix, train_sales)
    val_sales = create_availability_features(actual_matrix, val_sales)
    test = create_availability_features(actual_matrix, test)

    train_sales = create_holiday_features(train_sales)
    val_sales = create_holiday_features(val_sales)
    test = create_holiday_features(test)

    train_sales = create_online_features(online, train_sales)
    val_sales = create_online_features(online, val_sales)
    test = create_online_features(online, test)

    # ================================================================
    # R04 NEW Features
    # ================================================================
    log("--- R04 New Features ---")

    # ----------------------------------------------------------------
    # R04-1: YoY 364-day features
    # ----------------------------------------------------------------
    log("R04-1: YoY features...")
    train_sales = create_yoy_features(train_sales, train_sales)
    val_sales = create_yoy_features(full_train, val_sales)
    test = create_yoy_features(full_train, test)

    # ----------------------------------------------------------------
    # R04-2: Extended date features
    # ----------------------------------------------------------------
    log("R04-2: Extended date features...")
    train_sales = create_extended_date_features(train_sales)
    val_sales = create_extended_date_features(val_sales)
    test = create_extended_date_features(test)

    # ----------------------------------------------------------------
    # R04-3: Item lifecycle features
    # ----------------------------------------------------------------
    log("R04-3: Item lifecycle features...")
    train_sales = create_lifecycle_features(train_sales, train_sales)
    val_sales = create_lifecycle_features(full_train, val_sales)
    test = create_lifecycle_features(full_train, test)

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
    store_features_r02 = ["store_daily_qty", "item_store_qty_ratio"]

    # R02 day-of-week statistics
    dow_features = ["item_dow_mean", "item_dow_std"]

    # R03 discount/promotion features
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

    # R04 YoY features
    yoy_features = [
        "qty_yoy_364", "qty_yoy_364_ratio", "qty_yoy_diff_7d",
        "item_year_mean", "item_store_year_trend",
    ]

    # R04 extended date features
    extended_date_features = [
        "week_of_year", "quarter", "day_of_year", "is_pay_day",
    ]

    # R04 item lifecycle features
    lifecycle_features = [
        "item_age_days", "item_total_sales", "item_n_stores",
        "is_new_item", "days_since_first_sale",
    ]

    feature_cols = (
        r01_features + rolling_features + store_features_r02 + dow_features
        + discount_features + price_hist_features + markdown_features
        + store_meta_features + availability_features
        + holiday_features + online_features
        + yoy_features + extended_date_features + lifecycle_features
    )

    log(f"Feature columns ({len(feature_cols)}):")
    log(f"  R01 features:       {len(r01_features)}")
    log(f"  Rolling features:   {len(rolling_features)}")
    log(f"  Store agg features: {len(store_features_r02)}")
    log(f"  DOW features:       {len(dow_features)}")
    log(f"  Discount features:  {len(discount_features)}")
    log(f"  Price hist features:{len(price_hist_features)}")
    log(f"  Markdown features:  {len(markdown_features)}")
    log(f"  Store meta features:{len(store_meta_features)}")
    log(f"  Availability:       {len(availability_features)}")
    log(f"  Holiday features:   {len(holiday_features)}")
    log(f"  Online features:    {len(online_features)}")
    log(f"  --- R04 NEW ---")
    log(f"  YoY features:       {len(yoy_features)}")
    log(f"  Extended date:      {len(extended_date_features)}")
    log(f"  Lifecycle features: {len(lifecycle_features)}")
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
    # Train model with log1p target transform
    # ================================================================
    USE_LOG_TARGET = False
    model, val_rmse_original = train_catboost(
        X_train, y_train, X_val, y_val, CAT_FEATURES,
        use_log_target=USE_LOG_TARGET,
    )

    # ================================================================
    # Feature importance
    # ================================================================
    show_feature_importance(model, feature_cols)

    # ================================================================
    # Predict on test
    # ================================================================
    log("Predicting on test set...")
    test_pool = Pool(X_test, cat_features=CAT_FEATURES)
    preds_log = model.predict(test_pool)

    # ================================================================
    # Post-process (includes expm1 if USE_LOG_TARGET)
    # ================================================================
    preds = postprocess_predictions(preds_log, sales, test, catalog, use_log_target=USE_LOG_TARGET)

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
    best_score_log = model.get_best_score()["validation"]["RMSE"]
    log("=" * 70)
    log("R04b YoY + Raw Target Summary")
    log("=" * 70)
    log(f"Val RMSE (log scale):    {best_score_log:.4f}")
    log(f"Val RMSE (orig scale):   {val_rmse_original:.4f}")
    log(f"Best iter:               {model.get_best_iteration()}")
    log(f"Log1p target:            {USE_LOG_TARGET}")
    log(f"Train time:              {elapsed:.1f}s")
    log(f"Total features:          {len(feature_cols)}")
    log(f"  R01:                   {len(r01_features)}")
    log(f"  Rolling:               {len(rolling_features)}")
    log(f"  Store agg:             {len(store_features_r02)}")
    log(f"  DOW stats:             {len(dow_features)}")
    log(f"  Discounts:             {len(discount_features)}")
    log(f"  Price hist:            {len(price_hist_features)}")
    log(f"  Markdowns:             {len(markdown_features)}")
    log(f"  Store meta:            {len(store_meta_features)}")
    log(f"  Availability:          {len(availability_features)}")
    log(f"  Holidays:              {len(holiday_features)}")
    log(f"  Online:                {len(online_features)}")
    log(f"  YoY (NEW):             {len(yoy_features)}")
    log(f"  Extended date (NEW):   {len(extended_date_features)}")
    log(f"  Lifecycle (NEW):       {len(lifecycle_features)}")
    log(f"Pred mean:               {preds.mean():.4f}")
    log(f"Pred median:             {np.median(preds):.4f}")
    log(f"Submission:              {SUBMISSION_PATH}")
    log("=" * 70)

    return val_rmse_original


if __name__ == "__main__":
    score = main()
    sys.exit(0)
