#!/usr/bin/env python3
"""
R02 Rolling Features: CatBoost + Rolling Statistics for ML Zoomcamp 2024 Retail Demand Forecast.

Builds on R01 Baseline (14 features) by adding:
  1. Rolling window statistics per (item_id, store_id):
     - Rolling mean: 7, 14, 30, 60 day windows
     - Rolling std:  7, 14, 30 day windows
     - Rolling min/max: 7, 30 day windows
  2. Exponential weighted mean per (item_id, store_id):
     - EWM span=7, span=30
  3. Store-level aggregation features:
     - Store total daily quantity
     - Item's share of store daily quantity
  4. Day-of-week statistics per item:
     - Mean and std quantity per (item_id, day_of_week)

All R01 features are retained:
  - Date features, item/store identity, catalog features, lag features, price feature

Validation: time-based holdout (same as R01)
  Train: before 2024-08-27
  Val:   2024-08-27 ~ 2024-09-26 (30 days)
  Test:  2024-09-27 ~ 2024-10-26 (30 days)

Usage:
    python scripts/run_r02_rolling.py
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
SUBMISSION_PATH = OUTPUT_DIR / "submission_r02_rolling.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VAL_START = "2024-08-27"
VAL_END = "2024-09-26"

CAT_FEATURES = ["item_id", "store_id", "dept_name", "class_name"]
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


def log(msg: str) -> None:
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load sales, test, catalog, and sample submission."""
    log("Loading data...")
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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def filter_negative_quantities(sales: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with negative quantity."""
    before = len(sales)
    sales = sales[sales["quantity"] >= 0].copy()
    after = len(sales)
    log(f"Filtered negative quantities: {before} -> {after} (removed {before - after})")
    return sales


# ---------------------------------------------------------------------------
# Feature Engineering — R01 features (unchanged)
# ---------------------------------------------------------------------------
def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date-based features from the date column."""
    df = df.copy()
    dt = df["date"]
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon, 6=Sun
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
    """
    Create lag features for target_df using data from train_df only.

    For each row in target_df with date D, the lag_7 feature is the quantity
    from train_df at date (D - 7) for the same (item_id, store_id).

    Missing lags are filled with 0.
    """
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


# ---------------------------------------------------------------------------
# Feature Engineering — R02 new features
# ---------------------------------------------------------------------------
def _build_daily_series(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete daily time series per (item_id, store_id) from training data.

    Returns a DataFrame sorted by (item_id, store_id, date) with a 'qty' column.
    Dates with no sales are NOT filled (sparse representation) — rolling windows
    operate on observed data only, which is correct for sparse retail data.
    """
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
    """
    Create rolling window statistics for target_df using training data only.

    Strategy:
      1. Build daily qty series from train_df per (item_id, store_id).
      2. Compute rolling stats on the daily series using shift(1) to avoid
         using the current day's data.
      3. For each target row at date D, look up the rolling stat computed
         at the most recent training date <= D for that (item_id, store_id).
         This is done by forward-filling the rolling stats and mapping.

    To handle val/test rows whose dates fall beyond the last training date,
    we extend the daily series with zero-quantity entries for those dates so
    the rolling windows can be computed forward.
    """
    log("Creating rolling features...")

    daily = _build_daily_series(train_df)
    log(f"  Daily series: {len(daily)} rows, "
        f"{daily['item_id'].nunique()} items, "
        f"{daily['store_id'].nunique()} stores")

    # Compute rolling stats on the daily series (per group)
    # shift(1) ensures we don't use the current day's data
    grouped = daily.groupby(["item_id", "store_id"])

    # --- Rolling mean ---
    for w in ROLL_MEAN_WINDOWS:
        col = f"qty_roll_mean_{w}"
        daily[col] = (
            grouped["qty"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
            .values
        )

    # --- Rolling std ---
    for w in ROLL_STD_WINDOWS:
        col = f"qty_roll_std_{w}"
        daily[col] = (
            grouped["qty"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .std()
            .values
        )

    # --- Rolling min ---
    for w in ROLL_MIN_WINDOWS:
        col = f"qty_roll_min_{w}"
        daily[col] = (
            grouped["qty"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .min()
            .values
        )

    # --- Rolling max ---
    for w in ROLL_MAX_WINDOWS:
        col = f"qty_roll_max_{w}"
        daily[col] = (
            grouped["qty"]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .max()
            .values
        )

    # --- Exponential weighted mean ---
    for span in EWM_SPANS:
        col = f"qty_ewm_{span}"
        daily[col] = (
            grouped["qty"]
            .shift(1)
            .ewm(span=span, min_periods=1)
            .mean()
            .values
        )

    # Identify all rolling/ewm columns
    rolling_cols = (
        [f"qty_roll_mean_{w}" for w in ROLL_MEAN_WINDOWS]
        + [f"qty_roll_std_{w}" for w in ROLL_STD_WINDOWS]
        + [f"qty_roll_min_{w}" for w in ROLL_MIN_WINDOWS]
        + [f"qty_roll_max_{w}" for w in ROLL_MAX_WINDOWS]
        + [f"qty_ewm_{span}" for span in EWM_SPANS]
    )

    # For lookup: we need the rolling stat at the most recent date <= target date.
    # Build a lookup table: for each (item_id, store_id, date) -> rolling stats
    # Then merge target_df with this lookup using an asof-style approach.
    #
    # Efficient approach: expand daily to cover all dates in target_df per group,
    # forward-fill the rolling stats, then merge.

    # Get all unique dates from target_df
    target_dates = target_df["date"].unique()

    # For each (item_id, store_id) group, we need rolling stats at target dates.
    # Build a complete frame: for each group, create entries for all target dates,
    # concat with existing daily, sort, ffill rolling cols, then filter to target dates.
    log("  Extending daily series to cover target dates...")

    # Get unique (item_id, store_id) combos that appear in target_df
    target_groups = target_df[["item_id", "store_id"]].drop_duplicates()

    # Build empty target-date rows
    target_date_rows = []
    for date in target_dates:
        frame = target_groups.copy()
        frame["date"] = date
        frame["qty"] = 0  # placeholder, not used for rolling computation
        target_date_rows.append(frame)

    target_date_df = pd.concat(target_date_rows, ignore_index=True)

    # Mark which rows are original (have computed rolling stats) vs target-date rows
    daily["_is_original"] = True
    target_date_df["_is_original"] = False
    for col in rolling_cols:
        target_date_df[col] = np.nan

    # Concat and sort
    combined = pd.concat([daily, target_date_df], ignore_index=True)
    combined = combined.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    # Forward-fill rolling stats within each group
    log("  Forward-filling rolling stats...")
    for col in rolling_cols:
        combined[col] = combined.groupby(["item_id", "store_id"])[col].ffill()

    # Keep only target-date rows for the merge
    lookup = combined[~combined["_is_original"]].copy()
    lookup = lookup.drop(columns=["_is_original", "qty"])

    log(f"  Lookup table for merge: {len(lookup)} rows")

    # Merge rolling features into target_df
    result = target_df.merge(
        lookup,
        on=["item_id", "store_id", "date"],
        how="left",
    )

    # Fill NaN with 0 for items with no history
    for col in rolling_cols:
        fill_rate = result[col].notna().mean()
        result[col] = result[col].fillna(0)
        log(f"  {col}: fill rate={fill_rate:.4f}")

    return result


def create_store_aggregation_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create store-level aggregation features:
      - store_daily_qty: total quantity sold per store per day (from train)
      - item_store_qty_ratio: item's share of store daily quantity

    The store daily qty is computed from training data as a mean per
    (store, day_of_week) to capture weekly patterns, then mapped to target rows.
    """
    log("Creating store aggregation features...")

    # Compute store-level daily quantity stats from training data
    # Use day_of_week average for better generalization
    train_daily = (
        train_df.groupby(["store_id", "date"], as_index=False)
        .agg(store_qty=("quantity", "sum"))
    )
    train_daily["day_of_week"] = train_daily["date"].dt.dayofweek

    # Average store daily quantity per (store_id, day_of_week)
    store_dow_qty = (
        train_daily.groupby(["store_id", "day_of_week"], as_index=False)
        .agg(store_daily_qty=("store_qty", "mean"))
    )
    log(f"  store_dow_qty: {len(store_dow_qty)} rows")

    # Compute item-level mean daily quantity per (item_id, store_id, day_of_week)
    train_item_daily = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(item_qty=("quantity", "sum"))
    )
    train_item_daily["day_of_week"] = train_item_daily["date"].dt.dayofweek

    item_dow_qty = (
        train_item_daily.groupby(["item_id", "store_id", "day_of_week"], as_index=False)
        .agg(item_dow_mean_qty=("item_qty", "mean"))
    )

    # Merge to get ratio
    item_store_ratio = item_dow_qty.merge(store_dow_qty, on=["store_id", "day_of_week"], how="left")
    item_store_ratio["item_store_qty_ratio"] = np.where(
        item_store_ratio["store_daily_qty"] > 0,
        item_store_ratio["item_dow_mean_qty"] / item_store_ratio["store_daily_qty"],
        0,
    )
    item_store_ratio = item_store_ratio[["item_id", "store_id", "day_of_week",
                                         "store_daily_qty", "item_store_qty_ratio"]]

    # Merge into target_df
    result = target_df.copy()
    result = result.merge(
        item_store_ratio,
        on=["item_id", "store_id", "day_of_week"],
        how="left",
    )

    result["store_daily_qty"] = result["store_daily_qty"].fillna(0)
    result["item_store_qty_ratio"] = result["item_store_qty_ratio"].fillna(0)

    log(f"  store_daily_qty fill rate: {result['store_daily_qty'].notna().mean():.4f}")
    log(f"  item_store_qty_ratio fill rate: {result['item_store_qty_ratio'].notna().mean():.4f}")

    return result


def create_dow_statistics(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create day-of-week statistics per item:
      - item_dow_mean: mean quantity per (item_id, day_of_week)
      - item_dow_std:  std quantity per (item_id, day_of_week)
    """
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
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
    log("Feature importance (top 25):")
    for i in sorted_idx[:25]:
        log(f"  {feature_names[i]:35s} {importances[i]:8.2f}")


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess_predictions(
    preds: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    catalog: pd.DataFrame,
) -> np.ndarray:
    """
    Clip predictions to [0, max observed quantity].
    For cold items (not in train), predict class-level mean.
    """
    max_qty = train_df["quantity"].max()
    log(f"Clipping predictions to [0, {max_qty}]")
    preds = np.clip(preds, 0, max_qty)

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

    return preds


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> float:
    log("=" * 70)
    log("R02 Rolling Features: CatBoost + Rolling Statistics")
    log("=" * 70)

    t0 = time.time()

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    sales, test, catalog, sample_sub = load_data()

    # ----------------------------------------------------------------
    # 2. Filter negative quantities
    # ----------------------------------------------------------------
    sales = filter_negative_quantities(sales)

    # ----------------------------------------------------------------
    # 3. Split train/val (same as R01)
    # ----------------------------------------------------------------
    train_mask = sales["date"] < VAL_START
    val_mask = (sales["date"] >= VAL_START) & (sales["date"] <= VAL_END)

    train_sales = sales[train_mask].copy()
    val_sales = sales[val_mask].copy()

    log(f"Train split: {len(train_sales)} rows, "
        f"{train_sales['date'].min().date()} ~ {train_sales['date'].max().date()}")
    log(f"Val split:   {len(val_sales)} rows, "
        f"{val_sales['date'].min().date()} ~ {val_sales['date'].max().date()}")

    # ----------------------------------------------------------------
    # 4. Create date features (R01, unchanged)
    # ----------------------------------------------------------------
    log("Creating date features...")
    train_sales = create_date_features(train_sales)
    val_sales = create_date_features(val_sales)
    test = create_date_features(test)

    # ----------------------------------------------------------------
    # 5. Create lag features (R01, unchanged)
    # ----------------------------------------------------------------
    # For val/test: lags come from full sales data up to VAL_END
    full_train = sales[sales["date"] <= VAL_END].copy()

    train_sales = create_lag_features(train_sales, train_sales)
    val_sales = create_lag_features(full_train, val_sales)
    test = create_lag_features(full_train, test)

    # ----------------------------------------------------------------
    # 6. Price feature (R01, unchanged)
    # ----------------------------------------------------------------
    train_sales = create_price_feature(train_sales, train_sales)
    val_sales = create_price_feature(full_train, val_sales)
    test = create_price_feature(full_train, test)

    # ----------------------------------------------------------------
    # 7. Merge catalog (R01, unchanged)
    # ----------------------------------------------------------------
    train_sales = merge_catalog(train_sales, catalog)
    val_sales = merge_catalog(val_sales, catalog)
    test = merge_catalog(test, catalog)

    # ----------------------------------------------------------------
    # 8. R02 NEW: Rolling window features
    # ----------------------------------------------------------------
    log("--- R02 New Features ---")

    train_sales = create_rolling_features(train_sales, train_sales)
    val_sales = create_rolling_features(full_train, val_sales)
    test = create_rolling_features(full_train, test)

    # ----------------------------------------------------------------
    # 9. R02 NEW: Store-level aggregation features
    # ----------------------------------------------------------------
    train_sales = create_store_aggregation_features(train_sales, train_sales)
    val_sales = create_store_aggregation_features(full_train, val_sales)
    test = create_store_aggregation_features(full_train, test)

    # ----------------------------------------------------------------
    # 10. R02 NEW: Day-of-week statistics per item
    # ----------------------------------------------------------------
    train_sales = create_dow_statistics(train_sales, train_sales)
    val_sales = create_dow_statistics(full_train, val_sales)
    test = create_dow_statistics(full_train, test)

    # ----------------------------------------------------------------
    # 11. Define feature columns
    # ----------------------------------------------------------------
    # R01 features
    r01_features = DATE_FEATURES + [
        f"quantity_lag_{lag}" for lag in LAG_DAYS
    ] + ["price_base_latest"] + CAT_FEATURES

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

    feature_cols = r01_features + rolling_features + store_features + dow_features

    log(f"Feature columns ({len(feature_cols)}):")
    log(f"  R01 features:     {len(r01_features)}")
    log(f"  Rolling features: {len(rolling_features)}")
    log(f"  Store features:   {len(store_features)}")
    log(f"  DOW features:     {len(dow_features)}")
    log(f"  Total:            {len(feature_cols)}")

    # ----------------------------------------------------------------
    # 12. Prepare training data
    # ----------------------------------------------------------------
    X_train = train_sales[feature_cols].copy()
    y_train = train_sales["quantity"].values
    X_val = val_sales[feature_cols].copy()
    y_val = val_sales["quantity"].values
    X_test = test[feature_cols].copy()

    # Ensure categorical columns are string type for CatBoost
    for col in CAT_FEATURES:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # ----------------------------------------------------------------
    # 13. Train model (same CatBoost config as R01)
    # ----------------------------------------------------------------
    model = train_catboost(X_train, y_train, X_val, y_val, CAT_FEATURES)

    # ----------------------------------------------------------------
    # 14. Feature importance
    # ----------------------------------------------------------------
    show_feature_importance(model, feature_cols)

    # ----------------------------------------------------------------
    # 15. Predict on test
    # ----------------------------------------------------------------
    log("Predicting on test set...")
    test_pool = Pool(X_test, cat_features=CAT_FEATURES)
    preds = model.predict(test_pool)

    # ----------------------------------------------------------------
    # 16. Post-process
    # ----------------------------------------------------------------
    preds = postprocess_predictions(preds, sales, test, catalog)

    # ----------------------------------------------------------------
    # 17. Prediction distribution stats
    # ----------------------------------------------------------------
    log("Prediction distribution:")
    log(f"  mean={preds.mean():.4f}, median={np.median(preds):.4f}")
    log(f"  min={preds.min():.4f}, max={preds.max():.4f}")
    log(f"  std={preds.std():.4f}")
    log(f"  pct zeros: {(preds == 0).mean()*100:.2f}%")

    # ----------------------------------------------------------------
    # 18. Generate submission
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 19. Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t0
    best_score = model.get_best_score()["validation"]["RMSE"]
    log("=" * 70)
    log("R02 Rolling Features Summary")
    log("=" * 70)
    log(f"Val RMSE:     {best_score:.4f}")
    log(f"Best iter:    {model.get_best_iteration()}")
    log(f"Train time:   {elapsed:.1f}s")
    log(f"Features:     {len(feature_cols)}")
    log(f"  R01:        {len(r01_features)}")
    log(f"  Rolling:    {len(rolling_features)}")
    log(f"  Store agg:  {len(store_features)}")
    log(f"  DOW stats:  {len(dow_features)}")
    log(f"Pred mean:    {preds.mean():.4f}")
    log(f"Pred median:  {np.median(preds):.4f}")
    log(f"Submission:   {SUBMISSION_PATH}")
    log("=" * 70)

    return best_score


if __name__ == "__main__":
    score = main()
    sys.exit(0)
