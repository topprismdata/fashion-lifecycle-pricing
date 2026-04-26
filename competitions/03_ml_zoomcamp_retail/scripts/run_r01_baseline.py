#!/usr/bin/env python3
"""
R01 Baseline: CatBoost + Minimal Features for ML Zoomcamp 2024 Retail Demand Forecast.

Features:
  - Date features: day_of_week, day_of_month, month, is_weekend, is_month_start, is_month_end
  - Item identity: item_id, store_id (categorical)
  - Catalog: dept_name, class_name (categorical)
  - Lag features: quantity_lag_7, lag_14, lag_30 per (item_id, store_id)
  - Latest known price_base per (item_id, store_id)

Validation: time-based holdout
  Train: before 2024-08-27
  Val:   2024-08-27 ~ 2024-09-26 (30 days)
  Test:  2024-09-27 ~ 2024-10-26 (30 days)

Usage:
    python scripts/run_r01_baseline.py
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
SUBMISSION_PATH = OUTPUT_DIR / "submission_r01_baseline.csv"

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
# Feature Engineering
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

    This avoids the ts-lag-out-of-sample-trap: we only look at training data
    that existed BEFORE the prediction date.

    For test dates in [2024-09-27, 2024-10-26]:
      - lag_7 for 2024-10-03 -> looks at 2024-09-26 (in train) -> OK
      - lag_7 for 2024-10-26 -> looks at 2024-10-19 (NOT in train) -> missing
      - lag_30 has better coverage since it looks further back.

    Missing lags are filled with 0 (item had no sales on that day).
    """
    log("Creating lag features...")

    # Aggregate to daily level per (item_id, store_id) for efficiency
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

        # Create shifted lookup: map (item_id, store_id, lag_date) -> quantity
        # where lag_date = date + lag_days
        shifted = daily_qty[["item_id", "store_id", "date", "quantity"]].copy()
        shifted["lag_date"] = shifted["date"] + pd.Timedelta(days=lag)
        shifted = shifted.rename(columns={"quantity": lag_col})
        shifted = shifted[["item_id", "store_id", "lag_date", lag_col]]

        # Merge: target row at date D gets quantity from (D - lag)
        result = result.merge(
            shifted,
            left_on=["item_id", "store_id", "date"],
            right_on=["item_id", "store_id", "lag_date"],
            how="left",
        )
        result = result.drop(columns=["lag_date"])

        fill_rate = result[lag_col].notna().mean()
        log(f"    {lag_col} fill rate: {fill_rate:.4f}")

    # Fill missing lags with 0
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
    log("Feature importance (top 20):")
    for i in sorted_idx[:20]:
        log(f"  {feature_names[i]:30s} {importances[i]:8.2f}")


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

    # Identify cold items: in test but not in train
    train_items = set(train_df["item_id"].unique())
    cold_mask = ~test_df["item_id"].isin(train_items)
    n_cold = cold_mask.sum()
    n_cold_items = test_df.loc[cold_mask, "item_id"].nunique()
    log(f"Cold items: {n_cold_items} items ({n_cold} rows, {cold_mask.mean()*100:.2f}%)")

    if n_cold > 0:
        # Compute dept-level mean quantity as fallback for cold items
        train_item_mean = train_df.groupby("item_id")["quantity"].mean().reset_index()
        train_item_mean.columns = ["item_id", "item_mean_qty"]

        catalog_subset = catalog[["item_id", "dept_name"]].drop_duplicates(subset=["item_id"])
        train_item_mean = train_item_mean.merge(catalog_subset, on="item_id", how="left")

        dept_mean = train_item_mean.groupby("dept_name")["item_mean_qty"].mean()
        overall_mean = train_item_mean["item_mean_qty"].mean()

        # Apply dept-level mean to cold items
        cold_items = test_df.loc[cold_mask, "item_id"].unique()
        cold_catalog = catalog_subset[catalog_subset["item_id"].isin(cold_items)]
        cold_item_dept_mean = cold_catalog.set_index("item_id")["dept_name"].map(dept_mean)
        cold_item_dept_mean = cold_item_dept_mean.fillna(overall_mean)

        # Map back to rows
        cold_dept_means = test_df.loc[cold_mask, "item_id"].map(cold_item_dept_mean)
        preds[cold_mask] = cold_dept_means.values
        log(f"  Set cold item predictions to dept-level mean (overall mean={overall_mean:.4f})")

    return preds


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> float:
    log("=" * 70)
    log("R01 Baseline: CatBoost + Minimal Features")
    log("=" * 70)

    t0 = time.time()

    # 1. Load data
    sales, test, catalog, sample_sub = load_data()

    # 2. Filter negative quantities
    sales = filter_negative_quantities(sales)

    # 3. Split train/val
    train_mask = sales["date"] < VAL_START
    val_mask = (sales["date"] >= VAL_START) & (sales["date"] <= VAL_END)

    train_sales = sales[train_mask].copy()
    val_sales = sales[val_mask].copy()

    log(f"Train split: {len(train_sales)} rows, "
        f"{train_sales['date'].min().date()} ~ {train_sales['date'].max().date()}")
    log(f"Val split:   {len(val_sales)} rows, "
        f"{val_sales['date'].min().date()} ~ {val_sales['date'].max().date()}")

    # 4. Create date features
    log("Creating date features...")
    train_sales = create_date_features(train_sales)
    val_sales = create_date_features(val_sales)
    test = create_date_features(test)

    # 5. Create lag features
    # For train: lags come from train_sales itself
    # For val/test: lags come from full sales data up to VAL_END (avoids data leakage)
    full_train = sales[sales["date"] <= VAL_END].copy()

    train_sales = create_lag_features(train_sales, train_sales)
    val_sales = create_lag_features(full_train, val_sales)
    test = create_lag_features(full_train, test)

    # 6. Price feature
    train_sales = create_price_feature(train_sales, train_sales)
    val_sales = create_price_feature(full_train, val_sales)
    test = create_price_feature(full_train, test)

    # 7. Merge catalog
    train_sales = merge_catalog(train_sales, catalog)
    val_sales = merge_catalog(val_sales, catalog)
    test = merge_catalog(test, catalog)

    # 8. Define feature columns
    feature_cols = DATE_FEATURES + [
        f"quantity_lag_{lag}" for lag in LAG_DAYS
    ] + ["price_base_latest"] + CAT_FEATURES

    log(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # 9. Prepare training data
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

    # 10. Train model
    model = train_catboost(X_train, y_train, X_val, y_val, CAT_FEATURES)

    # 11. Feature importance
    show_feature_importance(model, feature_cols)

    # 12. Predict on test
    log("Predicting on test set...")
    test_pool = Pool(X_test, cat_features=CAT_FEATURES)
    preds = model.predict(test_pool)

    # 13. Post-process
    preds = postprocess_predictions(preds, sales, test, catalog)

    # 14. Prediction distribution stats
    log("Prediction distribution:")
    log(f"  mean={preds.mean():.4f}, median={np.median(preds):.4f}")
    log(f"  min={preds.min():.4f}, max={preds.max():.4f}")
    log(f"  std={preds.std():.4f}")
    log(f"  pct zeros: {(preds == 0).mean()*100:.2f}%")

    # 15. Generate submission
    log("Generating submission...")
    submission = pd.DataFrame({
        "row_id": test["row_id"].values,
        "quantity": preds,
    })

    # Verify format
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

    # 16. Summary
    elapsed = time.time() - t0
    best_score = model.get_best_score()["validation"]["RMSE"]
    log("=" * 70)
    log("R01 Baseline Summary")
    log("=" * 70)
    log(f"Val RMSE:     {best_score:.4f}")
    log(f"Best iter:    {model.get_best_iteration()}")
    log(f"Train time:   {elapsed:.1f}s")
    log(f"Features:     {len(feature_cols)}")
    log(f"Pred mean:    {preds.mean():.4f}")
    log(f"Pred median:  {np.median(preds):.4f}")
    log(f"Submission:   {SUBMISSION_PATH}")
    log("=" * 70)

    return best_score


if __name__ == "__main__":
    score = main()
    sys.exit(0)
