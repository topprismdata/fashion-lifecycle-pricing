#!/usr/bin/env python3
"""
R08b: Submit Track A AutoGluon predictions only.
The AutoGluon model is already trained and saved.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submissions"

SALES_PATH = DATA_DIR / "sales.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
ONLINE_PATH = DATA_DIR / "online.csv"
ACTUAL_MATRIX_PATH = DATA_DIR / "actual_matrix.csv"

SUBMISSION_PATH = OUTPUT_DIR / "submission_r08_autogluon.csv"
AG_PATH = OUTPUT_DIR / "r08_autogluon"

GRID_START = "2024-03-01"
VAL_END = "2024-09-26"

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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_rolling_sql(train_df):
    """Build rolling features via DuckDB SQL."""
    log("Computing rolling features via DuckDB SQL...")
    con = duckdb.connect()

    daily_qty = (
        train_df.groupby(["item_id", "store_id", "date"], as_index=False)
        .agg(total_qty=("total_qty", "sum"))
    )
    daily_qty = daily_qty.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    con.register("daily_qty", daily_qty)

    rolling_df = con.execute("""
        SELECT
            item_id,
            store_id,
            date,
            total_qty,
            COALESCE(AVG(total_qty) OVER (
                PARTITION BY item_id, store_id ORDER BY date
                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING), 0) AS rolling_7d_avg,
            COALESCE(AVG(total_qty) OVER (
                PARTITION BY item_id, store_id ORDER BY date
                ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING), 0) AS rolling_14d_avg,
            COALESCE(AVG(total_qty) OVER (
                PARTITION BY item_id, store_id ORDER BY date
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING), 0) AS rolling_30d_avg,
            COALESCE(AVG(total_qty) OVER (
                PARTITION BY item_id, store_id ORDER BY date
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING), 0) AS rolling_60d_avg,
            COALESCE(STDDEV(total_qty) OVER (
                PARTITION BY item_id, store_id ORDER BY date
                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING), 0) AS rolling_7d_std,
            COALESCE(STDDEV(total_qty) OVER (
                PARTITION BY item_id, store_id ORDER BY date
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING), 0) AS rolling_30d_std
        FROM daily_qty
        ORDER BY item_id, store_id, date
    """).fetchdf()
    con.close()
    return rolling_df


def main():
    log("=" * 70)
    log("R08b: Submit AutoGluon Track A Predictions")
    log("=" * 70)

    t0 = time.time()

    # 1. Load test + sample sub
    test = pd.read_csv(TEST_PATH, sep=";")
    test["date"] = pd.to_datetime(test["date"], format="%d.%m.%Y")
    test["store_id"] = test["store_id"].astype(str)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    log(f"Test: {test.shape}")

    # 2. Rebuild dense grid for rolling features (need training data)
    #    Actually we just need the daily quantity for rolling computation
    #    Load from original sales + online
    log("Loading training data for rolling features...")
    sales = pd.read_csv(SALES_PATH, usecols=["date", "item_id", "quantity", "store_id"])
    sales["date"] = pd.to_datetime(sales["date"])
    sales["store_id"] = sales["store_id"].astype(str)
    sales = sales[sales["quantity"] >= 0].copy()

    online = pd.read_csv(ONLINE_PATH, usecols=["date", "item_id", "quantity", "store_id"])
    online["date"] = pd.to_datetime(online["date"])
    online["store_id"] = online["store_id"].astype(str)
    online = online[online["quantity"] >= 0].copy()

    # Build daily total qty
    sales_daily = sales.groupby(["date", "item_id", "store_id"], as_index=False).agg(
        total_qty=("quantity", "sum")
    )
    online_daily = online.groupby(["date", "item_id", "store_id"], as_index=False).agg(
        total_qty=("quantity", "sum")
    )
    combined = pd.concat([sales_daily, online_daily])
    combined = combined.groupby(["date", "item_id", "store_id"], as_index=False).agg(
        total_qty=("total_qty", "sum")
    )
    combined = combined[combined["date"] >= GRID_START].copy()
    log(f"Training daily qty: {combined.shape}")

    # 3. Compute rolling features on training data
    rolling_df = build_rolling_sql(combined)

    # Get last rolling values per (item_id, store_id)
    last_rolling = rolling_df.groupby(["item_id", "store_id"]).last().reset_index()
    rolling_cols = ["rolling_7d_avg", "rolling_14d_avg", "rolling_30d_avg",
                    "rolling_60d_avg", "rolling_7d_std", "rolling_30d_std"]

    # 4. Prepare test features (same as Track A minimal)
    test_features = test.copy()
    test_features["day_of_week"] = test_features["date"].dt.dayofweek
    test_features["day"] = test_features["date"].dt.day
    test_features["week"] = test_features["date"].dt.isocalendar().week.astype(int)

    # Merge rolling features (use last known values)
    test_features = test_features.merge(
        last_rolling[["item_id", "store_id"] + rolling_cols],
        on=["item_id", "store_id"],
        how="left"
    )
    for col in rolling_cols:
        test_features[col] = test_features[col].fillna(0)

    feature_cols = [
        "item_id", "store_id", "day", "week", "day_of_week",
        "rolling_7d_avg", "rolling_14d_avg", "rolling_30d_avg",
        "rolling_60d_avg", "rolling_7d_std", "rolling_30d_std",
    ]

    X_test = test_features[feature_cols].copy()
    # Ensure categorical columns are string
    for col in ["item_id", "store_id"]:
        X_test[col] = X_test[col].astype(str)

    log(f"Test features: {X_test.shape}")

    # 5. Load AutoGluon model and predict
    from autogluon.tabular import TabularPredictor
    log(f"Loading AutoGluon model from {AG_PATH}...")
    predictor = TabularPredictor.load(str(AG_PATH))

    log("Predicting...")
    preds = predictor.predict(X_test)
    preds = np.clip(preds.values, 0, None)

    # 6. Post-process
    preds = np.nan_to_num(preds, nan=0.0)

    log("Prediction distribution:")
    log(f"  mean={preds.mean():.4f}, median={np.median(preds):.4f}")
    log(f"  min={preds.min():.4f}, max={preds.max():.4f}")
    log(f"  std={preds.std():.4f}")

    # 7. Generate submission
    submission = pd.DataFrame({
        "row_id": test["row_id"].values,
        "quantity": preds,
    })

    assert len(submission) == len(sample_sub), f"{len(submission)} vs {len(sample_sub)}"
    assert list(submission.columns) == list(sample_sub.columns)

    submission.to_csv(SUBMISSION_PATH, index=False)
    log(f"Submission saved to {SUBMISSION_PATH}")

    elapsed = time.time() - t0
    log("=" * 70)
    log(f"R08b Summary: AutoGluon only, {elapsed:.1f}s")
    log(f"Submission: {SUBMISSION_PATH}")
    log("=" * 70)


if __name__ == "__main__":
    main()
