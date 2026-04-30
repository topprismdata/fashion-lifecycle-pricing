#!/usr/bin/env python3
"""R27: MLOps Framework-Validated GBDT Ranking Pipeline

Replicates R21's GBDT ranking algorithm using the MLOps framework:
  1. CompetitionConfig + load_config() for typed configuration
  2. MLflow tracking for full experiment logging (params, metrics, artifacts)
  3. validate_features() at stage boundary + evaluation_gate() to block regressions
  4. ExperimentLogger replaces print() calls
  5. OOF predictions + model binary saved as artifacts

Target metrics (matching R21):
  - Val MAP@12 (active): ~0.05
  - Val MAP@12 (full): ~0.022-0.024

Usage:
    python scripts/run_r27_gbdt_framework.py
    python scripts/run_r27_gbdt_framework.py --smoke-test
    python scripts/run_r27_gbdt_framework.py --no-mlflow
"""
import sys
import argparse
import gc
import math
import json
import warnings
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

# ---- Framework imports ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_SRC = PROJECT_ROOT.parent.parent / "src"
sys.path.insert(0, str(SHARED_SRC))
sys.path.insert(0, str(PROJECT_ROOT))

from config import CompetitionConfig, load_config, apply_overrides
from pipeline.validate import validate_features, evaluation_gate, classify_failure
from utils.logging_utils import get_logger
from utils.submission import validate_and_save, get_submission_filename
from utils.paths import get_competition_dirs

import pandas as pd
import numpy as np

# ===========================================================================
# Constants
# ===========================================================================
RUN_NAME = "R27_gbdt_framework"
RANDOM_STATE = 42

OVERRIDES = {
    "model.default": "lightgbm",
    "model.lgb_params.objective": "binary",
    "model.lgb_params.metric": "average_precision",
    "model.lgb_params.num_leaves": 63,
    "model.lgb_params.learning_rate": 0.05,
    "model.lgb_params.feature_fraction": 0.8,
    "model.lgb_params.bagging_fraction": 0.8,
    "model.lgb_params.bagging_freq": 5,
    "model.lgb_params.min_child_samples": 50,
}

FEATURE_COLS = [
    # Recall features (11)
    "repurchase_score", "repurchase_rank", "itemcf_score", "itemcf_rank",
    "cooc_rank", "pop_rank",
    "is_repurchase", "is_itemcf", "is_cooc", "is_pop", "num_recall_sources",
    # User-item interaction (2)
    "user_item_buys", "user_item_last_day",
    # User activity (4)
    "user_total_buys", "user_unique_items", "user_last_day", "user_age",
    # Item popularity (3)
    "item_popularity", "item_total_buys", "item_unique_buyers",
    # Item category (5)
    "item_dept", "item_section", "item_garment", "item_category", "item_color",
    # Recent trend (4) — critical features from R18/R21
    "item_recent_week_buys", "item_last2w_buys", "item_last4w_buys",
    "item_sales_trend",
    # Repurchase ratio (1)
    "item_repurchase_ratio",
    # Age-demographic (3)
    "item_user_mean_age", "user_item_age_diff", "item_user_age_std",
    # Price match (3)
    "user_mean_price", "item_mean_price", "user_item_price_diff",
    # Item freshness (2)
    "item_freshness", "item_last_day",
    # Channel match (2)
    "user_mean_channel", "item_mean_channel",
]

FILL_0 = [
    "repurchase_score", "itemcf_score", "item_popularity",
    "item_total_buys", "item_unique_buyers", "item_recent_week_buys",
    "item_last2w_buys", "item_last4w_buys", "item_sales_trend",
    "item_repurchase_ratio", "item_user_mean_age", "item_user_age_std",
    "item_mean_price", "item_freshness", "user_item_buys",
    "user_total_buys", "user_unique_items", "user_age",
    "user_mean_price", "num_recall_sources", "user_mean_channel",
    "item_mean_channel",
]
FILL_99 = [
    "repurchase_rank", "itemcf_rank", "cooc_rank", "pop_rank",
    "user_item_last_day", "user_last_day", "item_last_day",
    "user_item_price_diff",
]


def apk(actual, predicted, k=12):
    """Average Precision at K."""
    if not actual:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


# ===========================================================================
# Stage 0: Configuration
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description=f"{RUN_NAME}: GBDT ranking with MLOps framework")
    parser.add_argument("--smoke-test", action="store_true", help="Run with sampled data")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--override", action="append", default=[], help="Override config key=value")
    return parser.parse_args()


def fill_nan(df):
    """Apply NaN filling strategy matching R21."""
    for col in FILL_0:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in FILL_99:
        if col in df.columns:
            df[col] = df[col].fillna(99)


def build_row(cid, aid, feat, lookup):
    """Build a single feature row for candidate (cid, aid)."""
    u_age = lookup["cust_age"].get(cid, 35)
    u_mean_pr = lookup["user_mean_price"].get(cid, 0)
    u_mean_ch = lookup["user_mean_channel"].get(cid, 1.5)
    i_mean_age = lookup["item_user_mean_age"].get(aid, 35)
    i_mean_pr = lookup["item_mean_price"].get(aid, 0)

    return {
        "customer_id": cid,
        "article_id": aid,
        "target": 1 if aid in lookup.get("actual", set()) else 0,
        # Recall features
        "repurchase_score": feat.get("repurchase_score", 0),
        "repurchase_rank": feat.get("repurchase_rank", 99),
        "itemcf_score": feat.get("itemcf_score", 0),
        "itemcf_rank": feat.get("itemcf_rank", 99),
        "cooc_rank": feat.get("cooc_rank", 99),
        "pop_rank": feat.get("pop_rank", 99),
        "is_repurchase": 1 if "repurchase_rank" in feat else 0,
        "is_itemcf": 1 if "itemcf_rank" in feat else 0,
        "is_cooc": 1 if "cooc_rank" in feat else 0,
        "is_pop": 1 if "pop_rank" in feat else 0,
        "num_recall_sources": sum(
            1 for k in ["repurchase_rank", "itemcf_rank", "cooc_rank", "pop_rank"]
            if k in feat
        ),
        # User-item interaction
        "user_item_buys": lookup["user_item_buy_count"].get((cid, aid), 0),
        "user_item_last_day": lookup["user_item_last_day"].get((cid, aid), 999),
        # User features
        "user_total_buys": lookup["user_total_buys"].get(cid, 0),
        "user_unique_items": lookup["user_unique_items"].get(cid, 0),
        "user_last_day": lookup["user_last_purchase_day"].get(cid, 999),
        "user_age": u_age,
        # Item popularity
        "item_popularity": lookup["pop_score_dict"].get(aid, 0),
        "item_total_buys": lookup["item_total_buys"].get(aid, 0),
        "item_unique_buyers": lookup["item_unique_buyers"].get(aid, 0),
        # Item category
        "item_dept": lookup["art_dept"].get(aid, -1),
        "item_section": lookup["art_section"].get(aid, -1),
        "item_garment": lookup["art_garment"].get(aid, -1),
        "item_category": lookup["art_category"].get(aid, -1),
        "item_color": lookup["art_color"].get(aid, -1),
        # Recent trend
        "item_recent_week_buys": lookup["item_recent_week_buys"].get(aid, 0),
        "item_last2w_buys": lookup["item_last2w_buys"].get(aid, 0),
        "item_last4w_buys": lookup["item_last4w_buys"].get(aid, 0),
        "item_sales_trend": lookup["item_sales_trend"].get(aid, 1.0),
        # Repurchase ratio
        "item_repurchase_ratio": lookup["item_repurchase_ratio"].get(aid, 1.0),
        # Age-demographic
        "item_user_mean_age": i_mean_age,
        "user_item_age_diff": abs(u_age - i_mean_age),
        "item_user_age_std": lookup["item_user_age_std"].get(aid, 10),
        # Price match
        "user_mean_price": u_mean_pr,
        "item_mean_price": i_mean_pr,
        "user_item_price_diff": (
            abs(u_mean_pr - i_mean_pr) if u_mean_pr > 0 and i_mean_pr > 0 else 999
        ),
        # Item freshness
        "item_freshness": lookup["item_freshness"].get(aid, 0),
        "item_last_day": lookup["item_last_day"].get(aid, 999),
        # Channel match
        "user_mean_channel": u_mean_ch,
        "item_mean_channel": lookup["item_mean_channel"].get(aid, 1.5),
    }


# ===========================================================================
# Main pipeline
# ===========================================================================
def main():
    args = parse_args()
    log = get_logger(RUN_NAME)

    # Reproducibility
    np.random.seed(RANDOM_STATE)

    log.separator(f"{RUN_NAME}: MLOps Framework-Validated GBDT Ranking")

    # Load config
    cfg = load_config(
        config_path=PROJECT_ROOT / "config.yaml",
        overrides=OVERRIDES,
        cli_args=args.override if args.override else None,
    )

    # Data path: use data_raw/ directly (not data/raw/)
    DATA = PROJECT_ROOT / "data_raw"

    # Ensure output directories
    dirs = get_competition_dirs(cfg)
    dirs.ensure_dirs()
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Data dir: {DATA}")

    # ==================================================================
    # Stage 1: Data Loading
    # ==================================================================
    log.section("Stage 1: Data Loading")

    txn = pd.read_csv(DATA / "transactions_train.csv", parse_dates=["t_dat"])
    articles = pd.read_csv(DATA / "articles.csv")
    customers = pd.read_csv(DATA / "customers.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")

    max_date = txn["t_dat"].max()
    txn["week"] = ((max_date - txn["t_dat"]).dt.days // 7).astype("int8")
    txn["days_ago"] = (max_date - txn["t_dat"]).dt.days
    txn = txn[txn["week"] <= 12].copy()
    gc.collect()

    if args.smoke_test:
        sample_cids = txn["customer_id"].drop_duplicates().sample(10000, random_state=RANDOM_STATE)
        txn = txn[txn["customer_id"].isin(sample_cids)].copy()
        sample_sub = sample_sub.head(10000)
        gc.collect()
        log.info("SMOKE TEST: using 10K sampled customers")

    log.info(f"Transactions (12w): {len(txn):,}")
    log.data_shape("transactions", txn)
    log.info(f"Date range: {txn['t_dat'].min().date()} to {txn['t_dat'].max().date()}")

    # Customer features
    customers["age"] = customers["age"].fillna(customers["age"].median())
    cust_age = customers.set_index("customer_id")["age"].to_dict()

    # Article lookup tables
    art_dept = articles.set_index("article_id")["department_no"].to_dict()
    art_section = articles.set_index("article_id")["section_no"].to_dict()
    art_garment = articles.set_index("article_id")["garment_group_no"].to_dict()
    art_category = articles.set_index("article_id")["index_group_no"].to_dict()
    art_color = articles.set_index("article_id")["colour_group_code"].to_dict()

    # ==================================================================
    # Stage 2: Feature Engineering + Candidate Generation
    # ==================================================================
    log.section("Stage 2: Feature Engineering + Candidate Generation")

    txn_features = txn[txn["week"] >= 1].copy()  # Weeks 1-12 for features

    # --- Item stats ---
    log.info("  Item stats...")
    item_total_buys = txn_features["article_id"].value_counts().to_dict()
    item_unique_buyers = txn_features.groupby("article_id")["customer_id"].nunique().to_dict()

    # Recent week buys (R18/R21 key feature)
    item_recent_week_buys = txn_features[txn_features["week"] == 1]["article_id"].value_counts().to_dict()
    item_last2w_buys = txn_features[txn_features["week"] <= 2]["article_id"].value_counts().to_dict()
    item_last4w_buys = txn_features[txn_features["week"] <= 4]["article_id"].value_counts().to_dict()

    # Sales trend
    item_weekly_avg = {aid: count / 12.0 for aid, count in item_total_buys.items()}
    item_sales_trend = {}
    for aid in item_recent_week_buys:
        avg = item_weekly_avg.get(aid, 1)
        item_sales_trend[aid] = item_recent_week_buys[aid] / max(avg, 0.1)

    # Repurchase ratio
    item_repurchase_ratio = {}
    for aid in item_total_buys:
        buyers = item_unique_buyers.get(aid, 1)
        buys = item_total_buys[aid]
        item_repurchase_ratio[aid] = buyers / max(buys, 1)

    # --- Age-demographic features ---
    log.info("  Age-demographic features...")
    txn_with_age = txn_features.merge(
        customers[["customer_id", "age"]], on="customer_id", how="left"
    )
    item_user_mean_age = txn_with_age.groupby("article_id")["age"].mean().to_dict()
    item_user_age_std = txn_with_age.groupby("article_id")["age"].std().fillna(0).to_dict()
    del txn_with_age
    gc.collect()

    # --- Price features ---
    log.info("  Price features...")
    user_mean_price = txn_features.groupby("customer_id")["price"].mean().to_dict()
    item_mean_price = txn_features.groupby("article_id")["price"].mean().to_dict()

    # --- Time features ---
    log.info("  Time features...")
    item_first_day = txn_features.groupby("article_id")["days_ago"].max().to_dict()
    item_last_day = txn_features.groupby("article_id")["days_ago"].min().to_dict()
    item_freshness = {aid: max_date.day - item_first_day.get(aid, 0) for aid in item_first_day}

    # --- Channel features ---
    user_mean_channel = txn_features.groupby("customer_id")["sales_channel_id"].mean().to_dict()
    item_mean_channel = txn_features.groupby("article_id")["sales_channel_id"].mean().to_dict()

    # --- Recall channels ---
    log.info("  Recall channels...")

    # Time-decay popular
    txn_features["decay"] = np.exp(-0.15 * txn_features["week"].astype(float))
    pop_scores = txn_features.groupby("article_id")["decay"].sum().sort_values(ascending=False)
    pop_top50 = list(pop_scores.head(50).index)
    pop_score_dict = pop_scores.to_dict()

    # Repurchase (6-week, recency-ordered)
    last_6w = txn_features[txn_features["week"] <= 7]
    cust_repurchase = (
        last_6w
        .sort_values("t_dat", ascending=False)
        .groupby("customer_id")["article_id"]
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
        .to_dict()
    )
    log.info(f"    Users with repurchase: {len(cust_repurchase):,}")

    # ItemCF
    cust_sequences = (
        txn_features[txn_features["week"] <= 11]
        .sort_values("t_dat")
        .groupby("customer_id")["article_id"]
        .apply(list)
        .to_dict()
    )

    sim_item = defaultdict(lambda: defaultdict(float))
    for cid, items in cust_sequences.items():
        n_items = len(items)
        if n_items < 2:
            continue
        log_len = math.log(1 + n_items)
        for i, item_i in enumerate(items):
            for j, item_j in enumerate(items):
                if i == j:
                    continue
                loc_alpha = 1.0 if j > i else 0.9
                loc_weight = loc_alpha * (0.7 ** (abs(j - i) - 1))
                sim_item[item_i][item_j] += loc_weight / log_len

    itemcf_lookup = {}
    for item, related in sim_item.items():
        sorted_related = sorted(related.items(), key=lambda x: -x[1])[:10]
        itemcf_lookup[item] = sorted_related
    log.info(f"    Articles with ItemCF: {len(itemcf_lookup):,}")
    del cust_sequences, sim_item
    gc.collect()

    # Co-occurrence
    buckets = txn_features[txn_features["week"] <= 11].groupby(
        ["t_dat", "customer_id", "sales_channel_id"]
    )["article_id"].apply(set).reset_index()
    buckets.columns = ["t_dat", "customer_id", "sales_channel_id", "article_set"]
    buckets = buckets[buckets["article_set"].apply(len) > 1]

    pair_counts = Counter()
    for arts in buckets["article_set"]:
        if len(arts) <= 10:
            for pair in combinations(arts, 2):
                pair_counts[pair] += 1

    freq_pairs = defaultdict(list)
    for (a, b), count in pair_counts.items():
        freq_pairs[a].append((b, count))
        freq_pairs[b].append((a, count))
    freq_pairs = {
        k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:5]]
        for k, v in freq_pairs.items()
    }
    log.info(f"    Articles with pairs: {len(freq_pairs):,}")
    del buckets, pair_counts
    gc.collect()

    # User activity stats
    user_total_buys = txn_features.groupby("customer_id").size().to_dict()
    user_unique_items = txn_features.groupby("customer_id")["article_id"].nunique().to_dict()
    user_last_purchase_day = txn_features.groupby("customer_id")["days_ago"].min().to_dict()

    # User-item interaction (use txn_features to avoid leaking week-0 target)
    user_item_buy_count = txn_features.groupby(["customer_id", "article_id"]).size().to_dict()
    user_item_last_day = txn_features.groupby(["customer_id", "article_id"])["days_ago"].min().to_dict()

    # Per-customer recall lookups
    cust_recent_items = (
        txn_features[txn_features["week"] <= 5]
        .sort_values("t_dat", ascending=False)
        .groupby("customer_id")["article_id"]
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
        .to_dict()
    )

    itemcf_user_recs = {}
    for cid, recent in cust_recent_items.items():
        cf_scores = defaultdict(float)
        for aid in recent[:5]:
            if aid in itemcf_lookup:
                for related_aid, score in itemcf_lookup[aid]:
                    cf_scores[related_aid] += score
        itemcf_user_recs[cid] = sorted(cf_scores.items(), key=lambda x: -x[1])

    cooc_user_recs = {}
    for cid, recent in cust_recent_items.items():
        cooc_items = []
        seen = set()
        for aid in recent[:5]:
            if aid in freq_pairs:
                for related in freq_pairs[aid]:
                    if related not in seen:
                        cooc_items.append(related)
                        seen.add(related)
        cooc_user_recs[cid] = cooc_items

    # Ground truth
    val_truth = txn[txn["week"] == 0].groupby("customer_id")["article_id"].apply(set).to_dict()
    log.info(f"  Val customers with purchases: {len(val_truth):,}")

    del txn_features
    gc.collect()

    # --- Build training data ---
    log.info("  Generating training data...")

    all_active = set(cust_repurchase.keys()) | set(cust_recent_items.keys())
    log.info(f"  Active users: {len(all_active):,}")

    lookup = {
        "cust_age": cust_age,
        "user_mean_price": user_mean_price,
        "user_mean_channel": user_mean_channel,
        "item_user_mean_age": item_user_mean_age,
        "item_mean_price": item_mean_price,
        "pop_score_dict": pop_score_dict,
        "item_total_buys": item_total_buys,
        "item_unique_buyers": item_unique_buyers,
        "art_dept": art_dept,
        "art_section": art_section,
        "art_garment": art_garment,
        "art_category": art_category,
        "art_color": art_color,
        "item_recent_week_buys": item_recent_week_buys,
        "item_last2w_buys": item_last2w_buys,
        "item_last4w_buys": item_last4w_buys,
        "item_sales_trend": item_sales_trend,
        "item_repurchase_ratio": item_repurchase_ratio,
        "item_user_age_std": item_user_age_std,
        "item_freshness": item_freshness,
        "item_last_day": item_last_day,
        "item_mean_channel": item_mean_channel,
        "user_item_buy_count": user_item_buy_count,
        "user_item_last_day": user_item_last_day,
        "user_total_buys": user_total_buys,
        "user_unique_items": user_unique_items,
        "user_last_purchase_day": user_last_purchase_day,
    }

    train_rows = []
    skipped = 0

    for cid in all_active:
        if cid not in val_truth:
            skipped += 1
            continue

        actual = val_truth[cid]
        candidates = {}

        # Repurchase
        for rank, aid in enumerate(cust_repurchase.get(cid, [])[:16]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["repurchase_rank"] = rank
            candidates[aid]["repurchase_score"] = 1.0 / (1.0 + rank)

        # ItemCF
        for rank, (aid, score) in enumerate(itemcf_user_recs.get(cid, [])[:20]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["itemcf_rank"] = rank
            candidates[aid]["itemcf_score"] = score

        # Co-occurrence
        for rank, aid in enumerate(cooc_user_recs.get(cid, [])[:10]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["cooc_rank"] = rank

        # Popular
        for rank, aid in enumerate(pop_top50[:30]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["pop_rank"] = rank

        # Build rows
        lookup["actual"] = actual
        for aid, feat in candidates.items():
            row = build_row(cid, aid, feat, lookup)
            train_rows.append(row)

    log.info(f"  Skipped (no val truth): {skipped:,}")
    log.info(f"  Training rows: {len(train_rows):,}")

    train_df = pd.DataFrame(train_rows)
    del train_rows
    gc.collect()

    # Fill NaN
    fill_nan(train_df)

    # Verify no NaN remain in feature columns
    nan_remaining = train_df[FEATURE_COLS].isna().sum().sum()
    assert nan_remaining == 0, f"NaN values remain after filling: {train_df[FEATURE_COLS].isna().sum().to_dict()}"

    # Validate features
    issues = validate_features(train_df, FEATURE_COLS, stage="train_features")
    if issues:
        log.warn(f"Feature validation issues: {issues}")

    # Log data stats
    pos_count = train_df["target"].sum()
    neg_count = (train_df["target"] == 0).sum()
    pos_rate = pos_count / len(train_df) * 100
    log.info(f"  Positive: {pos_count:,}, Negative: {neg_count:,}")
    log.info(f"  Positive rate: {pos_rate:.2f}%")

    # ==================================================================
    # Stage 3: Model Training
    # ==================================================================
    log.section("Stage 3: Model Training")

    import lightgbm as lgb

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]

    lgb_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": neg_count / max(pos_count, 1),
        "min_child_samples": 50,
        "deterministic": True,
        "verbose": -1,
        "n_jobs": -1,
        "seed": RANDOM_STATE,
    }

    log.info(f"  LightGBM params: {json.dumps(lgb_params, indent=2)}")
    log.info(f"  Boost rounds: 500 (no early stopping)")

    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(100)],
    )

    # Feature importance
    log.info("  Feature importance (top 20):")
    importance = sorted(
        zip(FEATURE_COLS, model.feature_importance()),
        key=lambda x: -x[1],
    )
    for fname, fimp in importance[:20]:
        log.info(f"    {fname}: {fimp}")

    # Save model binary
    model_path = dirs.models / "r27_gbdt_framework.lgb"
    model.save_model(str(model_path))
    log.info(f"  Model saved: {model_path}")

    del train_df, dtrain
    gc.collect()

    # ==================================================================
    # Val MAP@12 Evaluation
    # ==================================================================
    log.section("Val MAP@12 Evaluation")

    # Active-only MAP@12: only customers with week-0 purchases
    val_scores_active = []
    pop12_str = " ".join(f"{a:010d}" for a in pop_top50[:12])
    for cid in val_truth:
        # Reproduce predictions for validation customers
        candidates = {}
        for rank, aid in enumerate(cust_repurchase.get(cid, [])[:16]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["repurchase_rank"] = rank
            candidates[aid]["repurchase_score"] = 1.0 / (1.0 + rank)

        for rank, (aid, score) in enumerate(itemcf_user_recs.get(cid, [])[:20]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["itemcf_rank"] = rank
            candidates[aid]["itemcf_score"] = score

        for rank, aid in enumerate(cooc_user_recs.get(cid, [])[:10]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["cooc_rank"] = rank

        for rank, aid in enumerate(pop_top50[:30]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["pop_rank"] = rank

        if not candidates:
            pred = [int(x) for x in pop12_str.split()]
        else:
            rows = []
            aid_list = []
            lookup["actual"] = set()  # No target for prediction
            for aid, feat in candidates.items():
                row = build_row(cid, aid, feat, lookup)
                rows.append(row)
                aid_list.append(aid)

            X_pred = pd.DataFrame(rows)[FEATURE_COLS]
            fill_nan(X_pred)

            scores = model.predict(X_pred)
            sorted_idx = np.argsort(-scores)
            pred = [aid_list[i] for i in sorted_idx[:12]]
            used = set(pred)
            for aid in pop_top50:
                if len(pred) >= 12:
                    break
                if aid not in used:
                    pred.append(aid)

        val_scores_active.append(apk(val_truth[cid], pred, 12))

    map12_active = np.mean(val_scores_active)
    log.metric("MAP@12 (active)", map12_active)

    # Full-population MAP@12
    # Build lookup: cid -> active score for reuse
    val_truth_keys = list(val_truth.keys())
    cid_to_active_score = dict(zip(val_truth_keys, val_scores_active))

    val_scores_full = []
    all_customers = sample_sub["customer_id"].tolist()
    pop_pred = [int(x) for x in pop12_str.split()]

    for cid in all_customers:
        if cid in cid_to_active_score:
            val_scores_full.append(cid_to_active_score[cid])
        else:
            val_scores_full.append(apk(val_truth.get(cid, set()), pop_pred, 12))

    map12_full = np.mean(val_scores_full)
    log.metric("MAP@12 (full)", map12_full)

    log.info(f"  Active customers in val: {len(val_truth):,}")
    log.info(f"  Total customers: {len(all_customers):,}")

    # ==================================================================
    # Stage 4: Prediction + Submission
    # ==================================================================
    log.section("Stage 4: Prediction + Submission")

    # Rebuild from full data (including week 0)
    log.info("  Rebuilding features from full data...")
    last_6w_full = txn[txn["week"] <= 6]
    cust_repurchase_full = (
        last_6w_full
        .sort_values("t_dat", ascending=False)
        .groupby("customer_id")["article_id"]
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
        .to_dict()
    )

    user_item_buy_count_full = txn.groupby(["customer_id", "article_id"]).size().to_dict()
    user_item_last_day_full = txn.groupby(["customer_id", "article_id"])["days_ago"].min().to_dict()
    user_total_buys_full = txn.groupby("customer_id").size().to_dict()
    user_unique_items_full = txn.groupby("customer_id")["article_id"].nunique().to_dict()
    user_last_day_full = txn.groupby("customer_id")["days_ago"].min().to_dict()
    user_mean_price_full = txn.groupby("customer_id")["price"].mean().to_dict()
    item_mean_price_full = txn.groupby("article_id")["price"].mean().to_dict()

    # Pop scores from full data
    txn["decay"] = np.exp(-0.15 * txn["week"].astype(float))
    pop_scores_full = txn.groupby("article_id")["decay"].sum().sort_values(ascending=False)
    pop_top50_full = list(pop_scores_full.head(50).index)
    pop_score_dict_full = pop_scores_full.to_dict()
    popular_top12 = pop_top50_full[:12]

    # Recent week buys from full data
    item_recent_week_full = txn[txn["week"] == 0]["article_id"].value_counts().to_dict()

    # ItemCF recs from full data
    cust_recent_full = (
        txn[txn["week"] <= 4]
        .sort_values("t_dat", ascending=False)
        .groupby("customer_id")["article_id"]
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
        .to_dict()
    )

    itemcf_user_recs_full = {}
    for cid, recent in cust_recent_full.items():
        cf_scores = defaultdict(float)
        for aid in recent[:5]:
            if aid in itemcf_lookup:
                for related_aid, score in itemcf_lookup[aid]:
                    cf_scores[related_aid] += score
        itemcf_user_recs_full[cid] = sorted(cf_scores.items(), key=lambda x: -x[1])

    cooc_user_recs_full = {}
    for cid, recent in cust_recent_full.items():
        cooc_items = []
        seen = set()
        for aid in recent[:5]:
            if aid in freq_pairs:
                for related in freq_pairs[aid]:
                    if related not in seen:
                        cooc_items.append(related)
                        seen.add(related)
        cooc_user_recs_full[cid] = cooc_items

    # Channel from full data
    user_mean_channel_full = txn.groupby("customer_id")["sales_channel_id"].mean().to_dict()
    item_mean_channel_full = txn.groupby("article_id")["sales_channel_id"].mean().to_dict()

    del last_6w_full, cust_recent_full
    gc.collect()

    # Full-data lookup
    lookup_full = {
        "cust_age": cust_age,
        "user_mean_price": user_mean_price_full,
        "user_mean_channel": user_mean_channel_full,
        "item_user_mean_age": item_user_mean_age,
        "item_mean_price": item_mean_price_full,
        "pop_score_dict": pop_score_dict_full,
        "item_total_buys": item_total_buys,
        "item_unique_buyers": item_unique_buyers,
        "art_dept": art_dept,
        "art_section": art_section,
        "art_garment": art_garment,
        "art_category": art_category,
        "art_color": art_color,
        "item_recent_week_buys": item_recent_week_full,
        "item_last2w_buys": item_last2w_buys,
        "item_last4w_buys": item_last4w_buys,
        "item_sales_trend": item_sales_trend,
        "item_repurchase_ratio": item_repurchase_ratio,
        "item_user_age_std": item_user_age_std,
        "item_freshness": item_freshness,
        "item_last_day": item_last_day,
        "item_mean_channel": item_mean_channel_full,
        "user_item_buy_count": user_item_buy_count_full,
        "user_item_last_day": user_item_last_day_full,
        "user_total_buys": user_total_buys_full,
        "user_unique_items": user_unique_items_full,
        "user_last_purchase_day": user_last_day_full,
    }

    # Score all candidates
    log.info("  Scoring candidates...")
    predictions = {}
    gbdt_used = 0
    batch_count = 0

    for cid in sample_sub["customer_id"]:
        if cid not in cust_repurchase_full:
            predictions[cid] = " ".join(f"{a:010d}" for a in popular_top12)
            continue

        candidates = {}

        for rank, aid in enumerate(cust_repurchase_full.get(cid, [])[:16]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["repurchase_rank"] = rank
            candidates[aid]["repurchase_score"] = 1.0 / (1.0 + rank)

        for rank, (aid, score) in enumerate(itemcf_user_recs_full.get(cid, [])[:20]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["itemcf_rank"] = rank
            candidates[aid]["itemcf_score"] = score

        for rank, aid in enumerate(cooc_user_recs_full.get(cid, [])[:10]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["cooc_rank"] = rank

        for rank, aid in enumerate(pop_top50_full[:30]):
            if aid not in candidates:
                candidates[aid] = {}
            candidates[aid]["pop_rank"] = rank

        if not candidates:
            predictions[cid] = " ".join(f"{a:010d}" for a in popular_top12)
            continue

        lookup_full["actual"] = set()
        rows = []
        aid_list = []
        for aid, feat in candidates.items():
            row = build_row(cid, aid, feat, lookup_full)
            rows.append(row)
            aid_list.append(aid)

        X_pred = pd.DataFrame(rows)[FEATURE_COLS]
        fill_nan(X_pred)

        scores = model.predict(X_pred)
        sorted_idx = np.argsort(-scores)
        pred = [aid_list[i] for i in sorted_idx[:12]]
        gbdt_used += 12

        # Fill remaining with popular
        used = set(pred)
        for aid in pop_top50_full:
            if len(pred) >= 12:
                break
            if aid not in used:
                pred.append(aid)

        predictions[cid] = " ".join(f"{a:010d}" for a in pred[:12])

        batch_count += 1
        if batch_count % 100000 == 0:
            log.info(f"    Scored {batch_count:,} customers...")

    log.info(f"  GBDT-scored predictions: {gbdt_used:,}")

    # Build submission
    sub = sample_sub.copy()
    sub["prediction"] = sub["customer_id"].map(predictions)

    # Validate submission format
    submission_path = get_submission_filename(RUN_NAME, dirs.submissions)

    # Extra validation: each prediction must have exactly 12 ten-digit article IDs
    log.info("  Validating prediction format...")
    format_errors = 0
    for idx, pred_str in enumerate(sub["prediction"]):
        if pd.isna(pred_str):
            format_errors += 1
            continue
        parts = pred_str.split()
        if len(parts) != 12:
            format_errors += 1
            continue
        for p in parts:
            if len(p) != 10:
                format_errors += 1
                break
    if format_errors > 0:
        log.warn(f"  Format errors: {format_errors} rows")
    else:
        log.info("  All predictions have correct format (12 x 10-digit IDs)")

    # Fill any NaN predictions with popular_top12
    if sub["prediction"].isna().any():
        n_missing = sub["prediction"].isna().sum()
        log.warn(f"  Filling {n_missing} missing predictions with popular_top12")
        pop12_fallback = " ".join(f"{a:010d}" for a in popular_top12)
        sub["prediction"] = sub["prediction"].fillna(pop12_fallback)

    validate_and_save(sub, sample_sub, submission_path, check_nan=True)
    log.info(f"  Submission saved: {submission_path}")

    active_count = sum(
        1 for cid in sample_sub["customer_id"] if cid in cust_repurchase_full
    )
    log.info(f"  Active: {active_count:,}, Fallback: {len(sample_sub) - active_count:,}")

    # Save OOF predictions (top-12 per customer for efficiency)
    log.info("  Saving OOF predictions...")
    oof_path = dirs.oof / "oof_r27_gbdt_framework.csv"
    oof_rows = []
    for cid in val_truth:
        pred_str = predictions.get(cid, pop12_str)
        oof_rows.append({
            "customer_id": cid,
            "prediction": pred_str,
            "n_actual": len(val_truth[cid]),
        })
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(oof_path, index=False)
    log.info(f"  OOF saved: {oof_path} ({len(oof_df):,} rows)")

    # ==================================================================
    # MLflow Tracking
    # ==================================================================
    if not args.no_mlflow:
        log.section("MLflow Tracking")

        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name or cfg.slug or "hm_recommendations")

        with mlflow.start_run(run_name=RUN_NAME) as run:
            run_id = run.info.run_id
            log.info(f"  MLflow run_id: {run_id}")

            # Log parameters
            mlflow.log_params({
                "model": "lightgbm",
                "n_features": len(FEATURE_COLS),
                "boost_rounds": 500,
                "positive_rate": pos_rate,
                "scale_pos_weight": neg_count / max(pos_count, 1),
                "train_rows": len(X_train),
                "n_active_customers": len(all_active),
                "n_val_customers": len(val_truth),
                "smoke_test": args.smoke_test,
            })
            mlflow.log_params({
                f"lgb_{k}": v for k, v in lgb_params.items()
            })

            # Log features
            mlflow.log_text("\n".join(FEATURE_COLS), "features.txt")

            # Log metrics
            mlflow.log_metrics({
                "map12_active": map12_active,
                "map12_full": map12_full,
            })

            # Log feature importance
            importance_dict = [
                {"feature": fname, "importance": float(fimp)}
                for fname, fimp in importance
            ]
            mlflow.log_text(
                json.dumps(importance_dict, indent=2),
                "feature_importance.json",
            )

            # Log artifacts
            mlflow.log_artifact(str(submission_path), artifact_path="submissions")
            mlflow.log_artifact(str(oof_path), artifact_path="oof")
            mlflow.log_artifact(str(model_path), artifact_path="models")

            mlflow.set_tag("notes", "R27: Framework-validated GBDT ranking pipeline")
            mlflow.set_tag("top_feature", importance[0][0])

        log.info(f"  MLflow run_id: {run_id}")
    else:
        log.info("  MLflow disabled (--no-mlflow)")

    # ==================================================================
    # Evaluation Gate
    # ==================================================================
    log.section("Evaluation Gate")

    try:
        evaluation_gate(
            cv_score=map12_active,
            cv_std=0.0,  # Single fold, no std
            baseline_score=0.040,
            metric_direction="maximize",
            min_improvement=-0.005,  # Allow 5% tolerance below baseline
        )
        log.info("  Evaluation gate PASSED")
    except Exception as e:
        log.warn(f"  Evaluation gate FAILED: {e}")
        failure_info = classify_failure(str(e), cv_score=map12_active)
        log.warn(f"  Failure category: {failure_info['category']} ({failure_info['severity']})")

    # ==================================================================
    # Summary
    # ==================================================================
    log.separator(f"{RUN_NAME} Complete")
    log.metric("MAP@12 (active)", map12_active)
    log.metric("MAP@12 (full)", map12_full)
    log.info(f"  Features: {len(FEATURE_COLS)}")
    log.info(f"  Top feature: {importance[0][0]} ({importance[0][1]})")
    log.info(f"  R21 reference: active ~0.05, full ~0.024")
    log.separator()


if __name__ == "__main__":
    main()
