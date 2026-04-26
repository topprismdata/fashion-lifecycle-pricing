#!/usr/bin/env python3
"""
R26: Replicate 45th Place Solution (Wp-Zhang) -- LGB Recall 2 Pipeline
=======================================================================

Replicates the CF-based recall pipeline from the 45th place H&M solution.
Reference: https://github.com/Wp-Zhang/H-M-Fashion-RecSys
Target score: 0.02996 PB (LGB Recall 2 alone scores 0.03047).

Pipeline (faithfully replicating the notebook "LGB Recall 2"):
  1. Data loading & preprocessing (DataHelper)
  2. User grouping (purchase_ability, age_bins)
  3. Recall candidates per week (17 rules: ALS, BPR, ItemCF x3,
     UserGroupItemCF x6, UserGroupTimeHistory x3, OrderHistory,
     OrderHistoryDecay, TimeHistory)
  4. Pivot + score fusion + rank filter to top-200
  5. Feature engineering on all transactions (processed_inter)
  6. Merge features with candidates per week
  7. Extra time features + week-shift sales
  8. LightGBM binary training (weeks 2-5 train, week 1 valid)
  9. Validate MAP@12 on week 1
  10. Generate test predictions (week 0) in batches

Usage:
    python scripts/run_r26_replicate_45th.py
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import logging
import pickle
import gc
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REF_DIR = Path(
    "/Users/guohongbin/projects/fashion-lifecycle-pricing/"
    "competitions/02_hm_recommendations/reference_solution_45th"
)
DATA_DIR = REF_DIR / "data"
MODEL_DIR = REF_DIR / "models"
SCRIPT_DIR = Path(
    "/Users/guohongbin/projects/fashion-lifecycle-pricing/"
    "competitions/02_hm_recommendations/scripts"
)
SUBMISSION_DIR = SCRIPT_DIR.parent / "outputs" / "submissions"
VERSION_NAME = "recall2"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_WEEK_NUM = 4
WEEK_NUM = TRAIN_WEEK_NUM + 2  # 6
TEST = True  # Set True for final submission with test predictions
TEST_BATCH_SIZE = 70000
TEST_BATCH_NUM = 10

# ALS/BPR are extremely slow on CPU (~1.5h per rule for 1.37M users).
# Set True to include them (needed for full replication score).
INCLUDE_ALS_BPR = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("r26")


def log_section(title: str) -> None:
    log.info("=" * 70)
    log.info(f"  {title}")
    log.info("=" * 70)


def log_elapsed(start: float, label: str = "") -> None:
    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    msg = f"  Elapsed: {int(m)}m {s:.1f}s"
    if label:
        msg = f"  [{label}] {msg}"
    log.info(msg)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    log_section("R26: Replicate 45th Place -- LGB Recall 2 Pipeline")
    log.info(f"  VERSION_NAME   = {VERSION_NAME}")
    log.info(f"  WEEK_NUM       = {WEEK_NUM}")
    log.info(f"  TEST           = {TEST}")
    log.info(f"  INCLUDE_ALS_BPR= {INCLUDE_ALS_BPR}")

    # ==================================================================
    # Add reference solution src to sys.path
    # ==================================================================
    sys.path.insert(0, str(REF_DIR))

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from sklearn.preprocessing import MinMaxScaler
    from pandas.api.types import CategoricalDtype
    from itertools import chain

    tqdm.pandas()

    from src.data import DataHelper
    from src.data.metrics import map_at_k
    from src.retrieval.rules import (
        ALS, BPR, ItemCF, UserGroupItemCF,
        UserGroupTimeHistory, OrderHistory, OrderHistoryDecay, TimeHistory,
    )
    from src.retrieval.collector import RuleCollector
    from src.features.base_features import (
        full_sale, week_sale, period_sale, repurchase_ratio, popularity,
    )
    from src.utils import calc_valid_date, reduce_mem_usage, merge_week_data

    # ==================================================================
    # Step 0: Create output directories
    # ==================================================================
    log_section("Step 0: Creating output directories")
    for d in [
        f"interim/{VERSION_NAME}",
        f"processed/{VERSION_NAME}",
        "index_id_map",
    ]:
        os.makedirs(DATA_DIR / d, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==================================================================
    # Step 1: Data Loading & Preprocessing
    # ==================================================================
    log_section("Step 1: Data Loading & Preprocessing")
    t = time.time()

    dh = DataHelper(str(DATA_DIR))
    data = dh.preprocess_data(save=True, name="encoded_full")

    uid2idx = pickle.load(open(DATA_DIR / "index_id_map" / "user_id2index.pkl", "rb"))
    idx2uid = pickle.load(open(DATA_DIR / "index_id_map" / "user_index2id.pkl", "rb"))
    idx2iid = pickle.load(open(DATA_DIR / "index_id_map" / "item_index2id.pkl", "rb"))

    submission = pd.read_csv(DATA_DIR / "raw" / "sample_submission.csv")
    submission["customer_id"] = submission["customer_id"].map(uid2idx)

    log.info(f"  Users: {data['user'].shape[0]:,}  "
             f"Items: {data['item'].shape[0]:,}  "
             f"Transactions: {data['inter'].shape[0]:,}")
    log_elapsed(t, "Step 1")

    # ==================================================================
    # Step 2: User grouping features
    # ==================================================================
    log_section("Step 2: User Grouping Features")
    t = time.time()

    user_info = (
        data["inter"]
        .groupby(["customer_id"])["price"]
        .mean()
        .reset_index(name="mean_price")
    )
    user_info["purchase_ability"] = pd.qcut(user_info["mean_price"], 5, labels=False)
    del user_info["mean_price"]

    listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
    data["user"]["age_bins"] = pd.cut(data["user"]["age"], listBin)
    data["user"] = data["user"].merge(user_info, on="customer_id", how="left")

    log_elapsed(t, "Step 2")

    # ==================================================================
    # Step 3: Generate Recall Candidates per Week
    # ==================================================================
    log_section("Step 3: Generate Recall Candidates")
    t = time.time()

    for week in range(WEEK_NUM):
        if week == 0 and not TEST:
            continue

        tw = time.time()
        trans = data["inter"]
        start_date, end_date = calc_valid_date(week)
        log.info(f"\n  Week {week}: [{start_date}, {end_date})")

        train, valid = dh.split_data(trans, start_date, end_date)
        train = train.merge(
            data["user"][["customer_id", "age_bins", "user_gender"]],
            on="customer_id", how="left",
        )
        train = train.merge(user_info, on="customer_id", how="left")

        train["t_dat"] = pd.to_datetime(train["t_dat"])
        last_week = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=7)]
        last_2week = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=14)]
        last_60day = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=60)]
        last_80day = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=80)]

        if week != 0:
            customer_list = valid["customer_id"].values
        else:
            customer_list = submission["customer_id"].values

        log.info(f"    Customers: {len(customer_list):,}")

        # Define retrieval rules (same as original notebook)
        rules = []
        if INCLUDE_ALS_BPR:
            rules += [
                ALS(customer_list, last_60day, n=200, iter_num=25),
                BPR(customer_list, last_80day, n=200, iter_num=350),
            ]
        rules += [
            UserGroupTimeHistory(data, customer_list, last_week, ["age_bins"], n=200, scale=True, name="1"),
            UserGroupTimeHistory(data, customer_list, last_week, ["purchase_ability"], n=200, scale=True, name="2"),
            UserGroupTimeHistory(data, customer_list, last_week, ["user_gender"], n=200, scale=True, name="3"),
            OrderHistory(train, days=35, n=200),
            OrderHistoryDecay(train, days=7, n=200),
            TimeHistory(customer_list, last_week, n=200),
            ItemCF(last_80day, last_2week, top_k=10, name="1"),
            ItemCF(last_60day, last_2week, top_k=10, name="2"),
            ItemCF(last_2week, last_2week, top_k=10, name="3"),
            UserGroupItemCF(last_80day, last_2week, "age_bins", top_k=10, name="1"),
            UserGroupItemCF(last_60day, last_2week, "age_bins", top_k=10, name="2"),
            UserGroupItemCF(last_2week, last_2week, "age_bins", top_k=10, name="3"),
            UserGroupItemCF(last_80day, last_2week, "purchase_ability", top_k=10, name="4"),
            UserGroupItemCF(last_60day, last_2week, "purchase_ability", top_k=10, name="5"),
            UserGroupItemCF(last_2week, last_2week, "purchase_ability", top_k=10, name="6"),
        ]

        # Collect candidates
        # For validation weeks (1-5): small customer list, use compress=False + pivot
        # For test week (0): 1.3M customers, use compress=True to avoid OOM
        if week == 0 and len(customer_list) > 100000:
            log.info("    Large customer set detected, using compress=True...")
            compressed = RuleCollector().collect(
                week_num=week,
                trans_df=trans,
                customer_list=customer_list,
                rules=rules,
                min_pos_rate=0.0,
                norm=True,
                compress=True,
            )
            # Flatten: each row has customer_id -> list of article_ids
            candidates = compressed.explode("article_id").dropna(subset=["article_id"])
            candidates = candidates.drop_duplicates(["customer_id", "article_id"])
            candidates["score"] = 0.0
            candidates["rank"] = 1
            log.info(f"    Test candidates (unique pairs): {candidates.shape[0]:,}")
        else:
            candidates = RuleCollector().collect(
                week_num=week,
                trans_df=trans,
                customer_list=customer_list,
                rules=rules,
                min_pos_rate=0.0,
                norm=False,
                compress=False,
            )
            log.info(f"    Raw candidates: {candidates.shape[0]:,}")

            candidates, _ = reduce_mem_usage(candidates)

            # Normalize scores per method, then sum (replaces pivot_table which OOMs)
            # Group by method, normalize, then aggregate by (customer, article)
            log.info("    Normalizing and aggregating scores...")
            methods = candidates["method"].unique()
            for method in methods:
                mask = candidates["method"] == method
                scores = candidates.loc[mask, "score"].values.reshape(-1, 1)
                candidates.loc[mask, "score"] = MinMaxScaler().fit_transform(scores).ravel()
                del scores

            # Sum scores per (customer_id, article_id)
            candidates = candidates.groupby(
                ["customer_id", "article_id"], as_index=False
            )["score"].sum()

            candidates["rank"] = candidates.groupby(["customer_id"])["score"].rank(ascending=False)

        candidates = candidates[candidates["rank"] <= 200]

        log.info(f"    After pivot+rank200: {candidates.shape[0]:,}")

        # Save candidates
        if week == 0 and TEST:
            # Split test candidates into batches
            unique_customers = candidates["customer_id"].unique()
            log.info(f"    Test: splitting {len(unique_customers):,} customers into batches")
            for batch_idx in range(TEST_BATCH_NUM):
                start_idx = batch_idx * TEST_BATCH_SIZE
                end_idx = min((batch_idx + 1) * TEST_BATCH_SIZE, len(unique_customers))
                if start_idx >= len(unique_customers):
                    break
                batch_customers = unique_customers[start_idx:end_idx]
                batch_cands = candidates[candidates["customer_id"].isin(batch_customers)]
                batch_cands.to_parquet(
                    DATA_DIR / "interim" / VERSION_NAME / f"week{week}_candidate_{batch_idx}.pqt"
                )
        else:
            candidates.to_parquet(
                DATA_DIR / "interim" / VERSION_NAME / f"week{week}_candidate.pqt"
            )

        # Save validation labels
        if week != 0:
            valid.to_parquet(
                DATA_DIR / "processed" / VERSION_NAME / f"week{week}_label.pqt"
            )

        del train, valid, last_week, last_2week, last_60day, last_80day
        del candidates, customer_list, rules
        gc.collect()
        log_elapsed(tw, f"Week {week}")

    log_elapsed(t, "Step 3 (all weeks)")

    # ==================================================================
    # Step 4: Feature Engineering (processed_inter)
    # ==================================================================
    # This builds features on the full transaction history merged with
    # recall candidates, then saves as processed_inter.pqt.
    log_section("Step 4: Feature Engineering (processed_inter)")
    t = time.time()

    inter = data["inter"]
    inter["week"] = (pd.to_datetime("2020-09-29") - pd.to_datetime(inter["t_dat"])).dt.days // 7

    # Merge product_code + item features needed for feature engineering
    item_merge_cols = [
        "article_id", "product_code", "product_type_no", "product_group_name",
        "graphical_appearance_no", "colour_group_code",
        "perceived_colour_value_id", "perceived_colour_master_id",
    ]
    inter = inter.merge(
        data["item"][[c for c in item_merge_cols if c in data["item"].columns]],
        on="article_id", how="left",
    )

    # Verify item features are present
    log.info(f"  inter columns after item merge: {inter.columns.tolist()}")

    # Merge full candidates to transaction data
    log.info("  Merging full candidates to transaction data...")
    full_candidates = []
    for i in tqdm(range(WEEK_NUM)):
        fpath = DATA_DIR / "interim" / VERSION_NAME / f"week{i}_candidate.pqt"
        if os.path.exists(fpath):
            candidate = pd.read_parquet(fpath)
            full_candidates += candidate["article_id"].values.tolist()
    full_candidates = list(set(full_candidates))
    log.info(f"  Unique candidate articles: {len(full_candidates):,}")

    num_candidates = len(full_candidates)
    full_candidates = np.array(full_candidates)
    full_candidates = np.tile(full_candidates, WEEK_NUM + 1)
    weeks = np.repeat(np.arange(1, WEEK_NUM + 2), num_candidates)
    full_candidates = pd.DataFrame({"article_id": full_candidates, "week": weeks})

    inter["valid"] = 1
    in_train = inter[inter["week"] <= WEEK_NUM + 1]
    out_train = inter[inter["week"] > WEEK_NUM + 1]
    in_train = in_train.merge(full_candidates, on=["article_id", "week"], how="right")
    in_train["valid"] = in_train["valid"].fillna(0)
    # Fill NaN values for candidate-only rows (no customer_id, price, etc.)
    fillna_cols = ["customer_id", "price", "sales_channel_id"]
    for c in fillna_cols:
        if c in in_train.columns:
            if c == "customer_id":
                in_train[c] = in_train[c].fillna(0)
            elif c == "sales_channel_id":
                in_train[c] = in_train[c].fillna(0)
            else:
                in_train[c] = in_train[c].fillna(0)
    inter = pd.concat([in_train, out_train], ignore_index=True)
    inter = inter.sort_values(["valid"], ascending=False).reset_index(drop=True)
    del in_train, out_train, full_candidates
    gc.collect()
    log.info(f"  After candidate merge: {inter.shape[0]:,} rows")
    log.info(f"  inter columns after candidate merge: {inter.columns.tolist()}")

    # Ensure no NaN in numeric columns that will be used by base_features
    for col in inter.select_dtypes(include=[np.number]).columns:
        inter[col] = inter[col].fillna(0)
    inter["customer_id"] = inter["customer_id"].fillna(0).astype(np.int32)
    inter["article_id"] = inter["article_id"].fillna(0).astype(np.int32)

    # Period sales (14/21/28 days)
    log.info("  Computing period sales...")
    _, inter["i_1w_sale_rank"], inter["i_1w_sale_norm"] = period_sale(
        inter, ["article_id"], days=14, rank=True, norm=True, week_num=WEEK_NUM)
    _, inter["p_1w_sale_rank"], inter["p_1w_sale_norm"] = period_sale(
        inter, ["product_code"], days=14, rank=True, norm=True, week_num=WEEK_NUM)
    inter["i_2w_sale"], inter["i_2w_sale_rank"], inter["i_2w_sale_norm"] = period_sale(
        inter, ["article_id"], days=14, rank=True, norm=True, week_num=WEEK_NUM)
    inter["p_2w_sale"], inter["p_2w_sale_rank"], inter["p_2w_sale_norm"] = period_sale(
        inter, ["product_code"], days=14, rank=True, norm=True, week_num=WEEK_NUM)
    inter["i_3w_sale"], inter["i_3w_sale_rank"], inter["i_3w_sale_norm"] = period_sale(
        inter, ["article_id"], days=21, rank=True, norm=True, week_num=WEEK_NUM)
    inter["p_3w_sale"], inter["p_3w_sale_rank"], inter["p_3w_sale_norm"] = period_sale(
        inter, ["product_code"], days=21, rank=True, norm=True, week_num=WEEK_NUM)
    inter["i_4w_sale"], inter["i_4w_sale_rank"], inter["i_4w_sale_norm"] = period_sale(
        inter, ["article_id"], days=28, rank=True, norm=True, week_num=WEEK_NUM)
    inter["p_4w_sale"], inter["p_4w_sale_rank"], inter["p_4w_sale_norm"] = period_sale(
        inter, ["product_code"], days=28, rank=True, norm=True, week_num=WEEK_NUM)

    # Repurchase ratio
    inter["i_repurchase_ratio"] = repurchase_ratio(inter, ["article_id"], week_num=WEEK_NUM)
    inter["p_repurchase_ratio"] = repurchase_ratio(inter, ["product_code"], week_num=WEEK_NUM)

    # Replace inf with NaN, then fill all NaN with 0
    inter = inter.replace([np.inf, -np.inf], np.nan)
    inter = inter.fillna(0)
    inter, _ = reduce_mem_usage(inter)

    # Week sales
    log.info("  Computing week sales...")
    inter["i_sale"] = week_sale(inter, ["article_id"], week_num=WEEK_NUM)
    inter["p_sale"] = week_sale(inter, ["product_code"], week_num=WEEK_NUM)
    inter["i_sale_uni"] = week_sale(inter, ["article_id"], True, week_num=WEEK_NUM)
    inter["p_sale_uni"] = week_sale(inter, ["product_code"], True, week_num=WEEK_NUM)
    inter["lw_i_sale"] = week_sale(inter, ["article_id"], step=1, week_num=WEEK_NUM)
    inter["lw_p_sale"] = week_sale(inter, ["product_code"], step=1, week_num=WEEK_NUM)
    inter["lw_i_sale_uni"] = week_sale(inter, ["article_id"], True, step=1, week_num=WEEK_NUM)
    inter["lw_p_sale_uni"] = week_sale(inter, ["product_code"], True, step=1, week_num=WEEK_NUM)

    inter["i_sale_ratio"] = inter["i_sale"] / (inter["p_sale"] + 1e-6)
    inter["i_sale_uni_ratio"] = inter["i_sale_uni"] / (inter["p_sale_uni"] + 1e-6)
    inter["lw_i_sale_ratio"] = inter["lw_i_sale"] / (inter["lw_p_sale"] + 1e-6)
    inter["lw_i_sale_uni_ratio"] = inter["lw_i_sale_uni"] / (inter["lw_p_sale_uni"] + 1e-6)
    inter["i_uni_ratio"] = inter["i_sale"] / (inter["i_sale_uni"] + 1e-6)
    inter["p_uni_ratio"] = inter["p_sale"] / (inter["p_sale_uni"] + 1e-6)
    inter["lw_i_uni_ratio"] = inter["lw_i_sale"] / (inter["lw_i_sale_uni"] + 1e-6)
    inter["lw_p_uni_ratio"] = inter["lw_p_sale"] / (inter["lw_p_sale_uni"] + 1e-6)
    inter["i_sale_trend"] = (inter["i_sale"] - inter["lw_i_sale"]) / (inter["lw_i_sale"] + 1e-6)
    inter["p_sale_trend"] = (inter["p_sale"] - inter["lw_p_sale"]) / (inter["lw_p_sale"] + 1e-6)

    # Date features
    log.info("  Computing date features...")
    curr_date_dict = {x: calc_valid_date(x - 1)[0] for x in range(100)}
    current_dat = inter["week"].map(curr_date_dict)
    mask = inter["valid"] == 0
    inter.loc[mask, "t_dat"] = inter.loc[mask, "week"].map(curr_date_dict)
    first_date = inter.groupby("article_id")["t_dat"].min().reset_index(name="first_dat")
    inter = pd.merge(inter, first_date, on="article_id", how="left")
    inter["first_dat"] = (pd.to_datetime(current_dat) - pd.to_datetime(inter["first_dat"])).dt.days

    # Full sales + daily sales
    log.info("  Computing full sales + daily sales...")
    inter["i_full_sale"] = full_sale(inter, ["article_id"], week_num=WEEK_NUM)
    inter["p_full_sale"] = full_sale(inter, ["product_code"], week_num=WEEK_NUM)
    inter["i_daily_sale"] = inter["i_full_sale"] / inter["first_dat"]
    inter["p_daily_sale"] = inter["p_full_sale"] / inter["first_dat"]
    inter["i_daily_sale_ratio"] = inter["i_daily_sale"] / inter["p_daily_sale"]
    inter["i_w_full_sale_ratio"] = inter["i_sale"] / inter["i_full_sale"]
    inter["i_2w_full_sale_ratio"] = inter["i_2w_sale"] / inter["i_full_sale"]
    inter["p_w_full_sale_ratio"] = inter["p_sale"] / inter["p_full_sale"]
    inter["p_2w_full_sale_ratio"] = inter["p_2w_sale"] / inter["p_full_sale"]
    inter["i_week_above_daily_sale"] = inter["i_sale"] / 7 - inter["i_daily_sale"]
    inter["p_week_above_full_sale"] = inter["p_sale"] / 7 - inter["i_full_sale"]
    inter["i_2w_week_above_daily_sale"] = inter["i_2w_sale"] / 14 - inter["i_daily_sale"]
    inter["p_2w_week_above_daily_sale"] = inter["p_2w_sale"] / 14 - inter["p_daily_sale"]

    # Per-item-feature group daily sales ratios
    log.info("  Computing per-feature-group ratios...")
    item_feats_list = [
        "product_type_no", "product_group_name", "graphical_appearance_no",
        "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id",
    ]
    for f in tqdm(item_feats_list, desc="  Feature groups"):
        inter[f"{f}_full_sale"] = full_sale(inter, [f], week_num=WEEK_NUM)
        f_first_date = inter.groupby(f)["t_dat"].min().reset_index(name=f"{f}_first_dat")
        inter = inter.merge(f_first_date, on=f, how="left")
        inter[f"{f}_daily_sale"] = inter[f"{f}_full_sale"] / (
            pd.to_datetime(current_dat) - pd.to_datetime(inter[f"{f}_first_dat"])
        ).dt.days
        inter[f"i_{f}_daily_sale_ratio"] = inter["i_daily_sale"] / inter[f"{f}_daily_sale"]
        inter[f"p_{f}_daily_sale_ratio"] = inter["p_daily_sale"] / inter[f"{f}_daily_sale"]
        del inter[f"{f}_full_sale"], inter[f"{f}_first_dat"]
        gc.collect()

    for f in item_feats_list + ["i_full_sale", "p_full_sale"]:
        if f in inter.columns:
            del inter[f]

    # Popularity
    inter["i_pop"] = popularity(inter, "article_id", week_num=WEEK_NUM)
    inter["p_pop"] = popularity(inter, "product_code", week_num=WEEK_NUM)

    inter = inter.loc[inter["week"] <= WEEK_NUM + 2]
    inter.to_parquet(DATA_DIR / "processed" / "processed_inter.pqt")
    log.info(f"  processed_inter saved: {inter.shape}")
    del inter
    gc.collect()
    log_elapsed(t, "Step 4")

    # ==================================================================
    # Step 5: Merge features with candidates per week
    # ==================================================================
    log_section("Step 5: Merge features with candidates")
    t = time.time()

    inter = pd.read_parquet(DATA_DIR / "processed" / "processed_inter.pqt")
    for col in inter.columns:
        inter[col] = np.nan_to_num(inter[col])

    for i in range(1, WEEK_NUM):
        candidate = pd.read_parquet(DATA_DIR / "interim" / VERSION_NAME / f"week{i}_candidate.pqt")
        candidate = merge_week_data(data, inter, i, candidate)
        candidate.to_parquet(DATA_DIR / "processed" / VERSION_NAME / f"week{i}_candidate.pqt")
        log.info(f"  Week {i} done, shape: {candidate.shape}")
        gc.collect()

    del inter
    gc.collect()
    log_elapsed(t, "Step 5")

    # ==================================================================
    # Step 6: Load data for training
    # ==================================================================
    log_section("Step 6: Load & prepare training data")
    t = time.time()

    candidates = {}
    labels = {}
    for i in range(1, WEEK_NUM):
        candidates[i] = pd.read_parquet(DATA_DIR / "processed" / VERSION_NAME / f"week{i}_candidate.pqt")
        candidates[i] = candidates[i][candidates[i]["rank"] <= 50]
        labels[i] = pd.read_parquet(DATA_DIR / "processed" / VERSION_NAME / f"week{i}_label.pqt")

    # Feature columns
    feats = [
        x for x in candidates[1].columns
        if x not in ["label", "sales_channel_id", "t_dat", "week", "valid"]
    ]

    cat_features = [
        "customer_id", "article_id", "product_code", "FN", "Active",
        "club_member_status", "fashion_news_frequency", "age",
        "product_type_no", "product_group_name", "graphical_appearance_no",
        "colour_group_code", "perceived_colour_value_id", "perceived_colour_master_id",
        "user_gender", "article_gender", "season_type",
    ]

    # Categorical dtype setup
    cate_dict = {}
    for feat in tqdm(cat_features, desc="  Setting categorical dtypes"):
        if feat in data["user"].columns:
            value_set = set(data["user"][feat].unique())
        elif feat in data["item"].columns:
            value_set = set(data["item"][feat].unique())
        else:
            value_set = set(data["inter"][feat].unique())
        cate_dict[feat] = CategoricalDtype(categories=value_set)

    full_data = pd.concat([candidates[i] for i in range(1, WEEK_NUM)], ignore_index=True)
    del candidates
    gc.collect()

    # ==================================================================
    # Step 7: Extra time features
    # ==================================================================
    log_section("Step 7: Extra time features")
    t2 = time.time()

    inter_t = data["inter"]
    inter_t = inter_t[inter_t["t_dat"] < "2020-08-19"]
    inter_t["week"] = (pd.to_datetime("2020-09-29") - pd.to_datetime(inter_t["t_dat"])).dt.days // 7
    inter_t = inter_t.merge(data["item"][["article_id", "product_code"]], on="article_id", how="left")

    tmp = inter_t.groupby("article_id").week.mean()
    full_data["article_time_mean"] = full_data["article_id"].map(tmp)
    tmp = inter_t.groupby("customer_id").week.nth(-1)
    full_data["customer_id_last_time"] = full_data["customer_id"].map(tmp)
    tmp = inter_t.groupby("customer_id").week.nth(0)
    full_data["customer_id_first_time"] = full_data["customer_id"].map(tmp)
    tmp = inter_t.groupby("customer_id").week.mean()
    full_data["customer_id_time_mean"] = full_data["customer_id"].map(tmp)
    full_data["customer_id_gap"] = full_data["customer_id_first_time"] - full_data["customer_id_last_time"]
    tmp = inter_t.groupby("customer_id").size()
    full_data["customer_daily_bought"] = full_data["customer_id"].map(tmp) / full_data["customer_id_gap"]
    tmp = inter_t.groupby("customer_id").price.median()
    full_data["customer_id_price_median"] = full_data["customer_id"].map(tmp)
    full_data["customer_article_price_gap"] = full_data["customer_id_price_median"] - full_data["price"]

    # Week-shift sales
    def dict_union(L):
        return dict(chain.from_iterable(d.items() for d in L))

    dur = [52]
    for col in tqdm(["article_id", "product_code"], desc="  Week-shift features"):
        full_data[f"{col}_id_week"] = full_data[col].astype("str") + "_" + full_data["week"].astype("str")
        for j in dur:
            dict_list = []
            for i in range(5):
                tmp = inter_t[(inter_t["week"] >= (1 + i + 1)) & (inter_t["week"] < (1 + i + 2 + j))]
                tmp["week"] = i + 1
                tmp = tmp.groupby(["week", col]).size().reset_index()
                tmp.columns = ["week", col, "count_sales"]
                tmp[f"{col}_id_week"] = tmp[col].astype("str") + "_" + tmp["week"].astype("str")
                dict_list.append(dict(zip(tmp[f"{col}_id_week"], tmp["count_sales"])))
                del tmp
            dict_all = dict_union(dict_list)
            full_data[f"{col}_week_shift{j}"] = full_data[f"{col}_id_week"].map(dict_all)
            del dict_all
        gc.collect()

    full_data["article_id_week_1/52"] = full_data["i_sale"] / full_data["article_id_week_shift52"]
    full_data["product_code_week_1/52"] = full_data["p_sale"] / full_data["product_code_week_shift52"]

    extra_feats = [
        "article_time_mean", "customer_id_last_time", "customer_id_first_time",
        "customer_id_time_mean", "customer_id_gap", "customer_id_price_median",
        "customer_daily_bought", "customer_article_price_gap",
    ] + [f"{col}_week_shift{j}" for col in ["article_id", "product_code"] for j in dur] + [
        "article_id_week_1/52", "product_code_week_1/52",
    ]
    feats += extra_feats
    feats = list(dict.fromkeys(feats))  # deduplicate preserving order

    del inter_t
    gc.collect()
    log_elapsed(t2, "Step 7")

    # ==================================================================
    # Step 8: Categorical dtypes, train/valid split
    # ==================================================================
    log_section("Step 8: Train/valid split")
    t2 = time.time()

    for feat in tqdm(cat_features, desc="  Casting categoricals"):
        full_data[feat] = full_data[feat].astype(cate_dict[feat])

    train = full_data.loc[full_data["week"] > 1]
    valid = full_data.loc[full_data["week"] == 1]
    del full_data
    gc.collect()

    # Filter out customers with all-negative samples
    train["customer_id"] = train["customer_id"].astype(int)
    train["week_customer_id"] = train["customer_id"].astype(str) + "_" + train["week"].astype(str)
    valid_uids = train.groupby("week_customer_id")["label"].sum().reset_index(name="sum")
    train = train[train["week_customer_id"].isin(valid_uids.loc[valid_uids["sum"] > 0, "week_customer_id"])]
    del train["week_customer_id"]
    train["customer_id"] = train["customer_id"].astype(cate_dict["customer_id"])
    log.info(f"  Train positive rate: {train.label.mean():.4f}")

    train = train.sort_values(by=["week", "customer_id"], ascending=True).reset_index(drop=True)
    valid = valid.sort_values(by=["customer_id"], ascending=True).reset_index(drop=True)

    train = train[feats + ["label"]]
    valid = valid[feats + ["label"]]

    log.info(f"  Train: {train.shape[0]:,} rows, {train.shape[1]} cols")
    log.info(f"  Valid:  {valid.shape[0]:,} rows, {valid.shape[1]} cols")
    log.info(f"  Features: {len(feats)}")
    log_elapsed(t2, "Step 8")
    log_elapsed(t, "Steps 6-8")

    # ==================================================================
    # Step 9: Train LightGBM Binary Model
    # ==================================================================
    log_section("Step 9: Train LightGBM Binary")
    t = time.time()

    import lightgbm as lgb

    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "max_depth": 8,
        "num_leaves": 128,
        "learning_rate": 0.03,
        "verbose": -1,
        "eval_at": 12,
    }

    train_set = lgb.Dataset(
        data=train[feats], label=train["label"],
        feature_name=feats, categorical_feature=cat_features, params=params,
    )
    valid_set = lgb.Dataset(
        data=valid[feats], label=valid["label"],
        feature_name=feats, categorical_feature=cat_features, params=params,
    )

    ranker = lgb.train(
        params, train_set, num_boost_round=300,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(10)],
    )

    ranker.save_model(
        str(MODEL_DIR / "lgb_recall2_binary.model"),
        num_iteration=ranker.best_iteration,
    )
    log.info(f"  Best iteration: {ranker.best_iteration}")
    log_elapsed(t, "Step 9")

    # ==================================================================
    # Step 10: Feature Importance
    # ==================================================================
    log_section("Step 10: Feature Importance")
    feat_importance = pd.DataFrame({
        "feature": feats,
        "importance": ranker.feature_importance(),
    }).sort_values(by="importance", ascending=False)
    log.info(f"\n{feat_importance.head(20).to_string(index=False)}")

    # ==================================================================
    # Step 11: Validation MAP@12
    # ==================================================================
    log_section("Step 11: Validation MAP@12")
    t = time.time()

    val_candidates = valid.reset_index(drop=True)

    def predict(ranker, candidates, feat_list, batch_size=5_000_000):
        probs = np.zeros(candidates.shape[0])
        for batch in range(0, candidates.shape[0], batch_size):
            outputs = ranker.predict(candidates.loc[batch:batch + batch_size - 1, feat_list])
            probs[batch:batch + batch_size] = outputs
        candidates["prob"] = probs
        pred_lgb = candidates[["customer_id", "article_id", "prob"]]
        pred_lgb = pred_lgb.sort_values(by=["customer_id", "prob"], ascending=False).reset_index(drop=True)
        pred_lgb.rename(columns={"article_id": "prediction"}, inplace=True)
        pred_lgb = pred_lgb.drop_duplicates(["customer_id", "prediction"], keep="first")
        pred_lgb["customer_id"] = pred_lgb["customer_id"].astype(int)
        pred_lgb = pred_lgb.groupby("customer_id")["prediction"].progress_apply(list).reset_index()
        return pred_lgb

    pred = predict(ranker, val_candidates, feats)
    label = labels[1]
    label = pd.merge(label, pred, on="customer_id", how="left")
    val_map = map_at_k(label["article_id"], label["prediction"], k=12)
    log.info(f"  Validation MAP@12: {val_map:.6f}")
    log_elapsed(t, "Step 11")

    # ==================================================================
    # Step 12: Test Predictions (week 0)
    # ==================================================================
    if TEST:
        log_section("Step 12: Test Predictions")
        t = time.time()

        inter = pd.read_parquet(DATA_DIR / "processed" / "processed_inter.pqt")
        for col in inter.columns:
            inter[col] = np.nan_to_num(inter[col])

        test_l = []
        batch_files = sorted(
            (DATA_DIR / "interim" / VERSION_NAME).glob("week0_candidate_*.pqt")
        )
        if batch_files:
            for fpath in tqdm(batch_files, desc="  Test batches"):
                candidate = pd.read_parquet(fpath)
                candidate = merge_week_data(data, inter, 0, candidate)
                # Fill missing features with 0
                avail_feats = [f for f in feats if f in candidate.columns]
                missing_feats = [f for f in feats if f not in candidate.columns]
                if missing_feats:
                    log.info(f"  Missing {len(missing_feats)} features, filling with 0")
                    for f in missing_feats:
                        candidate[f] = 0
                # Predict
                probs = np.zeros(candidate.shape[0])
                for batch in range(0, candidate.shape[0], 5_000_000):
                    outputs = ranker.predict(candidate.loc[batch:batch + 5_000_000 - 1, feats])
                    probs[batch:batch + 5_000_000] = outputs
                candidate["prob"] = probs
                test_l.append(candidate[["customer_id", "article_id", "prob"]])
                gc.collect()
        else:
            candidate = pd.read_parquet(DATA_DIR / "interim" / VERSION_NAME / "week0_candidate.pqt")
            candidate = merge_week_data(data, inter, 0, candidate)
            # Fill missing features with 0
            missing_feats = [f for f in feats if f not in candidate.columns]
            if missing_feats:
                log.info(f"  Missing {len(missing_feats)} features, filling with 0")
                for f in missing_feats:
                    candidate[f] = 0
            probs = np.zeros(candidate.shape[0])
            for batch in range(0, candidate.shape[0], 5_000_000):
                outputs = ranker.predict(candidate.loc[batch:batch + 5_000_000 - 1, feats])
                probs[batch:batch + 5_000_000] = outputs
            candidate["prob"] = probs
            test_l.append(candidate[["customer_id", "article_id", "prob"]])

        del inter
        gc.collect()

        test_pred = pd.concat(test_l, ignore_index=True)

        # Sort and take top 12 per customer
        test_pred = test_pred.sort_values(
            by=["customer_id", "prob"], ascending=False
        ).reset_index(drop=True)
        test_pred.rename(columns={"article_id": "prediction"}, inplace=True)
        test_pred = test_pred.drop_duplicates(["customer_id", "prediction"], keep="first")
        test_pred["customer_id"] = test_pred["customer_id"].astype(int)
        test_pred = (
            test_pred
            .groupby("customer_id")["prediction"]
            .progress_apply(list)
            .reset_index()
        )

        # Convert encoded IDs back to original
        def parse_prediction(x):
            result = []
            for i in x[:12]:
                try:
                    result.append("0" + str(idx2iid[int(i)]))
                except (KeyError, ValueError):
                    pass
            popular_fallback = [
                "0706016002", "0372860001", "0610776002", "0759871002",
                "0464297007", "0370865001", "0156231001", "0751471042",
                "0708194001", "0448509015", "0708194003", "0157148005",
            ]
            while len(result) < 12:
                result.append(popular_fallback[len(result) % len(popular_fallback)])
            return " ".join(result[:12])

        test_pred["prediction"] = test_pred["prediction"].progress_apply(parse_prediction)

        # Merge with submission template
        sub = pd.read_csv(DATA_DIR / "raw" / "sample_submission.csv")
        sub = sub[["customer_id"]].merge(test_pred, on="customer_id", how="left")
        popular_default = (
            "0706016002 0372860001 0610776002 0759871002 "
            "0464297007 0370865001 0156231001 0751471042 "
            "0708194001 0448509015 0708194003 0157148005"
        )
        sub["prediction"] = sub["prediction"].fillna(popular_default)

        submission_path = SUBMISSION_DIR / "submission_r26_replicate_45th.csv"
        sub.to_csv(submission_path, index=False)
        log.info(f"  Submission saved: {submission_path}")
        log.info(f"  Shape: {sub.shape}")
        log_elapsed(t, "Step 12")
    else:
        log.info("  Skipping test predictions (TEST=False)")

    # ==================================================================
    # Final Summary
    # ==================================================================
    log_section("R26 Complete")
    log.info(f"  Validation MAP@12: {val_map:.6f}")
    log.info(f"  Total elapsed: {(time.time() - t0) / 60:.1f} minutes")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
