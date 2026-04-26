#!/usr/bin/env python3
"""
R26 Test v2: Memory-efficient test prediction pipeline.
Processes rules one at a time, normalizes scores, then does rank-filter
in chunks to avoid OOM from pivoting 1.2B rows.
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

TRAIN_WEEK_NUM = 4
WEEK_NUM = TRAIN_WEEK_NUM + 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("r26_test")

sys.path.insert(0, str(REF_DIR))

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

tqdm.pandas()

from src.data import DataHelper
from src.retrieval.rules import (
    ItemCF, UserGroupItemCF,
    UserGroupTimeHistory, OrderHistory, OrderHistoryDecay, TimeHistory,
)
from src.retrieval.collector import RuleCollector
from src.utils import calc_valid_date, reduce_mem_usage, merge_week_data

log.info("=" * 70)
log.info("  R26 Test v2: Memory-efficient test predictions")
log.info("=" * 70)

t0 = time.time()

# Load data
dh = DataHelper(str(DATA_DIR))
data = dh.preprocess_data(save=False, name="encoded_full")

uid2idx = pickle.load(open(DATA_DIR / "index_id_map" / "user_id2index.pkl", "rb"))
idx2uid = pickle.load(open(DATA_DIR / "index_id_map" / "user_index2id.pkl", "rb"))
idx2iid = pickle.load(open(DATA_DIR / "index_id_map" / "item_index2id.pkl", "rb"))

submission = pd.read_csv(DATA_DIR / "raw" / "sample_submission.csv")
submission["customer_id"] = submission["customer_id"].map(uid2idx)

# User grouping
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

# Generate week 0 candidates
week = 0
trans = data["inter"]
start_date, end_date = calc_valid_date(week)
log.info(f"Week 0: [{start_date}, {end_date})")

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

customer_list = submission["customer_id"].values
log.info(f"Test customers: {len(customer_list):,}")

rules = [
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

# Memory-efficient approach: process each rule, normalize, then merge incrementally
# Use groupby-based accumulation instead of per-row dict
log.info("Processing rules one at a time with normalization...")

accumulated = None  # Will be a DataFrame of (customer_id, article_id) -> score

for rule_idx, rule in enumerate(tqdm(rules, desc="Rules")):
    items = rule.retrieve()
    # Normalize
    scaler = MinMaxScaler()
    items["norm_score"] = scaler.fit_transform(items["score"].values.reshape(-1, 1)).flatten()

    # Group by (customer_id, article_id) and sum scores
    items_agg = items.groupby(["customer_id", "article_id"])["norm_score"].sum().reset_index()

    del items
    gc.collect()

    # Merge with accumulated scores
    if accumulated is None:
        accumulated = items_agg
    else:
        accumulated = pd.concat([accumulated, items_agg], ignore_index=True)
        # Re-aggregate to merge same (customer, item) pairs
        accumulated = accumulated.groupby(["customer_id", "article_id"])["norm_score"].sum().reset_index()

    del items_agg
    gc.collect()
    log.info(f"  Rule {rule_idx}: accumulated {accumulated.shape[0]:,} unique pairs")

log.info(f"Total unique (customer, item) pairs: {accumulated.shape[0]:,}")

# Rank and keep top 200 per customer
candidates = accumulated.rename(columns={"norm_score": "score"})
del accumulated
gc.collect()

log.info(f"Candidates DataFrame: {candidates.shape[0]:,} rows")

# Rank and keep top 200 per customer
candidates["rank"] = candidates.groupby(["customer_id"])["score"].rank(ascending=False)
candidates = candidates[candidates["rank"] <= 200]
log.info(f"After rank200: {candidates.shape[0]:,}")

# Save as single file
candidates.to_parquet(DATA_DIR / "interim" / VERSION_NAME / "week0_candidate.pqt")
log.info("Saved week0_candidate.pqt")

del train, valid, last_week, last_2week, last_60day, last_80day
gc.collect()
log.info(f"Candidate generation elapsed: {(time.time() - t0) / 60:.1f} min")

# Load model and features
import lightgbm as lgb

log.info("Loading model and processed_inter...")
ranker = lgb.Booster(model_file=str(MODEL_DIR / "lgb_recall2_binary.model"))

# Load processed_inter
inter = pd.read_parquet(DATA_DIR / "processed" / "processed_inter.pqt")
for col in inter.columns:
    inter[col] = np.nan_to_num(inter[col])

# Get feature list from training data
sample = pd.read_parquet(DATA_DIR / "processed" / VERSION_NAME / "week1_candidate.pqt")
feats = [x for x in sample.columns if x not in ["label", "sales_channel_id", "t_dat", "week", "valid"]]
del sample
gc.collect()

# Predict on candidates
log.info("Merging features and predicting...")
candidates = pd.read_parquet(DATA_DIR / "interim" / VERSION_NAME / "week0_candidate.pqt")
candidates = merge_week_data(data, inter, 0, candidates)

probs = np.zeros(candidates.shape[0])
batch_size = 5_000_000
for batch in range(0, candidates.shape[0], batch_size):
    outputs = ranker.predict(candidates.loc[batch:batch + batch_size - 1, feats])
    probs[batch:batch + batch_size] = outputs
candidates["prob"] = probs

# Sort and take top 12 per customer
test_pred = candidates[["customer_id", "article_id", "prob"]]
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
log.info(f"Submission saved: {submission_path}")
log.info(f"Shape: {sub.shape}")
log.info(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")
