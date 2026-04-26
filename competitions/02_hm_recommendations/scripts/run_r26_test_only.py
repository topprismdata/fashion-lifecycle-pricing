#!/usr/bin/env python3
"""
R26 Test-only: Generate test predictions using the trained R26 model.
Runs only the test week (week 0) recall + prediction pipeline.
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
WEEK_NUM = TRAIN_WEEK_NUM + 2  # 6
TEST_BATCH_SIZE = 70000
TEST_BATCH_NUM = 20

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
log.info("  R26 Test-Only: Generate test predictions")
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

# Define rules (same as training, minus ALS/BPR)
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

# Collect candidates
log.info("Collecting test candidates...")
candidates = RuleCollector().collect(
    week_num=week,
    trans_df=trans,
    customer_list=customer_list,
    rules=rules,
    min_pos_rate=0.0,
    norm=False,
    compress=False,
)
log.info(f"Raw candidates: {candidates.shape[0]:,}")

candidates, _ = reduce_mem_usage(candidates)

# Pivot + score fusion
candidates = (
    pd.pivot_table(
        candidates, values="score",
        index=["customer_id", "article_id"],
        columns=["method"], aggfunc=np.sum,
    )
    .reset_index()
)
rule_names = [x for x in candidates.columns if x not in ["customer_id", "article_id"]]
tmp = candidates[rule_names].copy()
for f in rule_names:
    tmp[f] = MinMaxScaler().fit_transform(tmp[f].values.reshape(-1, 1))
candidates["score"] = tmp[rule_names].sum(axis=1)
del tmp
gc.collect()

candidates["rank"] = candidates.groupby(["customer_id"])["score"].rank(ascending=False)
candidates = candidates[candidates["rank"] <= 200]
log.info(f"After rank200: {candidates.shape[0]:,}")

# Split into batches and save
unique_customers = candidates["customer_id"].unique()
actual_batch_num = (len(unique_customers) + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE
log.info(f"Splitting {len(unique_customers):,} customers into {actual_batch_num} batches")

for batch_idx in range(actual_batch_num):
    start_idx = batch_idx * TEST_BATCH_SIZE
    end_idx = min((batch_idx + 1) * TEST_BATCH_SIZE, len(unique_customers))
    batch_customers = unique_customers[start_idx:end_idx]
    batch_cands = candidates[candidates["customer_id"].isin(batch_customers)]
    batch_cands.to_parquet(
        DATA_DIR / "interim" / VERSION_NAME / f"week0_candidate_{batch_idx}.pqt"
    )
    log.info(f"  Batch {batch_idx}: {batch_cands.shape[0]:,} rows")

del candidates, train, valid, last_week, last_2week, last_60day, last_80day
gc.collect()
log.info(f"Candidate generation elapsed: {(time.time() - t0) / 60:.1f} min")

# Load model and processed_inter
import lightgbm as lgb

log.info("Loading model and processed_inter...")
ranker = lgb.Booster(model_file=str(MODEL_DIR / "lgb_recall2_binary.model"))
inter = pd.read_parquet(DATA_DIR / "processed" / "processed_inter.pqt")
for col in inter.columns:
    inter[col] = np.nan_to_num(inter[col])

# Load a processed candidate to get feature list
sample = pd.read_parquet(DATA_DIR / "processed" / VERSION_NAME / "week1_candidate.pqt")
feats = [x for x in sample.columns if x not in ["label", "sales_channel_id", "t_dat", "week", "valid"]]
del sample
gc.collect()

# Predict per batch
log.info("Predicting test batches...")
test_l = []
batch_files = sorted((DATA_DIR / "interim" / VERSION_NAME).glob("week0_candidate_*.pqt"))
for fpath in tqdm(batch_files, desc="Test batches"):
    candidate = pd.read_parquet(fpath)
    candidate = merge_week_data(data, inter, 0, candidate)
    # Predict
    probs = np.zeros(candidate.shape[0])
    batch_size = 5_000_000
    for batch in range(0, candidate.shape[0], batch_size):
        outputs = ranker.predict(candidate.loc[batch:batch + batch_size - 1, feats])
        probs[batch:batch + batch_size] = outputs
    candidate["prob"] = probs
    test_l.append(candidate[["customer_id", "article_id", "prob"]])
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
log.info(f"Submission saved: {submission_path}")
log.info(f"Shape: {sub.shape}")
log.info(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")
