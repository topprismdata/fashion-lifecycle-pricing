#!/usr/bin/env python3
"""
R26 Test v3: Batch-based test prediction pipeline.
Processes 100k customers at a time to manage memory.
"""

from __future__ import annotations
import sys, os, time, warnings, logging, pickle, gc
from pathlib import Path
warnings.filterwarnings("ignore")

REF_DIR = Path("/Users/guohongbin/projects/fashion-lifecycle-pricing/competitions/02_hm_recommendations/reference_solution_45th")
DATA_DIR = REF_DIR / "data"
MODEL_DIR = REF_DIR / "models"
SCRIPT_DIR = Path("/Users/guohongbin/projects/fashion-lifecycle-pricing/competitions/02_hm_recommendations/scripts")
SUBMISSION_DIR = SCRIPT_DIR.parent / "outputs" / "submissions"
VERSION_NAME = "recall2"
BATCH_SIZE = 100_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("r26_test")
sys.path.insert(0, str(REF_DIR))

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
tqdm.pandas()

from src.data import DataHelper
from src.retrieval.rules import (
    ItemCF, UserGroupItemCF, UserGroupTimeHistory,
    OrderHistory, OrderHistoryDecay, TimeHistory,
)
from src.retrieval.collector import RuleCollector
from src.utils import calc_valid_date, reduce_mem_usage, merge_week_data

log.info("=" * 70)
log.info("  R26 Test v3: Batch-based test predictions")
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
all_customers = submission["customer_id"].values

# User grouping
user_info = data["inter"].groupby(["customer_id"])["price"].mean().reset_index(name="mean_price")
user_info["purchase_ability"] = pd.qcut(user_info["mean_price"], 5, labels=False)
del user_info["mean_price"]
listBin = [-1, 19, 29, 39, 49, 59, 69, 119]
data["user"]["age_bins"] = pd.cut(data["user"]["age"], listBin)
data["user"] = data["user"].merge(user_info, on="customer_id", how="left")

# Prepare training data for rules
trans = data["inter"]
start_date, end_date = calc_valid_date(0)
train, _ = dh.split_data(trans, start_date, end_date)
train = train.merge(data["user"][["customer_id", "age_bins", "user_gender"]], on="customer_id", how="left")
train = train.merge(user_info, on="customer_id", how="left")
train["t_dat"] = pd.to_datetime(train["t_dat"])

last_week = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=7)]
last_2week = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=14)]
last_60day = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=60)]
last_80day = train[train["t_dat"] > train["t_dat"].max() - pd.Timedelta(days=80)]

del train
gc.collect()

# Load model and features
import lightgbm as lgb
ranker = lgb.Booster(model_file=str(MODEL_DIR / "lgb_recall2_binary.model"))

inter = pd.read_parquet(DATA_DIR / "processed" / "processed_inter.pqt")
for col in inter.columns:
    inter[col] = np.nan_to_num(inter[col])

sample = pd.read_parquet(DATA_DIR / "processed" / VERSION_NAME / "week1_candidate.pqt")
feats = [x for x in sample.columns if x not in ["label", "sales_channel_id", "t_dat", "week", "valid"]]

# Use model's feature list instead of candidate file's features
model_feats = ranker.feature_name()
log.info(f"  Candidate features: {len(feats)}, Model features: {len(model_feats)}")

# Use model features as the canonical list
feats = model_feats
del sample
gc.collect()

log.info(f"Setup complete. Processing {len(all_customers):,} customers in batches of {BATCH_SIZE:,}")
log.info(f"  Training features: {len(feats)}")
log.info(f"  Model expects: {ranker.num_feature()} features")

# Process in batches
popular_default = "0706016002 0372860001 0610776002 0759871002 0464297007 0370865001 0156231001 0751471042 0708194001 0448509015 0708194003 0157148005"

all_predictions = {}
n_batches = (len(all_customers) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(n_batches):
    tb = time.time()
    start = batch_idx * BATCH_SIZE
    end = min((batch_idx + 1) * BATCH_SIZE, len(all_customers))
    batch_customers = all_customers[start:end]

    log.info(f"\nBatch {batch_idx}/{n_batches}: {len(batch_customers):,} customers")

    # Generate candidates for this batch
    rules = [
        UserGroupTimeHistory(data, batch_customers, last_week, ["age_bins"], n=200, scale=True, name="1"),
        UserGroupTimeHistory(data, batch_customers, last_week, ["purchase_ability"], n=200, scale=True, name="2"),
        UserGroupTimeHistory(data, batch_customers, last_week, ["user_gender"], n=200, scale=True, name="3"),
        OrderHistory(last_80day[last_80day["customer_id"].isin(batch_customers)], days=35, n=200),
        OrderHistoryDecay(last_80day[last_80day["customer_id"].isin(batch_customers)], days=7, n=200),
        TimeHistory(batch_customers, last_week, n=200),
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

    # Collect candidates (no label needed for test)
    # Use compress=False to get raw candidates, then pivot+rank
    candidates = RuleCollector().collect(
        week_num=0,
        trans_df=trans,
        customer_list=batch_customers,
        rules=rules,
        min_pos_rate=0.0,
        norm=False,
        compress=False,
    )
    del rules
    gc.collect()

    if candidates.shape[0] == 0:
        log.info(f"  No candidates, using popular defaults")
        for cid in batch_customers:
            all_predictions[int(cid)] = popular_default
        continue

    # Pivot + normalize + sum scores
    candidates, _ = reduce_mem_usage(candidates)
    candidates = (
        pd.pivot_table(
            candidates, values="score",
            index=["customer_id", "article_id"],
            columns=["method"], aggfunc=np.sum,
        ).reset_index()
    )
    rule_names = [x for x in candidates.columns if x not in ["customer_id", "article_id"]]
    tmp = candidates[rule_names].copy()
    for f in rule_names:
        tmp[f] = MinMaxScaler().fit_transform(tmp[f].values.reshape(-1, 1))
    candidates["score"] = tmp[rule_names].sum(axis=1)
    del tmp, rule_names
    gc.collect()

    # Rank and filter top 200
    candidates["rank"] = candidates.groupby(["customer_id"])["score"].rank(ascending=False)
    candidates = candidates[candidates["rank"] <= 200]

    # Merge features
    candidates = merge_week_data(data, inter, 0, candidates)

    # Align features: add missing columns with 0, keep only training features
    log.info(f"  Candidates columns: {len(candidates.columns)}, Features list: {len(feats)}")
    missing_feats = [f for f in feats if f not in candidates.columns]
    if missing_feats:
        log.info(f"  Filling {len(missing_feats)} missing features with 0: {missing_feats[:5]}...")
        for f in missing_feats:
            candidates[f] = 0
    present_feats = [f for f in feats if f in candidates.columns]
    log.info(f"  Present features: {len(present_feats)}, Missing: {len(missing_feats)}")

    # Predict using numpy array to bypass pandas categorical checks
    X = candidates[feats].fillna(0).values.astype(np.float32)
    probs = np.zeros(X.shape[0])
    for batch in range(0, X.shape[0], 5_000_000):
        probs[batch:batch + 5_000_000] = ranker.predict(
            X[batch:batch + 5_000_000]
        )
    candidates["prob"] = probs

    # Sort and take top 12 per customer using groupby (much faster than per-customer loop)
    candidates = candidates.sort_values(
        by=["customer_id", "prob"], ascending=False
    ).reset_index(drop=True)
    candidates = candidates.drop_duplicates(["customer_id", "article_id"], keep="first")
    top12 = candidates.groupby("customer_id").head(12)

    # Build predictions dict from top12 using vectorized operations
    top12["article_str"] = "0" + top12["article_id"].astype(int).astype(str)
    # Map article indices to IDs
    top12["article_str"] = top12["article_id"].astype(int).map(lambda x: "0" + str(idx2iid.get(x, "")))
    # Group by customer and join predictions
    grouped = top12.groupby("customer_id")["article_str"].apply(list)
    for cid_int, articles in grouped.items():
        valid_articles = [a for a in articles if a != "0"]
        while len(valid_articles) < 12:
            valid_articles.append(popular_default.split()[len(valid_articles)])
        all_predictions[int(cid_int)] = " ".join(valid_articles[:12])

    # Fill customers with no candidates
    for cid in batch_customers:
        cid_int = int(cid)
        if cid_int not in all_predictions:
            all_predictions[cid_int] = popular_default

    del candidates
    gc.collect()
    log.info(f"  Batch {batch_idx} done in {(time.time() - tb) / 60:.1f} min, "
             f"predictions: {len(all_predictions):,}")

# Build submission
log.info("Building submission...")
sub = pd.read_csv(DATA_DIR / "raw" / "sample_submission.csv")
preds = []
for cid_str in tqdm(sub["customer_id"], desc="Formatting"):
    cid_int = uid2idx.get(cid_str, -1)
    if cid_int in all_predictions:
        preds.append(all_predictions[cid_int])
    else:
        preds.append(popular_default)

sub["prediction"] = preds
submission_path = SUBMISSION_DIR / "submission_r26_replicate_45th.csv"
sub.to_csv(submission_path, index=False)
log.info(f"Submission saved: {submission_path}")
log.info(f"Shape: {sub.shape}")
log.info(f"Total elapsed: {(time.time() - t0) / 60:.1f} min")
