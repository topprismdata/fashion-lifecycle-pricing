"""
H&M R07: R05 Base + Last-Week Trending Boost
R05 is best at 0.02224. R06 showed age-segment popular hurts.

Key insight: The time-decay popular in R05 may be smoothing too much.
Last-week trending captures the CURRENT week's hot items better.
Also try: interleave repurchase with trending (don't just append).

Changes from R05:
1. Same 6-week repurchase window (proven to help)
2. Last-7-days popular for fallback (R01-style, not time-decay)
3. Interleave: put last-1-week purchased items first, then older, then popular
4. No co-occurrence, no product code variants (they hurt in R04/R06)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("=" * 60)
print("R07: R05 Base + Last-Week Trending Boost")
print("=" * 60)

# ============================================================
# 1. Load Data
# ============================================================
print("\n--- Loading data ---")
txn = pd.read_csv(DATA / "transactions_train.csv", parse_dates=['t_dat'])
sample_sub = pd.read_csv(DATA / "sample_submission.csv")

max_date = txn['t_dat'].max()
txn['week'] = ((max_date - txn['t_dat']).dt.days // 7).astype('int8')
txn = txn[txn['week'] <= 12].copy()
gc.collect()
print(f"Transactions (12w): {len(txn):,}")

# ============================================================
# 2. Popular Articles — Multiple Time Windows
# ============================================================
print("\n--- Popular articles ---")
# Last 7 days (same as R01)
pop_1w = txn[txn['week'] <= 1]['article_id'].value_counts().head(12).index.tolist()
# Last 14 days
pop_2w = txn[txn['week'] <= 2]['article_id'].value_counts().head(20).index.tolist()
# Time-decay (same as R05)
txn_copy = txn.copy()
txn_copy['decay'] = np.exp(-0.15 * txn_copy['week'].astype(float))
pop_decay = txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_decay_12 = pop_decay.head(12).index.tolist()
pop_decay_50 = pop_decay.head(50).index.tolist()

print(f"Last-1w top-5: {pop_1w[:5]}")
print(f"Time-decay top-5: {pop_decay_12[:5]}")
print(f"Overlap: {len(set(pop_1w) & set(pop_decay_12))}/12")

# ============================================================
# 3. Repurchase — 6-Week Window with Recency Ordering
# ============================================================
print("\n--- Repurchase (6-week) ---")
cust_repurchase = (
    txn[txn['week'] <= 6]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)
print(f"Active customers: {len(cust_repurchase):,}")

# Also get last-1-week purchases specifically (most recent signal)
cust_last_week = (
    txn[txn['week'] <= 1]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:12])
    .to_dict()
)
print(f"Customers with last-1w purchases: {len(cust_last_week):,}")

# ============================================================
# 4. Build Predictions
# ============================================================
print("\n--- Building predictions ---")
predictions = {}

for cid in sample_sub['customer_id']:
    pred = []

    if cid in cust_repurchase:
        recent = cust_repurchase[cid]

        # Priority 1: Items from last 1 week (highest recency)
        last_week_items = cust_last_week.get(cid, [])
        for aid in last_week_items:
            if aid not in pred:
                pred.append(aid)

        # Priority 2: Items from weeks 2-6 (older repurchase)
        for aid in recent:
            if aid not in pred:
                pred.append(aid)

        # Priority 3: Time-decay popular to fill remaining
        remaining = 12 - len(pred)
        if remaining > 0:
            for aid in pop_decay_50:
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break
    else:
        # Fallback: Use time-decay popular (same as R05)
        pred = pop_decay_12[:]

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

# ============================================================
# 5. Save Submission
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r07_trending.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active = sum(1 for cid in sample_sub['customer_id'] if cid in cust_repurchase)
print(f"Active: {active:,}, Fallback: {len(sample_sub) - active:,}")

# ============================================================
# 6. Validation
# ============================================================
print("\n--- Validation ---")
def apk(actual, predicted, k=12):
    if not actual:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

val_truth = txn[txn['week'] == 0].groupby('customer_id')['article_id'].apply(set).to_dict()
val_txn = txn[txn['week'] > 0]

val_repurchase = (
    val_txn[val_txn['week'] <= 6]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)
val_last_week = (
    val_txn[val_txn['week'] <= 1]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:12])
    .to_dict()
)

val_txn_copy = val_txn.copy()
val_txn_copy['decay'] = np.exp(-0.15 * val_txn_copy['week'].astype(float))
val_pop = val_txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
val_pop12 = val_pop.head(12).index.tolist()
val_pop50 = val_pop.head(50).index.tolist()

all_scores = []
for cid in sample_sub['customer_id']:
    if cid in val_truth:
        actual = val_truth[cid]
        recent = val_repurchase.get(cid, [])

        if recent:
            pred = []
            lw = val_last_week.get(cid, [])
            for aid in lw:
                if aid not in pred:
                    pred.append(aid)
            for aid in recent:
                if aid not in pred:
                    pred.append(aid)
            remaining = 12 - len(pred)
            if remaining > 0:
                for aid in val_pop50:
                    if aid not in pred:
                        pred.append(aid)
                        remaining -= 1
                        if remaining <= 0:
                            break
        else:
            pred = val_pop12[:]
        all_scores.append(apk(actual, pred, 12))
    else:
        all_scores.append(apk(set(), val_pop12))

map12_all = np.mean(all_scores)
active_scores = [s for i, s in enumerate(all_scores) if sample_sub['customer_id'].iloc[i] in val_truth]
map12_active = np.mean(active_scores)

print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")

print("\n" + "=" * 60)
print("R07 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"R01 LB: 0.02207 | R05 LB: 0.02224 | R06 LB: 0.02112")
print("=" * 60)
