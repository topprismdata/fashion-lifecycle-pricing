"""
H&M R09: R05 Base + Last-Week Boost for Non-Repurchase Customers

Analysis shows R05 (0.02224) is best. All attempts to add complexity hurt.
The only change that helped: 6-week repurchase window + time-decay popular.

New idea: For customers who have purchases in the last week specifically,
prioritize those articles even higher. For customers with NO repurchase
candidates, use a mix of last-1-week popular + time-decay popular.

This is essentially R05 with a tweak to the fallback for semi-active customers
(those with purchases in weeks 2-6 but not week 1).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("=" * 60)
print("R09: R05 Base + Enhanced Semi-Active Fallback")
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
# 2. Compute Popular Article Lists
# ============================================================
print("\n--- Popular articles ---")
# Time-decay popular (R05 style)
txn_copy = txn.copy()
txn_copy['decay'] = np.exp(-0.15 * txn_copy['week'].astype(float))
global_pop = txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_12 = global_pop.head(12).index.tolist()
pop_50 = global_pop.head(50).index.tolist()

# Last-7-days popular (R01 style)
pop_1w = txn[txn['week'] <= 1]['article_id'].value_counts().head(12).index.tolist()

# Blend: Use time-decay top-12 as the primary fallback
print(f"Time-decay top-12: {pop_12}")
print(f"Last-7d top-12: {pop_1w}")

# ============================================================
# 3. Repurchase Windows
# ============================================================
print("\n--- Repurchase windows ---")

# Active: 6-week window
cust_6w = (
    txn[txn['week'] <= 6]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)

# Very active: 10-week window (more candidates for less active users)
cust_10w = (
    txn[txn['week'] <= 10]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:24])
    .to_dict()
)

# Last week specifically
cust_1w = (
    txn[txn['week'] <= 1]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:12])
    .to_dict()
)

print(f"Customers with 1w purchases: {len(cust_1w):,}")
print(f"Customers with 6w purchases: {len(cust_6w):,}")
print(f"Customers with 10w purchases: {len(cust_10w):,}")

# ============================================================
# 4. Co-occurrence (from R05, conservative)
# ============================================================
print("\n--- Co-occurrence ---")
buckets = txn[txn['week'] <= 10].groupby(['t_dat', 'customer_id', 'sales_channel_id'])['article_id'].apply(set).reset_index()
buckets.columns = ['t_dat', 'customer_id', 'sales_channel_id', 'article_set']
buckets = buckets[buckets['article_set'].apply(len) > 1]

pair_counts = Counter()
for arts in buckets['article_set']:
    if len(arts) <= 10:
        for pair in combinations(arts, 2):
            pair_counts[pair] += 1

freq_pairs = defaultdict(list)
for (a, b), count in pair_counts.items():
    freq_pairs[a].append((b, count))
    freq_pairs[b].append((a, count))
freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:5]] for k, v in freq_pairs.items()}
print(f"Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts
gc.collect()

# ============================================================
# 5. Build Predictions
# ============================================================
print("\n--- Building predictions ---")
predictions = {}

for cid in sample_sub['customer_id']:
    if cid in cust_6w:
        # Active customer: R05 strategy
        pred = cust_6w[cid][:12]

        # Co-occurrence (max 2, only if < 12)
        remaining = 12 - len(pred)
        if remaining > 0 and len(pred) > 0:
            pair_added = 0
            for aid in pred[:3]:
                if aid in freq_pairs:
                    for related in freq_pairs[aid]:
                        if related not in pred:
                            pred.append(related)
                            pair_added += 1
                            if pair_added >= min(2, remaining):
                                break
                if pair_added >= 2:
                    break

        # Time-decay popular fill
        remaining = 12 - len(pred)
        if remaining > 0:
            for aid in pop_50:
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break

        predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

    elif cid in cust_10w:
        # Semi-active: Use 10-week history + co-occurrence + popular
        pred = cust_10w[cid][:8]  # Fewer repurchase items (older)

        # More co-occurrence for semi-active (3 instead of 2)
        remaining = 12 - len(pred)
        if remaining > 0 and len(pred) > 0:
            pair_added = 0
            for aid in pred[:5]:
                if aid in freq_pairs:
                    for related in freq_pairs[aid]:
                        if related not in pred:
                            pred.append(related)
                            pair_added += 1
                            if pair_added >= min(3, remaining):
                                break
                if pair_added >= 3:
                    break

        # Time-decay popular fill
        remaining = 12 - len(pred)
        if remaining > 0:
            for aid in pop_50:
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break

        predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

    else:
        # Inactive: Time-decay popular
        predictions[cid] = ' '.join(f'{a:010d}' for a in pop_12)

# ============================================================
# 6. Save
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r09_semi_active.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active_6w = sum(1 for cid in sample_sub['customer_id'] if cid in cust_6w)
semi = sum(1 for cid in sample_sub['customer_id'] if cid not in cust_6w and cid in cust_10w)
cold = len(sample_sub) - active_6w - semi
print(f"Active (6w): {active_6w:,} | Semi-active (10w): {semi:,} | Cold: {cold:,}")

# ============================================================
# 7. Validation
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

val_6w = (val_txn[val_txn['week'] <= 6].sort_values('t_dat', ascending=False)
          .groupby('customer_id')['article_id']
          .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
          .to_dict())
val_10w = (val_txn[val_txn['week'] <= 10].sort_values('t_dat', ascending=False)
           .groupby('customer_id')['article_id']
           .apply(lambda x: list(dict.fromkeys(x.tolist()))[:24])
           .to_dict())

val_txn_copy = val_txn.copy()
val_txn_copy['decay'] = np.exp(-0.15 * val_txn_copy['week'].astype(float))
val_pop = val_txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
val_pop12 = val_pop.head(12).index.tolist()
val_pop50 = val_pop.head(50).index.tolist()

val_buckets = val_txn[val_txn['week'] <= 10].groupby(['t_dat', 'customer_id', 'sales_channel_id'])['article_id'].apply(set).reset_index()
val_buckets.columns = ['t_dat', 'customer_id', 'sales_channel_id', 'article_set']
val_buckets = val_buckets[val_buckets['article_set'].apply(len) > 1]
val_pair_counts = Counter()
for arts in val_buckets['article_set']:
    if len(arts) <= 10:
        for pair in combinations(arts, 2):
            val_pair_counts[pair] += 1
val_freq_pairs = defaultdict(list)
for (a, b), count in val_pair_counts.items():
    val_freq_pairs[a].append((b, count))
    val_freq_pairs[b].append((a, count))
val_freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:5]] for k, v in val_freq_pairs.items()}

all_scores = []
for cid in sample_sub['customer_id']:
    if cid in val_truth:
        actual = val_truth[cid]
        if cid in val_6w:
            pred = val_6w[cid][:12]
            remaining = 12 - len(pred)
            if remaining > 0 and len(pred) > 0:
                pair_added = 0
                for aid in pred[:3]:
                    if aid in val_freq_pairs:
                        for related in val_freq_pairs[aid]:
                            if related not in pred:
                                pred.append(related)
                                pair_added += 1
                                if pair_added >= min(2, remaining):
                                    break
                    if pair_added >= 2:
                        break
            remaining = 12 - len(pred)
            if remaining > 0:
                for aid in val_pop50:
                    if aid not in pred:
                        pred.append(aid)
                        remaining -= 1
                        if remaining <= 0:
                            break
        elif cid in val_10w:
            pred = val_10w[cid][:8]
            remaining = 12 - len(pred)
            if remaining > 0 and len(pred) > 0:
                pair_added = 0
                for aid in pred[:5]:
                    if aid in val_freq_pairs:
                        for related in val_freq_pairs[aid]:
                            if related not in pred:
                                pred.append(related)
                                pair_added += 1
                                if pair_added >= min(3, remaining):
                                    break
                    if pair_added >= 3:
                        break
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
print("R09 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"R05 LB: 0.02224 (best)")
print("=" * 60)
