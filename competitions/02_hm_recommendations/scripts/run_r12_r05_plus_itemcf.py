"""
H&M R12: R05 Core + ItemCF as Strict Fill-Only

LESSON from R10 (LB=0.01973) and R11 (LB=0.01965):
  Both ItemCF attempts scored WORSE than R05 (0.02224).
  The silver medal time-decay formula (a/sqrt(x) + b*exp(-c*x) - d) changes
  the repurchase ordering from recency-first to score-first, which HURTS.

KEY INSIGHT: R05 works because:
  1. Repurchase is sorted by RECENCY (most recent first), not by a score
  2. 6-week window (not 10-week) — more focused
  3. Simple exp(-0.15 * week) for popular, not the silver medal formula
  4. Co-occurrence is limited to 2 items, only fills empty slots

R12 Strategy: EXACTLY R05's repurchase + popular, but add ItemCF as fill-only
  with a very strict limit (max 2 ItemCF items per user).
  If ItemCF hurts even with 2 items max, then ItemCF is definitively useless here.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
import math
import gc
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("=" * 60)
print("R12: R05 Core + ItemCF Strict Fill-Only")
print("=" * 60)

# ============================================================
# 1. Load Data (EXACTLY like R05)
# ============================================================
print("\n--- Loading data ---")
txn = pd.read_csv(DATA / "transactions_train.csv", parse_dates=['t_dat'])
articles = pd.read_csv(DATA / "articles.csv")
sample_sub = pd.read_csv(DATA / "sample_submission.csv")

max_date = txn['t_dat'].max()
txn['week'] = ((max_date - txn['t_dat']).dt.days // 7).astype('int8')
txn = txn[txn['week'] <= 12].copy()
gc.collect()
print(f"Transactions (12w): {len(txn):,}")

# ============================================================
# 2. R05's Exact Time-Decay Popular
# ============================================================
print("\n--- Time-decay popular (R05 style) ---")
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
weighted_pop = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
popular_top12 = weighted_pop.head(12).index.tolist()
popular_top50 = weighted_pop.head(50).index.tolist()
print(f"Top 12 popular: {popular_top12[:5]}...")

# ============================================================
# 3. R05's Exact Repurchase (6-week, recency-ordered)
# ============================================================
print("\n--- Repurchase (6-week, recency-ordered) ---")
last_6w = txn[txn['week'] <= 6]
cust_repurchase = (
    last_6w
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)
print(f"Customers with repurchase (6w): {len(cust_repurchase):,}")

# ============================================================
# 4. R05's Exact Co-occurrence (top-5 per article)
# ============================================================
print("\n--- Co-occurrence (R05 style) ---")
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
# 5. ItemCF (NEW — from winning solution analysis)
# ============================================================
print("\n--- ItemCF with direction factor ---")
cust_sequences = (
    txn[txn['week'] <= 10]
    .sort_values('t_dat')
    .groupby('customer_id')['article_id']
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

# Keep top-10 similar items per article
itemcf_lookup = {}
for item, related in sim_item.items():
    sorted_related = sorted(related.items(), key=lambda x: -x[1])[:10]
    itemcf_lookup[item] = sorted_related

print(f"Articles with ItemCF: {len(itemcf_lookup):,}")
del cust_sequences, sim_item
gc.collect()

# Build per-customer ItemCF recommendation lists
print("    Generating ItemCF candidates per customer...")
active_customers = set(cust_repurchase.keys())
cust_recent_items = (
    txn[txn['week'] <= 4]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
    .to_dict()
)

itemcf_lists = {}
for cid in active_customers:
    recent_items = cust_recent_items.get(cid, [])
    if not recent_items:
        continue
    cf_scores = defaultdict(float)
    for aid in recent_items[:5]:
        if aid in itemcf_lookup:
            for related_aid, score in itemcf_lookup[aid]:
                cf_scores[related_aid] += score
    sorted_cf = sorted(cf_scores.items(), key=lambda x: -x[1])
    itemcf_lists[cid] = [aid for aid, _ in sorted_cf]

print(f"    Users with ItemCF: {len(itemcf_lists):,}")

# ============================================================
# 6. Build Predictions — R05 + ItemCF as Strict Fill
# ============================================================
print("\n--- Building predictions (R05 + ItemCF fill) ---")
predictions = {}
itemcf_used = 0
cooc_used = 0

for cid in sample_sub['customer_id']:
    pred = []

    if cid in cust_repurchase:
        # R05 EXACT: Repurchase first, recency-ordered
        repurchase = cust_repurchase[cid][:12]
        pred.extend(repurchase)

        # R05 EXACT: Co-occurrence (max 2 items) if repurchase < 10
        remaining = 12 - len(pred)
        if remaining > 2 and len(repurchase) > 0:
            pair_added = 0
            for aid in repurchase[:3]:
                if aid in freq_pairs:
                    for related in freq_pairs[aid]:
                        if related not in pred:
                            pred.append(related)
                            pair_added += 1
                            cooc_used += 1
                            if pair_added >= 2:
                                break
                if pair_added >= 2:
                    break

        # NEW: ItemCF (max 2 items) to fill remaining slots
        remaining = 12 - len(pred)
        if remaining > 0 and cid in itemcf_lists:
            cf_added = 0
            for aid in itemcf_lists[cid]:
                if aid not in pred:
                    pred.append(aid)
                    cf_added += 1
                    itemcf_used += 1
                    if cf_added >= 2:
                        break

        # R05: Time-decay popular to fill remaining
        remaining = 12 - len(pred)
        if remaining > 0:
            for aid in popular_top50:
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break
    else:
        # R05 EXACT: Fallback = time-decay popular
        pred = popular_top12[:]

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

print(f"  ItemCF items used: {itemcf_used:,}")
print(f"  Co-occurrence items used: {cooc_used:,}")

# ============================================================
# 7. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r12_r05_plus_itemcf.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

repurchase_count = sum(1 for cid in sample_sub['customer_id'] if cid in cust_repurchase)
print(f"Active: {repurchase_count:,}, Fallback: {len(sample_sub) - repurchase_count:,}")

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
pop12_str = ' '.join(f'{a:010d}' for a in popular_top12)
val_scores = []
for cid in val_truth:
    pred_str = predictions.get(cid, pop12_str)
    pred = [int(x) for x in pred_str.split()]
    val_scores.append(apk(val_truth[cid], pred, 12))

map12_active = np.mean(val_scores)
print(f"Val MAP@12 (active): {map12_active:.6f}")

print("\n" + "=" * 60)
print("R12 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"R05 LB: 0.02224 (best) | R10: 0.01973 | R11: 0.01965")
print("=" * 60)
