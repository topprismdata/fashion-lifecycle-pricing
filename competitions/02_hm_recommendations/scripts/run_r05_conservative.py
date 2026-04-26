"""
H&M R05: Conservative Improvement on R01
R04 was WORSE (0.02141 vs R01's 0.02207) because co-occurrence crowded out repurchase.

Key insight from analysis:
- Repurchase is the strongest signal. Don't let other strategies displace it.
- Co-occurrence items should only appear AFTER repurchase fills its slots.
- Out-of-stock filter may be removing items that are actually available.
- Time-decay on popular helps: use it for the global popular fallback.

Changes from R01:
1. Time-decay weighted popular (not just last-7-days count) — better fallback
2. Longer repurchase window (6 weeks instead of 4) — more coverage
3. Co-occurrence ONLY to fill remaining slots after repurchase, limited to 2 items
4. No out-of-stock filter (too aggressive, removing good items)
5. Same product code variants: only 1-2 to fill remaining slots
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
print("R05: Conservative Improvement on R01")
print("=" * 60)

# ============================================================
# 1. Load Data
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

# Article metadata
article_product_code = articles.set_index('article_id')['product_code'].to_dict()

# ============================================================
# 2. Time-Decay Weighted Popular (Better Fallback)
# ============================================================
print("\n--- Time-decay popular ---")
# Use exponential decay: week 0 = 1.0, week 1 = 0.85, ..., week 12 = 0.14
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
weighted_pop = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
popular_top12 = weighted_pop.head(12).index.tolist()
popular_top50 = weighted_pop.head(50).index.tolist()
print(f"Top 12 popular (time-decay): {popular_top12[:5]}...")

# R01 popular for comparison (last 7 days only)
r01_popular = txn[txn['week'] <= 1]['article_id'].value_counts().head(12).index.tolist()
print(f"R01 popular (last 7d): {r01_popular[:5]}...")
print(f"Overlap top-12: {len(set(popular_top12) & set(r01_popular))}/12")

# ============================================================
# 3. Repurchase — Longer Window (6 weeks)
# ============================================================
print("\n--- Repurchase (6-week window) ---")
last_6w = txn[txn['week'] <= 6]
# Dedup, keep most recent first
cust_repurchase = (
    last_6w
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)
print(f"Customers with repurchase (6w): {len(cust_repurchase):,}")

# Compare with 4-week
last_4w = txn[txn['week'] <= 4]
cust_repurchase_4w = (
    last_4w
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)
print(f"Customers with repurchase (4w): {len(cust_repurchase_4w):,}")
print(f"Extra from 6w window: {len(cust_repurchase) - len(cust_repurchase_4w):,}")

# ============================================================
# 4. Co-occurrence (Bucket-based, Top-5 Related per Article)
# ============================================================
print("\n--- Co-occurrence (frequent pairs) ---")
buckets = txn[txn['week'] <= 10].groupby(['t_dat', 'customer_id', 'sales_channel_id'])['article_id'].apply(set).reset_index()
buckets.columns = ['t_dat', 'customer_id', 'sales_channel_id', 'article_set']
buckets = buckets[buckets['article_set'].apply(len) > 1]
print(f"Multi-item receipts: {len(buckets):,}")

pair_counts = Counter()
for arts in buckets['article_set']:
    if len(arts) <= 10:
        for pair in combinations(arts, 2):
            pair_counts[pair] += 1

freq_pairs = defaultdict(list)
for (a, b), count in pair_counts.items():
    freq_pairs[a].append((b, count))
    freq_pairs[b].append((a, count))
# Keep top-5 (not 12) — more conservative
freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:5]] for k, v in freq_pairs.items()}
print(f"Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts
gc.collect()

# ============================================================
# 5. Product Code Variants (Same Product, Different Color)
# ============================================================
print("\n--- Product code variants ---")
product_groups = articles.groupby('product_code')['article_id'].apply(list).to_dict()
product_lookup = {}
for pc, aids in product_groups.items():
    if len(aids) >= 2:
        for a in aids:
            product_lookup[a] = [x for x in aids if x != a][:3]
print(f"Articles with variants: {len(product_lookup):,}")

# ============================================================
# 6. Build Predictions — Conservative Strategy
# ============================================================
print("\n--- Building predictions ---")
predictions = {}

for cid in sample_sub['customer_id']:
    pred = []

    if cid in cust_repurchase:
        # Slot 1-12: Repurchase (most important, fill up to 12)
        repurchase = cust_repurchase[cid][:12]
        pred.extend(repurchase)

        # Slot fill: ONLY if repurchase < 10, add co-occurrence (max 2 items)
        remaining = 12 - len(pred)
        if remaining > 2 and len(repurchase) > 0:
            pair_added = 0
            for aid in repurchase[:3]:
                if aid in freq_pairs:
                    for related in freq_pairs[aid]:
                        if related not in pred:
                            pred.append(related)
                            pair_added += 1
                            if pair_added >= 2:
                                break
                if pair_added >= 2:
                    break

        # Slot fill: Product code variant (max 1 item)
        remaining = 12 - len(pred)
        if remaining > 0 and len(repurchase) > 0:
            for aid in repurchase[:3]:
                if aid in product_lookup:
                    for variant in product_lookup[aid]:
                        if variant not in pred:
                            pred.append(variant)
                            break
                    break

        # Slot fill: Time-decay popular
        remaining = 12 - len(pred)
        if remaining > 0:
            for aid in popular_top50:
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break
    else:
        # Fallback: Time-decay popular (better than last-7-days)
        pred = popular_top12[:]

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

# ============================================================
# 7. Save Submission
# ============================================================
print("\n--- Saving submission ---")
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r05_conservative.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")

# Count stats
repurchase_count = sum(1 for cid in sample_sub['customer_id'] if cid in cust_repurchase)
print(f"Active: {repurchase_count:,}, Fallback: {len(sample_sub) - repurchase_count:,}")

# ============================================================
# 8. Validation
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

val_purchases = txn[txn['week'] == 0]
val_truth = val_purchases.groupby('customer_id')['article_id'].apply(set).to_dict()
print(f"Val customers with purchases: {len(val_truth):,}")

# Validation: use data before week 0
val_txn = txn[txn['week'] > 0]

# Val repurchase (6-week)
val_repurchase = (
    val_txn[val_txn['week'] <= 6]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)

# Val popular (time-decay)
val_txn_copy = val_txn.copy()
val_txn_copy['decay'] = np.exp(-0.15 * val_txn_copy['week'].astype(float))
val_pop = val_txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
val_pop12 = val_pop.head(12).index.tolist()
val_pop50 = val_pop.head(50).index.tolist()

# Val co-occurrence
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

# Score on all submission customers
all_scores = []
for cid in sample_sub['customer_id']:
    if cid in val_truth:
        actual = val_truth[cid]
        recent = val_repurchase.get(cid, [])

        if recent:
            pred = recent[:12]
            remaining = 12 - len(pred)
            if remaining > 2:
                pair_added = 0
                for aid in recent[:3]:
                    if aid in val_freq_pairs:
                        for related in val_freq_pairs[aid]:
                            if related not in pred:
                                pred.append(related)
                                pair_added += 1
                                if pair_added >= 2:
                                    break
                    if pair_added >= 2:
                        break
            if len(pred) < 12:
                for aid in recent[:3]:
                    if aid in product_lookup:
                        for variant in product_lookup[aid]:
                            if variant not in pred:
                                pred.append(variant)
                                break
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

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("R05 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"R01 LB: 0.02207 | R04 LB: 0.02141")
print("=" * 60)
