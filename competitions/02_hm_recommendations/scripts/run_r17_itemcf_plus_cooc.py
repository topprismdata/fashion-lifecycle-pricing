"""
H&M R17: R15 (6w + ItemCF max 12) + Co-occurrence as fill

R15 = 0.02279 (best). R16 (8w) = 0.02257 (worse: 8-week dilutes recency).
R12 had both co-occurrence (max 2) + ItemCF (max 2) and scored 0.02259.

Try: 6-week repurchase + ItemCF max 12 + co-occurrence max 4 (fill only).
Co-occurrence goes AFTER ItemCF since ItemCF proved more valuable.
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
print("R17: 6w Repurchase + ItemCF max 12 + Co-occurrence max 4")
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

# ============================================================
# 2. Time-Decay Popular
# ============================================================
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
weighted_pop = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
popular_top12 = weighted_pop.head(12).index.tolist()
popular_top50 = weighted_pop.head(50).index.tolist()

# ============================================================
# 3. Repurchase (6-week, recency-ordered)
# ============================================================
print("\n--- Repurchase (6-week) ---")
last_6w = txn[txn['week'] <= 6]
cust_repurchase = (
    last_6w
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)
print(f"Customers with repurchase: {len(cust_repurchase):,}")

# ============================================================
# 4. ItemCF
# ============================================================
print("\n--- ItemCF ---")
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

itemcf_lookup = {}
for item, related in sim_item.items():
    sorted_related = sorted(related.items(), key=lambda x: -x[1])[:10]
    itemcf_lookup[item] = sorted_related

print(f"Articles with ItemCF: {len(itemcf_lookup):,}")
del cust_sequences, sim_item
gc.collect()

# ItemCF per customer
print("    Generating ItemCF candidates...")
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
# 5. Co-occurrence (bucket-based)
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
# 6. Build Predictions
# ============================================================
print("\n--- Building predictions ---")
predictions = {}
itemcf_used = 0
cooc_used = 0

for cid in sample_sub['customer_id']:
    pred = []

    if cid in cust_repurchase:
        # Layer 1: Repurchase (6-week, recency-ordered)
        repurchase = cust_repurchase[cid][:12]
        pred.extend(repurchase)

        # Layer 2: ItemCF (max 12)
        remaining = 12 - len(pred)
        if remaining > 0 and cid in itemcf_lists:
            cf_added = 0
            for aid in itemcf_lists[cid]:
                if aid not in pred:
                    pred.append(aid)
                    cf_added += 1
                    itemcf_used += 1
                    if cf_added >= 12:
                        break

        # Layer 3: Co-occurrence (max 4, fill only)
        remaining = 12 - len(pred)
        if remaining > 0 and len(repurchase) > 0:
            cooc_added = 0
            for aid in repurchase[:5]:
                if aid in freq_pairs:
                    for related in freq_pairs[aid]:
                        if related not in pred:
                            pred.append(related)
                            cooc_added += 1
                            cooc_used += 1
                            if cooc_added >= 4:
                                break
                if cooc_added >= 4:
                    break

        # Layer 4: Time-decay popular
        remaining = 12 - len(pred)
        if remaining > 0:
            for aid in popular_top50:
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break
    else:
        pred = popular_top12[:]

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

print(f"  ItemCF items used: {itemcf_used:,}")
print(f"  Co-occurrence items used: {cooc_used:,}")

# ============================================================
# 7. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r17_itemcf_plus_cooc.csv"
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
print("R17 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"R15: 0.02279 (best) | R16: 0.02257 (8w)")
print("=" * 60)
