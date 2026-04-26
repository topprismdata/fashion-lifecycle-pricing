"""
H&M R13: R12 + More ItemCF Items (max 4) + Popular Ordering Fix

R12 hit NEW BEST at 0.02259 (+0.00035 over R05).
ItemCF added 497K useful items with max 2 per user.

Hypothesis: Maybe we can get more from ItemCF by allowing 3-4 items.
Also try: ordering ItemCF items by their similarity score for better precision.

Changes from R12:
1. ItemCF max items: 2 → 4
2. Co-occurrence max items: 2 → 1 (since ItemCF may be better)
3. Remove co-occurrence entirely (ItemCF captures the same signal better)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import math
import gc
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("=" * 60)
print("R15: R12 Core + ItemCF max 12 (full ItemCF fill)")
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
print("\n--- Time-decay popular ---")
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
# 4. ItemCF with Direction Factor
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

# Build per-customer ItemCF lists
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
# 5. Build Predictions — No Co-occurrence, More ItemCF
# ============================================================
print("\n--- Building predictions ---")
predictions = {}
itemcf_used_total = 0

for cid in sample_sub['customer_id']:
    pred = []

    if cid in cust_repurchase:
        # Repurchase first (recency-ordered)
        repurchase = cust_repurchase[cid][:12]
        pred.extend(repurchase)

        # ItemCF: max 4 items to fill empty slots
        remaining = 12 - len(pred)
        if remaining > 0 and cid in itemcf_lists:
            cf_added = 0
            for aid in itemcf_lists[cid]:
                if aid not in pred:
                    pred.append(aid)
                    cf_added += 1
                    itemcf_used_total += 1
                    if cf_added >= 12:
                        break

        # Time-decay popular to fill remaining
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

print(f"  ItemCF items used: {itemcf_used_total:,}")

# ============================================================
# 6. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r15_itemcf_max12.csv"
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
print("R13 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"R05: 0.02224 | R12: 0.02259 (best)")
print("=" * 60)
