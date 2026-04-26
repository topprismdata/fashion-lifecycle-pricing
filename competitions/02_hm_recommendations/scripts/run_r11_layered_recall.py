"""
H&M R11: Layered Recall with Repurchase Protection

LESSON from R10 (LB=0.01973, WORSE than R05=0.02224):
  Mixing all candidates into one pool and sorting by composite score lets
  ItemCF (7.5M items) and SPC candidates displace high-precision repurchase.
  The Candidate Diversity Paradox strikes again.

KEY INSIGHT from R04-R10 experiments:
  Repurchase items should NEVER be displaced by other candidate types.
  Use a LAYERED approach:
    Layer 1: Repurchase items (top priority, fill slots 1-N)
    Layer 2: ItemCF items (fill empty slots N+1 to 12)
    Layer 3: Age-bin popular + Global popular (fill remaining slots)

This is fundamentally different from R10's "mix and sort" approach.

Architecture:
  - Same 6-channel recall from R10
  - BUT: Layered slot-filling instead of composite scoring
  - Repurchase items ALWAYS go first
  - ItemCF and SPC only fill EMPTY slots
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
print("R11: Layered Recall with Repurchase Protection")
print("=" * 60)

# ============================================================
# 1. Load Data
# ============================================================
print("\n--- Loading data ---")
txn = pd.read_csv(DATA / "transactions_train.csv", parse_dates=['t_dat'])
articles = pd.read_csv(DATA / "articles.csv")
customers = pd.read_csv(DATA / "customers.csv")
sample_sub = pd.read_csv(DATA / "sample_submission.csv")

max_date = txn['t_dat'].max()
txn['week'] = ((max_date - txn['t_dat']).dt.days // 7).astype('int8')
txn['days_ago'] = (max_date - txn['t_dat']).dt.days
txn = txn[txn['week'] <= 12].copy()
gc.collect()
print(f"Transactions (12w): {len(txn):,}")

# Customer features
customers['age'] = customers['age'].fillna(customers['age'].median())
customers['age_bin'] = pd.cut(customers['age'], bins=[0, 21, 26, 31, 36, 46, 100],
                               labels=False).fillna(3).astype(int)
cust_age = customers.set_index('customer_id')['age'].to_dict()

# ============================================================
# 2. Build Recall Channels
# ============================================================
print("\n--- Building recall channels ---")

# --- Channel 1: Repurchase with Silver Medal Time-Decay Formula ---
print("\n  Channel 1: Repurchase (silver medal time-decay)...")
a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3

txn_10w = txn[txn['week'] <= 10].copy()
txn_10w['time_decay'] = a / np.sqrt(txn_10w['days_ago'].astype(float)) + b * np.exp(-c * txn_10w['days_ago'].astype(float)) - d
txn_10w.loc[txn_10w['time_decay'] < 0, 'time_decay'] = 0

repurchase_scores = txn_10w.groupby(['customer_id', 'article_id'])['time_decay'].sum().reset_index()
repurchase_scores.columns = ['customer_id', 'article_id', 'score_repurchase']
repurchase_scores = repurchase_scores.sort_values(['customer_id', 'score_repurchase'], ascending=[True, False])
repurchase_top = repurchase_scores.groupby('customer_id').head(12)
print(f"    Repurchase candidates: {len(repurchase_top):,}")
print(f"    Users with repurchase: {repurchase_top['customer_id'].nunique():,}")

# Build per-customer repurchase lists (ordered by score)
repurchase_lists = (
    repurchase_top
    .sort_values(['customer_id', 'score_repurchase'], ascending=[True, False])
    .groupby('customer_id')['article_id']
    .apply(list)
    .to_dict()
)

# --- Channel 2: ItemCF with Direction Factor ---
print("\n  Channel 2: ItemCF with direction factor...")
cust_sequences = (
    txn_10w.sort_values('t_dat')
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
    sorted_related = sorted(related.items(), key=lambda x: -x[1])[:20]
    itemcf_lookup[item] = sorted_related

print(f"    Articles with ItemCF: {len(itemcf_lookup):,}")
del cust_sequences
gc.collect()

# Generate ItemCF candidates per customer
print("    Generating ItemCF candidates per customer...")
active_customers = set(repurchase_top['customer_id'].unique())
cust_recent_items = (
    txn[txn['week'] <= 4]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
    .to_dict()
)

# Build per-customer ItemCF recommendation lists
itemcf_lists = {}
for cid in active_customers:
    recent_items = cust_recent_items.get(cid, [])
    if not recent_items:
        continue
    # Collect ItemCF recommendations with scores
    cf_scores = defaultdict(float)
    for aid in recent_items[:5]:
        if aid in itemcf_lookup:
            for related_aid, score in itemcf_lookup[aid]:
                cf_scores[related_aid] += score
    # Sort by score, take top items
    sorted_cf = sorted(cf_scores.items(), key=lambda x: -x[1])
    itemcf_lists[cid] = [aid for aid, _ in sorted_cf]

print(f"    Users with ItemCF: {len(itemcf_lists):,}")

del sim_item
gc.collect()

# --- Channel 3: Same Product Code Variants ---
print("\n  Channel 3: Same product code variants...")
product_groups = articles.groupby('product_code')['article_id'].apply(list).to_dict()
product_lookup = {}
for pc, aids in product_groups.items():
    if len(aids) >= 2:
        for a_idx, a in enumerate(aids):
            product_lookup[a] = [x for x in aids if x != a][:3]
print(f"    Articles with variants: {len(product_lookup):,}")

# Build per-customer SPC lists
spc_lists = {}
for cid in active_customers:
    recent_items = cust_recent_items.get(cid, [])
    variants = []
    seen = set()
    for aid in recent_items[:5]:
        if aid in product_lookup:
            for variant in product_lookup[aid]:
                if variant not in seen:
                    variants.append(variant)
                    seen.add(variant)
    if variants:
        spc_lists[cid] = variants

print(f"    Users with SPC: {len(spc_lists):,}")

# --- Channel 4: Global Popular ---
print("\n  Channel 4: Time-decay global popular...")
pop_decay = txn_10w.groupby('article_id')['time_decay'].sum().sort_values(ascending=False)
pop_top50 = list(pop_decay.head(50).index)
print(f"    Popular articles: {len(pop_top50)}")

# --- Channel 5: Age-bin Popular ---
print("\n  Channel 5: Age-bin popular...")
age_pop = {}
for ab in range(6):
    ab_custs = set(customers[customers['age_bin'] == ab]['customer_id'])
    ab_txn = txn_10w[txn_10w['customer_id'].isin(ab_custs)]
    ab_pop = ab_txn.groupby('article_id')['time_decay'].sum().sort_values(ascending=False).head(30)
    age_pop[ab] = list(ab_pop.index)
print(f"    Age bins covered: {len(age_pop)}")

del txn_10w
gc.collect()

# ============================================================
# 3. LAYERED PREDICTION (Key Innovation)
# ============================================================
print("\n--- LAYERED PREDICTION: Repurchase-protected slot filling ---")

# Pre-compute for speed
pop12_str = ' '.join(f'{a:010d}' for a in pop_top50[:12])
predictions = {}
layer1_total = 0
layer2_total = 0

for cid in sample_sub['customer_id']:
    pred = []
    used = set()

    # LAYER 1: Repurchase (top priority, NEVER displaced)
    for aid in repurchase_lists.get(cid, []):
        if len(pred) >= 12:
            break
        pred.append(aid)
        used.add(aid)
    n_l1 = len(pred)

    # LAYER 2: ItemCF (fill empty slots)
    if n_l1 < 12:
        for aid in itemcf_lists.get(cid, []):
            if len(pred) >= 12:
                break
            if aid not in used:
                pred.append(aid)
                used.add(aid)

    # LAYER 2b: Same product code variants
    if len(pred) < 12:
        for aid in spc_lists.get(cid, []):
            if len(pred) >= 12:
                break
            if aid not in used:
                pred.append(aid)
                used.add(aid)

    n_l2 = len(pred) - n_l1

    # LAYER 3: Global popular (fill remaining)
    if len(pred) < 12:
        for aid in pop_top50:
            if len(pred) >= 12:
                break
            if aid not in used:
                pred.append(aid)
                used.add(aid)

    layer1_total += n_l1
    layer2_total += n_l2
    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

n_total = len(sample_sub)
print(f"  Total customers: {n_total:,}")
print(f"  Layer 1 (repurchase) avg: {layer1_total / n_total:.2f}")
print(f"  Layer 2 (ItemCF+SPC) avg: {layer2_total / n_total:.2f}")

# ============================================================
# 4. Save Submission
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r11_layered_recall.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

# ============================================================
# 5. Validation
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
print(f"Val customers with purchases: {len(val_truth):,}")

val_scores = []
for cid in val_truth:
    pred_str = predictions.get(cid, pop12_str)
    pred = [int(x) for x in pred_str.split()]
    val_scores.append(apk(val_truth[cid], pred, 12))

map12_active = np.mean(val_scores)
print(f"Val MAP@12 (active, with week 0 data): {map12_active:.6f}")

print("\n" + "=" * 60)
print("R11 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"R05 LB: 0.02224 (best) | R10 LB: 0.01973")
print("=" * 60)
