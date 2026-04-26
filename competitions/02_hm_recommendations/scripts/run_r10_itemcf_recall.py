"""
H&M R10: Two-Stage Pipeline with ItemCF + Recall Score Features
Based on winning solutions analysis from competition wiki.

KEY INSIGHT from 3rd place team:
- Adding recall scores as ranking features boosted from 0.02855 → 0.03262 (+14%)
- ItemCF with direction factor captures sequential purchase intent
- Time-decay formula: a/sqrt(x) + b*exp(-c*x) - d

Architecture:
  Stage 1: Multi-channel recall (Repurchase, ItemCF, TimeDecayPopular, SaleTrend)
  Stage 2: Simple scoring using recall scores (no GBDT yet — validate candidates first)

Target: 0.022 → 0.024+
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
print("R10: Two-Stage Pipeline — ItemCF + Recall Scores")
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

# Article metadata
article_product_code = articles.set_index('article_id')['product_code'].to_dict()
article_dept = articles.set_index('article_id')['department_no'].to_dict()

# ============================================================
# 2. STAGE 1: Multi-Channel Recall
# ============================================================
print("\n--- STAGE 1: Multi-channel recall ---")

# --- Channel 1: Repurchase with Time-Decay (Silver Medal Formula) ---
print("\n  Channel 1: Repurchase with time-decay...")
a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3

txn_10w = txn[txn['week'] <= 10].copy()
txn_10w['time_decay'] = a / np.sqrt(txn_10w['days_ago'].astype(float)) + b * np.exp(-c * txn_10w['days_ago'].astype(float)) - d
txn_10w.loc[txn_10w['time_decay'] < 0, 'time_decay'] = 0

# Aggregate per (customer, article)
repurchase_scores = txn_10w.groupby(['customer_id', 'article_id'])['time_decay'].sum().reset_index()
repurchase_scores.columns = ['customer_id', 'article_id', 'score_repurchase']
repurchase_scores = repurchase_scores.sort_values(['customer_id', 'score_repurchase'], ascending=[True, False])
print(f"    Repurchase candidates: {len(repurchase_scores):,}")

# Top-N per customer (50 candidates)
repurchase_top = repurchase_scores.groupby('customer_id').head(50)
print(f"    Users with repurchase: {repurchase_top['customer_id'].nunique():,}")

# --- Channel 2: ItemCF with Direction Factor ---
print("\n  Channel 2: ItemCF with direction factor...")
# Build purchase sequences per customer (ordered by time)
cust_sequences = (
    txn_10w.sort_values('t_dat')
    .groupby('customer_id')['article_id']
    .apply(list)
    .to_dict()
)

# Compute ItemCF similarity with direction factor + distance decay
sim_item = defaultdict(lambda: defaultdict(float))
item_cnt = defaultdict(int)

for cid, items in cust_sequences.items():
    n_items = len(items)
    if n_items < 2:
        continue
    log_len = math.log(1 + n_items)

    for i, item_i in enumerate(items):
        item_cnt[item_i] += 1
        for j, item_j in enumerate(items):
            if i == j:
                continue
            # Direction factor: forward (j > i) gets 1.0, backward gets 0.9
            loc_alpha = 1.0 if j > i else 0.9
            # Distance decay: 0.7^(|j-i|-1)
            loc_weight = loc_alpha * (0.7 ** (abs(j - i) - 1))
            # Popularity penalty: divide by log(1 + sequence_length)
            sim_item[item_i][item_j] += loc_weight / log_len

# Keep top-20 similar items per article
itemcf_lookup = {}
for item, related in sim_item.items():
    sorted_related = sorted(related.items(), key=lambda x: -x[1])[:20]
    itemcf_lookup[item] = sorted_related  # list of (item, score)

print(f"    Articles with ItemCF: {len(itemcf_lookup):,}")
del cust_sequences
gc.collect()

# Generate ItemCF candidates for all active customers
print("    Generating ItemCF candidates...")
active_customers = set(repurchase_top['customer_id'].unique())
# Get each customer's recently bought items (last 4 weeks)
cust_recent_items = (
    txn[txn['week'] <= 4]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
    .to_dict()
)

itemcf_rows = []
for cid in active_customers:
    recent_items = cust_recent_items.get(cid, [])
    if not recent_items:
        continue
    for aid in recent_items[:5]:
        if aid in itemcf_lookup:
            for related_aid, score in itemcf_lookup[aid]:
                itemcf_rows.append({
                    'customer_id': cid,
                    'article_id': related_aid,
                    'score_itemcf': score
                })

itemcf_candidates = pd.DataFrame(itemcf_rows)
if len(itemcf_candidates) > 0:
    # Aggregate: sum scores per (customer, article)
    itemcf_candidates = itemcf_candidates.groupby(['customer_id', 'article_id'])['score_itemcf'].sum().reset_index()
    itemcf_candidates = itemcf_candidates.sort_values(['customer_id', 'score_itemcf'], ascending=[True, False])
    itemcf_top = itemcf_candidates.groupby('customer_id').head(30)
    print(f"    ItemCF candidates: {len(itemcf_top):,}")
else:
    itemcf_top = pd.DataFrame(columns=['customer_id', 'article_id', 'score_itemcf'])
    print("    ItemCF candidates: 0")

del itemcf_rows, sim_item
gc.collect()

# --- Channel 3: Same Product Code (Exchange Behavior) ---
print("\n  Channel 3: Same product code variants...")
product_groups = articles.groupby('product_code')['article_id'].apply(list).to_dict()
product_lookup = {}
for pc, aids in product_groups.items():
    if len(aids) >= 2:
        for a in aids:
            product_lookup[a] = [x for x in aids if x != a][:3]
print(f"    Articles with variants: {len(product_lookup):,}")

# Generate same-product-code candidates
spc_rows = []
for cid in active_customers:
    recent_items = cust_recent_items.get(cid, [])
    for aid in recent_items[:5]:
        if aid in product_lookup:
            for variant in product_lookup[aid]:
                spc_rows.append({'customer_id': cid, 'article_id': variant, 'score_spc': 1.0})

spc_candidates = pd.DataFrame(spc_rows)
if len(spc_candidates) > 0:
    spc_candidates = spc_candidates.drop_duplicates()
    print(f"    Same-product-code candidates: {len(spc_candidates):,}")
else:
    spc_candidates = pd.DataFrame(columns=['customer_id', 'article_id', 'score_spc'])

del spc_rows
gc.collect()

# --- Channel 4: Time-Decay Global Popular ---
print("\n  Channel 4: Time-decay global popular...")
pop_decay = txn_10w.groupby('article_id')['time_decay'].sum().sort_values(ascending=False)
pop_top50 = pop_decay.head(50).reset_index()
pop_top50.columns = ['article_id', 'score_popular']
print(f"    Popular articles: {len(pop_top50):,}")

# --- Channel 5: Sale Trend (accelerating popularity) ---
print("\n  Channel 5: Sale trend...")
last_14d = txn[txn['days_ago'] <= 13]
last_7d_counts = last_14d[last_14d['days_ago'] <= 6]['article_id'].value_counts().reset_index()
last_7d_counts.columns = ['article_id', 'count_recent']
prev_7d_counts = last_14d[last_14d['days_ago'] > 6]['article_id'].value_counts().reset_index()
prev_7d_counts.columns = ['article_id', 'count_prev']

trend = last_7d_counts.merge(prev_7d_counts, on='article_id', how='left')
trend['count_prev'] = trend['count_prev'].fillna(1)
trend['trend'] = (trend['count_recent'] - trend['count_prev']) / trend['count_prev']
trending = trend[trend['trend'] > 0.5].sort_values('trend', ascending=False).head(30)
trending['score_trend'] = trending['trend']
print(f"    Trending articles: {len(trending):,}")

# --- Channel 6: Age-Bin Popular ---
print("\n  Channel 6: Age-bin popular...")
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
# 3. STAGE 2: Score & Rank Candidates
# ============================================================
print("\n--- STAGE 2: Score & rank candidates ---")

# Merge all candidate sources
all_candidates = repurchase_top.copy()

if len(itemcf_top) > 0:
    all_candidates = all_candidates.merge(
        itemcf_top, on=['customer_id', 'article_id'], how='outer'
    )
else:
    all_candidates['score_itemcf'] = 0.0

all_candidates = all_candidates.merge(
    spc_candidates, on=['customer_id', 'article_id'], how='outer'
)

# Fill NaN scores with 0
for col in ['score_repurchase', 'score_itemcf', 'score_spc']:
    if col in all_candidates.columns:
        all_candidates[col] = all_candidates[col].fillna(0)

# Deduplicate
all_candidates = all_candidates.drop_duplicates(subset=['customer_id', 'article_id'])
print(f"  Total unique candidates: {len(all_candidates):,}")
print(f"  Users with candidates: {all_candidates['customer_id'].nunique():,}")

# Compute composite score: weighted sum of recall scores
# Weights tuned from wiki analysis:
# - Repurchase is strongest (weight 3.0)
# - ItemCF is second (weight 2.0)
# - Same product code (weight 1.0)
all_candidates['composite_score'] = (
    all_candidates['score_repurchase'] * 3.0 +
    all_candidates['score_itemcf'] * 2.0 +
    all_candidates['score_spc'] * 1.0
)

# Add popular/trend scores for items in global lists
pop_score_map = dict(zip(pop_top50['article_id'], pop_top50['score_popular']))
trend_score_map = dict(zip(trending['article_id'], trending['score_trend']))

all_candidates['score_popular'] = all_candidates['article_id'].map(pop_score_map).fillna(0)
all_candidates['score_trend'] = all_candidates['article_id'].map(trend_score_map).fillna(0)

# Final score with popular/trend bonus
all_candidates['final_score'] = (
    all_candidates['composite_score'] +
    all_candidates['score_popular'] * 0.5 +
    all_candidates['score_trend'] * 0.3
)

# ============================================================
# 4. Build Predictions
# ============================================================
print("\n--- Building predictions ---")

# For users with candidates: take top 12 by final_score
pred_with_candidates = (
    all_candidates
    .sort_values(['customer_id', 'final_score'], ascending=[True, False])
    .groupby('customer_id')
    .head(12)
)

user_preds = pred_with_candidates.groupby('customer_id')['article_id'].apply(list).to_dict()

# Global popular string for fallback
pop12_str = ' '.join(f'{a:010d}' for a in pop_top50['article_id'].head(12))

# Build submission
predictions = {}
for cid in sample_sub['customer_id']:
    if cid in user_preds:
        pred = user_preds[cid][:12]
        # Fill remaining slots with age-bin popular + global popular
        if len(pred) < 12:
            ab = cust_age.get(cid, 3)
            for aid in age_pop.get(ab, []):
                if aid not in pred:
                    pred.append(aid)
                    if len(pred) >= 12:
                        break
        if len(pred) < 12:
            for aid in pop_top50['article_id']:
                if aid not in pred:
                    pred.append(aid)
                    if len(pred) >= 12:
                        break
        predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])
    else:
        # Fallback: age-bin popular + global popular
        ab = cust_age.get(cid, 3)
        pred = []
        for aid in age_pop.get(ab, []):
            pred.append(aid)
            if len(pred) >= 6:
                break
        for aid in pop_top50['article_id']:
            if aid not in pred:
                pred.append(aid)
                if len(pred) >= 12:
                    break
        predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

# ============================================================
# 5. Save Submission
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r10_itemcf_recall.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active_count = len(user_preds)
print(f"With candidates: {active_count:,}, Fallback: {len(sample_sub) - active_count:,}")

# Print strategy stats
has_repurchase = (all_candidates['score_repurchase'] > 0).sum()
has_itemcf = (all_candidates['score_itemcf'] > 0).sum()
has_spc = (all_candidates['score_spc'] > 0).sum()
print(f"\nCandidate breakdown:")
print(f"  With repurchase score: {has_repurchase:,}")
print(f"  With ItemCF score: {has_itemcf:,}")
print(f"  With same-product-code: {has_spc:,}")

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
print(f"Val customers with purchases: {len(val_truth):,}")

# Quick validation: use predictions we already built (they use week 0 data for candidates)
# So this is a rough upper bound. Real validation would need to exclude week 0.
# Score on val customers only
val_scores = []
for cid in val_truth:
    pred_str = predictions.get(cid, pop12_str)
    pred = [int(x) for x in pred_str.split()]
    val_scores.append(apk(val_truth[cid], pred, 12))

map12_active = np.mean(val_scores)
print(f"Val MAP@12 (active, with week 0 data): {map12_active:.6f}")
print("(Note: inflated because candidates include week 0 repurchases)")

print("\n" + "=" * 60)
print("R10 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"R05 LB: 0.02224 (best) | R01 LB: 0.02207")
print("=" * 60)
