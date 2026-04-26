"""
H&M R06: Silver Medal Pattern — Better Time-Decay + Personalized Popular

R05 improved to 0.02224 (from 0.02207). Key improvements to try:
1. Time-decay formula from silver medal: 1/sqrt(days_ago) — sharper decay
2. Personalized popular by customer segment (age bin × region)
3. Repurchase with recency-weighted scoring
4. Last-7-days trending as separate signal

Based on 45th place silver medal solution patterns.
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
print("R06: Silver Medal — Better Time-Decay + Personalized Popular")
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
txn = txn[txn['week'] <= 12].copy()
gc.collect()
print(f"Transactions (12w): {len(txn):,}")

# Customer features
customers['age'] = customers['age'].fillna(customers['age'].median())
customers['age_bin'] = pd.cut(customers['age'], bins=[0, 21, 26, 31, 36, 46, 100], labels=False).fillna(3).astype(int)
customers['postal_cluster'] = customers['postal_code'].astype('category').cat.codes
cust_age = customers.set_index('customer_id')['age_bin'].to_dict()
cust_postal = customers.set_index('customer_id')['postal_cluster'].to_dict()

# Article metadata
article_product_code = articles.set_index('article_id')['product_code'].to_dict()
article_dept = articles.set_index('article_id')['department_no'].to_dict()
article_section = articles.set_index('article_id')['section_no'].to_dict()

# ============================================================
# 2. Time-Decay Weighted Repurchase (Silver Medal Formula)
# ============================================================
print("\n--- Repurchase with silver medal time-decay ---")
# Formula: weight = 1/sqrt(days_ago + 1) × purchase_count
# This gives sharper recency emphasis than 1/(1+days)
txn_10w = txn[txn['week'] <= 10].copy()
txn_10w['days_ago'] = (max_date - txn_10w['t_dat']).dt.days
txn_10w['recency_weight'] = 1.0 / np.sqrt(txn_10w['days_ago'].astype(float) + 1)

# Score = sum of recency weights per (customer, article)
cust_art_scores = txn_10w.groupby(['customer_id', 'article_id'])['recency_weight'].sum().reset_index()
cust_art_scores = cust_art_scores.sort_values(['customer_id', 'recency_weight'], ascending=[True, False])

# Top-24 per customer
cust_weighted = (
    cust_art_scores.groupby('customer_id')['article_id']
    .apply(lambda x: x.tolist()[:24])
    .to_dict()
)
del txn_10w, cust_art_scores
gc.collect()
print(f"Customers with weighted repurchase: {len(cust_weighted):,}")

# ============================================================
# 3. Time-Decay Popular (Global)
# ============================================================
print("\n--- Time-decay popular ---")
txn_copy = txn.copy()
txn_copy['decay'] = np.exp(-0.15 * txn_copy['week'].astype(float))
global_pop = txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
global_pop_12 = global_pop.head(12).index.tolist()
global_pop_50 = global_pop.head(50).index.tolist()

# Also compute last-7-days trending (for recency signal)
last_week = txn[txn['week'] <= 1]
trending_7d = last_week['article_id'].value_counts().head(20).index.tolist()
print(f"Global pop (time-decay) top-5: {global_pop_12[:5]}")
print(f"Last-7d trending top-5: {trending_7d[:5]}")

# ============================================================
# 4. Personalized Popular by Age Bin
# ============================================================
print("\n--- Personalized popular by age bin ---")
age_pop = {}
for ab in range(6):
    ab_customers = set(customers[customers['age_bin'] == ab]['customer_id'])
    ab_txn = txn[txn['customer_id'].isin(ab_customers)]
    ab_txn_copy = ab_txn.copy()
    ab_txn_copy['decay'] = np.exp(-0.15 * ab_txn_copy['week'].astype(float))
    ab_pop = ab_txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
    age_pop[ab] = ab_pop.head(30).index.tolist()
    print(f"  Age bin {ab}: {len(ab_customers):,} customers, top article = {age_pop[ab][0]}")

# ============================================================
# 5. Co-occurrence (Conservative, Top-3)
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
freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:3]] for k, v in freq_pairs.items()}
print(f"Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts
gc.collect()

# ============================================================
# 6. Build Predictions
# ============================================================
print("\n--- Building predictions ---")
predictions = {}

for cid in sample_sub['customer_id']:
    pred = []

    if cid in cust_weighted:
        # Strategy 1: Time-decay weighted repurchase (up to 10 items)
        repurchase = cust_weighted[cid][:10]
        pred.extend(repurchase)

        # Strategy 2: Co-occurrence (max 2, only if repurchase < 12)
        remaining = 12 - len(pred)
        if remaining > 0 and len(repurchase) > 0:
            pair_added = 0
            for aid in repurchase[:3]:
                if aid in freq_pairs:
                    for related in freq_pairs[aid]:
                        if related not in pred:
                            pred.append(related)
                            pair_added += 1
                            if pair_added >= min(2, remaining):
                                break
                if pair_added >= 2:
                    break

        # Strategy 3: Fill with age-segment popular
        remaining = 12 - len(pred)
        if remaining > 0:
            ab = cust_age.get(cid, 3)
            for aid in age_pop.get(ab, global_pop_50):
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break

        # Strategy 4: Global popular as final fill
        remaining = 12 - len(pred)
        if remaining > 0:
            for aid in global_pop_50:
                if aid not in pred:
                    pred.append(aid)
                    remaining -= 1
                    if remaining <= 0:
                        break
    else:
        # Inactive: Use age-segment popular + global popular
        ab = cust_age.get(cid, 3)
        age_pred = age_pop.get(ab, global_pop_12)[:6]
        pred = age_pred.copy()
        for aid in global_pop_50:
            if aid not in pred:
                pred.append(aid)
                if len(pred) >= 12:
                    break

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

# ============================================================
# 7. Save Submission
# ============================================================
print("\n--- Saving submission ---")
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r06_personalized.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")

active = sum(1 for cid in sample_sub['customer_id'] if cid in cust_weighted)
print(f"Active: {active:,}, Fallback: {len(sample_sub) - active:,}")

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

val_txn = txn[txn['week'] > 0]

# Val weighted repurchase
val_txn_10w = val_txn[val_txn['week'] <= 10].copy()
val_txn_10w['days_ago'] = (max_date - val_txn_10w['t_dat']).dt.days
val_txn_10w['recency_weight'] = 1.0 / np.sqrt(val_txn_10w['days_ago'].astype(float) + 1)
val_scores = val_txn_10w.groupby(['customer_id', 'article_id'])['recency_weight'].sum().reset_index()
val_scores = val_scores.sort_values(['customer_id', 'recency_weight'], ascending=[True, False])
val_weighted = val_scores.groupby('customer_id')['article_id'].apply(lambda x: x.tolist()[:24]).to_dict()
del val_txn_10w, val_scores
gc.collect()

# Val popular
val_txn_copy = val_txn.copy()
val_txn_copy['decay'] = np.exp(-0.15 * val_txn_copy['week'].astype(float))
val_pop = val_txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
val_pop12 = val_pop.head(12).index.tolist()
val_pop50 = val_pop.head(50).index.tolist()

# Val age-segment popular
val_age_pop = {}
for ab in range(6):
    ab_customers = set(customers[customers['age_bin'] == ab]['customer_id'])
    ab_txn = val_txn[val_txn['customer_id'].isin(ab_customers)]
    ab_txn_copy = ab_txn.copy()
    ab_txn_copy['decay'] = np.exp(-0.15 * ab_txn_copy['week'].astype(float))
    ab_pop = ab_txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
    val_age_pop[ab] = ab_pop.head(30).index.tolist()

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
val_freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:3]] for k, v in val_freq_pairs.items()}

# Score all customers
all_scores = []
for cid in sample_sub['customer_id']:
    if cid in val_truth:
        actual = val_truth[cid]
        recent = val_weighted.get(cid, [])

        if recent:
            pred = recent[:10]
            remaining = 12 - len(pred)
            if remaining > 0:
                pair_added = 0
                for aid in recent[:3]:
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
                ab = cust_age.get(cid, 3)
                for aid in val_age_pop.get(ab, val_pop50):
                    if aid not in pred:
                        pred.append(aid)
                        remaining -= 1
                        if remaining <= 0:
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
            ab = cust_age.get(cid, 3)
            age_pred = val_age_pop.get(ab, val_pop12)[:6]
            pred = age_pred.copy()
            for aid in val_pop50:
                if aid not in pred:
                    pred.append(aid)
                    if len(pred) >= 12:
                        break
        all_scores.append(apk(actual, pred, 12))
    else:
        all_scores.append(apk(set(), val_pop12))

map12_all = np.mean(all_scores)
active_scores = [s for i, s in enumerate(all_scores) if sample_sub['customer_id'].iloc[i] in val_truth]
map12_active = np.mean(active_scores)

print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")

print("\n" + "=" * 60)
print("R06 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"R01 LB: 0.02207 | R05 LB: 0.02224")
print("=" * 60)
