"""
H&M R04: Time-Decay Heuristics + Co-occurrence + Trending
Based on top solution analysis (silver medal 45th, 1st place patterns).

Key improvements over R01:
1. Time-decay weighted repurchase (recent purchases ranked higher)
2. Co-occurrence / frequent pairs (items bought together)
3. Same product code (different color/size variants)
4. Time-decay weighted popular (trending items)
5. Out-of-stock filter (items not sold in last 2 weeks)

Target: 0.022 → 0.024+
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
print("R04: Time-Decay Heuristics + Co-occurrence")
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

# Only keep last 12 weeks for efficiency (fashion trends change fast)
txn_full = txn.copy()  # keep full for validation
txn = txn[txn['week'] <= 12].copy()
gc.collect()

print(f"Transactions (12w): {len(txn):,}")
print(f"Max date: {max_date.date()}")

# Precompute article metadata
article_product_code = articles.set_index('article_id')['product_code'].to_dict()
article_dept = articles.set_index('article_id')['department_no'].to_dict()
article_section = articles.set_index('article_id')['section_no'].to_dict()
article_garment = articles.set_index('article_id')['garment_group_no'].to_dict()

# ============================================================
# 2. Out-of-Stock Filter
# ============================================================
print("\n--- Out-of-stock filter ---")
# Items sold in last 2 weeks are likely still available
recent_2w = set(txn[txn['week'] <= 2]['article_id'].unique())
all_articles_in_txn = set(txn['article_id'].unique())
print(f"Articles sold in last 2 weeks: {len(recent_2w):,}")
print(f"Total articles in 12w: {len(all_articles_in_txn):,}")
print(f"Filtered out (no recent sales): {len(all_articles_in_txn) - len(recent_2w):,}")

# ============================================================
# 3. Frequent Pairs (Co-occurrence)
# ============================================================
print("\n--- Frequent pairs (co-occurrence) ---")
# Bucket-based: same receipt = same day + customer + channel
recent_10w = txn[txn['week'] <= 10]
buckets = recent_10w.groupby(['t_dat', 'customer_id', 'sales_channel_id'])['article_id'].apply(set).reset_index()
buckets.columns = ['t_dat', 'customer_id', 'sales_channel_id', 'article_set']
buckets = buckets[buckets['article_set'].apply(len) > 1]
print(f"Multi-item receipts: {len(buckets):,}")

pair_counts = Counter()
for arts in buckets['article_set']:
    if len(arts) <= 10:
        for pair in combinations(arts, 2):
            pair_counts[pair] += 1

# Build lookup: article → list of related articles sorted by frequency
freq_pairs = defaultdict(list)
for (a, b), count in pair_counts.items():
    freq_pairs[a].append((b, count))
    freq_pairs[b].append((a, count))

# Keep top-12 related items per article
freq_pairs = {
    k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:12]]
    for k, v in freq_pairs.items()
}
print(f"Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts, recent_10w
gc.collect()

# ============================================================
# 4. Same Product Code Lookup
# ============================================================
print("\n--- Same product code lookup ---")
# Group articles by product_code (different colors/sizes)
product_code_groups = articles.groupby('product_code')['article_id'].apply(list).to_dict()
# Filter: only keep groups with 2+ articles and recent sales
product_code_lookup = {}
for pc, aids in product_code_groups.items():
    recent_aids = [a for a in aids if a in recent_2w]
    if len(recent_aids) >= 2:
        for a in recent_aids:
            product_code_lookup[a] = [x for x in recent_aids if x != a]
print(f"Articles with product code variants: {len(product_code_lookup):,}")

# ============================================================
# 5. Time-Decay Weighted Popular Articles
# ============================================================
print("\n--- Time-decay popular articles ---")
# Weight recent sales more heavily: decay factor per week
# Formula: weight = exp(-0.1 * week_number)  ≈ recent weeks dominate
def time_decay_popular(txn_df, n=50):
    """Compute time-decay weighted popularity."""
    txn_df = txn_df[txn_df['article_id'].isin(recent_2w)].copy()
    # Exponential decay: week 0 gets weight 1.0, week 1 gets 0.9, etc.
    txn_df['decay'] = np.exp(-0.15 * txn_df['week'].astype(float))
    weighted = txn_df.groupby('article_id')['decay'].sum().sort_values(ascending=False)
    return weighted.head(n).index.tolist()

trending_50 = time_decay_popular(txn, 50)
print(f"Trending top-50 (time-decay): first 5 = {trending_50[:5]}")

# Also compute per-department trending (for active customers)
dept_trending = {}
for dept in txn['article_id'].map(article_dept).dropna().unique():
    dept_articles = txn[txn['article_id'].map(article_dept) == dept]
    dept_trending[dept] = time_decay_popular(dept_articles, 20)

# ============================================================
# 6. Build Predictions
# ============================================================
print("\n--- Building predictions ---")

# Time-decay weighted repurchase for active customers
# Weight: 1 / (1 + days_ago), so recent purchases get much higher weight
def get_weighted_repurchase(customer_txns, max_items=24):
    """Get time-decay weighted repurchase candidates for a customer."""
    if len(customer_txns) == 0:
        return []

    # Score each article by recency-weighted frequency
    article_scores = defaultdict(float)
    for _, row in customer_txns.iterrows():
        days_ago = (max_date - row['t_dat']).days
        weight = 1.0 / (1.0 + days_ago)  # time decay
        article_scores[row['article_id']] += weight

    # Sort by score descending, filter out-of-stock
    sorted_arts = sorted(article_scores.items(), key=lambda x: -x[1])
    result = [a for a, s in sorted_arts if a in recent_2w][:max_items]
    return result

# Precompute active customers and their recent purchases
last_10w_customers = txn[txn['week'] <= 10]
active_customers = last_10w_customers['customer_id'].unique()
print(f"Active customers (last 10 weeks): {len(active_customers):,}")

# Build per-customer purchase history (sorted by recency)
cust_history = (
    txn[txn['week'] <= 10]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:30])  # dedup, keep order, up to 30
    .to_dict()
)

# Compute weighted repurchase scores using vectorized operations
print("Computing time-decay repurchase scores...")
# Precompute time-decay weight per transaction row
txn_10w = txn[txn['week'] <= 10].copy()
txn_10w['days_ago'] = (max_date - txn_10w['t_dat']).dt.days
txn_10w['decay_weight'] = 1.0 / (1.0 + txn_10w['days_ago'].astype(float))

# Filter to in-stock articles only
txn_10w = txn_10w[txn_10w['article_id'].isin(recent_2w)]

# Group by (customer, article) and sum weights
cust_art_scores = txn_10w.groupby(['customer_id', 'article_id'])['decay_weight'].sum().reset_index()
cust_art_scores = cust_art_scores.sort_values(['customer_id', 'decay_weight'], ascending=[True, False])

# Take top-24 per customer
cust_weighted = (
    cust_art_scores.groupby('customer_id')['article_id']
    .apply(lambda x: x.tolist()[:24])
    .to_dict()
)
del txn_10w, cust_art_scores
gc.collect()
print(f"Customers with weighted repurchase: {len(cust_weighted):,}")

# Global popular fallback string
popular_str = ' '.join(f'{a:010d}' for a in trending_50[:12])

# Build predictions for all submission customers
predictions = {}
stats = {'repurchase': 0, 'pairs': 0, 'product_code': 0, 'dept_pop': 0, 'fallback': 0}

for cid in sample_sub['customer_id']:
    pred = []

    if cid in cust_weighted and len(cust_weighted[cid]) > 0:
        # --- Active customer: multi-strategy ---

        # Strategy 1: Time-decay weighted repurchase
        repurchase = cust_weighted[cid][:12]
        pred.extend(repurchase)
        stats['repurchase'] += len(repurchase)

        # Strategy 2: Co-occurrence (items bought together with repurchase items)
        recent_top = cust_weighted[cid][:5]  # use top-5 for pair expansion
        pair_candidates = []
        for aid in recent_top:
            if aid in freq_pairs:
                for related in freq_pairs[aid]:
                    if related in recent_2w and related not in pred and related not in pair_candidates:
                        pair_candidates.append(related)
        if pair_candidates:
            remaining = 12 - len(pred)
            added = pair_candidates[:remaining]
            pred.extend(added)
            stats['pairs'] += len(added)

        # Strategy 3: Same product code variants (different color/size)
        if len(pred) < 12:
            for aid in cust_weighted[cid][:5]:
                if aid in product_code_lookup:
                    for variant in product_code_lookup[aid]:
                        if variant not in pred:
                            pred.append(variant)
                            stats['product_code'] += 1
                            if len(pred) >= 12:
                                break
                if len(pred) >= 12:
                    break

        # Strategy 4: Department-level trending
        if len(pred) < 12:
            # Get customer's preferred departments
            cust_depts = [article_dept.get(a, -1) for a in cust_weighted[cid][:10]]
            if cust_depts:
                dept_counts = Counter(d for d in cust_depts if d != -1)
                top_depts = [d for d, _ in dept_counts.most_common(3)]
                for d in top_depts:
                    if d in dept_trending:
                        for aid in dept_trending[d]:
                            if aid not in pred:
                                pred.append(aid)
                                if len(pred) >= 12:
                                    break
                    if len(pred) >= 12:
                        break
                stats['dept_pop'] += max(0, len(pred) - 12 + len([1 for a in pred[12-len(pred):] if a not in cust_weighted.get(cid, [])]))

        # Strategy 5: Fill with global trending
        if len(pred) < 12:
            for aid in trending_50:
                if aid not in pred:
                    pred.append(aid)
                    if len(pred) >= 12:
                        break
    else:
        # --- Inactive customer: global trending ---
        pred = trending_50[:12]
        stats['fallback'] += 1

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

# ============================================================
# 7. Save Submission
# ============================================================
print("\n--- Saving submission ---")
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r04_time_decay.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")
print(f"Total: {len(predictions):,}")

# Print strategy stats
print(f"\nStrategy stats:")
print(f"  Repurchase items: {stats['repurchase']:,}")
print(f"  Co-occurrence items: {stats['pairs']:,}")
print(f"  Product code variants: {stats['product_code']:,}")
print(f"  Dept trending items: {stats['dept_pop']:,}")
print(f"  Fallback customers: {stats['fallback']:,}")

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

# Validate on week 0 (last week of training data)
val_purchases = txn[txn['week'] == 0]
val_truth = val_purchases.groupby('customer_id')['article_id'].apply(set).to_dict()
print(f"Val week 0 purchases: {len(val_purchases):,}")
print(f"Val customers with purchases: {len(val_truth):,}")

# Build validation predictions using data before week 0
val_txn = txn[txn['week'] > 0]  # exclude week 0

# Val repurchase (weighted)
val_cust_history = (
    val_txn[val_txn['week'] <= 10]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:30])
    .to_dict()
)

# Val time-decay popular
val_popular = []
val_txn_filtered = val_txn[val_txn['article_id'].isin(recent_2w)].copy()
val_txn_filtered['decay'] = np.exp(-0.15 * val_txn_filtered['week'].astype(float))
val_weighted = val_txn_filtered.groupby('article_id')['decay'].sum().sort_values(ascending=False)
val_popular = val_weighted.head(50).index.tolist()

# Val frequent pairs (recompute from val data)
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
val_freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:12]] for k, v in val_freq_pairs.items()}

# Val predictions
val_scores = []
for cid in val_truth:
    actual = val_truth[cid]
    recent_arts = val_cust_history.get(cid, [])

    if not recent_arts:
        pred = val_popular[:12]
    else:
        # Time-decay repurchase
        scored = []
        for aid in recent_arts:
            if aid in recent_2w:
                scored.append(aid)
        pred = scored[:12]

        # Co-occurrence
        if len(pred) < 12:
            for aid in scored[:5]:
                if aid in val_freq_pairs:
                    for related in val_freq_pairs[aid]:
                        if related in recent_2w and related not in pred:
                            pred.append(related)
                            if len(pred) >= 12:
                                break
                if len(pred) >= 12:
                    break

        # Product code variants
        if len(pred) < 12:
            for aid in scored[:5]:
                if aid in product_code_lookup:
                    for variant in product_code_lookup[aid]:
                        if variant not in pred:
                            pred.append(variant)
                            if len(pred) >= 12:
                                break
                if len(pred) >= 12:
                    break

        # Fill with val popular
        if len(pred) < 12:
            for aid in val_popular:
                if aid not in pred:
                    pred.append(aid)
                    if len(pred) >= 12:
                        break

    val_scores.append(apk(actual, pred, 12))

map12 = np.mean(val_scores)
print(f"Val MAP@12: {map12:.6f}")

# Also compute on all submission customers
all_scores = []
val_pred_cids = set(val_truth.keys())
for cid in sample_sub['customer_id']:
    if cid in val_truth:
        # Same prediction as above (approximate — use full prediction logic)
        recent_arts = val_cust_history.get(cid, [])
        if recent_arts:
            scored = [a for a in recent_arts if a in recent_2w]
            pred = scored[:12]
            for aid in scored[:5]:
                if aid in val_freq_pairs:
                    for related in val_freq_pairs[aid]:
                        if related in recent_2w and related not in pred:
                            pred.append(related)
                            if len(pred) >= 12:
                                break
                if len(pred) >= 12:
                    break
            for aid in val_popular:
                if aid not in pred:
                    pred.append(aid)
                    if len(pred) >= 12:
                        break
        else:
            pred = val_popular[:12]
        all_scores.append(apk(val_truth[cid], pred, 12))
    else:
        # Inactive → popular fallback
        all_scores.append(apk(set(), val_popular[:12]))  # 0.0

map12_all = np.mean(all_scores)
print(f"Val MAP@12 (all customers): {map12_all:.6f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("R04 Complete")
print(f"Val MAP@12 (active): {map12:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"R01 LB: 0.02207 | R02 LB: 0.01800 | R03 LB: 0.01712")
print("=" * 60)
