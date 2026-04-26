"""
H&M R08: R05 + ALS Matrix Factorization for Cold-Start
R05 is best at 0.02224. The main gap is cold-start (80% inactive customers).

ALS from implicit library:
- Build user-item interaction matrix from transactions
- Train ALS model to learn latent factors
- For inactive customers: use ALS to recommend based on learned factors
- For active customers: keep R05's repurchase + popular

Target: Improve inactive customer predictions (the 80% that get popular fallback)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
import implicit
from collections import defaultdict
import gc
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("=" * 60)
print("R08: R05 + ALS Matrix Factorization")
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

# ============================================================
# 2. Build User-Item Matrix for ALS
# ============================================================
print("\n--- Building user-item matrix ---")
# Use time-decay weighted interactions
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))

# Create user and item ID mappings (0-based index for sparse matrix)
# Only use customers/articles that appear in transactions
all_customers = txn['customer_id'].unique()
all_articles = txn['article_id'].unique()

cust_to_idx = {c: i for i, c in enumerate(all_customers)}
idx_to_cust = {i: c for c, i in cust_to_idx.items()}
art_to_idx = {a: i for i, a in enumerate(all_articles)}
idx_to_art = {i: a for i, a in art_to_idx.items()}

print(f"Total customers: {len(cust_to_idx):,}")
print(f"Total articles: {len(art_to_idx):,}")

# Aggregate interactions: sum of time-decay weights per (customer, article)
interactions = txn.groupby(['customer_id', 'article_id'])['decay'].sum().reset_index()
interactions['user_idx'] = interactions['customer_id'].map(cust_to_idx)
interactions['item_idx'] = interactions['article_id'].map(art_to_idx)

# Build sparse matrix
user_items = csr_matrix(
    (interactions['decay'].values, (interactions['user_idx'].values, interactions['item_idx'].values)),
    shape=(len(cust_to_idx), len(art_to_idx))
)
print(f"Sparse matrix shape: {user_items.shape}")
print(f"Non-zero entries: {user_items.nnz:,}")
del interactions
gc.collect()

# ============================================================
# 3. Train ALS
# ============================================================
print("\n--- Training ALS ---")
model = implicit.als.AlternatingLeastSquares(
    factors=128,
    regularization=0.01,
    iterations=20,
    use_gpu=False,
    random_state=42,
)
# implicit 0.7.x: fit expects user_items matrix (users × items)
model.fit(user_items)
print("ALS training complete")

# ============================================================
# 4. R05 Components (Repurchase + Time-Decay Popular)
# ============================================================
print("\n--- R05 components ---")
cust_repurchase = (
    txn[txn['week'] <= 6]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:16])
    .to_dict()
)

# Time-decay popular
txn_copy = txn.copy()
txn_copy['decay'] = np.exp(-0.15 * txn_copy['week'].astype(float))
global_pop = txn_copy.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_12 = global_pop.head(12).index.tolist()
pop_50 = global_pop.head(50).index.tolist()
print(f"Active customers (6w): {len(cust_repurchase):,}")

# ============================================================
# 5. Co-occurrence (Conservative, from R05)
# ============================================================
print("\n--- Co-occurrence ---")
from itertools import combinations
from collections import Counter

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

# Submission customers
sub_customers = sample_sub['customer_id'].tolist()
active_set = set(cust_repurchase.keys())

# Pre-compute ALS recommendations for all customers
print("Computing ALS recommendations...")
als_recs = {}
batch_size = 1000
for i in range(0, len(sub_customers), batch_size):
    batch = sub_customers[i:i+batch_size]
    for cid in batch:
        if cid not in active_set and cid in cust_to_idx:
            # Inactive customer: use ALS recommendations
            user_idx = cust_to_idx[cid]
            ids, scores = model.recommend(user_idx, user_items[user_idx], N=12, filter_already_liked_items=False)
            als_recs[cid] = [idx_to_art[int(idx)] for idx in ids if int(idx) in idx_to_art]
print(f"ALS recommendations for inactive: {len(als_recs):,}")
del model
gc.collect()

# Build final predictions
for cid in sub_customers:
    if cid in cust_repurchase:
        # Active: R05 strategy (repurchase + co-occurrence + time-decay popular)
        pred = cust_repurchase[cid][:12]

        # Co-occurrence (max 2 items, only if < 12)
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

    elif cid in als_recs:
        # Inactive but has transaction history: use ALS
        als_pred = als_recs[cid][:12]
        predictions[cid] = ' '.join(f'{a:010d}' for a in als_pred)
    else:
        # Completely cold: global popular
        predictions[cid] = ' '.join(f'{a:010d}' for a in pop_12)

# ============================================================
# 7. Save Submission
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r08_als.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active_count = sum(1 for cid in sub_customers if cid in active_set)
als_count = sum(1 for cid in sub_customers if cid not in active_set and cid in als_recs)
cold_count = len(sub_customers) - active_count - als_count
print(f"Active (R05): {active_count:,} | ALS: {als_count:,} | Cold popular: {cold_count:,}")

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

val_truth = txn[txn['week'] == 0].groupby('customer_id')['article_id'].apply(set).to_dict()
print(f"Val customers with purchases: {len(val_truth):,}")

# How many of the val customers are active vs ALS?
val_active = sum(1 for cid in val_truth if cid in active_set)
val_als = sum(1 for cid in val_truth if cid not in active_set and cid in als_recs)
val_cold = len(val_truth) - val_active - val_als
print(f"Val active: {val_active:,} | ALS: {val_als:,} | Cold: {val_cold:,}")

# Score on all submission customers
all_scores = []
active_scores = []
als_scores = []

for cid in sub_customers:
    pred_str = predictions[cid]
    pred = [int(x) for x in pred_str.split()]

    if cid in val_truth:
        actual = val_truth[cid]
        score = apk(actual, pred, 12)
        all_scores.append(score)
        if cid in active_set:
            active_scores.append(score)
        elif cid in als_recs:
            als_scores.append(score)
    else:
        all_scores.append(apk(set(), pred))

map12_all = np.mean(all_scores)
map12_active = np.mean(active_scores) if active_scores else 0
map12_als = np.mean(als_scores) if als_scores else 0

print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (ALS inactive): {map12_als:.6f}")

print("\n" + "=" * 60)
print("R08 Complete")
print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (ALS inactive): {map12_als:.6f}")
print(f"R01 LB: 0.02207 | R05 LB: 0.02224")
print("=" * 60)
