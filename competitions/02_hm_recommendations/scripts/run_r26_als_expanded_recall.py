"""
H&M R26: R21 + ALS + Expanded Recall Channels

R21 (0.02365) was the best GBDT model. R26 adds:
  - ALS Matrix Factorization recall channel
  - User-group popular (by age bins)
  - Sale trend items (acceleration detection)
  - Expanded feature set for new recall channels

Target: 0.0300+
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
print("R26: R21 + ALS + Expanded Recall Channels")
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
cust_age = customers.set_index('customer_id')['age'].to_dict()

# Article lookup tables
art_dept = articles.set_index('article_id')['department_no'].to_dict()
art_section = articles.set_index('article_id')['section_no'].to_dict()
art_garment = articles.set_index('article_id')['garment_group_no'].to_dict()
art_product = articles.set_index('article_id')['product_code'].to_dict()
art_category = articles.set_index('article_id')['index_group_no'].to_dict()
art_color = articles.set_index('article_id')['colour_group_code'].to_dict()

# ============================================================
# 2. Pre-compute Feature Data (from weeks 1+ for training)
# ============================================================
print("\n--- Pre-computing features ---")
txn_features = txn[txn['week'] >= 1].copy()  # Weeks 1-12 for feature building

# --- Item stats ---
print("  Item stats...")
item_total_buys = txn_features['article_id'].value_counts().to_dict()
item_unique_buyers = txn_features.groupby('article_id')['customer_id'].nunique().to_dict()

# CRITICAL: Recent week buys (R18's key feature that R19 removed)
item_recent_week_buys = txn_features[txn_features['week'] == 1]['article_id'].value_counts().to_dict()
item_last2w_buys = txn_features[txn_features['week'] <= 2]['article_id'].value_counts().to_dict()
item_last4w_buys = txn_features[txn_features['week'] <= 4]['article_id'].value_counts().to_dict()

# Sales trend: recent week vs average
item_weekly_avg = {aid: count / 12.0 for aid, count in item_total_buys.items()}
item_sales_trend = {}
for aid in item_recent_week_buys:
    avg = item_weekly_avg.get(aid, 1)
    item_sales_trend[aid] = item_recent_week_buys[aid] / max(avg, 0.1)

# Repurchase ratio
item_repurchase_ratio = {}
for aid in item_total_buys:
    buyers = item_unique_buyers.get(aid, 1)
    buys = item_total_buys[aid]
    item_repurchase_ratio[aid] = buyers / max(buys, 1)

# --- Age-demographic features (Polimi's #1) ---
print("  Age-demographic features...")
# Merge age into transactions
txn_with_age = txn_features.merge(
    customers[['customer_id', 'age']], on='customer_id', how='left'
)
item_user_mean_age = txn_with_age.groupby('article_id')['age'].mean().to_dict()
item_user_age_std = txn_with_age.groupby('article_id')['age'].std().fillna(0).to_dict()
del txn_with_age
gc.collect()

# --- Price features ---
print("  Price features...")
user_mean_price = txn_features.groupby('customer_id')['price'].mean().to_dict()
user_std_price = txn_features.groupby('customer_id')['price'].std().fillna(0).to_dict()
item_mean_price = txn_features.groupby('article_id')['price'].mean().to_dict()
item_std_price = txn_features.groupby('article_id')['price'].std().fillna(0).to_dict()

# --- Time features ---
print("  Time features...")
item_first_day = txn_features.groupby('article_id')['days_ago'].max().to_dict()  # oldest transaction
item_last_day = txn_features.groupby('article_id')['days_ago'].min().to_dict()   # most recent
item_freshness = {aid: max_date.day - item_first_day.get(aid, 0) for aid in item_first_day}

# --- Channel features ---
user_mean_channel = txn_features.groupby('customer_id')['sales_channel_id'].mean().to_dict()
item_mean_channel = txn_features.groupby('article_id')['sales_channel_id'].mean().to_dict()

# --- Recall channels (same as R21) ---
print("  Recall channels...")

# Time-decay popular
txn_features['decay'] = np.exp(-0.15 * txn_features['week'].astype(float))
pop_scores = txn_features.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50 = list(pop_scores.head(50).index)
pop_score_dict = pop_scores.to_dict()

# Repurchase (6-week, recency-ordered)
last_6w = txn_features[txn_features['week'] <= 7]
cust_repurchase = (
    last_6w
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
    .to_dict()
)
print(f"    Users with repurchase: {len(cust_repurchase):,}")

# ItemCF
cust_sequences = (
    txn_features[txn_features['week'] <= 11]
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
print(f"    Articles with ItemCF: {len(itemcf_lookup):,}")
del cust_sequences, sim_item
gc.collect()

# Co-occurrence
buckets = txn_features[txn_features['week'] <= 11].groupby(
    ['t_dat', 'customer_id', 'sales_channel_id'])['article_id'].apply(set).reset_index()
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
print(f"    Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts
gc.collect()

# ============================================================
# ALS Matrix Factorization Recall
# ============================================================
print("  ALS recall...")
import implicit
from scipy.sparse import coo_matrix

# Map customer_id and article_id to integer indices
# Use ALL data for ALS (including week 0) since ALS doesn't need a "target" concept
als_txn = txn[txn['week'] <= 8].copy()  # Last ~9 weeks for relevance
als_users = als_txn['customer_id'].unique()
als_items = als_txn['article_id'].unique()
user_to_idx = {u: i for i, u in enumerate(als_users)}
idx_to_user = {i: u for i, u in enumerate(als_users)}
item_to_idx = {it: i for i, it in enumerate(als_items)}
idx_to_item = {i: it for i, it in enumerate(als_items)}

# Build interaction matrix with time decay weighting
als_txn['als_weight'] = np.exp(-0.1 * als_txn['week'].astype(float))
user_item_weights = als_txn.groupby(['customer_id', 'article_id'])['als_weight'].sum().reset_index()
user_item_weights.columns = ['customer_id', 'article_id', 'weight']

row = user_item_weights['customer_id'].map(user_to_idx).values
col = user_item_weights['article_id'].map(item_to_idx).values
data = user_item_weights['weight'].values
interaction_matrix = coo_matrix((data, (row, col)),
                                shape=(len(als_users), len(als_items)))

# Train ALS
als_model = implicit.als.AlternatingLeastSquares(
    factors=128,
    iterations=15,
    regularization=0.01,
    random_state=42,
    num_threads=-1,
)
als_model.fit(interaction_matrix)

# Generate ALS recommendations for active users (limit to top-1000 items for speed)
pop_1000_items = set(als_txn['article_id'].value_counts().head(1000).index)
pop_1000_item_indices = [item_to_idx[it] for it in pop_1000_items if it in item_to_idx]
pop_1000_item_indices_set = set(pop_1000_item_indices)

# Store ALS recs per user
als_user_recs = {}
print(f"    Generating ALS recs for {len(user_to_idx):,} users...")
for uid, uidx in user_to_idx.items():
    # Get top-N ALS recommendations
    ids, scores = als_model.recommend(
        uidx,
        interaction_matrix.tocsr()[uidx],
        N=100,
        filter_already_liked_items=False,
    )
    # Filter to only popular items (for tractability)
    filtered = [(idx_to_item[int(i)], float(s)) for i, s in zip(ids, scores)
                if int(i) in pop_1000_item_indices_set]
    als_user_recs[uid] = filtered[:50]

print(f"    Users with ALS recs: {len(als_user_recs):,}")
del als_txn, user_item_weights, interaction_matrix
gc.collect()

# ============================================================
# User-group popular by age bins
# ============================================================
print("  User-group popular...")
# Create age bins
customers['age_bin'] = pd.cut(customers['age'], bins=[0, 20, 25, 30, 35, 40, 50, 100], labels=False)
cust_age_bin = customers.set_index('customer_id')['age_bin'].to_dict()

# Popular items per age bin (last 4 weeks)
recent_4w = txn[txn['week'] <= 4]
recent_4w_with_age = recent_4w.merge(customers[['customer_id', 'age_bin']], on='customer_id', how='left')
age_bin_popular = {}
for ab in range(7):
    ab_items = recent_4w_with_age[recent_4w_with_age['age_bin'] == ab]['article_id'].value_counts().head(50)
    age_bin_popular[ab] = list(ab_items.index)

del recent_4w, recent_4w_with_age
gc.collect()

# ============================================================
# Sale trend items (acceleration detection)
# ============================================================
print("  Sale trend items...")
week1_buys = txn[txn['week'] == 1]['article_id'].value_counts().to_dict()
week2_buys = txn[txn['week'] == 2]['article_id'].value_counts().to_dict()
week3_buys = txn[txn['week'] == 3]['article_id'].value_counts().to_dict()
# Items with strong positive trend (week1 > week2 > week3)
trending_items = []
for aid in set(week1_buys.keys()) | set(week2_buys.keys()):
    w1 = week1_buys.get(aid, 0)
    w2 = week2_buys.get(aid, 0)
    w3 = week3_buys.get(aid, 0)
    if w1 > w2 and w2 > w3 and w1 >= 10:  # Accelerating upward
        trending_items.append((aid, w1 / max(w3, 1)))
trending_items.sort(key=lambda x: -x[1])
trending_top50 = [x[0] for x in trending_items[:50]]
print(f"    Trending items: {len(trending_top50)}")

# User activity stats
user_total_buys = txn_features.groupby('customer_id').size().to_dict()
user_unique_items = txn_features.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_purchase_day = txn_features.groupby('customer_id')['days_ago'].min().to_dict()

# User-item interaction
user_item_buy_count = txn.groupby(['customer_id', 'article_id']).size().to_dict()
user_item_last_day = txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()

# Build per-customer recall
cust_recent_items = (
    txn_features[txn_features['week'] <= 5]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
    .to_dict()
)

itemcf_user_recs = {}
for cid, recent in cust_recent_items.items():
    cf_scores = defaultdict(float)
    for aid in recent[:5]:
        if aid in itemcf_lookup:
            for related_aid, score in itemcf_lookup[aid]:
                cf_scores[related_aid] += score
    itemcf_user_recs[cid] = sorted(cf_scores.items(), key=lambda x: -x[1])

cooc_user_recs = {}
for cid, recent in cust_recent_items.items():
    cooc_items = []
    seen = set()
    for aid in recent[:5]:
        if aid in freq_pairs:
            for related in freq_pairs[aid]:
                if related not in seen:
                    cooc_items.append(related)
                    seen.add(related)
    cooc_user_recs[cid] = cooc_items

# Ground truth for validation
val_truth = txn[txn['week'] == 0].groupby('customer_id')['article_id'].apply(set).to_dict()
print(f"  Val customers with purchases: {len(val_truth):,}")

del txn_features
gc.collect()

# ============================================================
# 3. Generate Training Data (week 0 users, like R21)
# ============================================================
print("\n--- Generating training data ---")

all_active = set(cust_repurchase.keys()) | set(cust_recent_items.keys()) | set(als_user_recs.keys())
print(f"  Active users: {len(all_active):,}")

train_rows = []
skipped = 0

for cid in all_active:
    if cid not in val_truth:
        skipped += 1
        continue

    actual = val_truth[cid]
    candidates = {}

    # Repurchase
    for rank, aid in enumerate(cust_repurchase.get(cid, [])[:16]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['repurchase_rank'] = rank
        candidates[aid]['repurchase_score'] = 1.0 / (1.0 + rank)

    # ItemCF
    for rank, (aid, score) in enumerate(itemcf_user_recs.get(cid, [])[:20]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['itemcf_rank'] = rank
        candidates[aid]['itemcf_score'] = score

    # Co-occurrence
    for rank, aid in enumerate(cooc_user_recs.get(cid, [])[:10]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['cooc_rank'] = rank

    # Popular
    for rank, aid in enumerate(pop_top50[:30]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['pop_rank'] = rank

    # ALS candidates
    for rank, (aid, score) in enumerate(als_user_recs.get(cid, [])[:50]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['als_rank'] = rank
        candidates[aid]['als_score'] = score

    # User-group popular (age-bin specific)
    ab = cust_age_bin.get(cid, 3)
    for rank, aid in enumerate(age_bin_popular.get(ab, [])[:30]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['user_group_pop_rank'] = rank

    # Sale trend items
    for rank, aid in enumerate(trending_top50[:20]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['trend_rank'] = rank

    # Build feature rows
    u_age = cust_age.get(cid, 35)
    u_mean_pr = user_mean_price.get(cid, 0)
    u_std_pr = user_std_price.get(cid, 0)
    u_mean_ch = user_mean_channel.get(cid, 1.5)

    for aid, feat in candidates.items():
        i_mean_age = item_user_mean_age.get(aid, 35)
        i_mean_pr = item_mean_price.get(aid, 0)

        row = {
            'customer_id': cid,
            'article_id': aid,
            'target': 1 if aid in actual else 0,
            # Recall features
            'repurchase_score': feat.get('repurchase_score', 0),
            'repurchase_rank': feat.get('repurchase_rank', 99),
            'itemcf_score': feat.get('itemcf_score', 0),
            'itemcf_rank': feat.get('itemcf_rank', 99),
            'cooc_rank': feat.get('cooc_rank', 99),
            'pop_rank': feat.get('pop_rank', 99),
            # ALS features
            'als_score': feat.get('als_score', 0),
            'als_rank': feat.get('als_rank', 99),
            'is_als': 1 if 'als_rank' in feat else 0,
            # User-group popular
            'user_group_pop_rank': feat.get('user_group_pop_rank', 99),
            'is_user_group_pop': 1 if 'user_group_pop_rank' in feat else 0,
            # Trending
            'trend_rank': feat.get('trend_rank', 99),
            'is_trending': 1 if 'trend_rank' in feat else 0,
            # Recall meta
            'is_repurchase': 1 if 'repurchase_rank' in feat else 0,
            'is_itemcf': 1 if 'itemcf_rank' in feat else 0,
            'is_cooc': 1 if 'cooc_rank' in feat else 0,
            'is_pop': 1 if 'pop_rank' in feat else 0,
            'num_recall_sources': sum(1 for k in ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank', 'als_rank', 'user_group_pop_rank', 'trend_rank'] if k in feat),
            # User-item interaction
            'user_item_buys': user_item_buy_count.get((cid, aid), 0),
            'user_item_last_day': user_item_last_day.get((cid, aid), 999),
            # User features
            'user_total_buys': user_total_buys.get(cid, 0),
            'user_unique_items': user_unique_items.get(cid, 0),
            'user_last_day': user_last_purchase_day.get(cid, 999),
            'user_age': u_age,
            # Item popularity
            'item_popularity': pop_score_dict.get(aid, 0),
            'item_total_buys': item_total_buys.get(aid, 0),
            'item_unique_buyers': item_unique_buyers.get(aid, 0),
            # Item category
            'item_dept': art_dept.get(aid, -1),
            'item_section': art_section.get(aid, -1),
            'item_garment': art_garment.get(aid, -1),
            'item_category': art_category.get(aid, -1),
            'item_color': art_color.get(aid, -1),
        }

        # === NEW FEATURES (R21 additions) ===

        # 1. Recent week buys (R18's key feature, restored)
        row['item_recent_week_buys'] = item_recent_week_buys.get(aid, 0)
        row['item_last2w_buys'] = item_last2w_buys.get(aid, 0)
        row['item_last4w_buys'] = item_last4w_buys.get(aid, 0)

        # 2. Sales trend (acceleration)
        row['item_sales_trend'] = item_sales_trend.get(aid, 1.0)

        # 3. Repurchase ratio
        row['item_repurchase_ratio'] = item_repurchase_ratio.get(aid, 1.0)

        # 4. Age-demographic match (Polimi's #1 feature)
        row['item_user_mean_age'] = i_mean_age
        row['user_item_age_diff'] = abs(u_age - i_mean_age)
        row['item_user_age_std'] = item_user_age_std.get(aid, 10)

        # 5. Price match
        row['user_mean_price'] = u_mean_pr
        row['item_mean_price'] = i_mean_pr
        row['user_item_price_diff'] = abs(u_mean_pr - i_mean_pr) if u_mean_pr > 0 and i_mean_pr > 0 else 999

        # 6. Item freshness (days since first appeared)
        row['item_freshness'] = item_freshness.get(aid, 0)
        row['item_last_day'] = item_last_day.get(aid, 999)

        # 7. Channel match
        row['user_mean_channel'] = u_mean_ch
        row['item_mean_channel'] = item_mean_channel.get(aid, 1.5)

        train_rows.append(row)

print(f"  Skipped (no val truth): {skipped:,}")
print(f"  Training rows: {len(train_rows):,}")

train_df = pd.DataFrame(train_rows)
del train_rows
gc.collect()

# Fill NaN
FILL_0 = ['repurchase_score', 'itemcf_score', 'als_score', 'item_popularity',
          'item_total_buys', 'item_unique_buyers', 'item_recent_week_buys',
          'item_last2w_buys', 'item_last4w_buys', 'item_sales_trend',
          'item_repurchase_ratio', 'item_user_mean_age', 'item_user_age_std',
          'item_mean_price', 'item_freshness', 'user_item_buys',
          'user_total_buys', 'user_unique_items', 'user_age',
          'user_mean_price', 'num_recall_sources', 'user_mean_channel',
          'item_mean_channel']
FILL_99 = ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank',
           'als_rank', 'trend_rank', 'user_group_pop_rank',
           'user_item_last_day', 'user_last_day', 'item_last_day',
           'user_item_price_diff']

for col in FILL_0:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)
for col in FILL_99:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(99)

# ============================================================
# 4. Define Features & Train LightGBM
# ============================================================
print("\n--- Training LightGBM ---")
import lightgbm as lgb

FEATURE_COLS = [
    # Recall features
    'repurchase_score', 'repurchase_rank', 'itemcf_score', 'itemcf_rank',
    'cooc_rank', 'pop_rank', 'als_score', 'als_rank',
    'trend_rank', 'user_group_pop_rank',
    'is_repurchase', 'is_itemcf', 'is_cooc', 'is_pop', 'is_als',
    'is_user_group_pop', 'is_trending', 'num_recall_sources',
    # User-item interaction
    'user_item_buys', 'user_item_last_day',
    # User features
    'user_total_buys', 'user_unique_items', 'user_last_day', 'user_age',
    # Item popularity
    'item_popularity', 'item_total_buys', 'item_unique_buyers',
    # Item category
    'item_dept', 'item_section', 'item_garment', 'item_category', 'item_color',
    # Recent trend
    'item_recent_week_buys', 'item_last2w_buys', 'item_last4w_buys',
    'item_sales_trend',
    # Repurchase ratio
    'item_repurchase_ratio',
    # Age-demographic match
    'item_user_mean_age', 'user_item_age_diff', 'item_user_age_std',
    # Price match
    'user_mean_price', 'item_mean_price', 'user_item_price_diff',
    # Item freshness
    'item_freshness', 'item_last_day',
    # Channel match
    'user_mean_channel', 'item_mean_channel',
]

X_train = train_df[FEATURE_COLS]
y_train = train_df['target']

pos_count = y_train.sum()
neg_count = (y_train == 0).sum()
print(f"  Positive: {pos_count:,}, Negative: {neg_count:,}, Ratio: 1:{neg_count/max(pos_count,1):.1f}")

lgb_params = {
    'objective': 'binary',
    'metric': 'average_precision',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': neg_count / max(pos_count, 1),
    'min_child_samples': 50,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

dtrain = lgb.Dataset(X_train, label=y_train)
model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dtrain],
    callbacks=[lgb.log_evaluation(100)],
)

# Feature importance
print("\nFeature importance (top 20):")
importance = sorted(zip(FEATURE_COLS, model.feature_importance()), key=lambda x: -x[1])
for fname, fimp in importance[:20]:
    print(f"  {fname}: {fimp}")

del train_df, dtrain
gc.collect()

# ============================================================
# 5. Generate Predictions (Full Data)
# ============================================================
print("\n--- Generating predictions ---")

# Rebuild from full data (including week 0)
last_6w_full = txn[txn['week'] <= 6]
cust_repurchase_full = (
    last_6w_full
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
    .to_dict()
)

user_item_buy_count_full = txn.groupby(['customer_id', 'article_id']).size().to_dict()
user_item_last_day_full = txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()
user_total_buys_full = txn.groupby('customer_id').size().to_dict()
user_unique_items_full = txn.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_day_full = txn.groupby('customer_id')['days_ago'].min().to_dict()

# Price from full data
user_mean_price_full = txn.groupby('customer_id')['price'].mean().to_dict()
item_mean_price_full = txn.groupby('article_id')['price'].mean().to_dict()

# Pop scores from full data
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
pop_scores_full = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50_full = list(pop_scores_full.head(50).index)
pop_score_dict_full = pop_scores_full.to_dict()
popular_top12 = pop_top50_full[:12]

# Recent week buys from full data (now includes week 0)
item_recent_week_full = txn[txn['week'] == 0]['article_id'].value_counts().to_dict()

# ItemCF recs from full data
cust_recent_full = (
    txn[txn['week'] <= 4]
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
    .to_dict()
)

itemcf_user_recs_full = {}
for cid, recent in cust_recent_full.items():
    cf_scores = defaultdict(float)
    for aid in recent[:5]:
        if aid in itemcf_lookup:
            for related_aid, score in itemcf_lookup[aid]:
                cf_scores[related_aid] += score
    itemcf_user_recs_full[cid] = sorted(cf_scores.items(), key=lambda x: -x[1])

cooc_user_recs_full = {}
for cid, recent in cust_recent_full.items():
    cooc_items = []
    seen = set()
    for aid in recent[:5]:
        if aid in freq_pairs:
            for related in freq_pairs[aid]:
                if related not in seen:
                    cooc_items.append(related)
                    seen.add(related)
    cooc_user_recs_full[cid] = cooc_items

# Channel from full data
user_mean_channel_full = txn.groupby('customer_id')['sales_channel_id'].mean().to_dict()
item_mean_channel_full = txn.groupby('article_id')['sales_channel_id'].mean().to_dict()

del last_6w_full, cust_recent_full
gc.collect()

# --- Score all candidates ---
print("  Scoring candidates...")
predictions = {}
gbdt_used = 0

for cid in sample_sub['customer_id']:
    if cid not in cust_repurchase_full and cid not in als_user_recs:
        predictions[cid] = ' '.join(f'{a:010d}' for a in popular_top12)
        continue

    candidates = {}

    for rank, aid in enumerate(cust_repurchase_full.get(cid, [])[:16]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['repurchase_rank'] = rank
        candidates[aid]['repurchase_score'] = 1.0 / (1.0 + rank)

    for rank, (aid, score) in enumerate(itemcf_user_recs_full.get(cid, [])[:20]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['itemcf_rank'] = rank
        candidates[aid]['itemcf_score'] = score

    for rank, aid in enumerate(cooc_user_recs_full.get(cid, [])[:10]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['cooc_rank'] = rank

    for rank, aid in enumerate(pop_top50_full[:30]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['pop_rank'] = rank

    # ALS candidates
    for rank, (aid, score) in enumerate(als_user_recs.get(cid, [])[:50]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['als_rank'] = rank
        candidates[aid]['als_score'] = score

    # User-group popular
    ab = cust_age_bin.get(cid, 3)
    for rank, aid in enumerate(age_bin_popular.get(ab, [])[:30]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['user_group_pop_rank'] = rank

    # Trending
    for rank, aid in enumerate(trending_top50[:20]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['trend_rank'] = rank

    if not candidates:
        predictions[cid] = ' '.join(f'{a:010d}' for a in popular_top12)
        continue

    u_age = cust_age.get(cid, 35)
    u_mean_pr = user_mean_price_full.get(cid, 0)
    u_mean_ch = user_mean_channel_full.get(cid, 1.5)

    rows = []
    aid_list = []
    for aid, feat in candidates.items():
        i_mean_age = item_user_mean_age.get(aid, 35)
        i_mean_pr = item_mean_price_full.get(aid, 0)

        row = {
            'repurchase_score': feat.get('repurchase_score', 0),
            'repurchase_rank': feat.get('repurchase_rank', 99),
            'itemcf_score': feat.get('itemcf_score', 0),
            'itemcf_rank': feat.get('itemcf_rank', 99),
            'cooc_rank': feat.get('cooc_rank', 99),
            'pop_rank': feat.get('pop_rank', 99),
            # ALS features
            'als_score': feat.get('als_score', 0),
            'als_rank': feat.get('als_rank', 99),
            'is_als': 1 if 'als_rank' in feat else 0,
            # User-group popular
            'user_group_pop_rank': feat.get('user_group_pop_rank', 99),
            'is_user_group_pop': 1 if 'user_group_pop_rank' in feat else 0,
            # Trending
            'trend_rank': feat.get('trend_rank', 99),
            'is_trending': 1 if 'trend_rank' in feat else 0,
            # Recall meta
            'is_repurchase': 1 if 'repurchase_rank' in feat else 0,
            'is_itemcf': 1 if 'itemcf_rank' in feat else 0,
            'is_cooc': 1 if 'cooc_rank' in feat else 0,
            'is_pop': 1 if 'pop_rank' in feat else 0,
            'num_recall_sources': sum(1 for k in ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank', 'als_rank', 'user_group_pop_rank', 'trend_rank'] if k in feat),
            'user_item_buys': user_item_buy_count_full.get((cid, aid), 0),
            'user_item_last_day': user_item_last_day_full.get((cid, aid), 999),
            'user_total_buys': user_total_buys_full.get(cid, 0),
            'user_unique_items': user_unique_items_full.get(cid, 0),
            'user_last_day': user_last_day_full.get(cid, 999),
            'user_age': u_age,
            'item_popularity': pop_score_dict_full.get(aid, 0),
            'item_total_buys': item_total_buys.get(aid, 0),
            'item_unique_buyers': item_unique_buyers.get(aid, 0),
            'item_dept': art_dept.get(aid, -1),
            'item_section': art_section.get(aid, -1),
            'item_garment': art_garment.get(aid, -1),
            'item_category': art_category.get(aid, -1),
            'item_color': art_color.get(aid, -1),
            # NEW features
            'item_recent_week_buys': item_recent_week_full.get(aid, 0),
            'item_last2w_buys': item_last2w_buys.get(aid, 0),
            'item_last4w_buys': item_last4w_buys.get(aid, 0),
            'item_sales_trend': item_sales_trend.get(aid, 1.0),
            'item_repurchase_ratio': item_repurchase_ratio.get(aid, 1.0),
            'item_user_mean_age': i_mean_age,
            'user_item_age_diff': abs(u_age - i_mean_age),
            'item_user_age_std': item_user_age_std.get(aid, 10),
            'user_mean_price': u_mean_pr,
            'item_mean_price': i_mean_pr,
            'user_item_price_diff': abs(u_mean_pr - i_mean_pr) if u_mean_pr > 0 and i_mean_pr > 0 else 999,
            'item_freshness': item_freshness.get(aid, 0),
            'item_last_day': item_last_day.get(aid, 999),
            'user_mean_channel': u_mean_ch,
            'item_mean_channel': item_mean_channel_full.get(aid, 1.5),
        }
        rows.append(row)
        aid_list.append(aid)

    X_pred = pd.DataFrame(rows)[FEATURE_COLS]
    # Fill NaN
    for col in FILL_0:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna(0)
    for col in FILL_99:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna(99)

    scores = model.predict(X_pred)

    # Sort by GBDT score (like R21)
    sorted_idx = np.argsort(-scores)
    pred = [aid_list[i] for i in sorted_idx[:12]]
    gbdt_used += 12

    # Fill remaining with popular
    used = set(pred)
    for aid in pop_top50_full:
        if len(pred) >= 12:
            break
        if aid not in used:
            pred.append(aid)

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

print(f"  GBDT-scored predictions: {gbdt_used:,}")

# ============================================================
# 6. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r26_als_expanded_recall.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active_count = sum(1 for cid in sample_sub['customer_id'] if cid in cust_repurchase_full or cid in als_user_recs)
print(f"Active: {active_count:,}, Fallback: {len(sample_sub) - active_count:,}")

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

val_scores = []
pop12_str = ' '.join(f'{a:010d}' for a in popular_top12)
for cid in val_truth:
    pred_str = predictions.get(cid, pop12_str)
    pred = [int(x) for x in pred_str.split()]
    val_scores.append(apk(val_truth[cid], pred, 12))

map12_active = np.mean(val_scores)
print(f"Val MAP@12 (active): {map12_active:.6f}")

print("\n" + "=" * 60)
print("R26 Complete")
print(f"R21: 0.02365 | R18: 0.02318 | Target: 0.0300")
print("=" * 60)
