"""
H&M R22: CatBoost with R21's Rich Features

Polimi thesis showed CatBoost (0.0308) >> LightGBM (0.0286) for H&M.
Key: CatBoost handles categorical features natively (dept, section, color, etc.)

R22 = R21 features + CatBoost model + LGB ensemble
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
print("R22: CatBoost + LGB Ensemble with Rich Features")
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
# 2. Pre-compute Feature Data
# ============================================================
print("\n--- Pre-computing features ---")
txn_features = txn[txn['week'] >= 1].copy()

# Item stats
print("  Item stats...")
item_total_buys = txn_features['article_id'].value_counts().to_dict()
item_unique_buyers = txn_features.groupby('article_id')['customer_id'].nunique().to_dict()
item_recent_week_buys = txn_features[txn_features['week'] == 1]['article_id'].value_counts().to_dict()
item_last2w_buys = txn_features[txn_features['week'] <= 2]['article_id'].value_counts().to_dict()
item_last4w_buys = txn_features[txn_features['week'] <= 4]['article_id'].value_counts().to_dict()
item_weekly_avg = {aid: count / 12.0 for aid, count in item_total_buys.items()}
item_sales_trend = {aid: item_recent_week_buys.get(aid, 0) / max(item_weekly_avg.get(aid, 0.1), 0.1) for aid in item_total_buys}
item_repurchase_ratio = {aid: item_unique_buyers.get(aid, 1) / max(item_total_buys.get(aid, 1), 1) for aid in item_total_buys}

# Age-demographic
print("  Age-demographic...")
txn_with_age = txn_features.merge(customers[['customer_id', 'age']], on='customer_id', how='left')
item_user_mean_age = txn_with_age.groupby('article_id')['age'].mean().to_dict()
item_user_age_std = txn_with_age.groupby('article_id')['age'].std().fillna(0).to_dict()
del txn_with_age
gc.collect()

# Price
print("  Price...")
user_mean_price = txn_features.groupby('customer_id')['price'].mean().to_dict()
user_std_price = txn_features.groupby('customer_id')['price'].std().fillna(0).to_dict()
item_mean_price = txn_features.groupby('article_id')['price'].mean().to_dict()

# Time
print("  Time features...")
item_first_day = txn_features.groupby('article_id')['days_ago'].max().to_dict()
item_last_day_stat = txn_features.groupby('article_id')['days_ago'].min().to_dict()
item_freshness = {aid: max_date.day - item_first_day.get(aid, 0) for aid in item_first_day}

# Channel
user_mean_channel = txn_features.groupby('customer_id')['sales_channel_id'].mean().to_dict()
item_mean_channel = txn_features.groupby('article_id')['sales_channel_id'].mean().to_dict()

# Recall channels
print("  Recall channels...")
txn_features['decay'] = np.exp(-0.15 * txn_features['week'].astype(float))
pop_scores = txn_features.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50 = list(pop_scores.head(50).index)
pop_score_dict = pop_scores.to_dict()

last_6w = txn_features[txn_features['week'] <= 7]
cust_repurchase = (
    last_6w.sort_values('t_dat', ascending=False)
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
    itemcf_lookup[item] = sorted(related.items(), key=lambda x: -x[1])[:10]
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

# User stats
user_total_buys = txn_features.groupby('customer_id').size().to_dict()
user_unique_items = txn_features.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_purchase_day = txn_features.groupby('customer_id')['days_ago'].min().to_dict()
user_item_buy_count = txn.groupby(['customer_id', 'article_id']).size().to_dict()
user_item_last_day = txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()

# Per-customer recall
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

val_truth = txn[txn['week'] == 0].groupby('customer_id')['article_id'].apply(set).to_dict()
print(f"  Val truth: {len(val_truth):,}")
del txn_features
gc.collect()

# ============================================================
# 3. Generate Training Data
# ============================================================
print("\n--- Generating training data ---")
all_active = set(cust_repurchase.keys()) | set(cust_recent_items.keys())
print(f"  Active users: {len(all_active):,}")

train_rows = []
skipped = 0
for cid in all_active:
    if cid not in val_truth:
        skipped += 1
        continue
    actual = val_truth[cid]
    candidates = {}
    for rank, aid in enumerate(cust_repurchase.get(cid, [])[:16]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['repurchase_rank'] = rank
        candidates[aid]['repurchase_score'] = 1.0 / (1.0 + rank)
    for rank, (aid, score) in enumerate(itemcf_user_recs.get(cid, [])[:20]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['itemcf_rank'] = rank
        candidates[aid]['itemcf_score'] = score
    for rank, aid in enumerate(cooc_user_recs.get(cid, [])[:10]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['cooc_rank'] = rank
    for rank, aid in enumerate(pop_top50[:30]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['pop_rank'] = rank

    u_age = cust_age.get(cid, 35)
    u_mean_pr = user_mean_price.get(cid, 0)
    u_mean_ch = user_mean_channel.get(cid, 1.5)

    for aid, feat in candidates.items():
        i_mean_age = item_user_mean_age.get(aid, 35)
        i_mean_pr = item_mean_price.get(aid, 0)
        row = {
            'target': 1 if aid in actual else 0,
            'repurchase_score': feat.get('repurchase_score', 0),
            'repurchase_rank': feat.get('repurchase_rank', 99),
            'itemcf_score': feat.get('itemcf_score', 0),
            'itemcf_rank': feat.get('itemcf_rank', 99),
            'cooc_rank': feat.get('cooc_rank', 99),
            'pop_rank': feat.get('pop_rank', 99),
            'is_repurchase': 1 if 'repurchase_rank' in feat else 0,
            'is_itemcf': 1 if 'itemcf_rank' in feat else 0,
            'is_cooc': 1 if 'cooc_rank' in feat else 0,
            'is_pop': 1 if 'pop_rank' in feat else 0,
            'num_recall_sources': sum(1 for k in ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank'] if k in feat),
            'user_item_buys': user_item_buy_count.get((cid, aid), 0),
            'user_item_last_day': user_item_last_day.get((cid, aid), 999),
            'user_total_buys': user_total_buys.get(cid, 0),
            'user_unique_items': user_unique_items.get(cid, 0),
            'user_last_day': user_last_purchase_day.get(cid, 999),
            'user_age': u_age,
            'item_popularity': pop_score_dict.get(aid, 0),
            'item_total_buys': item_total_buys.get(aid, 0),
            'item_unique_buyers': item_unique_buyers.get(aid, 0),
            'item_dept': art_dept.get(aid, -1),
            'item_section': art_section.get(aid, -1),
            'item_garment': art_garment.get(aid, -1),
            'item_category': art_category.get(aid, -1),
            'item_color': art_color.get(aid, -1),
            'item_recent_week_buys': item_recent_week_buys.get(aid, 0),
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
            'item_last_day': item_last_day_stat.get(aid, 999),
            'user_mean_channel': u_mean_ch,
            'item_mean_channel': item_mean_channel.get(aid, 1.5),
        }
        train_rows.append(row)

print(f"  Skipped: {skipped:,}, Training rows: {len(train_rows):,}")
train_df = pd.DataFrame(train_rows)
del train_rows
gc.collect()

# Fill NaN
FILL_0 = ['repurchase_score', 'itemcf_score', 'item_popularity',
          'item_total_buys', 'item_unique_buyers', 'item_recent_week_buys',
          'item_last2w_buys', 'item_last4w_buys', 'item_sales_trend',
          'item_repurchase_ratio', 'item_user_mean_age', 'item_user_age_std',
          'item_mean_price', 'item_freshness', 'user_item_buys',
          'user_total_buys', 'user_unique_items', 'user_age',
          'user_mean_price', 'num_recall_sources', 'user_mean_channel',
          'item_mean_channel']
FILL_99 = ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank',
           'user_item_last_day', 'user_last_day', 'item_last_day',
           'user_item_price_diff']
for col in FILL_0:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)
for col in FILL_99:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(99)

FEATURE_COLS = [
    'repurchase_score', 'repurchase_rank', 'itemcf_score', 'itemcf_rank',
    'cooc_rank', 'pop_rank',
    'is_repurchase', 'is_itemcf', 'is_cooc', 'is_pop', 'num_recall_sources',
    'user_item_buys', 'user_item_last_day',
    'user_total_buys', 'user_unique_items', 'user_last_day', 'user_age',
    'item_popularity', 'item_total_buys', 'item_unique_buyers',
    'item_dept', 'item_section', 'item_garment', 'item_category', 'item_color',
    'item_recent_week_buys', 'item_last2w_buys', 'item_last4w_buys',
    'item_sales_trend', 'item_repurchase_ratio',
    'item_user_mean_age', 'user_item_age_diff', 'item_user_age_std',
    'user_mean_price', 'item_mean_price', 'user_item_price_diff',
    'item_freshness', 'item_last_day',
    'user_mean_channel', 'item_mean_channel',
]

# Categorical features for CatBoost
CAT_FEATURES = ['item_dept', 'item_section', 'item_garment', 'item_category', 'item_color']

X_train = train_df[FEATURE_COLS]
y_train = train_df['target']

pos_count = y_train.sum()
neg_count = (y_train == 0).sum()
print(f"  Positive: {pos_count:,}, Negative: {neg_count:,}, Ratio: 1:{neg_count/max(pos_count,1):.1f}")

# ============================================================
# 4. Train LightGBM (same as R21)
# ============================================================
print("\n--- Training LightGBM ---")
import lightgbm as lgb

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
model_lgb = lgb.train(lgb_params, dtrain, num_boost_round=500,
                       valid_sets=[dtrain], callbacks=[lgb.log_evaluation(100)])

print("\nLGB Feature importance (top 10):")
for fname, fimp in sorted(zip(FEATURE_COLS, model_lgb.feature_importance()), key=lambda x: -x[1])[:10]:
    print(f"  {fname}: {fimp}")

# ============================================================
# 5. Train CatBoost
# ============================================================
print("\n--- Training CatBoost ---")
from catboost import CatBoostClassifier, Pool

cat_feature_idx = [FEATURE_COLS.index(c) for c in CAT_FEATURES]
print(f"  Cat features at indices: {cat_feature_idx}")

train_pool = Pool(X_train, label=y_train, cat_features=cat_feature_idx)

model_cb = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    auto_class_weights='Balanced',
    eval_metric='PRAUC',
    random_seed=42,
    verbose=100,
    thread_count=-1,
)
model_cb.fit(train_pool)

print("\nCB Feature importance (top 10):")
cb_importance = model_cb.get_feature_importance()
for fname, fimp in sorted(zip(FEATURE_COLS, cb_importance), key=lambda x: -x[1])[:10]:
    print(f"  {fname}: {fimp:.2f}")

del train_df, dtrain, train_pool
gc.collect()

# ============================================================
# 6. Generate Predictions (Ensemble LGB + CB)
# ============================================================
print("\n--- Generating predictions (ensemble) ---")

# Rebuild from full data
last_6w_full = txn[txn['week'] <= 6]
cust_repurchase_full = (
    last_6w_full.sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
    .to_dict()
)
user_item_buy_count_full = txn.groupby(['customer_id', 'article_id']).size().to_dict()
user_item_last_day_full = txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()
user_total_buys_full = txn.groupby('customer_id').size().to_dict()
user_unique_items_full = txn.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_day_full = txn.groupby('customer_id')['days_ago'].min().to_dict()
user_mean_price_full = txn.groupby('customer_id')['price'].mean().to_dict()
item_mean_price_full = txn.groupby('article_id')['price'].mean().to_dict()
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
pop_scores_full = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50_full = list(pop_scores_full.head(50).index)
pop_score_dict_full = pop_scores_full.to_dict()
popular_top12 = pop_top50_full[:12]
item_recent_week_full = txn[txn['week'] == 0]['article_id'].value_counts().to_dict()

cust_recent_full = (
    txn[txn['week'] <= 4].sort_values('t_dat', ascending=False)
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

user_mean_channel_full = txn.groupby('customer_id')['sales_channel_id'].mean().to_dict()
item_mean_channel_full = txn.groupby('article_id')['sales_channel_id'].mean().to_dict()

del last_6w_full, cust_recent_full
gc.collect()

# Score with both models
print("  Scoring candidates with LGB + CB...")
predictions = {}
W_LGB = 0.4
W_CB = 0.6  # CatBoost gets more weight (Polimi showed CB >> LGB)

for cid in sample_sub['customer_id']:
    if cid not in cust_repurchase_full:
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
            'is_repurchase': 1 if 'repurchase_rank' in feat else 0,
            'is_itemcf': 1 if 'itemcf_rank' in feat else 0,
            'is_cooc': 1 if 'cooc_rank' in feat else 0,
            'is_pop': 1 if 'pop_rank' in feat else 0,
            'num_recall_sources': sum(1 for k in ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank'] if k in feat),
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
            'item_last_day': item_last_day_stat.get(aid, 999),
            'user_mean_channel': u_mean_ch,
            'item_mean_channel': item_mean_channel_full.get(aid, 1.5),
        }
        rows.append(row)
        aid_list.append(aid)

    X_pred = pd.DataFrame(rows)[FEATURE_COLS]
    for col in FILL_0:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna(0)
    for col in FILL_99:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna(99)

    # Ensemble: weighted average of LGB and CB scores
    scores_lgb = model_lgb.predict(X_pred)
    pred_pool = Pool(X_pred, cat_features=cat_feature_idx)
    scores_cb = model_cb.predict_proba(pred_pool)[:, 1]

    scores = W_LGB * scores_lgb + W_CB * scores_cb

    sorted_idx = np.argsort(-scores)
    pred = [aid_list[i] for i in sorted_idx[:12]]

    used = set(pred)
    for aid in pop_top50_full:
        if len(pred) >= 12:
            break
        if aid not in used:
            pred.append(aid)

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

print("  Done!")

# ============================================================
# 7. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r22_catboost_ensemble.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active_count = sum(1 for cid in sample_sub['customer_id'] if cid in cust_repurchase_full)
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
print("R22 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Features: {len(FEATURE_COLS)}, Models: LGB({W_LGB}) + CB({W_CB})")
print(f"R18: 0.02318 | R21: 0.02365 | Target: 0.0300")
print("=" * 60)
