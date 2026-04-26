"""
H&M R18: Two-Stage Pipeline with GBDT Ranking + Recall Score Features

Based on analysis of 1st-5th place solutions:
- 1st place: 16 recall strategies + GBDT binary classification, ~100 candidates/user
- 3rd place key insight: recall scores as features (+14% boost)
- Our R17 heuristic: 0.02282 (already beats wiki's best heuristic of 0.02263)

Architecture:
  Stage 1: Generate ~100 candidates per user from multiple channels
  Stage 2: GBDT binary classification (will user buy this item?)
           Features: recall scores, user-item interaction, item popularity, recency

Key difference from R02/R03 (which failed at 0.018):
  - R02/R03 used LGBMRanker which replaced repurchase ordering
  - R18 uses LGBM binary classifier, preserves layered structure
  - R18 includes recall scores as features (the 3rd place breakthrough)
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
print("R18: GBDT Ranking + Recall Score Features")
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

# Article lookup tables
art_dept = articles.set_index('article_id')['department_no'].to_dict()
art_section = articles.set_index('article_id')['section_no'].to_dict()
art_garment = articles.set_index('article_id')['garment_group_no'].to_dict()
art_product_code = articles.set_index('article_id')['product_code'].to_dict()
art_category = articles.set_index('article_id')['index_group_no'].to_dict()

# ============================================================
# 2. Train/Test Split (Temporal)
# ============================================================
# For GBDT training: use week > 0 data for features, week 0 as target
# For final submission: use all data
TRAIN_WEEKS = list(range(1, 13))  # Weeks 1-12 for building features
TARGET_WEEK = 0  # Week 0 is the target

print("\n--- Preparing training data ---")

# Ground truth for validation
val_truth = txn[txn['week'] == TARGET_WEEK].groupby('customer_id')['article_id'].apply(set).to_dict()
print(f"Val customers with purchases: {len(val_truth):,}")

# ============================================================
# 3. Build Recall Channels (using weeks 1+)
# ============================================================
print("\n--- Building recall channels ---")
txn_features = txn[txn['week'] >= 1].copy()  # Exclude target week for feature building

# --- Channel 1: Repurchase (6-week, recency-ordered) ---
print("  Channel 1: Repurchase...")
last_6w = txn_features[txn_features['week'] <= 7]  # week 1-7 (6 weeks before target)
cust_repurchase = (
    last_6w
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
    .to_dict()
)
print(f"    Users with repurchase: {len(cust_repurchase):,}")

# Repurchase scores per (user, item)
repurchase_score_map = {}
for cid, items in cust_repurchase.items():
    for rank, aid in enumerate(items):
        repurchase_score_map[(cid, aid)] = 1.0 / (1.0 + rank)  # Recency-based score

# --- Channel 2: ItemCF ---
print("  Channel 2: ItemCF...")
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

# --- Channel 3: Co-occurrence ---
print("  Channel 3: Co-occurrence...")
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

# --- Channel 4: Time-decay Popular ---
print("  Channel 4: Popular...")
txn_features['decay'] = np.exp(-0.15 * txn_features['week'].astype(float))
pop_scores = txn_features.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50 = list(pop_scores.head(50).index)
pop_score_dict = pop_scores.to_dict()
print(f"    Popular articles: {len(pop_scores):,}")

# --- Channel 5: Item popularity features ---
print("  Channel 5: Item stats...")
item_total_buys = txn_features['article_id'].value_counts().to_dict()
item_week0_buys = txn_features[txn_features['week'] <= 1]['article_id'].value_counts().to_dict()
item_unique_buyers = txn_features.groupby('article_id')['customer_id'].nunique().to_dict()

# ============================================================
# 4. Generate Candidates (Training: week > 0)
# ============================================================
print("\n--- Generating training candidates ---")

# User activity stats
user_total_buys = txn_features.groupby('customer_id').size().to_dict()
user_unique_items = txn_features.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_purchase_day = txn_features.groupby('customer_id')['days_ago'].min().to_dict()

# Build ItemCF recs per customer
print("    Building ItemCF recs per customer...")
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

# Co-occurrence recs per customer
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

del txn_features
gc.collect()

# --- Generate candidate rows for TRAINING ---
print("    Generating candidate rows...")

# For training, we need positive and negative examples
# Positive: items bought in week 0
# Negative: candidates that were NOT bought in week 0

# Get all active users (anyone with recent activity)
all_active = set(cust_repurchase.keys()) | set(cust_recent_items.keys())
print(f"    Active users: {len(all_active):,}")

# User-item interaction features
user_item_buy_count = (
    txn.groupby(['customer_id', 'article_id']).size().to_dict()
)
user_item_last_day = (
    txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()
)

train_rows = []
skipped = 0

for cid in all_active:
    if cid not in val_truth:
        skipped += 1
        continue  # Skip users not in validation for training

    actual = val_truth[cid]
    candidates = {}  # aid -> features dict

    # Repurchase candidates
    for rank, aid in enumerate(cust_repurchase.get(cid, [])[:16]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['repurchase_rank'] = rank
        candidates[aid]['repurchase_score'] = 1.0 / (1.0 + rank)

    # ItemCF candidates
    for rank, (aid, score) in enumerate(itemcf_user_recs.get(cid, [])[:20]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['itemcf_rank'] = rank
        candidates[aid]['itemcf_score'] = score

    # Co-occurrence candidates
    for rank, aid in enumerate(cooc_user_recs.get(cid, [])[:10]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['cooc_rank'] = rank

    # Popular candidates (top 30)
    for rank, aid in enumerate(pop_top50[:30]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['pop_rank'] = rank

    # Build features for each candidate
    for aid, feat in candidates.items():
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
            'is_repurchase': 1 if 'repurchase_rank' in feat else 0,
            'is_itemcf': 1 if 'itemcf_rank' in feat else 0,
            'is_cooc': 1 if 'cooc_rank' in feat else 0,
            'is_pop': 1 if 'pop_rank' in feat else 0,
            'num_recall_sources': sum([1 for k in ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank'] if k in feat]),
        }
        train_rows.append(row)

print(f"    Skipped users (no val truth): {skipped:,}")
print(f"    Training rows: {len(train_rows):,}")

train_df = pd.DataFrame(train_rows)
del train_rows
gc.collect()

# Fill NaN
for col in ['repurchase_score', 'repurchase_rank', 'itemcf_score', 'itemcf_rank',
            'cooc_rank', 'pop_rank']:
    train_df[col] = train_df[col].fillna(0 if 'score' in col else 99)

# Add item features
train_df['item_popularity'] = train_df['article_id'].map(pop_score_dict).fillna(0)
train_df['item_total_buys'] = train_df['article_id'].map(item_total_buys).fillna(0)
train_df['item_week0_buys'] = train_df['article_id'].map(item_week0_buys).fillna(0)
train_df['item_unique_buyers'] = train_df['article_id'].map(item_unique_buyers).fillna(0)
train_df['item_dept'] = train_df['article_id'].map(art_dept).fillna(-1)
train_df['item_section'] = train_df['article_id'].map(art_section).fillna(-1)

# Add user-item interaction features
train_df['user_item_buys'] = train_df.apply(
    lambda r: user_item_buy_count.get((r['customer_id'], r['article_id']), 0), axis=1
)
train_df['user_item_last_day'] = train_df.apply(
    lambda r: user_item_last_day.get((r['customer_id'], r['article_id']), 999), axis=1
)

# Add user features
train_df['user_total_buys'] = train_df['customer_id'].map(user_total_buys).fillna(0)
train_df['user_unique_items'] = train_df['customer_id'].map(user_unique_items).fillna(0)
train_df['user_last_day'] = train_df['customer_id'].map(user_last_purchase_day).fillna(999)

print(f"    Positive: {train_df['target'].sum():,}")
print(f"    Negative: {(train_df['target'] == 0).sum():,}")
print(f"    Positive rate: {train_df['target'].mean():.4f}")

# ============================================================
# 5. Train LightGBM
# ============================================================
print("\n--- Training LightGBM ---")
import lightgbm as lgb

feature_cols = [
    'repurchase_score', 'repurchase_rank', 'itemcf_score', 'itemcf_rank',
    'cooc_rank', 'pop_rank',
    'is_repurchase', 'is_itemcf', 'is_cooc', 'is_pop', 'num_recall_sources',
    'item_popularity', 'item_total_buys', 'item_week0_buys', 'item_unique_buyers',
    'item_dept', 'item_section',
    'user_item_buys', 'user_item_last_day',
    'user_total_buys', 'user_unique_items', 'user_last_day',
]

X_train = train_df[feature_cols]
y_train = train_df['target']

# Compute positive weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / max(pos_count, 1)

lgb_params = {
    'objective': 'binary',
    'metric': 'average_precision',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale_pos,
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
print("\nFeature importance (top 10):")
importance = sorted(zip(feature_cols, model.feature_importance()), key=lambda x: -x[1])
for fname, fimp in importance[:10]:
    print(f"  {fname}: {fimp}")

del train_df, dtrain
gc.collect()

# ============================================================
# 6. Generate Predictions (Full Data)
# ============================================================
print("\n--- Generating predictions with GBDT ---")

# Rebuild recall channels with FULL data (including week 0) for submission
# Repurchase from full data
last_6w_full = txn[txn['week'] <= 6]
cust_repurchase_full = (
    last_6w_full
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
    .to_dict()
)

# User-item features from full data
user_item_buy_count_full = txn.groupby(['customer_id', 'article_id']).size().to_dict()
user_item_last_day_full = txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()
user_total_buys_full = txn.groupby('customer_id').size().to_dict()
user_unique_items_full = txn.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_day_full = txn.groupby('customer_id')['days_ago'].min().to_dict()

# Pop scores from full data
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
pop_scores_full = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50_full = list(pop_scores_full.head(50).index)
pop_score_dict_full = pop_scores_full.to_dict()
popular_top12 = pop_top50_full[:12]

# ItemCF recs from full data (reuse itemcf_lookup, build per-customer with full data)
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

del last_6w_full, cust_recent_full
gc.collect()

# --- Score all candidates with GBDT ---
print("    Scoring candidates...")
predictions = {}
gbdt_used = 0
repurchase_preserved = 0

for cid in sample_sub['customer_id']:
    # For inactive users: just use popular
    if cid not in cust_repurchase_full:
        predictions[cid] = ' '.join(f'{a:010d}' for a in popular_top12)
        continue

    # Generate candidates
    candidates = {}  # aid -> features

    # Repurchase
    for rank, aid in enumerate(cust_repurchase_full.get(cid, [])[:16]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['repurchase_rank'] = rank
        candidates[aid]['repurchase_score'] = 1.0 / (1.0 + rank)

    # ItemCF
    for rank, (aid, score) in enumerate(itemcf_user_recs_full.get(cid, [])[:20]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['itemcf_rank'] = rank
        candidates[aid]['itemcf_score'] = score

    # Co-occurrence
    for rank, aid in enumerate(cooc_user_recs_full.get(cid, [])[:10]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['cooc_rank'] = rank

    # Popular top 30
    for rank, aid in enumerate(pop_top50_full[:30]):
        if aid not in candidates:
            candidates[aid] = {}
        candidates[aid]['pop_rank'] = rank

    if not candidates:
        predictions[cid] = ' '.join(f'{a:010d}' for a in popular_top12)
        continue

    # Build feature matrix
    rows = []
    aid_list = []
    is_repurchase_list = []
    for aid, feat in candidates.items():
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
            'num_recall_sources': sum([1 for k in ['repurchase_rank', 'itemcf_rank', 'cooc_rank', 'pop_rank'] if k in feat]),
            'item_popularity': pop_score_dict_full.get(aid, 0),
            'item_total_buys': item_total_buys.get(aid, 0),
            'item_week0_buys': item_week0_buys.get(aid, 0),
            'item_unique_buyers': item_unique_buyers.get(aid, 0),
            'item_dept': art_dept.get(aid, -1),
            'item_section': art_section.get(aid, -1),
            'user_item_buys': user_item_buy_count_full.get((cid, aid), 0),
            'user_item_last_day': user_item_last_day_full.get((cid, aid), 999),
            'user_total_buys': user_total_buys_full.get(cid, 0),
            'user_unique_items': user_unique_items_full.get(cid, 0),
            'user_last_day': user_last_day_full.get(cid, 999),
        }
        rows.append(row)
        aid_list.append(aid)
        is_repurchase_list.append(1 if 'repurchase_rank' in feat else 0)

    X_pred = pd.DataFrame(rows)[feature_cols]
    scores = model.predict(X_pred)

    # Sort by GBDT score (descending)
    sorted_idx = np.argsort(-scores)
    pred = []
    used = set()

    # Take top 12 by GBDT score
    for idx in sorted_idx[:12]:
        pred.append(aid_list[idx])
        used.add(aid_list[idx])
        gbdt_used += 1

    # Fill remaining with popular
    for aid in pop_top50_full:
        if len(pred) >= 12:
            break
        if aid not in used:
            pred.append(aid)

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

print(f"  GBDT-scored predictions: {gbdt_used:,}")

# ============================================================
# 7. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r18_gbdt_ranking.csv"
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
print("R18 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"R17: 0.02282 (best heuristic)")
print("=" * 60)
