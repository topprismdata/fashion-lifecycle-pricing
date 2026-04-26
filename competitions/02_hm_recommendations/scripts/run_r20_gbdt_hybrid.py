"""
H&M R20: GBDT Hybrid — Repurchase Locked + GBDT-ranked Fill

R19 scored 0.01769 (WORSE than heuristic 0.02282) because GBDT displaced
high-precision repurchase items with lower-precision candidates.
This is the "mix and sort" anti-pattern.

R20 fix: HYBRID approach
  - Layer 1: Repurchase items LOCKED in recency order (NEVER displaced)
  - Layer 2: GBDT scores remaining candidates (ItemCF, cooc, popular, variants)
  - Layer 3: Popular fills any remaining slots

Training identical to R19 (3-fold temporal CV, 31 features).
Only prediction generation changes.
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
print("R20: GBDT Hybrid — Repurchase Locked + GBDT-ranked Fill")
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

# Article lookups
art_dept = articles.set_index('article_id')['department_no'].to_dict()
art_section = articles.set_index('article_id')['section_no'].to_dict()
art_prod = articles.set_index('article_id')['product_code'].to_dict()
art_cat = articles.set_index('article_id')['index_group_no'].to_dict()
art_garment = articles.set_index('article_id')['garment_group_no'].to_dict()
art_color = articles.set_index('article_id')['colour_group_code'].to_dict()

# ============================================================
# 2. Pre-compute ItemCF (from full transaction history)
# ============================================================
print("\n--- ItemCF ---")
cust_sequences = (
    txn[txn['week'] <= 11]
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

# ============================================================
# 3. Pre-compute Co-occurrence
# ============================================================
print("\n--- Co-occurrence ---")
buckets = txn[txn['week'] <= 11].groupby(
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
print(f"Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts
gc.collect()

# ============================================================
# 4. Pre-compute Global Stats
# ============================================================
print("\n--- Global item/user stats ---")
# Item stats
item_total_buys = txn['article_id'].value_counts().to_dict()
item_unique_buyers = txn.groupby('article_id')['customer_id'].nunique().to_dict()

# Time-decay popular
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
pop_scores = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50 = list(pop_scores.head(50).index)
pop_score_dict = pop_scores.to_dict()

# Product code groups for variants
product_groups = articles.groupby('product_code')['article_id'].apply(list).to_dict()
product_variants = {}
for pc, aids in product_groups.items():
    if len(aids) >= 2:
        for a in aids:
            product_variants[a] = [x for x in aids if x != a][:3]

print(f"Items: {len(item_total_buys):,}")

# ============================================================
# 5. Helper: Generate Candidates + Features for a Target Week
# ============================================================
def generate_candidates_and_features(txn_df, target_week, for_training=True):
    """Generate candidates with features for a given target week.

    If for_training: use data AFTER target_week only for features
    If for_prediction: use all data
    """
    if for_training:
        feature_txn = txn_df[txn_df['week'] > target_week].copy()
    else:
        feature_txn = txn_df.copy()

    max_feature_week = feature_txn['week'].max()

    # Time-decay popular from feature data
    feature_txn['feat_decay'] = np.exp(-0.15 * (feature_txn['week'] - (target_week + 1)).astype(float).clip(lower=0))
    feat_pop = feature_txn.groupby('article_id')['feat_decay'].sum().sort_values(ascending=False)
    feat_pop_top50 = list(feat_pop.head(50).index)
    feat_pop_dict = feat_pop.to_dict()

    # Repurchase (6 weeks before target)
    rep_window_end = target_week + 6
    rep_txn = feature_txn[(feature_txn['week'] > target_week) & (feature_txn['week'] <= rep_window_end)]

    cust_repurchase = (
        rep_txn.sort_values('t_dat', ascending=False)
        .groupby('customer_id')['article_id']
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
        .to_dict()
    )

    # Recent items (for ItemCF/cooc lookup)
    cust_recent = (
        feature_txn[feature_txn['week'] <= target_week + 5]
        .sort_values('t_dat', ascending=False)
        .groupby('customer_id')['article_id']
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
        .to_dict()
    )

    # User stats from feature data
    user_total_buys = feature_txn.groupby('customer_id').size().to_dict()
    user_unique_items = feature_txn.groupby('customer_id')['article_id'].nunique().to_dict()
    user_last_day = feature_txn.groupby('customer_id')['days_ago'].min().to_dict()

    # Item stats from feature data
    feat_item_buys = feature_txn['article_id'].value_counts().to_dict()
    feat_item_buyers = feature_txn.groupby('article_id')['customer_id'].nunique().to_dict()

    # User-item interaction stats
    user_item_count = feature_txn.groupby(['customer_id', 'article_id']).size().to_dict()
    user_item_last_day = feature_txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()

    # Item CF recs per customer
    itemcf_recs = {}
    for cid, recent in cust_recent.items():
        cf_scores = defaultdict(float)
        for aid in recent[:5]:
            if aid in itemcf_lookup:
                for rel_aid, score in itemcf_lookup[aid]:
                    cf_scores[rel_aid] += score
        itemcf_recs[cid] = sorted(cf_scores.items(), key=lambda x: -x[1])

    # Co-occurrence recs per customer
    cooc_recs = {}
    for cid, recent in cust_recent.items():
        cooc_items = []
        seen = set()
        for aid in recent[:5]:
            if aid in freq_pairs:
                for related in freq_pairs[aid]:
                    if related not in seen:
                        cooc_items.append(related)
                        seen.add(related)
        cooc_recs[cid] = cooc_items

    # Target truth
    if for_training:
        truth = txn_df[txn_df['week'] == target_week].groupby('customer_id')['article_id'].apply(set).to_dict()
    else:
        truth = {}

    return {
        'cust_repurchase': cust_repurchase,
        'cust_recent': cust_recent,
        'itemcf_recs': itemcf_recs,
        'cooc_recs': cooc_recs,
        'feat_pop_top50': feat_pop_top50,
        'feat_pop_dict': feat_pop_dict,
        'user_total_buys': user_total_buys,
        'user_unique_items': user_unique_items,
        'user_last_day': user_last_day,
        'feat_item_buys': feat_item_buys,
        'feat_item_buyers': feat_item_buyers,
        'user_item_count': user_item_count,
        'user_item_last_day': user_item_last_day,
        'truth': truth,
    }


def build_feature_row(cid, aid, ctx):
    """Build feature dict for a (customer, article) pair."""
    row = {}

    # Recall features
    repurchase = ctx['cust_repurchase'].get(cid, [])
    if aid in repurchase:
        rank = repurchase.index(aid)
        row['repurchase_rank'] = rank
        row['repurchase_score'] = 1.0 / (1.0 + rank)
    else:
        row['repurchase_rank'] = 99
        row['repurchase_score'] = 0

    cf_recs = ctx['itemcf_recs'].get(cid, [])
    cf_aid_scores = {a: s for a, s in cf_recs}
    if aid in cf_aid_scores:
        rank = list(cf_aid_scores.keys()).index(aid)
        row['itemcf_rank'] = rank
        row['itemcf_score'] = cf_aid_scores[aid]
    else:
        row['itemcf_rank'] = 99
        row['itemcf_score'] = 0

    cooc = ctx['cooc_recs'].get(cid, [])
    if aid in cooc:
        row['cooc_rank'] = cooc.index(aid)
    else:
        row['cooc_rank'] = 99

    if aid in ctx['feat_pop_top50']:
        row['pop_rank'] = ctx['feat_pop_top50'].index(aid)
    else:
        row['pop_rank'] = 99

    # Binary recall flags
    row['is_repurchase'] = 1 if aid in repurchase else 0
    row['is_itemcf'] = 1 if aid in cf_aid_scores else 0
    row['is_cooc'] = 1 if aid in cooc else 0
    row['is_pop'] = 1 if aid in ctx['feat_pop_top50'][:30] else 0
    row['num_recall_sources'] = row['is_repurchase'] + row['is_itemcf'] + row['is_cooc'] + row['is_pop']

    # Product variant
    recent = ctx['cust_recent'].get(cid, [])
    row['is_variant'] = 1 if any(aid in product_variants.get(r, []) for r in recent[:5]) else 0

    # User-item interaction features
    row['user_item_count'] = ctx['user_item_count'].get((cid, aid), 0)
    row['user_item_last_day'] = ctx['user_item_last_day'].get((cid, aid), 999)
    row['user_item_recency'] = max(0, 100 - row['user_item_last_day']) if row['user_item_last_day'] < 999 else 0

    # User features
    row['user_total_buys'] = ctx['user_total_buys'].get(cid, 0)
    row['user_unique_items'] = ctx['user_unique_items'].get(cid, 0)
    row['user_last_day'] = ctx['user_last_day'].get(cid, 999)
    row['user_activity'] = row['user_unique_items'] / max(row['user_last_day'] / 7, 1)
    row['user_age'] = cust_age.get(cid, 35)

    # Item features
    row['item_popularity'] = ctx['feat_pop_dict'].get(aid, 0)
    row['item_total_buys'] = ctx['feat_item_buys'].get(aid, 0)
    row['item_unique_buyers'] = ctx['feat_item_buyers'].get(aid, 0)
    row['item_dept'] = art_dept.get(aid, -1)
    row['item_section'] = art_section.get(aid, -1)
    row['item_cat'] = art_cat.get(aid, -1)
    row['item_garment'] = art_garment.get(aid, -1)
    row['item_color'] = art_color.get(aid, -1)

    # Global item stats
    row['item_global_buys'] = item_total_buys.get(aid, 0)
    row['item_global_buyers'] = item_unique_buyers.get(aid, 0)
    row['item_repurchase_ratio'] = row['item_global_buyers'] / max(row['item_global_buys'], 1)

    return row


# ============================================================
# 6. Generate Training Data (Multiple Folds)
# ============================================================
print("\n--- Generating training data (3 folds) ---")
NEG_PER_POS = 4  # 4 negatives per positive

all_train_rows = []

for target_week in [1, 2, 3]:
    print(f"  Fold: week {target_week} as target...")
    ctx = generate_candidates_and_features(txn, target_week, for_training=True)
    truth = ctx['truth']
    print(f"    Users with truth: {len(truth):,}")

    fold_rows = []
    for cid, actual_items in truth.items():
        # All candidates for this user
        candidates = set()

        # Repurchase candidates
        for aid in ctx['cust_repurchase'].get(cid, [])[:20]:
            candidates.add(aid)

        # ItemCF candidates
        for aid, _ in ctx['itemcf_recs'].get(cid, [])[:20]:
            candidates.add(aid)

        # Co-occurrence candidates
        for aid in ctx['cooc_recs'].get(cid, [])[:10]:
            candidates.add(aid)

        # Popular
        for aid in ctx['feat_pop_top50'][:30]:
            candidates.add(aid)

        # Separate into positives and negatives
        positives = [a for a in candidates if a in actual_items]
        negatives = [a for a in candidates if a not in actual_items]

        # Add positives
        for aid in positives:
            row = build_feature_row(cid, aid, ctx)
            row['target'] = 1
            row['week_fold'] = target_week
            fold_rows.append(row)

        # Sample negatives
        np.random.seed(target_week * 1000 + hash(cid) % 1000)
        n_neg = min(len(negatives), max(len(positives) * NEG_PER_POS, 4))
        sampled_negs = np.random.choice(negatives, size=n_neg, replace=False) if len(negatives) > 0 else []

        for aid in sampled_negs:
            row = build_feature_row(cid, aid, ctx)
            row['target'] = 0
            row['week_fold'] = target_week
            fold_rows.append(row)

    print(f"    Fold rows: {len(fold_rows):,} (pos: {sum(1 for r in fold_rows if r['target']==1):,})")
    all_train_rows.extend(fold_rows)

    del ctx, fold_rows
    gc.collect()

print(f"\n  Total training rows: {len(all_train_rows):,}")
train_df = pd.DataFrame(all_train_rows)
del all_train_rows
gc.collect()

# Feature columns
FEATURE_COLS = [
    'repurchase_score', 'repurchase_rank', 'itemcf_score', 'itemcf_rank',
    'cooc_rank', 'pop_rank',
    'is_repurchase', 'is_itemcf', 'is_cooc', 'is_pop', 'is_variant', 'num_recall_sources',
    'user_item_count', 'user_item_last_day', 'user_item_recency',
    'user_total_buys', 'user_unique_items', 'user_last_day', 'user_activity', 'user_age',
    'item_popularity', 'item_total_buys', 'item_unique_buyers',
    'item_dept', 'item_section', 'item_cat', 'item_garment', 'item_color',
    'item_global_buys', 'item_global_buyers', 'item_repurchase_ratio',
]

# Fill NaN
for col in FEATURE_COLS:
    train_df[col] = train_df[col].fillna(0 if 'score' in col or 'count' in col or 'buys' in col or 'buyers' in col else 99 if 'rank' in col or 'last_day' in col else 0)

X_train = train_df[FEATURE_COLS]
y_train = train_df['target']

pos_count = y_train.sum()
neg_count = (y_train == 0).sum()
print(f"  Positive: {pos_count:,}, Negative: {neg_count:,}, Ratio: 1:{neg_count/max(pos_count,1):.1f}")

# ============================================================
# 7. Train LightGBM with CV
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

# Use week_fold for CV
folds = []
for wk in [1, 2, 3]:
    train_idx = train_df[train_df['week_fold'] != wk].index
    val_idx = train_df[train_df['week_fold'] == wk].index
    folds.append((train_idx, val_idx))

cv_results = lgb.cv(
    lgb_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    folds=folds,
    callbacks=[lgb.log_evaluation(100)],
    return_cvbooster=True,
)

# Get best iteration
best_iter = len(cv_results['valid average_precision-mean'])
print(f"  Best iteration: {best_iter}")
print(f"  CV AP: {cv_results['valid average_precision-mean'][-1]:.4f}")

# Train final model on all data
dtrain = lgb.Dataset(X_train, label=y_train)
model = lgb.train(lgb_params, dtrain, num_boost_round=min(best_iter, 800))

# Feature importance
print("\nFeature importance (top 15):")
importance = sorted(zip(FEATURE_COLS, model.feature_importance()), key=lambda x: -x[1])
for fname, fimp in importance[:15]:
    print(f"  {fname}: {fimp}")

del train_df, dtrain, X_train, y_train
gc.collect()

# ============================================================
# 8. Generate Predictions
# ============================================================
print("\n--- Generating predictions ---")

# Use all data for prediction
ctx_pred = generate_candidates_and_features(txn, 0, for_training=False)
popular_top12 = ctx_pred['feat_pop_top50'][:12]

predictions = {}
gbdt_scored = 0

for cid in sample_sub['customer_id']:
    # Inactive users: global popular
    if cid not in ctx_pred['cust_repurchase']:
        predictions[cid] = ' '.join(f'{a:010d}' for a in popular_top12)
        continue

    # HYBRID: Layer 1 — Repurchase LOCKED in recency order
    repurchase_items = ctx_pred['cust_repurchase'].get(cid, [])[:12]
    pred = list(repurchase_items)  # recency-ordered, never displaced
    repurchase_set = set(pred)

    if len(pred) >= 12:
        predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])
        gbdt_scored += 12
        continue

    # HYBRID: Layer 2 — GBDT ranks non-repurchase candidates
    fill_candidates = set()

    for aid, _ in ctx_pred['itemcf_recs'].get(cid, [])[:20]:
        if aid not in repurchase_set:
            fill_candidates.add(aid)
    for aid in ctx_pred['cooc_recs'].get(cid, [])[:10]:
        if aid not in repurchase_set:
            fill_candidates.add(aid)
    for aid in ctx_pred['feat_pop_top50'][:30]:
        if aid not in repurchase_set:
            fill_candidates.add(aid)
    # Product variants
    for aid in ctx_pred['cust_recent'].get(cid, [])[:5]:
        if aid in product_variants:
            for v in product_variants[aid][:2]:
                if v not in repurchase_set:
                    fill_candidates.add(v)

    if not fill_candidates:
        # Fill remaining with popular
        for aid in ctx_pred['feat_pop_top50']:
            if len(pred) >= 12:
                break
            if aid not in repurchase_set:
                pred.append(aid)
        predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])
        continue

    # Build features for fill candidates only
    rows = []
    aid_list = []
    for aid in fill_candidates:
        row = build_feature_row(cid, aid, ctx_pred)
        rows.append(row)
        aid_list.append(aid)

    X_pred = pd.DataFrame(rows)[FEATURE_COLS]
    for col in FEATURE_COLS:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna(0 if 'score' in col or 'count' in col or 'buys' in col or 'buyers' in col else 99 if 'rank' in col or 'last_day' in col else 0)

    scores = model.predict(X_pred)

    # Sort fill candidates by GBDT score, take top to fill remaining slots
    remaining = 12 - len(pred)
    sorted_idx = np.argsort(-scores)
    gbdt_added = 0
    for idx in sorted_idx:
        if gbdt_added >= remaining:
            break
        pred.append(aid_list[idx])
        gbdt_added += 1

    # Layer 3: Popular for any remaining slots
    used = set(pred)
    for aid in ctx_pred['feat_pop_top50']:
        if len(pred) >= 12:
            break
        if aid not in used:
            pred.append(aid)

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])
    gbdt_scored += len(pred)

print(f"  GBDT-scored predictions: {gbdt_scored:,}")

# ============================================================
# 9. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r20_gbdt_hybrid.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active_count = sum(1 for cid in sample_sub['customer_id'] if cid in ctx_pred['cust_repurchase'])
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
print("R20 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"R17: 0.02282 (heuristic) | R18: 0.02318 (GBDT v1) | R19: 0.01769 (mix-and-sort)")
print("=" * 60)
