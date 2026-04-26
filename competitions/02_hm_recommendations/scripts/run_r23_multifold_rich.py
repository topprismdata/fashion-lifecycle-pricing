"""
H&M R23: Multi-fold Training + Rich Features + More Recall

R21 (0.02365) = LGB single fold + 40 features (best so far)
R22 (0.02175) = CatBoost ensemble failed (overfit single feature)

R23 strategy:
  1. Multi-fold training (weeks 1,2,3 as targets) like R19
  2. BUT with item_recent_week_buys (R19's missing key feature)
  3. Add department-level popular as recall channel
  4. Add out-of-stock filtering
  5. LightGBM only (CatBoost proven worse)
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
print("R23: Multi-fold + Rich Features + More Recall")
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

customers['age'] = customers['age'].fillna(customers['age'].median())
cust_age = customers.set_index('customer_id')['age'].to_dict()

art_dept = articles.set_index('article_id')['department_no'].to_dict()
art_section = articles.set_index('article_id')['section_no'].to_dict()
art_garment = articles.set_index('article_id')['garment_group_no'].to_dict()
art_product = articles.set_index('article_id')['product_code'].to_dict()
art_category = articles.set_index('article_id')['index_group_no'].to_dict()
art_color = articles.set_index('article_id')['colour_group_code'].to_dict()

# Out-of-stock detection: items whose week-0 sales dropped >80% vs week-1
print("\n--- Out-of-stock detection ---")
w0_buys = txn[txn['week'] == 0].groupby('article_id').size()
w1_buys = txn[txn['week'] == 1].groupby('article_id').size()
oos_items = set()
for aid in w1_buys.index:
    if w1_buys[aid] >= 5:  # Had meaningful sales in week 1
        r0 = w0_buys.get(aid, 0)
        if r0 < w1_buys[aid] * 0.2:  # Dropped >80%
            oos_items.add(aid)
print(f"  Out-of-stock items: {len(oos_items)}")

# Product code groups for variants
product_groups = articles.groupby('product_code')['article_id'].apply(list).to_dict()

# ============================================================
# 2. Build Global ItemCF + Co-occurrence (from all data)
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
    itemcf_lookup[item] = sorted(related.items(), key=lambda x: -x[1])[:10]
print(f"  Articles with ItemCF: {len(itemcf_lookup):,}")
del cust_sequences, sim_item
gc.collect()

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
print(f"  Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts
gc.collect()

# ============================================================
# 3. Helper: Build Features for a Target Week
# ============================================================
def build_fold_data(txn_df, target_week):
    """Build training data for a specific target week."""
    # Features from data AFTER target week (older data, available before target)
    feat_txn = txn_df[txn_df['week'] > target_week].copy()

    # Time-decay popular
    feat_txn['fdecay'] = np.exp(-0.15 * (feat_txn['week'] - (target_week + 1)).astype(float).clip(lower=0))
    pop_scores = feat_txn.groupby('article_id')['fdecay'].sum().sort_values(ascending=False)
    pop_top50 = list(pop_scores.head(50).index)
    pop_dict = pop_scores.to_dict()

    # Repurchase (6 weeks before target)
    rep_txn = feat_txn[(feat_txn['week'] > target_week) & (feat_txn['week'] <= target_week + 6)]
    cust_repurchase = (
        rep_txn.sort_values('t_dat', ascending=False)
        .groupby('customer_id')['article_id']
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
        .to_dict()
    )

    # Recent items for ItemCF/cooc lookup
    cust_recent = (
        feat_txn[feat_txn['week'] <= target_week + 5]
        .sort_values('t_dat', ascending=False)
        .groupby('customer_id')['article_id']
        .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10])
        .to_dict()
    )

    # ItemCF recs per customer
    itemcf_recs = {}
    for cid, recent in cust_recent.items():
        cf_scores = defaultdict(float)
        for aid in recent[:5]:
            if aid in itemcf_lookup:
                for rel_aid, score in itemcf_lookup[aid]:
                    cf_scores[rel_aid] += score
        itemcf_recs[cid] = sorted(cf_scores.items(), key=lambda x: -x[1])

    # Cooc recs per customer
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

    # Item stats from feature data
    item_total_buys_f = feat_txn['article_id'].value_counts().to_dict()
    item_unique_buyers_f = feat_txn.groupby('article_id')['customer_id'].nunique().to_dict()

    # CRITICAL: Recent week buys (first available week after target)
    first_week = target_week + 1
    item_recent_week = feat_txn[feat_txn['week'] == first_week]['article_id'].value_counts().to_dict()
    item_last2w = feat_txn[feat_txn['week'] <= first_week + 1]['article_id'].value_counts().to_dict()
    item_last4w = feat_txn[feat_txn['week'] <= first_week + 3]['article_id'].value_counts().to_dict()

    item_weekly_avg_f = {aid: count / max(12 - target_week, 1) for aid, count in item_total_buys_f.items()}
    item_sales_trend_f = {aid: item_recent_week.get(aid, 0) / max(item_weekly_avg_f.get(aid, 0.1), 0.1) for aid in item_total_buys_f}
    item_repurchase_ratio_f = {aid: item_unique_buyers_f.get(aid, 1) / max(item_total_buys_f.get(aid, 1), 1) for aid in item_total_buys_f}

    # Age-demographic
    feat_with_age = feat_txn.merge(customers[['customer_id', 'age']], on='customer_id', how='left')
    item_user_mean_age_f = feat_with_age.groupby('article_id')['age'].mean().to_dict()
    item_user_age_std_f = feat_with_age.groupby('article_id')['age'].std().fillna(0).to_dict()
    del feat_with_age

    # Price
    item_mean_price_f = feat_txn.groupby('article_id')['price'].mean().to_dict()

    # Time features
    item_first_day_f = feat_txn.groupby('article_id')['days_ago'].max().to_dict()
    item_last_day_f = feat_txn.groupby('article_id')['days_ago'].min().to_dict()
    item_freshness_f = {aid: max_date.day - item_first_day_f.get(aid, 0) for aid in item_first_day_f}

    # Channel
    item_mean_ch_f = feat_txn.groupby('article_id')['sales_channel_id'].mean().to_dict()

    # User stats
    user_total_buys_f = feat_txn.groupby('customer_id').size().to_dict()
    user_unique_items_f = feat_txn.groupby('customer_id')['article_id'].nunique().to_dict()
    user_last_day_f = feat_txn.groupby('customer_id')['days_ago'].min().to_dict()
    user_mean_price_f = feat_txn.groupby('customer_id')['price'].mean().to_dict()
    user_mean_ch_f = feat_txn.groupby('customer_id')['sales_channel_id'].mean().to_dict()

    # User-item interaction
    user_item_buys_f = feat_txn.groupby(['customer_id', 'article_id']).size().to_dict()
    user_item_last_f = feat_txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()

    # Department-level popular (NEW recall channel)
    # For each department, get top items by time-decay score
    dept_pop = {}
    feat_with_dept = feat_txn.merge(articles[['article_id', 'department_no']], on='article_id', how='left')
    for dept in feat_with_dept['department_no'].unique():
        dept_items = feat_with_dept[feat_with_dept['department_no'] == dept]
        dept_scores = dept_items.groupby('article_id')['fdecay'].sum().sort_values(ascending=False)
        dept_pop[dept] = list(dept_scores.head(6).index)
    del feat_with_dept

    # User's preferred departments
    user_depts = {}
    for cid in cust_recent:
        recent_items = cust_recent[cid][:10]
        dept_counts = Counter()
        for aid in recent_items:
            d = art_dept.get(aid, -1)
            if d >= 0:
                dept_counts[d] += 1
        user_depts[cid] = [d for d, _ in dept_counts.most_common(3)]

    # Truth
    truth = txn_df[txn_df['week'] == target_week].groupby('customer_id')['article_id'].apply(set).to_dict()

    return {
        'pop_top50': pop_top50, 'pop_dict': pop_dict,
        'cust_repurchase': cust_repurchase, 'cust_recent': cust_recent,
        'itemcf_recs': itemcf_recs, 'cooc_recs': cooc_recs,
        'item_total_buys': item_total_buys_f, 'item_unique_buyers': item_unique_buyers_f,
        'item_recent_week': item_recent_week, 'item_last2w': item_last2w, 'item_last4w': item_last4w,
        'item_sales_trend': item_sales_trend_f, 'item_repurchase_ratio': item_repurchase_ratio_f,
        'item_user_mean_age': item_user_mean_age_f, 'item_user_age_std': item_user_age_std_f,
        'item_mean_price': item_mean_price_f,
        'item_first_day': item_first_day_f, 'item_last_day': item_last_day_f,
        'item_freshness': item_freshness_f, 'item_mean_ch': item_mean_ch_f,
        'user_total_buys': user_total_buys_f, 'user_unique_items': user_unique_items_f,
        'user_last_day': user_last_day_f, 'user_mean_price': user_mean_price_f,
        'user_mean_ch': user_mean_ch_f,
        'user_item_buys': user_item_buys_f, 'user_item_last': user_item_last_f,
        'dept_pop': dept_pop, 'user_depts': user_depts,
        'truth': truth,
    }


def build_feature_row(cid, aid, ctx):
    u_age = cust_age.get(cid, 35)
    i_mean_age = ctx['item_user_mean_age'].get(aid, 35)
    u_mean_pr = ctx['user_mean_price'].get(cid, 0)
    i_mean_pr = ctx['item_mean_price'].get(aid, 0)
    u_mean_ch = ctx['user_mean_ch'].get(cid, 1.5)
    i_mean_ch = ctx['item_mean_ch'].get(aid, 1.5)

    repurchase = ctx['cust_repurchase'].get(cid, [])
    cf_recs = ctx['itemcf_recs'].get(cid, [])
    cf_dict = {a: s for a, s in cf_recs}
    cooc = ctx['cooc_recs'].get(cid, [])

    row = {
        'repurchase_score': 1.0 / (1.0 + repurchase.index(aid)) if aid in repurchase else 0,
        'repurchase_rank': repurchase.index(aid) if aid in repurchase else 99,
        'itemcf_score': cf_dict.get(aid, 0),
        'itemcf_rank': list(cf_dict.keys()).index(aid) if aid in cf_dict else 99,
        'cooc_rank': cooc.index(aid) if aid in cooc else 99,
        'pop_rank': ctx['pop_top50'].index(aid) if aid in ctx['pop_top50'] else 99,
        'is_repurchase': 1 if aid in repurchase else 0,
        'is_itemcf': 1 if aid in cf_dict else 0,
        'is_cooc': 1 if aid in cooc else 0,
        'is_pop': 1 if aid in ctx['pop_top50'][:30] else 0,
        'num_recall_sources': (1 if aid in repurchase else 0) + (1 if aid in cf_dict else 0) + (1 if aid in cooc else 0) + (1 if aid in ctx['pop_top50'][:30] else 0),
        'user_item_buys': ctx['user_item_buys'].get((cid, aid), 0),
        'user_item_last_day': ctx['user_item_last'].get((cid, aid), 999),
        'user_total_buys': ctx['user_total_buys'].get(cid, 0),
        'user_unique_items': ctx['user_unique_items'].get(cid, 0),
        'user_last_day': ctx['user_last_day'].get(cid, 999),
        'user_age': u_age,
        'item_popularity': ctx['pop_dict'].get(aid, 0),
        'item_total_buys': ctx['item_total_buys'].get(aid, 0),
        'item_unique_buyers': ctx['item_unique_buyers'].get(aid, 0),
        'item_dept': art_dept.get(aid, -1),
        'item_section': art_section.get(aid, -1),
        'item_garment': art_garment.get(aid, -1),
        'item_category': art_category.get(aid, -1),
        'item_color': art_color.get(aid, -1),
        'item_recent_week_buys': ctx['item_recent_week'].get(aid, 0),
        'item_last2w_buys': ctx['item_last2w'].get(aid, 0),
        'item_last4w_buys': ctx['item_last4w'].get(aid, 0),
        'item_sales_trend': ctx['item_sales_trend'].get(aid, 1.0),
        'item_repurchase_ratio': ctx['item_repurchase_ratio'].get(aid, 1.0),
        'item_user_mean_age': i_mean_age,
        'user_item_age_diff': abs(u_age - i_mean_age),
        'item_user_age_std': ctx['item_user_age_std'].get(aid, 10),
        'user_mean_price': u_mean_pr,
        'item_mean_price': i_mean_pr,
        'user_item_price_diff': abs(u_mean_pr - i_mean_pr) if u_mean_pr > 0 and i_mean_pr > 0 else 999,
        'item_freshness': ctx['item_freshness'].get(aid, 0),
        'item_last_day': ctx['item_last_day'].get(aid, 999),
        'user_mean_channel': u_mean_ch,
        'item_mean_channel': i_mean_ch,
    }
    return row


# ============================================================
# 4. Multi-fold Training
# ============================================================
print("\n--- Multi-fold training (weeks 1,2,3 as targets) ---")

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

all_train_rows = []
for target_week in [1, 2, 3]:
    print(f"  Fold: week {target_week} as target...")
    ctx = build_fold_data(txn, target_week)
    truth = ctx['truth']
    print(f"    Users with truth: {len(truth):,}")

    fold_rows = 0
    fold_pos = 0
    for cid, actual_items in truth.items():
        candidates = set()
        for aid in ctx['cust_repurchase'].get(cid, [])[:16]:
            candidates.add(aid)
        for aid, _ in ctx['itemcf_recs'].get(cid, [])[:20]:
            candidates.add(aid)
        for aid in ctx['cooc_recs'].get(cid, [])[:10]:
            candidates.add(aid)
        for aid in ctx['pop_top50'][:30]:
            candidates.add(aid)
        # Department popular (NEW)
        for dept in ctx['user_depts'].get(cid, []):
            for aid in ctx['dept_pop'].get(dept, [])[:4]:
                candidates.add(aid)

        for aid in candidates:
            row = build_feature_row(cid, aid, ctx)
            row['target'] = 1 if aid in actual_items else 0
            if row['target'] == 1:
                fold_pos += 1
            all_train_rows.append(row)
            fold_rows += 1

    print(f"    Fold rows: {fold_rows:,} (pos: {fold_pos:,})")
    del ctx
    gc.collect()

print(f"\n  Total training rows: {len(all_train_rows):,}")
train_df = pd.DataFrame(all_train_rows)
del all_train_rows
gc.collect()

for col in FILL_0:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)
for col in FILL_99:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(99)

X_train = train_df[FEATURE_COLS]
y_train = train_df['target']

pos_count = y_train.sum()
neg_count = (y_train == 0).sum()
print(f"  Positive: {pos_count:,}, Negative: {neg_count:,}, Ratio: 1:{neg_count/max(pos_count,1):.1f}")

# ============================================================
# 5. Train LightGBM with CV
# ============================================================
print("\n--- Training LightGBM with 3-fold CV ---")
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

# Temporal CV
folds = []
n_per_fold = len(train_df) // 3
for i in range(3):
    val_idx = list(range(i * n_per_fold, min((i + 1) * n_per_fold, len(train_df))))
    train_idx = list(range(0, i * n_per_fold)) + list(range((i + 1) * n_per_fold, len(train_df)))
    folds.append((train_idx, val_idx))

cv_results = lgb.cv(
    lgb_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    folds=folds,
    callbacks=[lgb.log_evaluation(100)],
    return_cvbooster=True,
)

best_iter = len(cv_results['valid average_precision-mean'])
print(f"  Best iteration: {best_iter}")
print(f"  CV AP: {cv_results['valid average_precision-mean'][-1]:.4f}")

# Train final model on all data
dtrain = lgb.Dataset(X_train, label=y_train)
model = lgb.train(lgb_params, dtrain, num_boost_round=min(best_iter, 800))

print("\nFeature importance (top 15):")
for fname, fimp in sorted(zip(FEATURE_COLS, model.feature_importance()), key=lambda x: -x[1])[:15]:
    print(f"  {fname}: {fimp}")

del train_df, dtrain
gc.collect()

# ============================================================
# 6. Generate Predictions
# ============================================================
print("\n--- Generating predictions ---")
ctx_pred = build_fold_data(txn, 0)
popular_top12 = ctx_pred['pop_top50'][:12]

# Filter OOS from popular
popular_filtered = [a for a in ctx_pred['pop_top50'] if a not in oos_items]
if len(popular_filtered) < 12:
    popular_filtered = ctx_pred['pop_top50']

predictions = {}

for cid in sample_sub['customer_id']:
    if cid not in ctx_pred['cust_repurchase']:
        pred = [a for a in popular_filtered[:12] if a not in oos_items]
        if len(pred) < 12:
            for a in ctx_pred['pop_top50']:
                if len(pred) >= 12:
                    break
                if a not in pred and a not in oos_items:
                    pred.append(a)
        predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])
        continue

    candidates = set()
    for aid in ctx_pred['cust_repurchase'].get(cid, [])[:16]:
        candidates.add(aid)
    for aid, _ in ctx_pred['itemcf_recs'].get(cid, [])[:20]:
        candidates.add(aid)
    for aid in ctx_pred['cooc_recs'].get(cid, [])[:10]:
        candidates.add(aid)
    for aid in ctx_pred['pop_top50'][:30]:
        candidates.add(aid)
    # Department popular
    for dept in ctx_pred['user_depts'].get(cid, []):
        for aid in ctx_pred['dept_pop'].get(dept, [])[:4]:
            candidates.add(aid)
    # Product variants
    for aid in ctx_pred['cust_recent'].get(cid, [])[:5]:
        pc = art_product.get(aid)
        if pc and pc in product_groups and len(product_groups[pc]) >= 2:
            for v in product_groups[pc]:
                if v != aid:
                    candidates.add(v)

    # Filter OOS
    candidates = candidates - oos_items

    if not candidates:
        predictions[cid] = ' '.join(f'{a:010d}' for a in popular_filtered[:12])
        continue

    rows = []
    aid_list = []
    for aid in candidates:
        rows.append(build_feature_row(cid, aid, ctx_pred))
        aid_list.append(aid)

    X_pred = pd.DataFrame(rows)[FEATURE_COLS]
    for col in FILL_0:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna(0)
    for col in FILL_99:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].fillna(99)

    scores = model.predict(X_pred)
    sorted_idx = np.argsort(-scores)
    pred = [aid_list[i] for i in sorted_idx[:12]]

    # Fill with popular (filtered)
    used = set(pred)
    for aid in popular_filtered:
        if len(pred) >= 12:
            break
        if aid not in used:
            pred.append(aid)

    predictions[cid] = ' '.join(f'{a:010d}' for a in pred[:12])

# ============================================================
# 7. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r23_multifold_rich.csv"
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
pop12_str = ' '.join(f'{a:010d}' for a in popular_filtered[:12])
val_scores = []
for cid in val_truth:
    pred_str = predictions.get(cid, pop12_str)
    pred = [int(x) for x in pred_str.split()]
    val_scores.append(apk(val_truth[cid], pred, 12))

map12_active = np.mean(val_scores)
print(f"Val MAP@12 (active): {map12_active:.6f}")

print("\n" + "=" * 60)
print("R23 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"CV AP: {cv_results['valid average_precision-mean'][-1]:.4f}")
print(f"Features: {len(FEATURE_COLS)}, OOS items filtered: {len(oos_items)}")
print(f"R21: 0.02365 | Target: 0.0300")
print("=" * 60)
