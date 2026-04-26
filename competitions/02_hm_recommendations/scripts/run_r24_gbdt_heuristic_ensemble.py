"""
H&M R24: GBDT + Heuristic Ensemble

R21 (GBDT): 0.02365 — best learned model
R17 (heuristic): 0.02282 — best rule-based

They likely capture different signals:
  - R17: Repurchase always first, ItemCF fill, cooc fill, popular fill
  - R21: GBDT reorders by learned features (age, price, trend match)

Strategy: For each user, blend top-K from each model.
  - Take top 6 from R21 GBDT (learned ranking)
  - Take top 6 from R17 heuristic (recall-based)
  - Merge with dedup, maintain order
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
print("R24: GBDT + Heuristic Ensemble")
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

# ============================================================
# 2. Pre-compute Shared Data
# ============================================================
print("\n--- Pre-computing ---")
txn_features = txn[txn['week'] >= 1].copy()

# Item stats
item_total_buys = txn_features['article_id'].value_counts().to_dict()
item_unique_buyers = txn_features.groupby('article_id')['customer_id'].nunique().to_dict()
item_recent_week_buys = txn_features[txn_features['week'] == 1]['article_id'].value_counts().to_dict()
item_last2w_buys = txn_features[txn_features['week'] <= 2]['article_id'].value_counts().to_dict()
item_last4w_buys = txn_features[txn_features['week'] <= 4]['article_id'].value_counts().to_dict()
item_weekly_avg = {aid: count / 12.0 for aid, count in item_total_buys.items()}
item_sales_trend = {aid: item_recent_week_buys.get(aid, 0) / max(item_weekly_avg.get(aid, 0.1), 0.1) for aid in item_total_buys}
item_repurchase_ratio = {aid: item_unique_buyers.get(aid, 1) / max(item_total_buys.get(aid, 1), 1) for aid in item_total_buys}

# Age-demographic
txn_with_age = txn_features.merge(customers[['customer_id', 'age']], on='customer_id', how='left')
item_user_mean_age = txn_with_age.groupby('article_id')['age'].mean().to_dict()
item_user_age_std = txn_with_age.groupby('article_id')['age'].std().fillna(0).to_dict()
del txn_with_age
gc.collect()

# Price
user_mean_price = txn_features.groupby('customer_id')['price'].mean().to_dict()
item_mean_price = txn_features.groupby('article_id')['price'].mean().to_dict()

# Time
item_first_day = txn_features.groupby('article_id')['days_ago'].max().to_dict()
item_last_day_stat = txn_features.groupby('article_id')['days_ago'].min().to_dict()
item_freshness = {aid: max_date.day - item_first_day.get(aid, 0) for aid in item_first_day}

# Channel
user_mean_channel = txn_features.groupby('customer_id')['sales_channel_id'].mean().to_dict()
item_mean_channel = txn_features.groupby('article_id')['sales_channel_id'].mean().to_dict()

# Popular
txn_features['decay'] = np.exp(-0.15 * txn_features['week'].astype(float))
pop_scores = txn_features.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop_top50 = list(pop_scores.head(50).index)
pop_score_dict = pop_scores.to_dict()

# Repurchase
last_6w = txn_features[txn_features['week'] <= 7]
cust_repurchase = (
    last_6w.sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20])
    .to_dict()
)

# ItemCF
cust_sequences = (
    txn_features[txn_features['week'] <= 11].sort_values('t_dat')
    .groupby('customer_id')['article_id'].apply(list).to_dict()
)
sim_item = defaultdict(lambda: defaultdict(float))
for cid, items in cust_sequences.items():
    n = len(items)
    if n < 2: continue
    log_len = math.log(1 + n)
    for i, ii in enumerate(items):
        for j, jj in enumerate(items):
            if i == j: continue
            a = 1.0 if j > i else 0.9
            sim_item[ii][jj] += a * (0.7 ** (abs(j - i) - 1)) / log_len
itemcf_lookup = {k: sorted(v.items(), key=lambda x: -x[1])[:10] for k, v in sim_item.items()}
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
for (a, b), c in pair_counts.items():
    freq_pairs[a].append((b, c))
    freq_pairs[b].append((a, c))
freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:5]] for k, v in freq_pairs.items()}
del buckets, pair_counts
gc.collect()

# User stats
user_total_buys = txn_features.groupby('customer_id').size().to_dict()
user_unique_items = txn_features.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_purchase_day = txn_features.groupby('customer_id')['days_ago'].min().to_dict()
user_item_buy_count = txn.groupby(['customer_id', 'article_id']).size().to_dict()
user_item_last_day = txn.groupby(['customer_id', 'article_id'])['days_ago'].min().to_dict()

# Per-customer recall
cust_recent = (
    txn_features[txn_features['week'] <= 5].sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10]).to_dict()
)
itemcf_user_recs = {}
for cid, recent in cust_recent.items():
    cf = defaultdict(float)
    for aid in recent[:5]:
        if aid in itemcf_lookup:
            for r, s in itemcf_lookup[aid]:
                cf[r] += s
    itemcf_user_recs[cid] = sorted(cf.items(), key=lambda x: -x[1])

cooc_user_recs = {}
for cid, recent in cust_recent.items():
    items, seen = [], set()
    for aid in recent[:5]:
        if aid in freq_pairs:
            for r in freq_pairs[aid]:
                if r not in seen:
                    items.append(r)
                    seen.add(r)
    cooc_user_recs[cid] = items

val_truth = txn[txn['week'] == 0].groupby('customer_id')['article_id'].apply(set).to_dict()
del txn_features
gc.collect()

# ============================================================
# 3. Train GBDT (same as R21)
# ============================================================
print("\n--- Training GBDT ---")
import lightgbm as lgb

all_active = set(cust_repurchase.keys()) | set(cust_recent.keys())
train_rows = []
for cid in all_active:
    if cid not in val_truth: continue
    actual = val_truth[cid]
    candidates = {}
    for rank, aid in enumerate(cust_repurchase.get(cid, [])[:16]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['repurchase_rank'] = rank
        candidates[aid]['repurchase_score'] = 1.0 / (1.0 + rank)
    for rank, (aid, score) in enumerate(itemcf_user_recs.get(cid, [])[:20]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['itemcf_rank'] = rank
        candidates[aid]['itemcf_score'] = score
    for rank, aid in enumerate(cooc_user_recs.get(cid, [])[:10]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['cooc_rank'] = rank
    for rank, aid in enumerate(pop_top50[:30]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['pop_rank'] = rank

    u_age = cust_age.get(cid, 35)
    u_mpr = user_mean_price.get(cid, 0)
    u_mch = user_mean_channel.get(cid, 1.5)

    for aid, feat in candidates.items():
        i_mage = item_user_mean_age.get(aid, 35)
        i_mpr = item_mean_price.get(aid, 0)
        train_rows.append({
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
            'num_recall_sources': sum(1 for k in ['repurchase_rank','itemcf_rank','cooc_rank','pop_rank'] if k in feat),
            'user_item_buys': user_item_buy_count.get((cid, aid), 0),
            'user_item_last_day': user_item_last_day.get((cid, aid), 999),
            'user_total_buys': user_total_buys.get(cid, 0),
            'user_unique_items': user_unique_items.get(cid, 0),
            'user_last_day': user_last_purchase_day.get(cid, 999),
            'user_age': u_age,
            'item_popularity': pop_score_dict.get(aid, 0),
            'item_total_buys': item_total_buys.get(aid, 0),
            'item_unique_buyers': item_unique_buyers.get(aid, 0),
            'item_dept': art_dept.get(aid, -1), 'item_section': art_section.get(aid, -1),
            'item_garment': art_garment.get(aid, -1), 'item_category': art_category.get(aid, -1),
            'item_color': art_color.get(aid, -1),
            'item_recent_week_buys': item_recent_week_buys.get(aid, 0),
            'item_last2w_buys': item_last2w_buys.get(aid, 0),
            'item_last4w_buys': item_last4w_buys.get(aid, 0),
            'item_sales_trend': item_sales_trend.get(aid, 1.0),
            'item_repurchase_ratio': item_repurchase_ratio.get(aid, 1.0),
            'item_user_mean_age': i_mage, 'user_item_age_diff': abs(u_age - i_mage),
            'item_user_age_std': item_user_age_std.get(aid, 10),
            'user_mean_price': u_mpr, 'item_mean_price': i_mpr,
            'user_item_price_diff': abs(u_mpr - i_mpr) if u_mpr > 0 and i_mpr > 0 else 999,
            'item_freshness': item_freshness.get(aid, 0), 'item_last_day': item_last_day_stat.get(aid, 999),
            'user_mean_channel': u_mch, 'item_mean_channel': item_mean_channel.get(aid, 1.5),
        })

train_df = pd.DataFrame(train_rows)
del train_rows
gc.collect()

FILL_0 = ['repurchase_score','itemcf_score','item_popularity','item_total_buys','item_unique_buyers',
          'item_recent_week_buys','item_last2w_buys','item_last4w_buys','item_sales_trend',
          'item_repurchase_ratio','item_user_mean_age','item_user_age_std','item_mean_price',
          'item_freshness','user_item_buys','user_total_buys','user_unique_items','user_age',
          'user_mean_price','num_recall_sources','user_mean_channel','item_mean_channel']
FILL_99 = ['repurchase_rank','itemcf_rank','cooc_rank','pop_rank','user_item_last_day',
           'user_last_day','item_last_day','user_item_price_diff']
for c in FILL_0:
    if c in train_df.columns: train_df[c] = train_df[c].fillna(0)
for c in FILL_99:
    if c in train_df.columns: train_df[c] = train_df[c].fillna(99)

FEATURE_COLS = [
    'repurchase_score','repurchase_rank','itemcf_score','itemcf_rank','cooc_rank','pop_rank',
    'is_repurchase','is_itemcf','is_cooc','is_pop','num_recall_sources',
    'user_item_buys','user_item_last_day','user_total_buys','user_unique_items','user_last_day','user_age',
    'item_popularity','item_total_buys','item_unique_buyers',
    'item_dept','item_section','item_garment','item_category','item_color',
    'item_recent_week_buys','item_last2w_buys','item_last4w_buys','item_sales_trend','item_repurchase_ratio',
    'item_user_mean_age','user_item_age_diff','item_user_age_std',
    'user_mean_price','item_mean_price','user_item_price_diff',
    'item_freshness','item_last_day','user_mean_channel','item_mean_channel',
]

X_train = train_df[FEATURE_COLS]
y_train = train_df['target']
pos = y_train.sum()
neg = (y_train == 0).sum()
print(f"  Rows: {len(train_df):,}, Pos: {pos:,}, Neg: {neg:,}, Ratio 1:{neg/max(pos,1):.1f}")

dtrain = lgb.Dataset(X_train, label=y_train)
model = lgb.train(
    {'objective':'binary','metric':'average_precision','boosting_type':'gbdt',
     'num_leaves':63,'learning_rate':0.05,'feature_fraction':0.8,'bagging_fraction':0.8,
     'bagging_freq':5,'scale_pos_weight':neg/max(pos,1),'min_child_samples':50,
     'verbose':-1,'n_jobs':-1,'seed':42},
    dtrain, num_boost_round=500, valid_sets=[dtrain], callbacks=[lgb.log_evaluation(100)])

del train_df, dtrain
gc.collect()

# ============================================================
# 4. Generate BOTH Predictions and Ensemble
# ============================================================
print("\n--- Generating ensemble predictions ---")

# Rebuild from full data
last_6w_full = txn[txn['week'] <= 6]
cust_rep_full = (
    last_6w_full.sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:20]).to_dict()
)
user_item_buys_full = txn.groupby(['customer_id','article_id']).size().to_dict()
user_item_last_full = txn.groupby(['customer_id','article_id'])['days_ago'].min().to_dict()
user_buys_full = txn.groupby('customer_id').size().to_dict()
user_uniq_full = txn.groupby('customer_id')['article_id'].nunique().to_dict()
user_last_full = txn.groupby('customer_id')['days_ago'].min().to_dict()
user_mpr_full = txn.groupby('customer_id')['price'].mean().to_dict()
item_mpr_full = txn.groupby('article_id')['price'].mean().to_dict()
txn['decay'] = np.exp(-0.15 * txn['week'].astype(float))
pop_full = txn.groupby('article_id')['decay'].sum().sort_values(ascending=False)
pop50_full = list(pop_full.head(50).index)
pop_dict_full = pop_full.to_dict()
pop12 = pop50_full[:12]
recent_w0 = txn[txn['week'] == 0]['article_id'].value_counts().to_dict()
user_mch_full = txn.groupby('customer_id')['sales_channel_id'].mean().to_dict()
item_mch_full = txn.groupby('article_id')['sales_channel_id'].mean().to_dict()

cust_recent_full = (
    txn[txn['week'] <= 4].sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:10]).to_dict()
)
itemcf_full = {}
for cid, recent in cust_recent_full.items():
    cf = defaultdict(float)
    for aid in recent[:5]:
        if aid in itemcf_lookup:
            for r, s in itemcf_lookup[aid]: cf[r] += s
    itemcf_full[cid] = sorted(cf.items(), key=lambda x: -x[1])

cooc_full = {}
for cid, recent in cust_recent_full.items():
    items, seen = [], set()
    for aid in recent[:5]:
        if aid in freq_pairs:
            for r in freq_pairs[aid]:
                if r not in seen: items.append(r); seen.add(r)
    cooc_full[cid] = items

del last_6w_full, cust_recent_full
gc.collect()

# Generate predictions with blending
predictions = {}
N_GBDT = 6  # Top N from GBDT
N_HEUR = 6  # Top N from heuristic

for cid in sample_sub['customer_id']:
    if cid not in cust_rep_full:
        predictions[cid] = ' '.join(f'{a:010d}' for a in pop12)
        continue

    # === Heuristic prediction (R17 style) ===
    heur_pred = []
    repurchase = cust_rep_full.get(cid, [])[:12]
    heur_pred.extend(repurchase)

    remaining = 12 - len(heur_pred)
    if remaining > 0:
        cf_items = [a for a, _ in itemcf_full.get(cid, []) if a not in set(heur_pred)]
        for a in cf_items[:remaining]:
            heur_pred.append(a)
            remaining -= 1
            if remaining <= 0: break

    if remaining > 0:
        for a in cooc_full.get(cid, []):
            if remaining <= 0: break
            if a not in set(heur_pred):
                heur_pred.append(a)
                remaining -= 1

    if remaining > 0:
        used = set(heur_pred)
        for a in pop50_full:
            if remaining <= 0: break
            if a not in used:
                heur_pred.append(a)
                remaining -= 1

    # === GBDT prediction (R21 style) ===
    candidates = {}
    for rank, aid in enumerate(cust_rep_full.get(cid, [])[:16]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['repurchase_rank'] = rank
        candidates[aid]['repurchase_score'] = 1.0 / (1.0 + rank)
    for rank, (aid, score) in enumerate(itemcf_full.get(cid, [])[:20]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['itemcf_rank'] = rank
        candidates[aid]['itemcf_score'] = score
    for rank, aid in enumerate(cooc_full.get(cid, [])[:10]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['cooc_rank'] = rank
    for rank, aid in enumerate(pop50_full[:30]):
        if aid not in candidates: candidates[aid] = {}
        candidates[aid]['pop_rank'] = rank

    if not candidates:
        predictions[cid] = ' '.join(f'{a:010d}' for a in heur_pred[:12])
        continue

    u_age = cust_age.get(cid, 35)
    u_mpr = user_mpr_full.get(cid, 0)
    u_mch = user_mch_full.get(cid, 1.5)

    rows, aids = [], []
    for aid, feat in candidates.items():
        i_mage = item_user_mean_age.get(aid, 35)
        i_mpr = item_mpr_full.get(aid, 0)
        rows.append({
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
            'num_recall_sources': sum(1 for k in ['repurchase_rank','itemcf_rank','cooc_rank','pop_rank'] if k in feat),
            'user_item_buys': user_item_buys_full.get((cid, aid), 0),
            'user_item_last_day': user_item_last_full.get((cid, aid), 999),
            'user_total_buys': user_buys_full.get(cid, 0),
            'user_unique_items': user_uniq_full.get(cid, 0),
            'user_last_day': user_last_full.get(cid, 999),
            'user_age': u_age,
            'item_popularity': pop_dict_full.get(aid, 0),
            'item_total_buys': item_total_buys.get(aid, 0),
            'item_unique_buyers': item_unique_buyers.get(aid, 0),
            'item_dept': art_dept.get(aid, -1), 'item_section': art_section.get(aid, -1),
            'item_garment': art_garment.get(aid, -1), 'item_category': art_category.get(aid, -1),
            'item_color': art_color.get(aid, -1),
            'item_recent_week_buys': recent_w0.get(aid, 0),
            'item_last2w_buys': item_last2w_buys.get(aid, 0),
            'item_last4w_buys': item_last4w_buys.get(aid, 0),
            'item_sales_trend': item_sales_trend.get(aid, 1.0),
            'item_repurchase_ratio': item_repurchase_ratio.get(aid, 1.0),
            'item_user_mean_age': i_mage, 'user_item_age_diff': abs(u_age - i_mage),
            'item_user_age_std': item_user_age_std.get(aid, 10),
            'user_mean_price': u_mpr, 'item_mean_price': i_mpr,
            'user_item_price_diff': abs(u_mpr - i_mpr) if u_mpr > 0 and i_mpr > 0 else 999,
            'item_freshness': item_freshness.get(aid, 0), 'item_last_day': item_last_day_stat.get(aid, 999),
            'user_mean_channel': u_mch, 'item_mean_channel': item_mch_full.get(aid, 1.5),
        })
        aids.append(aid)

    X_pred = pd.DataFrame(rows)[FEATURE_COLS]
    for c in FILL_0:
        if c in X_pred.columns: X_pred[c] = X_pred[c].fillna(0)
    for c in FILL_99:
        if c in X_pred.columns: X_pred[c] = X_pred[c].fillna(99)

    scores = model.predict(X_pred)
    sorted_idx = np.argsort(-scores)
    gbdt_pred = [aids[i] for i in sorted_idx[:12]]

    # === Ensemble: interleave GBDT top-6 + Heuristic top-6 ===
    final = []
    seen = set()
    # Take top N_GBDT from GBDT
    for a in gbdt_pred[:N_GBDT]:
        if a not in seen:
            final.append(a)
            seen.add(a)
    # Take top N_HEUR from heuristic (deduped)
    for a in heur_pred[:N_HEUR + len(final)]:
        if len(final) >= 12: break
        if a not in seen:
            final.append(a)
            seen.add(a)
    # Fill with popular
    for a in pop50_full:
        if len(final) >= 12: break
        if a not in seen:
            final.append(a)
            seen.add(a)

    predictions[cid] = ' '.join(f'{a:010d}' for a in final[:12])

# ============================================================
# 5. Save & Validate
# ============================================================
sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(predictions)
sub_path = OUTPUTS / "submission_r24_gbdt_heuristic_ensemble.csv"
sub.to_csv(sub_path, index=False)
print(f"\nSubmission saved: {sub_path}")

active = sum(1 for cid in sample_sub['customer_id'] if cid in cust_rep_full)
print(f"Active: {active:,}, Fallback: {len(sample_sub) - active:,}")

print("\n--- Validation ---")
def apk(actual, predicted, k=12):
    if not actual: return 0.0
    score = 0.0
    nh = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            nh += 1.0
            score += nh / (i + 1.0)
    return score / min(len(actual), k)

vs = []
p12 = ' '.join(f'{a:010d}' for a in pop12)
for cid in val_truth:
    ps = predictions.get(cid, p12)
    vs.append(apk(val_truth[cid], [int(x) for x in ps.split()], 12))
print(f"Val MAP@12 (active): {np.mean(vs):.6f}")

print("\n" + "=" * 60)
print("R24 Complete")
print(f"Val MAP@12 (active): {np.mean(vs):.6f}")
print(f"Ensemble: GBDT top-{N_GBDT} + Heuristic top-{N_HEUR}")
print(f"R21: 0.02365 | R17: 0.02282 | Target: 0.0300")
print("=" * 60)
