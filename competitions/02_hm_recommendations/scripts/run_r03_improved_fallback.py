"""
H&M R03: R02 Ranker + R01-style fallback
R02 val MAP@12 was great (0.065) but LB was worse (0.018 vs R01's 0.022).
Root cause: 80% of customers get fallback predictions. Age-bin popular is worse than global popular.

Fix: Use R02's ranker for active customers + R01's global popular fallback for inactive.
Also: increase candidate count for active customers.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
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
print("R03: R02 Ranker + Better Fallback")
print("=" * 60)

# ============================================================
# 1. Load Data (last 12 weeks)
# ============================================================
print("\n--- Loading data ---")
articles = pd.read_csv(DATA / "articles.csv")
customers = pd.read_csv(DATA / "customers.csv")
sample_sub = pd.read_csv(DATA / "sample_submission.csv")

txn = pd.read_csv(DATA / "transactions_train.csv", parse_dates=['t_dat'])
max_date = txn['t_dat'].max()
txn['week'] = ((max_date - txn['t_dat']).dt.days // 7).astype('int8')
txn = txn[txn['week'] <= 12].copy()
gc.collect()

article_dept = articles.set_index('article_id')['department_no'].to_dict()
customers['age'] = customers['age'].fillna(customers['age'].median())
customers['age_bin'] = pd.cut(customers['age'], bins=[0, 18, 25, 35, 45, 55, 100], labels=False).fillna(3).astype(int)

print(f"Transactions: {len(txn):,}")

# ============================================================
# 2. Frequent Pairs
# ============================================================
print("\n--- Frequent pairs ---")
recent_all = txn[txn['week'] <= 10]
buckets = recent_all.groupby(['t_dat', 'customer_id', 'sales_channel_id'])['article_id'].apply(set).reset_index()
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
freq_pairs = {k: [p[0] for p in sorted(v, key=lambda x: -x[1])[:12]] for k, v in freq_pairs.items()}
print(f"Articles with pairs: {len(freq_pairs):,}")
del buckets, pair_counts; gc.collect()

# ============================================================
# 3. Feature Engineering
# ============================================================
print("\n--- Features ---")

def compute_features(txn_data, customers_df, articles_df):
    cf = customers_df[['customer_id', 'age', 'age_bin']].copy()
    cf['is_active'] = (customers_df['club_member_status'] == 'ACTIVE').astype(int)

    stats = txn_data.groupby('customer_id').agg(
        cust_purchases=('article_id', 'count'),
        cust_unique=('article_id', 'nunique'),
        cust_avg_price=('price', 'mean'),
        cust_recency=('week', 'min'),
    ).reset_index()

    r4 = txn_data[txn_data['week'] <= 4].groupby('customer_id').size().reset_index(name='cust_4w')
    r1 = txn_data[txn_data['week'] <= 1].groupby('customer_id').size().reset_index(name='cust_1w')

    cf = cf.merge(stats, on='customer_id', how='left')
    cf = cf.merge(r4, on='customer_id', how='left')
    cf = cf.merge(r1, on='customer_id', how='left')
    for col in ['cust_purchases', 'cust_unique', 'cust_avg_price', 'cust_recency', 'cust_4w', 'cust_1w']:
        cf[col] = cf[col].fillna(0)

    # Preferred dept
    recent_dept = txn_data[txn_data['week'] <= 10][['customer_id', 'article_id']].copy()
    recent_dept['dept'] = recent_dept['article_id'].map(article_dept)
    cust_pref = recent_dept.groupby('customer_id')['dept'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else -1
    ).reset_index(name='cust_pref_dept')
    cf = cf.merge(cust_pref, on='customer_id', how='left')
    cf['cust_pref_dept'] = cf['cust_pref_dept'].fillna(-1)
    del recent_dept

    af = articles_df[['article_id', 'product_code', 'product_type_no',
                       'colour_group_code', 'department_no', 'index_group_no',
                       'section_no', 'garment_group_no']].copy()

    art_stats = txn_data.groupby('article_id').agg(
        art_total=('customer_id', 'count'),
        art_buyers=('customer_id', 'nunique'),
        art_avg_price=('price', 'mean'),
    ).reset_index()

    for w, name in [(1, 'art_1w'), (2, 'art_2w'), (4, 'art_4w')]:
        ws = txn_data[txn_data['week'] <= w].groupby('article_id').size().reset_index(name=name)
        art_stats = art_stats.merge(ws, on='article_id', how='left')
        art_stats[name] = art_stats[name].fillna(0)
    art_stats['art_popularity'] = art_stats['art_1w'] * 4 + art_stats['art_2w'] * 2 + art_stats['art_4w']

    art_repeat = txn_data.groupby(['customer_id', 'article_id']).size().reset_index(name='cnt')
    art_repeat = art_repeat[art_repeat['cnt'] > 1].groupby('article_id').size().reset_index(name='art_repeat_buyers')
    art_stats = art_stats.merge(art_repeat, on='article_id', how='left')
    art_stats['art_repeat_ratio'] = (art_stats['art_repeat_buyers'] / art_stats['art_buyers']).fillna(0)

    af = af.merge(art_stats, on='article_id', how='left')
    for col in ['art_total', 'art_buyers', 'art_avg_price', 'art_1w', 'art_2w', 'art_4w',
                'art_popularity', 'art_repeat_buyers', 'art_repeat_ratio']:
        af[col] = af[col].fillna(0)

    return cf, af

cf, af = compute_features(txn, customers, articles)
print(f"Customer features: {cf.shape[1]-1}, Article features: {af.shape[1]-1}")

# ============================================================
# 4. Candidate Generation
# ============================================================
print("\n--- Building datasets ---")

def generate_candidates(txn_all, target_week, target_customers=None):
    history = txn_all[(txn_all['week'] > target_week) & (txn_all['week'] <= target_week + 10)]

    if target_customers is None:
        target_customers = txn_all[txn_all['week'] == target_week]['customer_id'].unique()

    target_customers = set(target_customers)
    active_customers = set(history[history['week'] <= 4]['customer_id'].unique()) & target_customers

    records = defaultdict(lambda: {})

    # Strategy 1: Repurchase
    recent = history[history['week'] <= target_week + 4]
    cust_articles = recent.groupby('customer_id')['article_id'].apply(
        lambda x: list(dict.fromkeys(x.tolist()))
    ).to_dict()

    for cid in active_customers:
        for rank, aid in enumerate(cust_articles.get(cid, [])[:30]):
            records[(cid, aid)]['repurchase'] = rank + 1

    # Strategy 2: Frequent pairs
    for cid in active_customers:
        recent_arts = cust_articles.get(cid, [])[:5]
        pair_rank = 0
        for art in recent_arts:
            if art in freq_pairs:
                for related in freq_pairs[art]:
                    if (cid, related) not in records or 'item2item' not in records[(cid, related)]:
                        pair_rank += 1
                        records[(cid, related)]['item2item'] = pair_rank

    # Strategy 3: Global popular (top 100 from last week)
    pop100 = history[history['week'] <= target_week + 1]['article_id'].value_counts().head(100)
    for cid in active_customers:
        for rank, (aid, _) in enumerate(pop100.items()):
            records[(cid, aid)]['popular'] = rank + 1

    # Strategy 4: Department popular
    recent_depts = recent.copy()
    recent_depts['dept'] = recent_depts['article_id'].map(article_dept)
    dept_pop = recent_depts[recent_depts['week'] <= target_week + 1].groupby('dept')['article_id'].apply(
        lambda x: x.value_counts().head(30).index.tolist()
    ).to_dict()

    for cid in active_customers:
        cust_arts = cust_articles.get(cid, [])
        if not cust_arts:
            continue
        cust_depts = pd.Series([article_dept.get(a, -1) for a in cust_arts]).value_counts().head(3).index
        rank = 0
        for d in cust_depts:
            if d in dept_pop:
                for aid in dept_pop[d]:
                    if (cid, aid) not in records or 'dept_pop' not in records[(cid, aid)]:
                        rank += 1
                        records[(cid, aid)]['dept_pop'] = rank

    rows = []
    for (cid, aid), strategies in records.items():
        row = {'customer_id': cid, 'article_id': aid}
        row['n_strategies'] = len(strategies)
        for sname in ['repurchase', 'item2item', 'popular', 'dept_pop']:
            row[f'strat_{sname}'] = 1 if sname in strategies else 0
            row[f'rank_{sname}'] = strategies.get(sname, 999)
        rows.append(row)

    if not rows:
        cols = ['customer_id', 'article_id', 'n_strategies']
        for sname in ['repurchase', 'item2item', 'popular', 'dept_pop']:
            cols.extend([f'strat_{sname}', f'rank_{sname}'])
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)

def build_dataset(txn_all, target_week, cf_df, af_df):
    target_purchases = txn_all[txn_all['week'] == target_week]
    target_customers = target_purchases['customer_id'].unique()

    cand_df = generate_candidates(txn_all, target_week, target_customers)
    if len(cand_df) == 0:
        return None

    truth = target_purchases[['customer_id', 'article_id']].drop_duplicates()
    truth['target'] = 1
    cand_df = cand_df.merge(truth, on=['customer_id', 'article_id'], how='left')
    cand_df['target'] = cand_df['target'].fillna(0).astype(int)

    cand_df = cand_df.merge(cf_df, on='customer_id', how='left')
    cand_df = cand_df.merge(af_df, on='article_id', how='left')

    history = txn_all[(txn_all['week'] > target_week) & (txn_all['week'] <= target_week + 10)]

    hist_pairs = history[['customer_id', 'article_id']].drop_duplicates()
    hist_pairs['bought_before'] = 1
    cand_df = cand_df.merge(hist_pairs, on=['customer_id', 'article_id'], how='left')
    cand_df['bought_before'] = cand_df['bought_before'].fillna(0).astype(int)

    dept_small = history[['customer_id', 'article_id']].copy()
    dept_small['dept'] = dept_small['article_id'].map(article_dept)
    dept_counts = dept_small.groupby(['customer_id', 'dept']).size().reset_index(name='cust_dept_count')
    cand_df['dept'] = cand_df['article_id'].map(article_dept)
    cand_df = cand_df.merge(dept_counts, left_on=['customer_id', 'dept'], right_on=['customer_id', 'dept'], how='left')
    cand_df['cust_dept_count'] = cand_df['cust_dept_count'].fillna(0)
    cand_df = cand_df.drop(columns=['dept'])
    del dept_small, dept_counts

    return cand_df

# Train on weeks 1-3
train_dfs = []
for w in range(1, 4):
    print(f"  Week {w}...")
    wdf = build_dataset(txn, w, cf, af)
    if wdf is not None and len(wdf) > 0:
        total_actual = txn[txn['week'] == w][['customer_id', 'article_id']].drop_duplicates().shape[0]
        recalled = wdf['target'].sum()
        print(f"    {len(wdf):,} rows, {wdf['target'].mean()*100:.2f}% pos, recall={recalled/total_actual*100:.1f}%")
        train_dfs.append(wdf)

train_df = pd.concat(train_dfs, ignore_index=True)
print(f"\nTotal training: {len(train_df):,}")

# Validation (week 0)
print("\nVal (week 0)...")
val_df = build_dataset(txn, 0, cf, af)
total_val = txn[txn['week'] == 0][['customer_id', 'article_id']].drop_duplicates().shape[0]
print(f"Val: {len(val_df):,} rows, recall={val_df['target'].sum()/total_val*100:.1f}%")

# ============================================================
# 5. Train
# ============================================================
print("\n--- Training ---")
feature_cols = [c for c in train_df.columns if c not in ['customer_id', 'article_id', 'target']]
print(f"Features: {len(feature_cols)}")

train_df = train_df.sort_values('customer_id').reset_index(drop=True)
val_df = val_df.sort_values('customer_id').reset_index(drop=True)

train_groups = train_df.groupby('customer_id', sort=False).size().values
val_groups = val_df.groupby('customer_id', sort=False).size().values

ranker = lgb.LGBMRanker(
    objective='lambdarank', metric='ndcg', eval_at=[12],
    n_estimators=500, learning_rate=0.05, num_leaves=255,
    min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, reg_alpha=0.1, reg_lambda=0.1,
    verbosity=-1, random_state=42,
)

ranker.fit(
    train_df[feature_cols].values, train_df['target'].values, group=train_groups,
    eval_set=[(val_df[feature_cols].values, val_df['target'].values)],
    eval_group=[val_groups],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)
print(f"Best iteration: {ranker.best_iteration_}")

# ============================================================
# 6. Validate
# ============================================================
print("\n--- Validation ---")
val_df['score'] = ranker.predict(val_df[feature_cols].values)

def apk(actual, predicted, k=12):
    if not actual: return 0.0
    score = 0.0; num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0; score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

val_preds = val_df.sort_values('score', ascending=False).groupby('customer_id')['article_id'].apply(
    lambda x: list(x.head(12))).to_dict()
val_truth = txn[txn['week'] == 0].groupby('customer_id')['article_id'].apply(set).to_dict()

scores = [apk(val_truth.get(cid, set()), val_preds.get(cid, [])) for cid in val_preds]
map12_active = np.mean(scores)

# Also compute MAP@12 for ALL submission customers (including fallback)
# Use global popular as fallback for inactive customers
pop12_global = txn[txn['week'] <= 1]['article_id'].value_counts().head(12).index.tolist()

all_scores = []
# Active customers (ranker predictions)
for cid in val_preds:
    all_scores.append(apk(val_truth.get(cid, set()), val_preds[cid]))
# Inactive customers (global popular fallback) — check if they have purchases
val_all_customers = sample_sub['customer_id'].values
ranker_customers = set(val_preds.keys())
for cid in val_all_customers:
    if cid not in ranker_customers:
        all_scores.append(apk(val_truth.get(cid, set()), pop12_global))

map12_all = np.mean(all_scores)
print(f"Val MAP@12 (active only): {map12_active:.6f}")
print(f"Val MAP@12 (all customers): {map12_all:.6f}")
print(f"Active: {len(val_preds):,}, Inactive: {len(val_all_customers) - len(val_preds):,}")

# Feature importance
imp = pd.DataFrame({'feature': feature_cols, 'importance': ranker.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
print(f"\nTop 15 features:")
print(imp.head(15).to_string(index=False))

# ============================================================
# 7. Test Submission — R02 ranker + R01 global popular fallback
# ============================================================
print("\n--- Test submission ---")
active = txn[txn['week'] <= 10]['customer_id'].unique()
print(f"Active customers: {len(active):,}")

# Generate candidates for active customers
test_cand = generate_candidates(txn, -1, active)
test_cand = test_cand.merge(cf, on='customer_id', how='left')
test_cand = test_cand.merge(af, on='article_id', how='left')

# Interaction features
hist_pairs = txn[['customer_id', 'article_id']].drop_duplicates()
hist_pairs['bought_before'] = 1
test_cand = test_cand.merge(hist_pairs, on=['customer_id', 'article_id'], how='left')
test_cand['bought_before'] = test_cand['bought_before'].fillna(0).astype(int)

dept_small = txn[['customer_id', 'article_id']].copy()
dept_small['dept'] = dept_small['article_id'].map(article_dept)
dept_counts = dept_small.groupby(['customer_id', 'dept']).size().reset_index(name='cust_dept_count')
test_cand['dept'] = test_cand['article_id'].map(article_dept)
test_cand = test_cand.merge(dept_counts, left_on=['customer_id', 'dept'], right_on=['customer_id', 'dept'], how='left')
test_cand['cust_dept_count'] = test_cand['cust_dept_count'].fillna(0)
test_cand = test_cand.drop(columns=['dept'])

print(f"Test candidates: {len(test_cand):,}")

test_cand['score'] = ranker.predict(test_cand[feature_cols].values)
test_preds = (
    test_cand.sort_values('score', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: ' '.join(f'{a:010d}' for a in x.head(12)))
    .to_dict()
)

# Global popular fallback (R01-style)
popular_str = ' '.join(f'{a:010d}' for a in pop12_global)

sub = sample_sub.copy()
sub['prediction'] = sub['customer_id'].map(test_preds).fillna(popular_str)

sub_path = OUTPUTS / "submission_r03_improved_fallback.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}")
print(f"Ranker predictions: {len(test_preds):,}, Global popular fallback: {(sub['prediction'] == popular_str).sum():,}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"R03 Complete")
print(f"Val MAP@12 (active): {map12_active:.6f}")
print(f"Val MAP@12 (all): {map12_all:.6f}")
print(f"R01 LB: 0.02207 | R02 LB: 0.01800")
print("=" * 60)
