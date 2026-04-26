"""
H&M R01 Baseline: Repurchase + Recent Popular
Expected: ~0.020 MAP@12

Strategy:
- For each customer: take their last 4 weeks' purchased articles (repurchase signal)
- Fill remaining slots with globally popular articles from last week
- Output 12 predictions per customer
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA = Path(__file__).resolve().parent.parent / "data_raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("=" * 60)
print("R01: H&M Baseline — Repurchase + Recent Popular")
print("=" * 60)

# Load data
print("\nLoading data...")
txn = pd.read_csv(DATA / "transactions_train.csv", parse_dates=['t_dat'])
print(f"Transactions: {len(txn):,}")
print(f"Date range: {txn['t_dat'].min().date()} to {txn['t_dat'].max().date()}")

sub = pd.read_csv(DATA / "sample_submission.csv")
print(f"Submission customers: {len(sub):,}")

# Define time periods
max_date = txn['t_dat'].max()
print(f"\nMax date: {max_date.date()}")

# Last week for popular items
last_week_start = max_date - pd.Timedelta(days=6)
last_week = txn[txn['t_dat'] >= last_week_start]
print(f"Last 7 days: {len(last_week):,} transactions")

# Last 4 weeks for repurchase
repurchase_start = max_date - pd.Timedelta(weeks=4)
last_4w = txn[txn['t_dat'] >= repurchase_start]
print(f"Last 4 weeks: {len(last_4w):,} transactions")

# Strategy 1: Globally popular articles from last week
print("\n--- Strategy 1: Recent Popular (last 7 days) ---")
popular_articles = last_week['article_id'].value_counts().head(12).index.tolist()
print(f"Top 12 popular: {popular_articles}")

# Strategy 2: Per-customer repurchase from last 4 weeks
print("\n--- Strategy 2: Repurchase (last 4 weeks) ---")
# Group by customer, get their recent articles sorted by recency
cust_recent = (
    last_4w
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:12])  # dedup, keep order
    .to_dict()
)
print(f"Customers with recent purchases: {len(cust_recent):,} ({len(cust_recent)/len(sub)*100:.1f}%)")

# Build predictions
print("\n--- Building predictions ---")
predictions = []
no_recent = 0
for cid in sub['customer_id']:
    if cid in cust_recent:
        recent = cust_recent[cid]
        # Fill remaining slots with popular articles
        pred = recent[:12]
        for a in popular_articles:
            if len(pred) >= 12:
                break
            if a not in pred:
                pred.append(a)
        predictions.append(' '.join(f'{a:010d}' for a in pred[:12]))
    else:
        predictions.append(' '.join(f'{a:010d}' for a in popular_articles))
        no_recent += 1

print(f"Customers using popular fallback: {no_recent:,} ({no_recent/len(sub)*100:.1f}%)")

# Save submission
sub['prediction'] = predictions
sub_path = OUTPUTS / "submission_r01_baseline.csv"
sub.to_csv(sub_path, index=False)

print(f"\nSubmission saved: {sub_path}")
print(f"Total predictions: {len(predictions):,}")
print(f"Sample prediction: {predictions[0]}")

# Quick validation: MAP@12 on last week (holdout)
print("\n--- Validation: MAP@12 on last week ---")
val_start = max_date - pd.Timedelta(days=6)
val_txn = txn[txn['t_dat'] >= val_start]
val_ground_truth = val_txn.groupby('customer_id')['article_id'].apply(set).to_dict()

# Use data before val_start for prediction
train_txn = txn[txn['t_dat'] < val_start]
train_recent_start = val_start - pd.Timedelta(weeks=4)
train_recent = train_txn[train_txn['t_dat'] >= train_recent_start]

train_popular = train_txn[train_txn['t_dat'] >= val_start - pd.Timedelta(days=6)]['article_id'].value_counts().head(12).index.tolist()
train_cust_recent = (
    train_recent
    .sort_values('t_dat', ascending=False)
    .groupby('customer_id')['article_id']
    .apply(lambda x: list(dict.fromkeys(x.tolist()))[:12])
    .to_dict()
)

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

scores = []
for cid, actual in val_ground_truth.items():
    if cid in train_cust_recent:
        pred = train_cust_recent[cid][:12]
        for a in train_popular:
            if len(pred) >= 12:
                break
            if a not in pred:
                pred.append(a)
    else:
        pred = train_popular
    scores.append(apk(actual, pred, 12))

map12 = np.mean(scores)
print(f"Val MAP@12: {map12:.6f}")
print(f"Val customers with purchases: {len(val_ground_truth):,}")

print("\n" + "=" * 60)
print("R01 Complete")
print("=" * 60)
