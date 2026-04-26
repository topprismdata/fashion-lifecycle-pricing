"""Quick EDA for H&M dataset - bash-based for speed"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data_raw"

print("=" * 60)
print("H&M Dataset Quick EDA")
print("=" * 60)

# 1. Articles
print("\n--- Articles ---")
articles = pd.read_csv(DATA / 'articles.csv')
print(f"Shape: {articles.shape}")
print(f"Columns: {list(articles.columns)}")
print(f"\nKey columns unique values:")
for col in ['product_code', 'product_type_no', 'graphical_appearance_no',
            'colour_group_code', 'perceived_colour_value_id',
            'department_no', 'index_code', 'index_group_no',
            'section_no', 'garment_group_no']:
    print(f"  {col}: {articles[col].nunique()}")

print(f"\nindex_group_name distribution:")
print(articles['index_group_name'].value_counts())

# 2. Customers
print("\n--- Customers ---")
customers = pd.read_csv(DATA / 'customers.csv')
print(f"Shape: {customers.shape}")
print(f"Columns: {list(customers.columns)}")
print(f"\nAge stats: {customers['age'].describe()}")
print(f"Age NaN: {customers['age'].isna().sum()} ({customers['age'].isna().mean()*100:.1f}%)")
print(f"\nclub_member_status:\n{customers['club_member_status'].value_counts()}")
print(f"\nfashion_news_frequency:\n{customers['fashion_news_frequency'].value_counts()}")
print(f"FN: {customers['FN'].notna().mean()*100:.1f}% have")
print(f"Active: {customers['Active'].notna().mean()*100:.1f}% active")
print(f"Postal code unique: {customers['postal_code'].nunique()}")

# 3. Transactions (sample)
print("\n--- Transactions (first 2M rows) ---")
txn = pd.read_csv(DATA / 'transactions_train.csv', nrows=2_000_000, parse_dates=['t_dat'])
print(f"Shape (sample): {txn.shape}")
print(f"Date range: {txn['t_dat'].min()} to {txn['t_dat'].max()}")
print(f"Unique customers: {txn['customer_id'].nunique():,}")
print(f"Unique articles: {txn['article_id'].nunique():,}")
print(f"\nsales_channel_id:\n{txn['sales_channel_id'].value_counts()}")
print(f"\nprice stats:\n{txn['price'].describe()}")

# 4. Sample submission
print("\n--- Sample Submission ---")
sub = pd.read_csv(DATA / 'sample_submission.csv', nrows=5)
print(f"Columns: {list(sub.columns)}")
print(sub.head())
preds = sub['prediction'].iloc[0].split()
print(f"\nPredictions per customer: {len(preds)}")
print(f"Sample: {preds[:5]}")

# 5. Repurchase analysis (using sample)
print("\n--- Repurchase Analysis (from 2M sample) ---")
# Group by customer, find articles bought multiple times
cust_articles = txn.groupby('customer_id')['article_id'].apply(list)
repeat_pct = cust_articles.apply(lambda x: len(x) != len(set(x))).mean()
print(f"Customers with repeat article purchases: {repeat_pct*100:.1f}%")

# Most popular articles in the sample
top_articles = txn['article_id'].value_counts().head(20)
print(f"\nTop 20 articles (in 2M sample):")
for aid, cnt in top_articles.items():
    row = articles[articles['article_id'] == aid]
    if len(row) > 0:
        name = row.iloc[0]['prod_name']
        dept = row.iloc[0]['department_name']
        print(f"  {aid}: {cnt:,} purchases — {name[:40]} ({dept})")

print("\n" + "=" * 60)
print("EDA Complete")
print("=" * 60)
