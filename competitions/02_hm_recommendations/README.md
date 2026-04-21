# 02 - H&M Personalized Fashion Recommendations

## Competition Info
- **URL**: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
- **Status**: Closed (2022)
- **Task**: Predict 12 articles each customer will buy in next 7 days
- **Metric**: MAP@12 (Mean Average Precision at 12)
- **Role in Project**: Recommendation as markdown avoidance strategy

## Data Files
| File | Rows | Size | Description |
|------|------|------|-------------|
| transactions_train.csv | ~31.8M | ~2.7GB | 2-year purchase history |
| articles.csv | ~105K | small | 25-column product metadata |
| customers.csv | ~1.37M | medium | 7-column customer info |
| images/ | ~105K | large | Product images |

## Architecture: Two-Stage Retrieval-Ranking Pipeline
```
Stage 1: Candidate Generation → 50-200 candidates/customer
Stage 2: LGBMRanker Ranking → top-12 predictions
```

## Key Techniques
1. Repurchase signal (strongest)
2. Item2Item collaborative filtering
3. Recent popularity + trending items
4. LGBMRanker (lambdarank objective)
5. Feature engineering: popularity, interaction, recency, trending

## ML Workflow Checklist
- [ ] EDA: customer profiles, article attributes, purchase patterns
- [ ] Baseline: popular items recommendation
- [ ] Candidate generation: repurchase + Item2Item + popular + trending
- [ ] Feature engineering: user-item interaction, popularity, recency
- [ ] Ranking model: LGBMRanker with group parameter
- [ ] Ensemble: multi-recall × multi-ranker
- [ ] Time-based validation (7-day window)
- [ ] Submit and record LB
