# 03 - ML Zoomcamp 2024: Retail Demand Forecasting

## Competition Info
- **URL**: https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition
- **Course**: DataTalksClub ML Zoomcamp 2024
- **Task**: Predict next-month demand for 28,182 items × 4 stores
- **Metric**: RMSE
- **Data Size**: ~872MB, 10 CSV files
- **Time Span**: 25 months (Aug 2022 - Sep 2024)
- **Role in Project**: Causal inference lab, price elasticity estimation

## Data Files
| File | Description | Key Columns |
|------|-------------|-------------|
| sales.csv | Historical sales | date, item_id, store_id, quantity, price |
| online.csv | Online sales | Similar to sales.csv |
| markdowns.csv | Discount events | date, item_id, store_id |
| prices.csv | Item pricing | date, item_id, price |
| stores.csv | Store metadata | store_id, attributes |
| items.csv | Product catalog | item_id, attributes |
| test.csv | Test set | item_id, store_id, date |

## Key Challenges
1. Price data available → enables price elasticity analysis
2. Markdown data → discount effect on demand
3. Item catalog incomplete for some items
4. 28K items × 4 stores = large prediction space

## Extended Goal: Price Elasticity via DML
Beyond the competition's RMSE target:
- Use Double Machine Learning (DML) to estimate true price elasticity
- Separate causal effect of price on demand from confounders
- Build Expected Sales Curve (ESC) baseline

## Top Solutions
- 1st: AutoML (~11.9 RMSE)
- 4th: Feature engineering + Boosting (~11.95 RMSE)

## ML Workflow Checklist
- [ ] EDA: sales patterns, price distributions, markdown effects
- [ ] Baseline: LightGBM with basic features
- [ ] Feature engineering: lag, rolling, price change, temporal
- [ ] Standard competition: optimize RMSE
- [ ] Extended: DML for price elasticity estimation
- [ ] Extended: ESC (Expected Sales Curve) construction
- [ ] Submit and record LB
