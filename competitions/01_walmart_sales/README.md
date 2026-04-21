# 01 - Walmart Recruiting: Store Sales Forecasting

## Competition Info
- **URL**: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
- **Status**: Closed (2014)
- **Task**: Predict weekly sales for 45 Walmart stores × ~98 departments
- **Metric**: WMAE (holiday weeks weighted 5x)
- **Role in Project**: Seasonal demand prediction baseline, missing value handling

## Data Files
| File | Description |
|------|-------------|
| train.csv | Store, Dept, Date, Weekly_Sales, IsHoliday |
| test.csv | Prediction target weeks |
| features.csv | Temperature, Fuel_Price, MarkDown 1-5 (>60% missing), CPI, Unemployment |
| stores.csv | Store Type, Size |

## Key Challenges
1. MarkDown 1-5 missing rate >60%
2. Holiday weeks: 5x weight (Super Bowl, Labor Day, Thanksgiving, Christmas)
3. ~4000 store-department combinations
4. Negative sales (returns)
5. Test period entirely out-of-sample (no overlap with training)

## Top Solutions
- 1st: 6 time-series models averaged (STLF, ETS, Fourier ARIMA, seasonal naive) + Christmas shift post-processing - R
- 2nd: Per-department GBM with cross-store features, sqrt transform
- Common: SVD noise reduction, holiday distance features

## Experiment Results

| Version | CV WMAE | Public LB | Private LB | Key Technique | Notes |
|---------|---------|-----------|------------|---------------|-------|
| R01 | 1568.88 | 3102.97 | 3215.57 | LightGBM baseline, 28 features | First baseline |
| R02 | 187.77 | 5960.12 | 6162.20 | Lag+Rolling, combined train+test | **FAIL**: data leakage |
| R03 | 1257.87 | 15050.74 | 15116.63 | Lag from train-only, fillna(0) | **FAIL**: test lag all zeros |
| R04 | 2296.69 | 5910.07 | 6075.48 | Multi-model ensemble, feat mismatch | **FAIL**: column mismatch |
| R05 | 2261.57 | 3196.53 | 3312.05 | 69 features, aggregated stats | Worse than R01 |
| R06 | 1392.27 | 2815.12 | 2912.12 | 40 features, seasonal+sd_median | Good improvement |
| **R08** | **1410.64** | **2720.96** | **2838.91** | **R06+YoY52+sd_mean** | **BEST** |
| R07 | 1334.98 | 3533.66 | 3508.43 | YoY+Christmas shift+sqrt+5seed | FAIL: sqrt transform |
| R09 | 1081.26 | — | — | Leak-free stats+naive blend | Not submitted (leakage in val) |
| R10 | 1570.92 | 3025.14 | 3148.09 | Leak-free stats+3seed | Worse than R08 |
| R11 | 1178.47 | 2907.02 | 3016.09 | LGB+XGB+Naive ensemble | Worse than R08 |
| R12 | 1414.75 | 2820.51 | 2933.14 | R08+5seed avg | Slightly worse than R08 |

## Key Findings

### 1. Lag Features Are Harmful (ts-lag-out-of-sample-trap)
Test period (2012-11-02 to 2013-07-26) extends BEYOND training period (2010-02-05 to 2012-10-26).
Lag features referencing recent history (lag_1 through lag_52) are unavailable for test rows.
R02-R04 all failed because lag features degraded from helpful-in-CV to harmful-in-LB.

### 2. Safe Features for Out-of-Sample Test
- Calendar features (Week, Month, Quarter) — always computable
- Seasonal sin/cos encoding — mathematical transform
- Holiday proximity — based on fixed calendar dates
- Store-Dept median/mean from train — static statistic
- YoY same-week-last-year (offset 52) — references known train data (98.2% valid rate in test)
- MarkDown aggregates — based on test's own values

### 3. Unsafe Features for Out-of-Sample Test
- Lag features (any offset) — no history available
- Rolling statistics — depends on recent history
- Seasonal naive baseline from full train — leakage in CV

### 4. What Worked (R06→R08 Improvement Path)
- R06 over R01: +sin/cos encoding, +holiday proximity, +sd_median (+287 LB improvement)
- R08 over R06: +YoY same-week-last-year, +sd_mean (+94 LB improvement)

### 5. What Didn't Work
- sqrt transform (R07): changed loss landscape, hurt WMAE
- Multi-model ensemble (R05, R11): XGBoost much worse than LightGBM for this task
- 69 features with many aggregated stats (R05): overfitting
- Christmas shift post-processing (R07): unclear benefit, combined with sqrt hurt
- Multi-seed averaging (R12): models converged to same solution
- Seasonal naive blending (R11): leakage in naive baseline

## 经验总结 (Lessons Learned)

### 核心教训: 先判断 Train-Test 时间关系

**在做任何特征工程之前**, 必须先检查训练集和测试集的时间关系:

```python
train_max = train['Date'].max()
test_min = test['Date'].min()
if test_min > train_max:
    print("WARNING: Test is entirely out-of-sample!")
    print("→ Don't use lag/rolling features")
    print("→ Use calendar/seasonal/static stats instead")
```

**判断树**:
```
Test 是否与 Train 时间重叠?
├── 是 → lag 特征有效, 正常使用 (如: ts-lag-nan-cascade-bug)
└── 否 → lag 特征有害!
    ├── 用 Calendar features (Week, Month, Quarter, sin/cos)
    ├── 用 Holiday proximity (基于固定日期)
    ├── 用 Store/Dept static stats (从 train 计算的均值/中位数)
    └── 用 YoY same-week-last-year (offset 52, 引用已知 train 数据)
```

### CV-LB 方向不一致是危险信号

| 模式 | 含义 | 行动 |
|------|------|------|
| CV↓ LB↓ | 特征有效 | 继续优化 |
| CV↓ LB↑ | 特征在 CV 中泄漏 | **立即停止**, 检查数据泄漏 |
| CV↑ LB↓ | 模型过拟合 CV | 减少特征, 正则化 |

R02 是最典型的例子: CV 从 1568→187 (大降!), 但 LB 从 3102→5960 (大升!)。
这说明 lag 特征在 CV 中有完美信息, 但在 test 中完全不可用。

### Less is More (特征数量 vs 质量)

| 版本 | 特征数 | Public LB | 诊断 |
|------|--------|-----------|------|
| R01 | 28 | 3102 | 基线 — 所有特征始终有效 |
| R05 | 69 | 3196 | 过多聚合特征, 过拟合 |
| R06 | 40 | 2815 | 精选特征, 显著提升 |
| R08 | 43 | **2720** | +YoY52, 最佳平衡点 |

**规律**: 从 28→40 特征有效 (+382), 从 40→69 特征反而退化 (-97)。
特征工程不是堆数量, 而是验证每个特征在 test 场景下的可用性。

### 特征工程验证清单

添加新特征前, 必须回答:
1. **"在 test 预测时, 这个特征的值从哪来?"** — 如果答案是 "需要 test 数据", 则泄漏
2. **"test 中有多少行的特征值是 NaN/0?"** — 如果 >50%, 则特征不可靠
3. **"CV 和 LB 的改善方向一致吗?"** — 如果不一致, 说明有泄漏或过拟合

### 技术债务: 列对齐问题

R04/R05 都遇到了 train 和 test 列数不一致的问题。根因:
- `create_features(train)` 没有 `train_ref` → 不生成聚合特征
- `create_features(test, train_ref=train)` 有 `train_ref` → 生成聚合特征
- **必须在 train 和 test 上使用完全相同的特征列**

**解决**: 总是用 `train_ref=train` 处理 train, 确保特征列一致。

### Best Practice: 从简单到复杂

成功的迭代路径:
```
R01 (基线, 28 feat) → R06 (+seasonal, +holiday, +sd_median) → R08 (+YoY52)
  LB: 3102              LB: 2815                               LB: 2720
```

每次只加少量特征, 验证 LB 是否改善。不要一次性加很多特征。

## ML Workflow Checklist
- [x] EDA: data format, missing values, sales distribution
- [x] Baseline: LightGBM with basic features (R01: LB=3102.97)
- [x] Feature engineering: seasonal, holiday, aggregated stats, YoY
- [x] Model optimization: hyperparameter tuning, multi-model comparison
- [x] Key insight: lag features harmful for out-of-sample test
- [x] Skill extraction: ts-lag-out-of-sample-trap
- [x] Target: LB < 2800 (top 10%) — **ACHIEVED: R08 LB=2720.96**
- [ ] Further improvement: per-department models, SVD noise reduction
