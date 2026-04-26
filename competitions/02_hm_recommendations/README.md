# 02 - H&M Personalized Fashion Recommendations

## Competition Info
- **URL**: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
- **Status**: Closed (2022)
- **Task**: Predict 12 articles each customer will buy in next 7 days
- **Metric**: MAP@12 (Mean Average Precision at 12)
- **Role in Project**: Recommendation systems, candidate generation + ranking pipeline

## Data Files
| File | Rows | Size | Description |
|------|------|------|-------------|
| transactions_train.csv | ~31.8M | 3.2GB | 2 years of purchases (2018-09 to 2020-09) |
| articles.csv | 105,542 | 34MB | 25 columns of article metadata |
| customers.csv | 1,371,980 | 198MB | 7 columns of customer info |
| sample_submission.csv | 1,371,980 | 258MB | Submission template |

## Key Challenges
1. **Massive scale**: 1.37M customers, 105K articles, 31.8M transactions
2. **Cold start**: ~80% of test customers have no recent purchases
3. **Fashion trends**: Rapid change, old data less useful
4. **Low predictability**: Most purchases are essentially unpredictable
5. **Submission format**: 12 space-separated 10-digit article IDs per customer

## Experiment Results

| Version | Val MAP@12 | Public LB | Private LB | Key Technique | Notes |
|---------|-----------|-----------|------------|---------------|-------|
| **R15** | n/a | **0.02279** | **0.02271** | R12 core + ItemCF max 12 (no co-occurrence) | **BEST** |
| R14 | n/a | 0.02277 | 0.02261 | R12 core + ItemCF max 6 | Near-best |
| R13 | n/a | 0.02267 | 0.02250 | R12 core + ItemCF max 4 | Good |
| R12 | n/a | 0.02259 | 0.02240 | R05 core + ItemCF max 2 fill-only | Breakthrough: ItemCF helps |
| R05 | 0.0261 | 0.02224 | 0.02196 | 6w repurchase + time-decay popular + limited co-occurrence | Previous best |
| R01 | 0.0265 | 0.02207 | 0.02131 | Repurchase 4w + last-7d popular | Strong baseline |
| R09 | 0.0261 | 0.02210 | 0.02182 | R05 + 10w semi-active layer | Same as R05 |
| R10 | n/a | 0.01973 | 0.01965 | ItemCF composite scoring (mix & sort) | Worse: displaces repurchase |
| R11 | n/a | 0.01965 | 0.01976 | ItemCF layered recall (silver medal formula) | Worse: wrong repurchase ordering |
| R04 | 0.0261 | 0.02141 | 0.02188 | Time-decay + co-occurrence + out-of-stock filter | Worse: co-occurrence displaces repurchase |
| R08 | n/a | 0.02143 | 0.02094 | R05 active + ALS inactive | Worse: ALS worse than global popular for inactive |
| R07 | 0.0248 | 0.02136 | 0.02089 | Last-1w priority ordering | Worse: priority reordering hurts |
| R06 | 0.0243 | 0.02112 | 0.02158 | Age-segment personalized popular | Worse: segmentation hurts fallback |
| R02 | 0.0649 (active) | 0.01800 | 0.01829 | LGBMRanker + age-bin fallback | Worse: ranker hurts active users |
| R03 | 0.0638 (active) | 0.01712 | 0.01750 | R02 ranker + global popular fallback | Worse: same issue |

## Key Findings

### 1. ItemCF with Repurchase Protection is the Key Breakthrough (R12-R15)
R12 discovered that ItemCF (item-to-item collaborative filtering from purchase sequences with direction factor)
helps significantly — BUT only when repurchase items are NEVER displaced:
- R10/R11 tried mixing ItemCF with repurchase → WORSE (0.01973/0.01965 vs 0.02224)
- R12 used layered approach: repurchase first, ItemCF fills empty slots → 0.02259 (+0.00035)
- R15 allowed full ItemCF fill (max 12) → 0.02279 (+0.00055 over R05)
- ItemCF direction factor captures sequential purchase intent (forward=1.0, backward=0.9)
- Distance decay: 0.7^(|j-i|-1), popularity penalty: 1/log(1+len(sequence))

### 2. How You Add Candidates Matters More Than What You Add
- R10/R11: "mix and sort" approach → ItemCF displaces repurchase → WORSE
- R12-R15: "layered fill" approach → ItemCF only fills empty slots → BETTER
- The same ItemCF algorithm that hurt in R10/R11 helped in R12-R15
- **Critical lesson: never displace high-confidence predictions**

### 2. Candidate Recall is the Fundamental Bottleneck
With 4 candidate strategies (repurchase, item2item, popular, dept_pop), recall is only 10.1%.
89.9% of actual purchases are not in the candidate set at all.
But adding more candidates (co-occurrence, ALS, product variants) dilutes the strong repurchase signal.

### 3. Active vs Inactive Customer Split
- ~25% of customers have recent purchases (active)
- ~75% are inactive → need fallback (global popular)
- MAP@12 for active customers: ~0.026
- MAP@12 for ALL customers: ~0.001 (inactive customers dilute the score heavily)

### 4. Personalization Hurts Inactive Users
- Age-segment popular (R06): worse than global popular
- ALS for inactive (R08): worse than global popular
- The "one-size-fits-all" popular list is hard to beat for cold-start

### 5. Co-occurrence and Product Variants Add Noise (but ItemCF helps)
- Bucket-based co-occurrence items displace high-confidence repurchase predictions
- Product code variants (different color/size) rarely match actual purchases
- Out-of-stock filter is too aggressive (removes 16K articles)

### 6. CV-LB Divergence
Val MAP@12 on active customers (0.065 with LGBMRanker) doesn't translate to good LB (0.018).
The metric on all customers is dominated by the 80% inactive users who get fallback predictions.

### 7. Memory Constraints
- Full 31.8M transactions → OOM on 64GB Mac when doing multiple groupby operations
- Solution: Only load last 12 weeks (3.9M rows) — fashion data older than 12 weeks is not useful
- Category dtype can cause groupby issues with named aggregation

### 8. Article ID Formatting
Article IDs must be zero-padded to 10 digits in submission (e.g., `0541518023`, not `541518023`).
Missing this causes LB = 0.00000.

## 经验总结 (Lessons Learned)

### 核心教训 1: 推荐系统中召回质量 > 排序质量

在此竞赛中, 限制因素是让正确的项目进入候选集, 而不是对它们进行正确排序。
一个简单的 "过去6周购买量 + 前12个时间衰减热门" 击败了所有复杂方法, 因为:
1. 重复购买项目具有非常高的精度 (之前购买过 → 可能会再次购买)
2. 全球热门项目捕捉到了趋势项目
3. 排序器通过重新排序这些高置信度预测来增加噪声

### 核心教训 2: 添加候选项 ≠ 提高召回率

我们尝试了6种不同的候选扩展策略:
- 共现 (R04): 添加了320万个候选项, 但没有提高分数
- 同产品编码变体 (R04): 1.5万个变体, 几乎没有命中
- 年龄段热门 (R06): 分段后热门商品比全局热门差
- ALS 矩阵分解 (R08): 对非活跃用户不如全局热门
- 10周半活跃层 (R09): 没有明显改善
- 部门级热门 (R04): 同样不如全局热门

所有这些方法都通过稀释强信号 (复购+全局热门) 来降低性能。

### 时间窗口对时尚很重要
- 仅在过去12周的数据上训练就足够了
- 超过12周的旧数据不提供有用的信号 (时尚趋势变化快)
- 6周复购窗口比4周多覆盖7万客户
- 这与沃尔玛销售形成对比，后者历史模式更稳定

### 非活跃用户的"个性化悖论"
对于非活跃用户, 个性化推荐比全局热门更差:
- 全局热门 (前12个热门商品) 是一个很强的基线
- 任何形式的个性化 (年龄分段, ALS, 部门热门) 都会降低分数
- 原因: 非活跃用户的信息太少, 个性化引入了噪声

## ML Workflow Checklist
- [x] EDA: data format, distributions, article/customer characteristics
- [x] Baseline: R01 repurchase + popular (LB=0.02207)
- [x] LGBMRanker pipeline: candidate generation + ranking (R02-R03)
- [x] Time-decay improvements: R05 (LB=0.02224, best)
- [x] Candidate expansion experiments: co-occurrence, ALS, product variants (R04, R06-R09)
- [x] Key insight: simple heuristics beat all complex approaches
- [x] Further improvement: ItemCF from purchase sequences (R12-R15, LB=0.02279)
- [ ] Word2Vec embeddings for items
- [ ] Target: LB > 0.025 (top 30%)
