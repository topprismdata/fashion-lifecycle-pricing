# Idea Library — ML Zoomcamp 2024 Retail Demand Forecast

## 研究发现 (R05 后更新)

### Top Solutions 关键技术 (2024-04-25 研究)

| Rank | Score | 核心技术 |
|------|-------|---------|
| 1st | 8.96 | 未公开 |
| 2nd | 9.23 | AutoGluon + DuckDB + **date×store×item cross-join** |
| 4th | 9.47 | CatBoost depth=12 + cyclical encoding + GDP feature |
| 5th | 9.56 | XGBoost + Optuna HPO + target encoding |
| 6th | 9.87 | RandomForest + actual_matrix filter |

**最大差距**: 我们 17.17 vs top 8.96 = ~48% gap
**最关键技术**: 密集时间网格 (cross-join)、分位数异常值清洗、AutoGluon 多模型集成

---

## ALGO 级想法

### A0: 密集时间网格 (Cross-Join) ← TOP PRIORITY
- **来源**: 2nd place (ArturG) 的核心创新
- **描述**: 创建 date×store×item 完整交叉表，缺失销量填 0。让模型学习"没有销售"= 0 而非缺失
- **预期收益**: 极高 — 估计 -3 到 -5 RMSE
- **风险**: 数据量从 7.4M 增长到 ~40M+，需要内存优化
- **状态**: **IN_PROGRESS** → R06
- **实现轮次**: R06

### A1: 两阶段零膨胀模型
- **来源**: Walmart R21 (zero-inflated-two-stage-forecasting skill)
- **描述**: 先用二分类预测 quantity>0，再对非零样本回归。零值比例高时有效。
- **预期收益**: 高 — Walmart 实验证明零膨胀能提升 3-5%
- **风险**: 如果零值比例不高（EDA 显示 0% 负值），可能无效
- **状态**: TODO
- **实现轮次**: R07

### A2: Transformer 时序模型
- **来源**: AutoSOTA paper-88 (ChannelNorm, +15.2%)
- **描述**: 用 PatchTST/iTransformer 替代 GBDT，捕捉长程依赖
- **预期收益**: 中 — 时序特有模式可能被 GBDT 遗漏
- **风险**: 训练成本高，Apple Silicon 上可能不够快
- **状态**: TODO
- **实现轮次**: 可能 R08+

### A3: 门店间迁移学习
- **来源**: AutoSOTA cross-pollination 思路
- **描述**: 在数据量大的门店(store 1,4)上训练，微调到小门店(store 2,3)
- **预期收益**: 中 — store 2/3 只有 129K/58K 行，可能受益于迁移
- **风险**: 门店差异可能导致负迁移
- **状态**: TODO
- **实现轮次**: R06+

## CODE 级想法

### C1: 滚动窗口统计 (7/14/30/60天)
- **来源**: Top solution (4th place CatBoost), ts-forecasting-stale-lag-methodology skill
- **描述**: 对每个 (item, store) 计算 rolling mean/std/min/max
- **预期收益**: 高 — Top solutions 共识，最重要的特征类型
- **风险**: 计算量大，需要内存优化
- **状态**: TODO
- **实现轮次**: R02

### C2: 多源数据整合
- **来源**: Top solution (2nd place AutoGluon + DuckDB)
- **描述**: 整合 discounts_history, price_history, markdowns, online, actual_matrix
- **预期收益**: 高 — 促销对销量影响大
- **风险**: 特征冗余 (external-data-fusion-redundancy skill)
- **状态**: TODO
- **实现轮次**: R03

### C3: 俄罗斯假期特征
- **来源**: Top solution (4th place), holidays 库
- **描述**: is_holiday, days_to_holiday, is_pay_day (俄罗期每月发薪日)
- **预期收益**: 中 — 零售销量受假期影响显著
- **风险**: 俄罗斯假期可能数据中不明显
- **状态**: TODO
- **实现轮次**: R03

### C4: YoY 364天特征
- **来源**: yoy-364day-features skill
- **描述**: quantity_yoy_364 = qty[t] - qty[t-364]，消除年季节性
- **预期收益**: 中 — 25个月数据足够计算一次 YoY
- **风险**: 只有12个月 overlap，覆盖率可能不够
- **状态**: TODO
- **实现轮次**: R04

### C5: 目标变换实验
- **来源**: Store Sales R20 (RMSLE)
- **描述**: 对比 log1p(qty), sqrt(qty), raw 的效果
- **预期收益**: 中 — 高度右偏分布 (mean=5.6, max=4952)
- **风险**: 变换后需要反变换，引入偏差
- **状态**: TODO
- **实现轮次**: R04

### C6: 线上+线下销量合并
- **来源**: Top solution (2nd place)
- **描述**: 将 online.csv (54MB) 合并到 sales 中，增加训练数据
- **预期收益**: 中 — 但 online 只有 store 1
- **风险**: 线上线下消费模式不同
- **状态**: TODO
- **实现轮次**: R03

## PARAM 级想法

### P1: CatBoost Optuna 调参
- **来源**: 标准流程
- **描述**: learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature
- **预期收益**: 低-中 — GBDT 对超参不敏感
- **风险**: 过拟合验证集
- **状态**: TODO
- **实现轮次**: R05

### P2: 多模型集成权重搜索
- **来源**: kaggle-optimal-blending skill
- **描述**: CatBoost + LightGBM + XGBoost 加权平均
- **预期收益**: 中 — 多样性收益
- **风险**: ensemble-model-correlation-trap
- **状态**: TODO
- **实现轮次**: R08
