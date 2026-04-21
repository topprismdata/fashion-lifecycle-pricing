# 执行计划 — Fashion Lifecycle Pricing

> 基于 DESIGN.md 的具体执行计划
> 创建时间: 2026-04-21

---

## 总体策略

按技术难度递增顺序，依次完成 7 个竞赛/数据集。
每完成一个，提取 skills 并积累到共享模块。
最终目标是构建完整的 "Retail AI Brain" 架构。

---

## Stage 1: 时序预测基础

### 1.1 Walmart Store Sales Forecasting

**目标**: 掌握零售时序预测和极端季节性处理

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1.1.1 | 竞赛分析: 理解数据格式、评估指标(WMAE)、5倍假期权重 | |
| 1.1.2 | EDA: 缺失值分析(MarkDown 1-5, 60%+缺失), 季节性模式 | |
| 1.1.3 | Baseline: LightGBM 基准模型 | |
| 1.1.4 | 特征工程: 假期特征, 滚动统计, FFT频域特征 | |
| 1.1.5 | 缺失值策略: 对比删除/填充/indicator column | |
| 1.1.6 | 模型优化: 多模型(XGB/CB/LGB), 集成策略 | |
| 1.1.7 | 提交并记录LB分数 | |

**关键学习点**:
- WMAE评估函数 (假期权重5x)
- 高缺失率特征的处理策略
- 零售促销信号的提取

### 1.2 H&M Personalized Fashion Recommendations

**目标**: 掌握推荐系统和"避免降价"策略

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1.2.1 | 竞赛分析: 理解数据(客户×文章×交易), MAP@12评估 | |
| 1.2.2 | EDA: 客户画像, 商品属性, 购买模式 | |
| 1.2.3 | Baseline: Popular items baseline | |
| 1.2.4 | 候选生成: 协同过滤 + 内容相似 + 流行度 | |
| 1.2.5 | 排序模型: LGBMRanker 两阶段管道 | |
| 1.2.6 | 特征工程: 用户历史, 商品属性, 交互特征 | |
| 1.2.7 | 提交并记录LB分数 | |

**关键学习点**:
- LGBMRanker 的使用
- 两阶段管道 (召回+排序)
- MAP@12 评估指标优化

---

## Stage 2: 冷启动与产品知识

### 2.1 iMaterialist Fashion Attribute Recognition

**目标**: 掌握计算机视觉在时尚领域的应用

| 步骤 | 内容 | 状态 |
|------|------|------|
| 2.1.1 | 数据分析: 100万+图像, 228细粒度属性 | |
| 2.1.2 | 预训练模型: ResNet50/ViT 特征提取 | |
| 2.1.3 | 属性分类: 多标签分类 pipeline | |
| 2.1.4 | Embedding: 视觉Embeddings → 相似款匹配 | |
| 2.1.5 | 冷启动: 新品 → 匹配历史款 → 估计初始弹性 | |

**关键学习点**:
- 时尚领域的细粒度属性识别
- 视觉Embeddings在零售中的应用
- 冷启动问题的CV解决方案

### 2.2 DeepFashion / 跨品类关联

**目标**: 掌握图神经网络在零售关系发现中的应用

| 步骤 | 内容 | 状态 |
|------|------|------|
| 2.2.1 | 构建商品关系图 (替代品/互补品) | |
| 2.2.2 | GCN特征提取 | |
| 2.2.3 | 跨品类关联分析 | |
| 2.2.4 | 交叉定价和捆绑销售策略 | |

---

## Stage 3: 因果推断

### 3.1 ML Zoomcamp 2024 Retail Demand Forecast

**目标**: 掌握因果推断在定价中的应用

| 步骤 | 内容 | 状态 |
|------|------|------|
| 3.1.1 | 竞赛分析: 25个月数据, 显性价格变动 | |
| 3.1.2 | 内生性问题分析: 价格与需求的双向因果 | |
| 3.1.3 | DML实现: EconML/DoubleML库使用 | |
| 3.1.4 | 价格弹性估计: 真实弹性 vs 朴素弹性 | |
| 3.1.5 | 期望销售曲线 (ESC) 构建 | |
| 3.1.6 | "销售滞后"指标和偏差曲线 | |

**关键学习点**:
- Double Machine Learning 两阶段估计
- 内生性偏差的诊断和消除
- 价格弹性的因果解释

---

## Stage 4: 竞争博弈

### 4.1 INFORMS RMP Challenge

**目标**: 掌握考虑竞争对手状态的动态定价

| 步骤 | 内容 | 状态 |
|------|------|------|
| 4.1.1 | 竞赛分析: 竞争对手数据, 选择模型 | |
| 4.1.2 | 消费者选择模型 (MNL/混合Logit) | |
| 4.1.3 | 多臂老虎机 (MAB) 策略 | |
| 4.1.4 | 竞争对手缺货状态的定价优化 | |

### 4.2 INFORMS Dynamic Pricing Challenge

**目标**: 掌握多智能体强化学习

| 步骤 | 内容 | 状态 |
|------|------|------|
| 4.2.1 | MARL环境搭建 | |
| 4.2.2 | MAPPO/MADDPG/QMIX 算法实现 | |
| 4.2.3 | CTDE架构: 集中训练 + 分布执行 | |
| 4.2.4 | 价格战避免策略 | |
| 4.2.5 | 全局优化引擎 (拉格朗日分解 + MIP) | |

---

## 跨阶段任务

### Skills 提取计划

| 时机 | 预期 Skills |
|------|------------|
| Stage 1 完成 | retail-missing-value-strategies, lgmb-ranker-two-stage-pipeline |
| Stage 2 完成 | fashion-cv-embeddings, cold-start-anchor-matching, gcn-product-relations |
| Stage 3 完成 | double-ml-price-elasticity, expected-sales-curve |
| Stage 4 完成 | marl-pricing-framework, lagrangian-mip-decomposition |

### 共享模块积累

| 模块 | 来源阶段 | 说明 |
|------|---------|------|
| `src/data/cleaner.py` | Stage 1 | 高缺失值清洗策略 |
| `src/features/ts_features.py` | Stage 1 | 时序特征工程 |
| `src/features/cv_features.py` | Stage 2 | 视觉特征提取 |
| `src/models/price_elasticity.py` | Stage 3 | DML价格弹性 |
| `src/models/marl_agents.py` | Stage 4 | MARL代理 |
| `src/models/optimizer.py` | Stage 4 | MIP优化 |

---

## 当前优先级

**立即开始**: Stage 1.1 — Walmart Store Sales Forecasting

理由:
1. 技术难度最低，适合建立项目框架
2. 缺失值处理和时序特征是后续阶段的基础
3. 有明确的 Kaggle LB 可以及时验证
4. 与 S6E4 经验 (LightGBM/XGB/集成) 最直接相关
