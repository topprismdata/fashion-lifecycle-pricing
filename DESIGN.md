# 机器学习驱动的服饰全生命周期决策优化研究

## ML-Driven Fashion Lifecycle Decision Optimization

> 基于 NotebookLM 研究笔记 (36b5e898) 的详细设计方案
> 创建时间: 2026-04-21

---

## 一、项目愿景

### 1.1 核心目标

推动零售定价策略从传统依赖人工经验的"被动打折 (Reactive Discounting)"
向"基于轨迹的算法定价治理 (Trajectory-Based Pricing Governance)"范式转移。

实现 **3000 家门店 × 数万 SKU** 的全生命周期动态折扣优化:
- 最大化全生命周期总利润 (LTP)
- 最大化全价售罄率
- 保护品牌毛利率和市场份额

### 1.2 核心范式转移

```
传统模式:  经验驱动 → 统一折扣 → 季末清仓
算法定价:  数据驱动 → 个性定价 → 全程优化
```

### 1.3 研究范围

体育用品/快时尚行业，覆盖商品从 **录入 → 上市 → 销售 → 清仓** 的完整生命周期，
通过 7 个竞赛/数据集构建 "零售 AI 大脑 (Retail AI Brain)" 的完整架构。

---

## 二、研究体系架构

### 2.1 全生命周期闭环

```
┌─────────────────────────────────────────────────────────────────┐
│                    Retail AI Brain                               │
├─────────┬───────────┬──────────────┬────────────────────────────┤
│ Phase 1 │ Phase 2   │ Phase 3      │ Phase 4                    │
│ 冷启动  │ 需求感知  │ 精准分发     │ 动态博弈                   │
│         │           │              │                            │
│ iMate-  │ Walmart + │ H&M          │ INFORMS RMP +              │
│ rialist │ ML Zoom-  │ Recomme-     │ Dynamic Pricing            │
│ + Deep- │ camp      │ ndations     │                            │
│ Fashion │           │              │                            │
│         │           │              │                            │
│ CV特征  │ 因果推断  │ 避免降价     │ MARL多智能体               │
│ 提取    │ 弹性计算  │ 精准匹配     │ 竞争博弈                   │
├─────────┴───────────┴──────────────┴────────────────────────────┤
│              数据基建: 特征库 + MLOps + 分布式计算               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 七个竞赛/数据集的角色

| # | 竞赛/数据集 | 角色 | 解决的子问题 | 核心技术 |
|---|------------|------|-------------|---------|
| 1 | iMaterialist Fashion (Kaggle) | CV特征基石 | 新SKU冷启动，视觉属性提取 | ResNet/ViT, 228细粒度属性 |
| 2 | DeepFashion Dataset | 消费者关系图谱 | 跨品类关联，替代/互补品发现 | GCN, 图卷积网络 |
| 3 | Walmart Store Sales (Kaggle) | 季节性压力测试 | 极端季节性预测，60%缺失值处理 | LightGBM, FFT/PCA |
| 4 | ML Zoomcamp 2024 Retail | 因果推断实验室 | 内生性偏差消除，真实价格弹性 | DML (Double ML) |
| 5 | H&M Personalized Recs (Kaggle) | 个性化引擎 | 避免降价，精准匹配替代打折 | LGBMRanker, 两阶段管道 |
| 6 | INFORMS RMP Challenge | 消费者选择模型 | 竞争对手缺货状态的动态定价 | MAB (多臂老虎机) |
| 7 | INFORMS Dynamic Pricing | MARL终极测试 | 寡头市场博弈，避免价格战 | MAPPO/MADDPG/QMIX |

### 2.3 子问题依赖关系

```
冷启动定价 ──→ 需求预测 ──→ 弹性计算 ──→ 优化引擎
    │              │             │              │
  CV/NLP      时序特征      因果推断       组合优化
    │              │             │              │
    └──────→ 精准分发 ←─────┘              │
                │                           │
              推荐排序                    竞争博弈
                │                           │
                └──────── 全局优化 ←─────────┘
```

---

## 三、核心研究问题

### 3.1 冷启动与无历史数据预测

**问题**: 新品SKU生命周期极短，缺乏历史销售数据，如何预测需求模式和初始价格弹性?

**方法**:
- 计算机视觉提取产品属性 (颜色、轮廓、面料)
- NLP提取文本描述语义特征
- 聚类算法匹配历史相似款 (Anchor Products)
- 映射生成新品的价格弹性曲线
- Bass扩散模型评估新品渗透率

### 3.2 因果推断与内生性偏差消除

**问题**: 传统销量预测混淆了"因为需求低所以降价"的因果关系，如何剥离出真实价格弹性?

**方法**:
- 双重机器学习 (DML) 两阶段估计
- 期望销售曲线 (ESC) 作为基准
- "销售滞后 (Sale Lag)" 和 "偏差曲线" 指标排除断码噪音
- 残差回归计算纯粹的价格-需求因果效应

### 3.3 竞争博弈与动态平衡

**问题**: 多方博弈市场中，定价不是孤立的，如何平衡自身利润最大化与全局价格稳定性?

**方法**:
- 多智能体强化学习 (MARL) 仿真竞争环境
- MAPPO: 低方差稳定收益优化
- MADDPG: 多主体利润公平分配
- QMIX: 捕获多智能体联合行动协同效应
- CTDE架构: 集中式训练，去中心化执行

### 3.4 超大规模组合优化

**问题**: 如何在3小时内求解3000门店×数万SKU的混合整数规划 (MIP) 问题?

**方法**:
- 拉格朗日分解将全局约束拆解为独立子问题
- 最大违规启发式割平面法加速求解
- 毫秒级近似最优解
- 并行求解与云端编排

---

## 四、技术架构设计

### 4.1 分层架构

```
┌──────────────────────────────────────────────────┐
│  Layer 4: 应用层 — 定价决策仪表板 + A/B测试       │
├──────────────────────────────────────────────────┤
│  Layer 3: 优化层 — MIP求解器 + MARL代理           │
│           Pyomo/GLPK/IPOPT + MAPPO/MADDPG         │
├──────────────────────────────────────────────────┤
│  Layer 2: 模型层 — 预测 + 因果推断 + 推荐         │
│           LightGBM/XGB + DML + LGBMRanker         │
├──────────────────────────────────────────────────┤
│  Layer 1: 特征层 — Feature Store + CV Pipeline    │
│           在线/离线特征库 + ResNet/ViT Embeddings   │
├──────────────────────────────────────────────────┤
│  Layer 0: 数据层 — 数据湖 + ETL + 数据质量        │
│           PySpark + Pandas + NumPy/Numba           │
└──────────────────────────────────────────────────┘
```

### 4.2 技术栈选型

| 层级 | 技术选型 | 理由 |
|------|---------|------|
| **数据层** | PySpark + S3数据湖 | 海量POS流水水平扩展 |
| **特征层** | Feast/Tecton + PyTorch | 在线低延迟(10-20ms) + 离线批量 |
| **模型层** | LightGBM + EconML (DML) | 表格数据高效 + 因果推断标准库 |
| **优化层** | Pyomo + GLPK/IPOPT | 开源MIP求解器 + Python原生 |
| **强化学习** | PyTorch + Ray RLlib | MARL算法族 + 分布式训练 |
| **计算机视觉** | PyTorch + ResNet/ViT | 产品图像特征提取 |
| **编排** | AWS Step Functions + SageMaker | 工作流编排 + 模型注册 |
| **监控** | MLflow + Prometheus | 实验跟踪 + 系统监控 |

### 4.3 关键算法详解

#### 4.3.1 双重机器学习 (DML)

```python
# 阶段1: 纠缠模型
Y_hat = f(X, Z)  # 预测销量 (X=控制变量, Z=价格)
T_hat = g(X)     # 预测价格处理

# 阶段2: 残差回归
Y_residual = Y - Y_hat
T_residual = T - T_hat
theta = (Y_residual × T_residual) / (T_residual × T_residual)
# theta = 真实价格弹性 (无混淆)
```

#### 4.3.2 拉格朗日分解

```python
# 原问题: max Σ profit(sku, store, discount)
#   s.t. 全局折扣率限制, 库存约束, 品牌保护约束

# 分解为:
# L(λ) = Σ subproblem_i(discount_i) + λ × (全局约束违反)

# 迭代:
for iteration in range(max_iter):
    for sku_store in parallel_subproblems:
        solve_local(sku_store, lambda)  # 独立求解
    lambda = update_multipliers()       # 更新拉格朗日乘子
    if violation < tolerance: break
```

#### 4.3.3 MARL架构 (CTDE)

```
训练阶段 (集中式):
  全局状态 S → Critic网络 → 全局价值函数 Q(S, A1, A2, ..., An)
  每个Agent观测 oi → Actor网络 → 动作 ai (折扣深度)

执行阶段 (去中心化):
  每个门店Agent独立:
  观测 (本地库存, 竞争对手价格, 市场份额) → 独立决策折扣深度
```

---

## 五、数据需求

### 5.1 内部数据

| 数据类型 | 来源 | 频率 | 说明 |
|---------|------|------|------|
| POS交易流水 | 收银系统 | 日度 | 门店×SKU×日期粒度 |
| 折扣历史 | 促销系统 | 日度 | 折扣深度、促销类型 |
| 动态库存 | WMS/RFID | 日度 | 尺码×颜色×门店级库存 |
| 产品图像 | PIM系统 | 一次性 | 高清产品图 |
| 产品元数据 | PIM系统 | 一次性 | 属性字典 (品类、材质、颜色等) |
| 促销日历 | 运营系统 | 周度 | 计划促销活动 |

### 5.2 外部数据

| 数据类型 | 来源 | 说明 |
|---------|------|------|
| 竞争对手价格 | 价格抓取/第三方 | 历史价格配置 |
| 天气预报 | 气象API | 局部天气预报 |
| 社交媒体趋势 | 社交平台API | 时尚趋势信号 |
| 宏观经济 | 公开数据 | CPI、失业率等 |

### 5.3 数据质量挑战

- **稀疏性**: 降价特征缺失率高达 60%
- **内生性**: 价格与需求的双向因果关系
- **延迟**: 多源数据集成延迟
- **一致性**: 全渠道数据格式统一

---

## 六、项目结构与代码组织

```
fashion-lifecycle-pricing/
├── DESIGN.md                    # 本文件 — 项目详细设计
├── PLAN.md                      # 执行计划与里程碑
├── CLAUDE.md                    # Claude Code 工作指引
│
├── src/                         # 共享模块
│   ├── __init__.py
│   ├── config.py                # 全局配置
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # 数据加载 (各竞赛)
│   │   ├── cleaner.py           # 数据清洗
│   │   └── validator.py         # 数据质量验证
│   ├── features/
│   │   ├── __init__.py
│   │   ├── cv_features.py       # 计算机视觉特征
│   │   ├── ts_features.py       # 时序特征
│   │   ├── graph_features.py    # 图特征 (GCN)
│   │   └── causal_features.py   # 因果推断特征
│   ├── models/
│   │   ├── __init__.py
│   │   ├── demand_forecast.py   # 需求预测模型
│   │   ├── price_elasticity.py  # 价格弹性 (DML)
│   │   ├── recommender.py       # 推荐系统 (LGBMRanker)
│   │   ├── marl_agents.py       # MARL代理
│   │   └── optimizer.py         # MIP优化求解
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # 评估指标
│       └── visualization.py     # 可视化工具
│
├── competitions/                # 7个竞赛独立目录
│   ├── 01_walmart_sales/        # Walmart Store Sales
│   │   ├── scripts/
│   │   ├── notebooks/
│   │   ├── outputs/
│   │   └── README.md
│   ├── 02_hm_recommendations/   # H&M Personalized Recs
│   ├── 03_ml_zoomcamp_retail/   # ML Zoomcamp 2024
│   ├── 04_imaterialist/         # iMaterialist Fashion
│   ├── 05_informs_rmp/          # INFORMS RMP Challenge
│   ├── 06_dynamic_pricing/      # INFORMS Dynamic Pricing
│   └── 07_deepfashion/          # DeepFashion
│
├── research/                    # 研究笔记与文献
│   ├── papers/                  # 论文笔记
│   ├── experiments/             # 实验记录
│   └── insights/                # 洞察总结
│
├── shared/                      # 跨竞赛共享资源
│   ├── pretrained_models/       # 预训练模型
│   └── external_data/           # 外部数据集
│
└── notebooks/                   # 探索性分析
    └── eda/
```

---

## 七、实施路线图

### 7.1 四阶段路线图

```
Phase 1: 数据基建 + ESC构建        (月 0-3)
  │
  ▼
Phase 2: 真实弹性建模 + MVP        (月 3-6)
  │
  ▼
Phase 3: MARL + 全局优化引擎       (月 6-9)
  │
  ▼
Phase 4: 规模化部署 + MLOps闭环    (月 9-12)
```

### 7.2 学习与复现顺序

基于技术难度和依赖关系，推荐以下学习路径:

#### Stage 1: 时序预测基础 (入门)

**Walmart Store Sales + H&M Recommendations**

| 项目 | 学习重点 |
|------|---------|
| Walmart | 60%缺失值处理, 极端季节性, WMAE评估, 5倍假期权重 |
| H&M | 两阶段管道 (候选生成+排序), LGBMRanker, 避免降价的推荐 |

**交付物**:
- 能够处理零售稀疏数据的完整pipeline
- 缺失值处理策略库
- LGBMRanker两阶段推荐框架

#### Stage 2: 冷启动与产品知识 (进阶)

**iMaterialist + DeepFashion**

| 项目 | 学习重点 |
|------|---------|
| iMaterialist | ResNet/ViT图像特征, 228细粒度属性提取, 聚类匹配 |
| DeepFashion | GCN图卷积, 跨品类关联, 替代/互补品发现 |

**交付物**:
- 产品图像 → 视觉Embeddings → 相似款匹配 pipeline
- 跨品类关系图谱
- 冷启动定价基准模型

#### Stage 3: 因果推断 (高级)

**ML Zoomcamp 2024 Retail Demand Forecast**

| 项目 | 学习重点 |
|------|---------|
| ML Zoomcamp | DML双重机器学习, 内生性偏差, 真实价格弹性 |

**交付物**:
- 价格弹性因果推断模型
- 期望销售曲线 (ESC) 基准
- "销售滞后"指标和偏差曲线

#### Stage 4: 竞争博弈 (专家)

**INFORMS RMP + INFORMS Dynamic Pricing**

| 项目 | 学习重点 |
|------|---------|
| INFORMS RMP | MAB多臂老虎机, 消费者选择模型, 竞争对手缺货状态 |
| Dynamic Pricing | MAPPO/MADDPG/QMIX, CTDE架构, 避免价格战 |

**交付物**:
- 多智能体强化学习训练框架
- 市场仿真环境
- 全局优化引擎 (拉格朗日分解 + MIP)

### 7.3 MVP 定义

**范围限定**:
- 选择收入和利润率排名前 20 的 SKU 类别
- 在 50-100 家代表性旗舰店实施
- 基于树模型的销售预测 (LightGBM)
- 静态规则辅助降价推荐
- 暂不引入 MARL 动态博弈

**验证目标**:
- 算法估算的价格弹性准确性
- 全价售罄率提升 (vs 对照组)
- 数据集成延迟发现和修复
- 内部团队信任建立

### 7.4 关键里程碑

| 里程碑 | 验证指标 (KPI) |
|--------|---------------|
| M1: 数据质量 | 数据完整性指数 ≥ 95% |
| M2: 预测精度 | 需求预测 R² ≥ 0.74, WMAE显著降低 |
| M3: 商业验证 | 试点组全价售罄率提升, 毛利提升 |
| M4: 系统时效 | 单请求 < 0.2s, 全局优化 < 3h |

---

## 八、与 Kaggle PS S6E4 经验的映射

本团队在 S6E4 灌溉需求预测竞赛中积累了丰富的实战经验，以下经验可直接迁移:

| S6E4 经验 | 本项目映射 |
|-----------|-----------|
| Pairwise TE (135特征) | 跨品类交互特征 (品类×区域×季节) |
| 10模型集成 + LR Stacking | 多模型预测融合 |
| 单轮Pseudo-labeling (thr=0.9) | 高置信度需求预测用于弹性估计 |
| 阈值优化 (Balanced Accuracy) | 折扣深度阈值优化 |
| CV-LB Gap 控制 | A/B测试离线/在线评估一致性 |
| Sigmoid Smoothing TE | 类别编码平滑 (新品属性→价格弹性) |
| ML Pipeline Unit Testing | 数据质量和特征验证 |
| 特征工程7大模式 | 零售特征工程 (比值/交互/分箱等) |

---

## 九、研究来源

本项目设计基于 NotebookLM 笔记本中的 40+ 篇文献和案例分析，包括:

**学术论文**:
- Deep learning for new fashion product demand prediction
- Graph Neural Network-Driven Demand Forecasting
- Pricing in Agent Economies Using Multi-Agent Q-Learning
- Algorithmic Governance and ML Architectures in Global Retail
- AI-Driven Cold Start Modeling for Retail Demand Forecasting
- An AI pipeline for garment price projection using computer vision
- An Integrated AI Model for Analyzing Consumer Fashion Preferences

**行业案例**:
- The Rise of SHEIN: Navigating the Digital Era of Fast Fashion
- 10 Ways Adidas is Using AI (2026)
- Artificial Intelligence at Nike: Current Use-Cases
- Case Study on Zara Apparel Manufacturing and Retail
- Case Study: Zara's Inventory Management for Fast Fashion

**技术实践**:
- Complete guide to ML in retail demand forecasting (RELEX)
- 4 Algorithms to Automate Daily Replenishment at Scale
- Beyond Discounts: A Comprehensive Guide to Markdown Optimization
- Building a Dynamic Inventory Optimisation System
- Competition-Based Dynamic Pricing in Online Retailing
- Data for Fashion Retailers: The Four Problems Nobody Talks About

---

## 十、下一步行动

1. **初始化项目结构** — 创建代码骨架和共享模块
2. **从 Stage 1 开始** — Walmart + H&M 竞赛
3. **建立实验跟踪** — MLflow集成
4. **迭代积累** — 每完成一个竞赛提取 skills
5. **跨竞赛整合** — 逐步构建完整 Retail AI Brain

---

*本设计文档将随着项目进展持续更新。*
