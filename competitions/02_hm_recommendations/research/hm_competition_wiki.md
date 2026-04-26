# H&M Competition Local Wiki

> 本地维基：H&M Personalized Fashion Recommendations Kaggle 竞赛全攻略
> 最后更新：2026-04-22

---

## 1. 概览 (Overview)

### 1.1 竞赛基本信息

| 项目 | 详情 |
|------|------|
| **竞赛名称** | H&M Personalized Fashion Recommendations |
| **主办方** | H&M Group |
| **竞赛类型** | Featured Prediction Competition |
| **评估指标** | MAP@12 (Mean Average Precision @ 12) |
| **参赛队伍** | 3,006 teams |
| **预测目标** | 为每个 customer 推荐 12 个 articles |
| **预测时间窗口** | 数据结束后 7 天 (2020-09-23 至 2020-09-30) |
| **Public LB** | 5% 测试数据 |
| **Private LB** | 95% 测试数据（最终排名依据） |

### 1.2 数据统计

| 数据集 | 规模 |
|--------|------|
| **Transactions** | 31,788,324 条交易记录 |
| **Customers** | 1,371,980 个客户 |
| **Articles** | 105,542 个商品 |
| **Product Code** | 47,224 个产品代码（article_id 前6位） |
| **Product Type** | 131 种产品类型 |
| **Product Group** | 19 个产品组 |
| **时间范围** | 2018-09-20 至 2020-09-22 (约2年) |
| **Department** | 299 个部门 |
| **Color Group** | 50 种颜色组 |
| **Graphical Appearance** | 30 种图案 |

### 1.3 数据文件

- `transactions_train.csv`: 交易记录 (t_dat, customer_id, article_id, price, sales_channel_id)
- `articles.csv`: 商品元数据 (25个特征, 其中14个文本特征, 11个数值特征)
- `customers.csv`: 客户元数据 (7个特征: FN, Active, club_member_status, fashion_news_frequency, age, postal_code)
- `sample_submission.csv`: 提交样例

### 1.4 MAP@12 公式

```
MAP@12 = (1/|U|) * Σ_{u=1}^{|U|} (1/min(m,12)) * Σ_{k=1}^{min(n,12)} P(k) * rel(k)
```

其中:
- |U| = 客户数量
- P(k) = 前 k 个预测的 precision
- rel(k) = 第 k 个预测是否正确 (0/1)
- m = 该客户实际购买数, n = 预测数

### 1.5 关键数据洞察

**客户行为特征 (来自 Polimi 论文分析):**

- 50% 的客户在2年内购买 <10 次
- 日均交易量: 25,000 - 80,000
- 复购行为:
  - 同 article_id: 1周内复购 5.1%, 2周 6.2%, 3周 6.6%
  - 同 product_code: 1周内复购 6.8%, 2周 8.5%, 3周 9.4%
  - 同 category: 1周内复购 10.8%, 2周 14.3%, 3周 16.5%
- 年龄分布: 两个主要群体 20-30岁 和 45-55岁
- 33.85% 客户同时 FN=1 且 Active=1
- price 分布: 大多数商品 price < 0.1 (归一化)

---

## 2. 获奖方案详解 (Winning Solutions)

### 2.1 1st Place (PB: 0.03792, Public: 0.03716)

**来源:** [Kaggle Discussion 324070](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324070)

**团队描述:** "Our solution is using various retrieval strategies + feature engineering + GBDT, seems simple but powerful."

#### 架构: Two-Stage Pipeline

**Stage 1: Recall (候选生成)**
- 为每个用户生成约 **100 个候选商品**
- 使用多种 recall 策略（作者未完全公开具体策略列表）
- 从已知的 top solution 分析，推测包含:
  - Repurchase (复购)
  - ItemCF (Item-based Collaborative Filtering)
  - Popular items (热门商品)
  - Word2Vec embedding 相似度
  - Same product code (同产品代码不同颜色/尺码)
  - Co-occurrence (共现商品)

**Stage 2: Ranking (排序)**
- 使用 GBDT 模型进行排序
- 负采样 (Down-sampling) 技术平衡正负样本
- ~100 个候选 vs ~106,000 全量商品

**特征工程 (Table from Polimi 论文 Table 3.8):**

| 特征类型 | 描述 |
|----------|------|
| **Count** | User-item, user-category 的 last week/month/season/same week of last year/entire dataset 交互计数，以及基于时间距离加权的交易重要性 |
| **Time** | 每个商品的首次和最后交易日期 |
| **Mean/Max/Min** | 基于 age, price, sales_channel 的聚合统计 |
| **Difference/Ratio** | 用户年龄与购买该商品用户平均年龄的差值; 用户购买该商品次数与该商品总购买次数的比率 |
| **Similarity** | Item2Item CF score; Word2Vec item2item cosine similarity; User2item cosine similarity |

**模型细节:**
- Best single model: LightGBM (Public: 0.0367)
- Final ensemble: 5 LightGBM + 7 Catboost (PB: 0.03716 → 0.03792)
- Ensemble 后提升约 2%

### 2.2 2nd Place

**来源:** [Kaggle Discussion 324094](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324094)

**架构:** Two-stage Recall → Rank

**Recall Stage:**
- 为每个客户使用不同的 recall model 生成候选
- 4 种主要 recall 方法:
  1. **Item2Item CF**: 发现商品对之间的关系 (bought Y also bought Z)
  2. **Repurchase**: 每个客户最近购买的 20 个商品
  3. **Popular items**: 全局热门商品
  4. **Two Tower MMoE**: 能够为所有用户生成任意长度候选的深度学习模型

**Ranking Stage:**
- 特征包括: 用户-商品复购特征 (用户是否会复购某商品, 某商品是否会被复购)
- 高阶组合特征 (Higher-order combinatorial features)
- Recall strategy 本身作为特征 (recall strategy name + rank position)

**关键创新:** 使用 gating network 区分 active vs non-active customers，分配不同权重

### 2.3 3rd Place (PB: 0.03623, Public: ~0.0357)

**来源:** [Kaggle Discussion 324129](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324129)

**架构:** 类似 2nd Place，两阶段 pipeline

**核心特点:**
- **BPR Matrix Factorization**: 用于学习 user-article 相似度，在 ranking 中发挥重要作用
- **Item2Item Similarity Features**:
  - Co-occurrence count (共现计数)
  - Bought-together items count
- **Recall Strategy Features**:
  - 是否被某个特定策略 recall (binary feature)
  - 在该策略下的排名位置

**特征类别:**
1. 用户特征 (购买历史、偏好)
2. 商品特征 (流行度、类别、属性)
3. 交互特征 (user-item 交互统计)
4. Recall strategy 特征 (strategy name, rank under each strategy)

### 2.4 45th Place Silver Medal (LB: 0.0292, PB: 0.02996)

**来源:** [GitHub: Wp-Zhang/H-M-Fashion-RecSys](https://github.com/Wp-Zhang/H-M-Fashion-RecSys)

**团队:** Wp-Zhang, zhouyuanzhe, tarickMorty, Thomasyyj, ChenmienTan

#### 架构概览

```
┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│  Recall 1    │────▶│  LGB Ranker      │     │            │
│  (Strategy A)│────▶│  LGB Classifier  │────▶│  Ensemble  │──▶ Final 12 items
│              │────▶│  DNN             │     │            │
├──────────────┤     ├──────────────────┤     │            │
│  Recall 2    │────▶│  LGB Ranker      │     │            │
│  (Strategy B)│────▶│  LGB Classifier  │────▶│            │
│              │────▶│  DNN             │     │            │
└──────────────┘     └──────────────────┘     └────────────┘
```

**关键数据:**
- 2 种 recall 策略 × 3 个 ranking 模型 = 6 个子模型
- 单策略最高 LB: 0.0286
- Ensemble 后 LB: 0.0292 (提升 ~2%)
- 硬件限制: 50GB RAM
- 每用户平均 50 个候选
- 使用 4 周数据训练模型

**Pre-trained Embeddings:**
- DSSM: dssm_item_embd.npy, dssm_user_embd.npy
- YouTube DNN: yt_item_embd.npy, yt_user_embd.npy
- Word2Vec (CBOW): w2v_item_embd.npy, w2v_user_embd.npy, w2v_product_embd.npy
- Word2Vec (Skip-gram): w2v_skipgram_item_embd.npy, w2v_skipgram_user_embd.npy, w2v_skipgram_product_embd.npy
- Image: image_embd.npy

#### 项目结构

```
├── data
│   ├── external       <- Pre-trained embeddings
│   ├── interim        <- Candidates from recall strategies
│   ├── processed      <- Features merged dataframe
│   └── raw            <- Original dataset
├── notebooks          <- Jupyter notebooks
└── src
    ├── data           <- datahelper.py, metrics.py
    ├── features       <- base_features.py
    └── retrieval      <- collector.py, rules.py
```

---

## 3. 核心技术详解 (Core Techniques)

### 3.1 ItemCF with Direction Factor

#### 基本原理

ItemCF (Item-based Collaborative Filtering) 是本竞赛最核心的 recall 方法之一。不同于简单的 co-occurrence，ItemCF 考虑了**方向因子 (direction factor)**。

**标准 Co-occurrence:**
- 统计两个商品被同一用户购买的次数
- 对称关系: count(A,B) = count(B,A)

**带方向因子的 ItemCF:**
- 考虑购买顺序: 用户先买 A 后买 B
- 非对称关系: sim(A→B) ≠ sim(B→A)
- 捕获了 "用户买了 A 之后更可能买 B" 的时序模式

#### 代码实现 (Silver Medal Team)

```python
# 来自 Wp-Zhang/H-M-Fashion-RecSys 的 recall rules

class OrderHistory(PersonalRetrieveRule):
    """Retrieve recently bought items by the customer."""

    def __init__(self, trans_df, days=7, n=None, name="1", item_id="article_id"):
        self.iid = item_id
        self.trans_df = trans_df[["t_dat", "customer_id", item_id]]
        self.days = days
        self.n = n
        self.name = name

    def retrieve(self):
        df = self.trans_df.reset_index()
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        tmp = df.groupby("customer_id").t_dat.max().reset_index()
        tmp.columns = ["customer_id", "max_dat"]
        res = df.merge(tmp, on=["customer_id"], how="left")

        res["diff_dat"] = (res.max_dat - res.t_dat).dt.days
        res = res.loc[res["diff_dat"] < self.days].reset_index(drop=True)

        res = res.sort_values(by=["diff_dat"], ascending=True).reset_index(drop=True)
        res = res.groupby(["customer_id", self.iid], as_index=False).first()

        res = res.reset_index()
        res = res.sort_values(by="index", ascending=False).reset_index(drop=True)
        res["rank"] = res.groupby(["customer_id"])["index"].rank(
            ascending=True, method="first"
        )
        res["score"] = -res["diff_dat"]

        if self.n is not None:
            res = res.loc[res["rank"] <= self.n]

        res["method"] = f"OrderHistory_{self.name}"
        res = res[["customer_id", self.iid, "score", "method"]]
        return res
```

#### ItemCF 如何与 Co-occurrence 不同

| 方面 | Co-occurrence | ItemCF with Direction |
|------|---------------|----------------------|
| 计算方式 | count(A ∩ B) | sim(i,j) = count(i→j) / count(i→*) |
| 对称性 | 对称 | 非对称 (方向敏感) |
| 时序信息 | 无 | 有 (先买A后买B) |
| 归一化 | 无/简单归一化 | 除以 source item 总频次 |
| 典型用途 | 搭配推荐 | 顺序推荐 |

### 3.2 Word2Vec / Item2Vec

#### 原理

将用户的购买序列视为"句子"，每个商品视为"单词"，使用 Word2Vec 训练商品 embedding。

**训练数据构造:**
```
用户A: [item_123, item_456, item_789, item_012, ...]  → 一条"句子"
用户B: [item_345, item_678, ...]                       → 一条"句子"
```

#### 9th Place 方案的关键参数

- 模型: Word2Vec (CBOW + Skip-gram 均尝试)
- 训练数据: 用户购买序列 (按时间排序)
- 用途:
  1. 生成 item embedding → 计算 item2item 相似度
  2. 平均用户购买过的 item embedding 得到 user embedding
  3. 计算 user2item cosine similarity
  4. 使用 top-N 相似商品作为候选

#### Silver Medal 方案的 Embedding 训练

该团队训练了多种 embedding:

| Embedding 类型 | 维度 | 说明 |
|---------------|------|------|
| DSSM (item/user) | 64/128 | Deep Structured Semantic Model |
| YouTube DNN (item/user) | 64/128 | YouTube 推荐双塔模型 |
| W2V CBOW (item/user/product) | 64/128 | Word2Vec CBOW 模式 |
| W2V Skip-gram (item/user/product) | 64/128 | Word2Vec Skip-gram 模式 |
| Image embedding | 128 | 商品图片 CNN 特征 |

#### 1st Place 方案的 Similarity 特征

1st place 使用了三种 embedding 相似度作为特征:
- **Item2Item CF score**: 协同过滤相似度
- **Item2Item Word2Vec cosine similarity**: 基于 W2V embedding 的余弦相似度
- **User2Item cosine similarity**: 用户 embedding 与商品 embedding 的余弦相似度

### 3.3 Time-Decay Scoring Formula

#### Silver Medal 的核心时间衰减公式

这是 45th place 方案中最关键的公式之一，用于 TimeHistoryDecay 规则:

```python
# 来自 Wp-Zhang/H-M-Fashion-RecSys/src/retrieval/rules.py

class TimeHistoryDecay(GlobalRetrieveRule):
    """Retrieve popular items with time decay."""

    def retrieve(self):
        df = self.trans_df
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        last_ts = df["t_dat"].max()
        df["dat_gap"] = (last_ts - df["t_dat"]).dt.days

        # 计算周期销售量
        df["last_day"] = last_ts - (last_ts - df["t_dat"]).dt.floor(f"{self.days}D")
        period_sales = (
            df[["last_day", self.iid, "t_dat"]]
            .groupby(["last_day", self.iid])
            .count()
        )
        period_sales = period_sales.rename(columns={"t_dat": "period_sale"})
        df = df.join(period_sales, on=["last_day", self.iid])

        # 目标周期销售量
        period_sales = period_sales.reset_index().set_index(self.iid)
        last_day = last_ts.strftime("%Y-%m-%d")
        df = df.join(
            period_sales.loc[period_sales["last_day"] == last_day, ["period_sale"]],
            on=self.iid,
            rsuffix="_targ",
        )
        df["period_sale_targ"].fillna(0, inplace=True)

        # 周期销售比率 (当前周期/历史周期)
        df["quotient"] = df["period_sale_targ"] / df["period_sale"]

        # *** 核心时间衰减公式 ***
        df["dat_gap"][df["dat_gap"] < 1] = 1
        x = df["dat_gap"]

        a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
        # value = a/sqrt(x) + b*exp(-c*x) - d
        df["value"] = a / np.sqrt(x) + b * np.exp(-c * x) - d

        df["value"][df["value"] < 0] = 0
        # 乘以周期销售比率
        df["value"] = df["value"] * df["quotient"]

        # 按商品聚合得分
        df = df.groupby([self.iid], as_index=False)["value"].sum()
        df = df.sort_values(by="value", ascending=False).reset_index(drop=True)

        df["rank"] = df.index + 1
        df["score"] = df["value"]
        df = df[df["rank"] <= self.n]
        df["method"] = f"TimeHistoryDecay_{self.name}"

        return self.merge(df)
```

#### 公式详解

```
score(item) = Σ_t  (a/sqrt(x_t) + b*exp(-c*x_t) - d) * quotient_t
```

其中:
- `x_t` = 该交易距最后一天的天数差 (dat_gap)
- `a = 2.5e4` (25,000): 控制短期权重衰减的速度
- `b = 1.5e5` (150,000): 控制指数衰减的幅度
- `c = 2e-1` (0.2): 指数衰减速率
- `d = 1e3` (1,000): 基础偏移量 (截断低价值)
- `quotient_t` = 当前周期销量 / 历史周期销量 (趋势因子)

**公式特点:**
1. `a/sqrt(x)`: 缓慢衰减的短期记忆，平方根使近期权重快速上升
2. `b*exp(-c*x)`: 指数衰减的长期记忆，随时间快速归零
3. `-d`: 截断低价值交易，只有足够"新鲜"的交易才有正分
4. `* quotient`: 叠加趋势因子，近期增长的商品获得额外加分

**衰减曲线关键值:**
| x (days) | a/sqrt(x) | b*exp(-c*x) | value (before -d) | value (after -d) |
|----------|-----------|-------------|-------------------|------------------|
| 1 | 25,000 | 122,774 | 147,774 | 146,774 |
| 7 | 9,449 | 36,845 | 46,294 | 45,294 |
| 14 | 6,689 | 9,042 | 15,731 | 14,731 |
| 30 | 4,564 | 1,009 | 5,573 | 4,573 |
| 60 | 3,226 | 7 | 3,233 | 2,233 |
| 90 | 2,635 | 0 | 2,635 | 1,635 |

### 3.4 Two-Stage Pipeline Architecture

#### 通用架构 (所有 Top Team 共用)

```
┌──────────────────────────────────────────────────────┐
│                    Recall Stage                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Repurchase│ │ ItemCF  │ │Popular  │ │ W2V/ALS │   │
│  │         │ │         │ │Items    │ │ Emb Sim │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       └──────┬────┴──────┬────┴──────┬────┘         │
│              │           │           │               │
│        ┌─────▼───────────▼───────────▼─────┐         │
│        │     Candidate Union & Dedup       │         │
│        │     (~100-300 items per user)      │         │
│        └──────────────┬────────────────────┘         │
└───────────────────────┼──────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────┐
│                Ranking Stage                          │
│        ┌──────────────▼────────────────────┐         │
│        │     Feature Engineering           │         │
│        │  (~100-500 features per pair)     │         │
│        └──────────────┬────────────────────┘         │
│        ┌──────────────▼────────────────────┐         │
│        │     GBDT Model (LightGBM/CatBoost)│         │
│        │     Binary Classification         │         │
│        └──────────────┬────────────────────┘         │
│        ┌──────────────▼────────────────────┐         │
│        │     Top 12 by Score → Submission  │         │
│        └──────────────────────────────────┘         │
└──────────────────────────────────────────────────────┘
```

#### 各队伍 Recall 策略对比

| Recall 策略 | 1st | 2nd | 3rd | 4th | 5th | 45th | Polimi Thesis |
|-------------|-----|-----|-----|-----|-----|------|---------------|
| Repurchase | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (36 items) |
| Item2Item CF | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (36 items) |
| Popular Items | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (60 items) |
| Word2Vec Similarity | ✓ | | | | ✓ | ✓ | |
| Same Product Code | ✓ | | | | | ✓ | ✓ |
| Co-occurrence | ✓ | | ✓ | | ✓ | ✓ | ✓ |
| Age-based | | | | | ✓ | | ✓ (12 items) |
| Popularity per Dept | | | | | | | ✓ (6 items) |
| BPR Matrix Factorization | | | ✓ | | | | |
| Two Tower MMoE | | | | ✓ | | | |
| SWIM Transformer | | | | | ✓ | | |
| Sentence Transformer | | | | | ✓ | | |
| DSSM | | | | | | ✓ | |
| YouTube DNN | | | | | | ✓ | |
| Image Embedding | | | | | ✓ | ✓ | |
| Sale Trend | | | | | | ✓ | |
| Out-of-Stock Filter | ✓ | | | | | ✓ | ✓ |

#### 候选数量统计

| 队伍 | 每用户候选数 | 训练数据周数 |
|------|-------------|-------------|
| 1st Place | ~100 | - |
| 45th Place | ~50 | 4 weeks |
| Polimi Thesis | ~300 | 6 weeks |

### 3.5 BPR / ALS Matrix Factorization

#### ALS (Alternating Least Squares)

**在竞赛中的表现 (来自 Polimi 论文):**

| 模型 | Public | Private |
|------|--------|---------|
| ALS | 0.01413 | 0.01406 |
| PureSVD | 0.00431 | 0.00426 |
| Top Popular 12 | 0.02163 | 0.02119 |
| Item KNN CF | 0.00345 | 0.00357 |
| P3Alpha | 0.00431 | 0.00426 |
| RP3Beta | 0.00425 | 0.00453 |

**关键发现:**
- ALS 是最强的个性化 CF baseline
- 但 ALS (0.014) 远低于 Top Popular 12 (0.021)
- 非个性化方法在这个竞赛中表现异常好
- 独立的 CF 模型都不如 heuristic 方法
- 只有结合 GBDT ranking 后才能大幅提升

#### BPR (Bayesian Personalized Ranking)

**3rd Place 的 BPR 用法:**
- 用于学习 user-article 相似度矩阵
- 学习的是排序而非评分
- 使用 triplet (u, i, j) 训练: 用户 u 偏好 i 胜过 j
- BPR embedding 作为 ranking stage 的特征输入

**BPR vs ALS:**
| 方面 | ALS | BPR |
|------|-----|-----|
| 优化目标 | 重建评分矩阵 | 学习偏好排序 |
| 输入 | 显式/隐式评分 | 隐式反馈 (购买=正例) |
| 负采样 | 不需要 | 需要 (未购买=负例) |
| 适用场景 | 预测评分 | 排序推荐 |
| 本竞赛用途 | 作为 baseline/特征 | 3rd place 的核心特征 |

### 3.6 Negative Sampling Strategy

#### 1st Place 的负采样策略

- 每个 user 有 ~100 个正样本 (被 recall 的候选中，在 test week 实际购买的)
- 从 106,000 个未购买商品中采样负样本
- 下采样 (down-sampling) 平衡正负样本比例

#### 通用负采样方法

1. **Random Negative Sampling**: 随机从未交互商品中采样
2. **Popularity-based Sampling**: 按商品流行度采样 (热门但不被该用户购买的商品是更有信息量的负例)
3. **In-batch Negative Sampling**: 批内其他用户的正样本作为负样本
4. **Time-aware Sampling**: 只在 test week 之前可用的商品中采样

---

## 4. 特征工程 (Feature Engineering)

### 4.1 Count 特征

| 特征 | 描述 | 来源 |
|------|------|------|
| user_item_count | 用户购买该商品的次数 | 1st |
| user_category_count | 用户购买该类别的次数 | 1st |
| last_week_count | 上周交互次数 | 1st |
| last_month_count | 上月交互次数 | 1st |
| last_season_count | 上季度交互次数 | 1st |
| same_week_last_year_count | 去年同周交互次数 | 1st |
| full_sale | 累计销量 | 45th |
| week_sale | 当周销量 | 45th |
| period_sale | 近N天销量 | 45th |

**Silver Medal 的销量计算代码:**

```python
def full_sale(trans, groupby_cols, unique=False, week_num=6):
    """Calculate cumulative sales of each item unit."""
    inter = trans[["customer_id", "week", "valid", *groupby_cols]]
    if unique:
        inter = inter.drop_duplicates(["customer_id", "week", *groupby_cols])

    tmp_l = []
    for week in range(1, week_num + 1):
        df = inter[inter["week"] >= week]
        df = df.groupby([*groupby_cols])["valid"].sum().reset_index(name="_SALE")
        df["week"] = week
        tmp_l.append(df)

    df = pd.concat(tmp_l, ignore_index=True)
    inter = trans[["customer_id", "week", *groupby_cols]].merge(
        df, on=["week", *groupby_cols], how="left"
    )
    inter["_SALE"] = inter["_SALE"].fillna(0).astype("int")
    return inter["_SALE"].values
```

### 4.2 Time 特征

| 特征 | 描述 | 来源 |
|------|------|------|
| first_transaction_day | 商品首次出现日期 | Polimi |
| last_transaction_day | 商品最后交易日期 | Polimi |
| user_first_day | 用户首次交易日期 | Polimi |
| user_item_first_day | 用户-商品首次交互日期 | Polimi |
| item_freshness | 商品"新鲜度"(首次出现距今天数) | Polimi |
| user_freshness | 用户"新鲜度" | Polimi |
| user_item_freshness | 用户-商品对"新鲜度" | Polimi |
| time_weighted_importance | 基于时间距离加权的交易重要性 | 1st |

### 4.3 Aggregation 特征

| 特征 | 描述 | 来源 |
|------|------|------|
| user_mean_price | 用户平均消费金额 | Polimi/45th |
| user_std_price | 用户消费金额标准差 | Polimi |
| user_mean_channel | 用户平均购买渠道 | Polimi |
| item_mean_price | 商品平均售价 | Polimi/45th |
| item_std_price | 商品售价标准差 | Polimi |
| item_mean_channel | 商品平均销售渠道 | Polimi |
| item_user_mean_age | 购买该商品用户的平均年龄 | Polimi |
| item_user_std_age | 购买该商品用户的年龄标准差 | Polimi |
| item_age_range | 商品的主要购买年龄范围 | Polimi |
| repurchase_ratio | 商品复购率 | 45th |
| popularity | 商品流行度因子 | 45th |

**Silver Medal 的复购率计算:**

```python
def repurchase_ratio(trans, groupby_cols, week_num=6):
    """Calculate repurchase ratio of item units."""
    tmp_l = []
    for week in tqdm(range(1, week_num + 1)):
        tmp_df = trans[trans["week"] >= week]
        item_user_sale = (
            tmp_df[tmp_df["valid"] == 1]
            .groupby(["customer_id", *groupby_cols])
            .size()
            .reset_index(name="_SALE")
        )
        item_sale = (
            item_user_sale.groupby(groupby_cols).size().reset_index(name="_I_SALE")
        )
        item_user_sale = (
            item_user_sale[item_user_sale["_SALE"] > 1]
            .groupby(groupby_cols)
            .size()
            .reset_index(name="_MULTI_SALE")
        )
        item_sale = item_sale.merge(item_user_sale, on=groupby_cols, how="left")
        item_sale["_RATIO"] = item_sale["_MULTI_SALE"] / (item_sale["_I_SALE"] + 1e-6)
        item_sale = item_sale[[*groupby_cols, "_RATIO"]]
        item_sale["week"] = week
        tmp_l.append(item_sale)

    df = trans[["week", *groupby_cols]]
    item_sale = pd.concat(tmp_l, ignore_index=True)
    df = df.merge(item_sale, on=["week", *groupby_cols], how="left")
    df["_RATIO"] = df["_RATIO"].fillna(0)
    return df["_RATIO"].values
```

### 4.4 Similarity 特征

| 特征 | 描述 | 来源 |
|------|------|------|
| item2item_cf_score | CF 相似度分数 | 1st |
| item2item_w2v_cosine | Word2Vec 余弦相似度 | 1st |
| user2item_cosine | 用户-商品 embedding 余弦相似度 | 1st |
| dssm_similarity | DSSM 模型相似度 | 45th |
| yt_similarity | YouTube DNN 模型相似度 | 45th |
| image_similarity | 图片 embedding 相似度 | 45th |

### 4.5 Recall-Score-as-Feature 突破

**这是本竞赛最重要的发现之一。**

将 recall 阶段的分数和排名直接作为 ranking 阶段的特征输入:

| 特征 | 描述 | 效果 |
|------|------|------|
| is_recalled_by_strategy_A | 是否被策略A recall (0/1) | 高 |
| rank_under_strategy_B | 在策略B下的排名 | 高 |
| score_from_strategy_C | 策略C给出的分数 | 高 |
| recall_method_name | recall 方法名称 (categorical) | 中 |

**原理:** Recall 阶段的分数已经包含了大量关于 user-item 匹配度的信息。直接将这些分数作为特征让 GBDT 学习如何加权组合不同 recall 策略，比手动调参有效得多。

---

## 5. 实验经验 (Experimental Insights)

### 5.1 模型分数对比表 (Polimi 论文 Table 5.1 & 5.3)

| 方法 | Public | Private | 备注 |
|------|--------|---------|------|
| **1st Place** | **0.03716** | **0.03792** | 5 LGB + 7 CB ensemble |
| 4th Place | 0.03544 | 0.03563 | Two Tower MMoE |
| 5th Place | 0.03536 | 0.03553 | SWIM + Sentence Transformer |
| **Two-Stage CatBoost** | 0.0308 | **0.0298** | Polimi 最佳 (约第10名) |
| **Two-Stage LightGBM** | 0.0286 | 0.0279 | Polimi 单模型 |
| **45th Place Ensemble** | **0.0292** | **0.02996** | 2 recall × 3 models |
| Trending repurchase heuristic | 0.02263 | 0.02291 | 最佳 heuristic |
| Items purchased together | 0.02169 | 0.02159 | Association rule |
| Top popular 12 items | 0.02163 | 0.02119 | 非个性化 baseline |
| Recommend last 3 weeks | 0.01854 | 0.01850 | Heuristic |
| Top popular by age | 0.01949 | 0.01962 | Association rule |
| Top popular by category | 0.01973 | 0.01992 | Trousers/Sweater/Cardigan |
| Top popular by age+discount | 0.01478 | 0.01482 | Association rule |
| ALS | 0.01413 | 0.01406 | 最强个性化 CF |
| Heuristic trendy+popular | 0.00613 | 0.00642 | Heuristic |
| Top popular new items | 0.0064 | 0.00676 | Heuristic |
| P3Alpha | 0.00431 | 0.00426 | Graph-based |
| PureSVD | 0.00431 | 0.00426 | Matrix factorization |
| RP3Beta | 0.00425 | 0.00453 | Graph-based |
| Item KNN CF | 0.00345 | 0.00357 | CF |
| Top popular Sep 2020 | 0.00407 | 0.00384 | Season-specific |

### 5.2 关键发现

**什么有效:**
1. **Two-stage pipeline 是王道**: 所有 top team 都使用 Recall + Ranking 架构
2. **GBDT (LightGBM/CatBoost) > Deep Learning**: 在特征工程充分的情况下
3. **Recall 策略的多样性**: 多种互补的 recall 策略比单一复杂策略更有效
4. **Recall score 作为 ranking 特征**: 这是最重要的突破之一
5. **Ensemble**: 单模型到 ensemble 通常有 2-5% 的提升
6. **Time decay**: 时间衰减加权对 recall 效果至关重要
7. **Repurchase**: 快时尚领域最强的单一信号
8. **Feature simplicity**: 候选生成后，简单特征 (mean, count) 就能带来很大提升

**什么不太有效:**
1. **单独的 CF 模型**: ALS (0.014) 远低于 Top Popular 12 (0.021)
2. **纯 Deep Learning**: 复杂模型 (Transformer, DNN) 在单模型上不如 GBDT
3. **季节性推荐**: Top Popular Aug/Sep 2020 (0.004) 远低于全局 Top Popular (0.021)
4. **Graph-based 方法**: P3Alpha, RP3Beta 单独效果差
5. **Trendy color**: 与热门商品结合后效果反而下降
6. **Ensemble of weak baselines**: 多个弱模型 ensemble 不会变强

### 5.3 常见陷阱

1. **忽略时间窗口**: 使用太旧的数据训练会导致模型过时
   - 推荐: 只使用最近 4-7 周数据
2. **候选数量过多/过少**:
   - 过多: 内存和计算压力大
   - 过少: recall 覆盖不够
   - 推荐: 每用户 50-300 个候选
3. **正负样本不平衡**:
   - 每个 user 只有少数正样本 vs 大量未交互商品
   - 必须使用 down-sampling 或 weighted sampling
4. **忽略 Out-of-Stock 商品**:
   - 已停产商品不应出现在推荐列表中
   - Silver medal 方案有专门的 OutOfStock 过滤规则
5. **数据泄漏**: 确保只用 test week 之前的数据

### 5.4 特征重要性 (Polimi 论文)

GBDT 模型中最重要的特征 (按重要性排序):

1. **user_age**: 用户年龄
2. **item_user_mean_age**: 购买该商品用户的平均年龄
3. **item_age_ranges**: 商品的主要购买年龄范围
4. **color_group_code**: 商品颜色组
5. **garment_group_number**: 服装组编号
6. **product_type**: 产品类型
7. **item_volumes**: 商品交易量
8. **user_item_volume**: 用户-商品交互量

**关键洞察:** 年龄相关特征最重要，说明 H&M 客户的购买行为与年龄高度相关。

---

## 6. 代码参考 (Code References)

### 6.1 Silver Medal Team 关键代码

**Recall Rules (rules.py):**
- `OrderHistory`: 用户最近购买的商品 (带时间窗口)
- `OrderHistoryDecay`: 带时间衰减的复购推荐
- `TimeHistory`: 时间窗口内的热门商品
- `TimeHistoryDecay`: 带时间衰减的热门商品 (**核心公式**)
- `SaleTrend`: 销售趋势商品
- `UGSaleTrend`: 用户分组销售趋势
- `OutOfStock`: 过滤已停产商品

**Feature Engineering (base_features.py):**
- `full_sale()`: 累计销量
- `week_sale()`: 周销量
- `period_sale()`: 近N天销量
- `repurchase_ratio()`: 复购率
- `popularity()`: 流行度因子

### 6.2 SaleTrend 实现 (趋势检测)

```python
class SaleTrend(GlobalRetrieveRule):
    """Retrieve trending items - items with accelerating sales."""

    def retrieve(self):
        item_sale = self.trans_df
        item_sale["t_dat"] = pd.to_datetime(item_sale["t_dat"])
        item_sale["dat_gap"] = (item_sale.t_dat.max() - item_sale.t_dat).dt.days

        # 取最近 2*days 天数据，分前后两组
        item_sale = item_sale[item_sale["dat_gap"] <= 2 * self.days - 1]
        group_a = item_sale[item_sale["dat_gap"] > self.days - 1]  # 较早的组
        group_b = item_sale[item_sale["dat_gap"] <= self.days - 1]  # 较近的组

        # 计算趋势: (近期销量 - 早期销量) / 早期销量
        log = pd.merge(group_b, group_a, on=[self.iid], how="left")
        log["trend"] = (log["count_x"] - log["count_y"]) / log["count_y"]

        # 只保留上升趋势 > threshold 的商品
        log = log[log["trend"] > self.t]
        return log
```

### 6.3 Popularity 因子计算

```python
def popularity(trans, item_id="article_id", week_num=6):
    """Time-decay weighted popularity."""
    df = trans[[item_id, "t_dat", "week", "valid"]]
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    tmp_l = []
    name = "Popularity_" + item_id
    for week in range(1, week_num + 1):
        tmp_df = df[df["week"] >= week][df["valid"] == 1]
        last_day = tmp_df["t_dat"].max()
        # 核心: 越近期的交易权重越大
        tmp_df[name] = 1 / ((last_day - tmp_df["t_dat"]).dt.days + 1)
        tmp_df = tmp_df.groupby([item_id])[name].sum().reset_index()
        tmp_df["week"] = week
        tmp_l.append(tmp_df)

    info = pd.concat(tmp_l)[[item_id, name, "week"]]
    df = df.merge(info, on=[item_id, "week"], how="left")
    return df[name].values
```

### 6.4 Key Repositories & Links

| 资源 | 链接 | 说明 |
|------|------|------|
| 1st Place Discussion | https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324070 | 冠军方案 |
| 2nd Place Discussion | https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324094 | 亚军方案 |
| 3rd Place Discussion | https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324129 | 季军方案 |
| 4th Place Discussion | https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324094 | Two Tower MMoE |
| 5th Place Writeup | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/hao-5th-place-solution | SWIM + Sentence Transformer |
| 9th Place Discussion | https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324127 | Word2Vec 方案 |
| 45th Place Repo | https://github.com/Wp-Zhang/H-M-Fashion-RecSys | Silver Medal 完整代码 |
| 45th Place Notebook | https://www.kaggle.com/code/weipengzhang/h-m-silver-medal-solution-45-3006 | Silver Medal Kaggle Notebook |
| Product Embeddings NB | https://www.kaggle.com/code/konradb/product-embeddings | 商品 Embedding Notebook |
| Implicit ALS NB | https://www.kaggle.com/code/julian3833/h-m-implicit-als-model-0-014 | ALS 实现 (0.014) |
| Polimi Thesis | https://www.politesi.polimi.it/retrieve/ab0b82e5-ed1c-4e5c-bd4f-ea79d4afd334/Master_Thesis_With_Summary_Arcangelo-Pisa.pdf | 学术分析 |
| Competition Page | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations | 竞赛主页 |
| Medium Silver Medal | https://ajisamudra.medium.com/silver-medal-solution-on-kaggle-h-m-personalized-fashion-recommendations-a0878e1eae63 | 另一篇银牌方案 |
| 46th Place (Item2Vec) | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324205 | Item2Vec + Transformer |

### 6.5 Out-of-Stock Filter 实现

```python
class OutOfStock(FilterRule):
    """Filter items that are out of stock."""

    def __init__(self, trans_df, item_id="article_id"):
        self.iid = item_id
        mask = trans_df["t_dat"] >= "2020-08-01"
        self.trans_df = trans_df.loc[mask, ["customer_id", self.iid, "t_dat"]]

    def _off_stock_items(self):
        sale = self.trans_df
        sale["t_dat"] = pd.to_datetime(sale["t_dat"])
        sale["year_month"] = (
            (sale["t_dat"].dt.year).astype(str)
            + "_"
            + (sale["t_dat"].dt.month).astype(str)
        )
        sale = (
            sale.groupby([self.iid, "year_month"])["customer_id"]
            .count()
            .reset_index(name="count")
        )
        sale = pd.pivot_table(
            sale, values="count", index=self.iid, columns="year_month"
        )
        sale = sale.fillna(0)
        # 如果 9月销量相比 8月下降超过 80%，或者 9月销量为 0
        mask = ((sale["2020_9"] - sale["2020_8"]) / sale["2020_8"]) < -0.8
        mask2 = sale["2020_9"] == 0
        return list(sale[mask | mask2].index)

    def retrieve(self):
        return self._off_stock_items()
```

---

## 附录: 竞赛关键时间线

| 日期 | 事件 |
|------|------|
| 2018-09-20 | 训练数据开始 |
| 2020-09-22 | 训练数据结束 |
| 2020-09-23 ~ 09-30 | 测试周 (预测目标) |
| 2022-05 | 竞赛结束，结果公布 |

---

> **参考来源:**
> - Kaggle Discussion 324070 (1st place)
> - Kaggle Discussion 324094 (2nd place)
> - Kaggle Discussion 324129 (3rd place)
> - GitHub: Wp-Zhang/H-M-Fashion-RecSys (45th place)
> - Politecnico di Milano Thesis: "Data-Driven Recommendations for H&M" by Arcangelo Pisa
> - Kaggle Competition Page: h-and-m-personalized-fashion-recommendations
> - Medium: Silver Medal Solution by ajisamudra
> - Kaggle Notebook: konradb/product-embeddings
