# H&M 个性化时尚推荐 — 竞赛研究报告

## 竞赛基本信息
- **URL**: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
- **状态**: 已关闭 (2022年)
- **任务**: 为每个客户预测未来7天最可能购买的12件商品
- **评估指标**: MAP@12 (Mean Average Precision at 12)
- **参赛人数**: ~2,952 teams
- **奖金**: $50,000

## 数据规模
| 文件 | 行数 | 大小 | 说明 |
|------|------|------|------|
| transactions_train.csv | ~3180万 | ~2.7GB | 2年购买历史 (2018-09 to 2020-09) |
| articles.csv | ~105,542 | 小 | 25列商品元数据 |
| customers.csv | ~1,371,980 | 中 | 7列客户信息 |
| images/ | ~105K张 | ~26GB | 商品图片 |
| sample_submission.csv | ~1,371,980 | 小 | 提交模板 |

## 通用架构: 两阶段召回-排序管道

所有 Top 解决方案使用相同架构:
```
Stage 1: 候选生成 (Recall/Candidate Generation)
  → 每个客户生成 100-300 个候选商品

Stage 2: 排序 (Ranking with LGBMRanker)
  → 对候选商品打分排序
  → 取 top-12 作为最终预测
```

## Top 解决方案详情

### 1st Place (SENKIN13) — MAP@12 ~0.037
- 核心策略: 近期热门商品 + GBDT 排序
- 训练数据: 仅用最近6周 (时尚趋势变化快)
- 负采样: 召回但未购买的商品作为负样本
- 关键洞察: 近期热门是最强的召回信号

### 2nd Place (wht1996)
- 用户-商品复购特征 + 高阶组合特征
- 混合 binary 和 ranking 目标

### 3rd Place (SIRIUS)
- 召回元特征 (meta-features): 每个商品被哪个策略召回, 排名多少
- 如果一个商品被6个策略中的5个召回, 购买概率极高

### 9th Place
- RAPIDS cuDF 加速
- 每个客户200个候选
- 负样本从最近20周随机采样一半

### 25th Place
- LambdaMART + 人气/交互/趋势特征
- 标准实现参考

## 候选生成策略 (按重要性排序)

### Tier 1 — 必须实现 (最高召回)
1. **复购 (Repurchase)**: 客户最近1-4周购买过的商品 (10.8%客户1周内复购)
2. **近期热门 (Recent Popular)**: 最近1-2周全球畅销商品
3. **分类热门**: 按 sales_channel_id, index_group_no, department 分组的热门商品

### Tier 2 — 强召回提升
4. **Item-to-Item 协同过滤**: 共现矩阵 (同一客户同一周购买的商品对)
5. **同产品线**: 相同 product_code 不同颜色/尺码
6. **Word2Vec/Item2Vec**: 购买序列训练嵌入

### Tier 3 — 锦上添花
7. **ALS 矩阵分解**: Bayesian Personalized Ranking
8. **图神经网络**: customer-item 二部图
9. **双塔模型**: Neural retrieval

## 排序特征工程

### 客户特征
- age, age_group, club_member_status, fashion_news_frequency
- 历史购买次数 (lifetime, 最近4周, 最近1周)
- 平均购买价格, 偏好部门/类别
- RFM 评分 (Recency, Frequency, Monetary)

### 商品特征
- product_type_no, colour_group_code, department_no
- 商品价格 (当前价格, 历史均价)
- 商品年龄 (首次出现距今天数)

### 人气/趋势特征
- 全局销售量 (最近1/2/4/8/16周)
- 部门级销售量 (最近1周)
- 时间衰减人气分数
- 新品标记

### 客户-商品交互特征 (最重要!)
- 客户是否购买过此商品 (二值)
- 距上次购买天数
- 购买次数
- 是否购买过同 product_code / product_type_no / department_no 的商品
- 客户在此商品类别的购买总数
- 客户在此价格区间的平均消费

### 召回元特征 (高影响力 — 3rd Place 创新)
- 每个策略是否召回了此商品 (每策略一列)
- 此商品在每个策略中的排名
- 召回此商品的策略数量 (多样性分数)

## LGBMRanker 设置

```python
from lightgbm import LGBMRanker

ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[12],
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=255,
)

# group 参数: 每个客户-周的候选数量
train_baskets = train_df.groupby(['week', 'customer_id']).size().values
ranker.fit(X_train, y_train, group=train_baskets)
```

## 验证策略
- 时间切分: 最后7天作为验证集
- 训练集只用最近4-20周 (时尚趋势变化快)
- 评估指标直接用 MAP@12 (不是 NDCG)

## 常见陷阱
1. **负采样中的假负样本**: 热门商品作为负样本可能实际被购买
2. **过拟合最后一周**: 时尚趋势变化快
3. **冷启动用户 (~50%)**: 近半数测试用户近期无交易, 需要兜底策略
4. **使用过多历史数据**: 18个月前的购买模式已不相关
5. **忽视季节性**: 夏装冬天不卖

## 推荐实现顺序
1. Baseline: 复购 + 热门商品 (~0.020 MAP@12)
2. R01: LGBMRanker 基础版 + 基本特征 (~0.025)
3. R02: 扩展候选生成 + 交互特征 (~0.030)
4. R03: 深度特征工程 + 召回元特征 (~0.033)
5. R04: 多策略集成 + 冷启动处理 (~0.035)
