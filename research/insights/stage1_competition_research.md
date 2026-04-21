# Stage 1 竞赛调研报告

> 整理时间: 2026-04-21
> 来源: Kaggle官方文档、论坛讨论、Top解决方案分析

---

## 一、Walmart Recruiting - Store Sales Forecasting

### 基本信息
- **URL**: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting
- **状态**: 已关闭 (2014年)
- **任务**: 预测45家Walmart门店各部门的周销售额
- **评估指标**: WMAE (Weighted Mean Absolute Error)

### WMAE 公式
```
WMAE = (1 / Σwi) * Σ wi * |yi - ŷi|

其中:
  wi = 5 (假期周)
  wi = 1 (非假期周)
```

### 假期日期 (权重5倍)
| 假期 | 2010 | 2011 | 2012 | 2013 |
|------|------|------|------|------|
| Super Bowl | 12-Feb | 11-Feb | 10-Feb | 8-Feb |
| Labor Day | 10-Sep | 9-Sep | 7-Sep | 6-Sep |
| Thanksgiving | 26-Nov | 25-Nov | 23-Nov | 29-Nov |
| Christmas | 31-Dec | 30-Dec | 28-Dec | 27-Dec |

### 数据文件
| 文件 | 说明 |
|------|------|
| train.csv | 历史销售数据 (Store, Dept, Date, Weekly_Sales, IsHoliday) |
| test.csv | 预测目标周
| features.csv | 补充特征 (Temperature, Fuel_Price, MarkDown 1-5, CPI, Unemployment) |
| stores.csv | 门店信息 (Type, Size) |

### 关键挑战
1. **MarkDown 1-5 缺失率 >60%** — 匿名促销特征
2. **假期权重5倍** — 假期预测精度对排名影响极大
3. **门店-部门组合** — 45店 × ~98部门 = ~4000个独立序列
4. **负销售额** — 部分退货导致负值

### Top 解决方案
| 排名 | 方法 | 核心技术 |
|------|------|---------|
| 1st (David Thaler) | 8个时序模型简单平均 | STLF, ETS, 线性季节模型, R实现 |
| 2nd | GBDT集成 | Extra Trees + Gradient Boosting + Random Forest |
| 常见方法 | XGBoost/LightGBM | Lag特征, 滚动统计, 假期特征 |

---

## 二、H&M Personalized Fashion Recommendations

### 基本信息
- **URL**: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
- **状态**: 已关闭 (2022年)
- **任务**: 为每个客户预测未来7天最可能购买的12件商品
- **评估指标**: MAP@12 (Mean Average Precision at 12)
- **参赛人数**: ~3,006 teams
- **奖金**: $50,000

### MAP@12 公式
```
AP@12(u) = (1 / min(m, 12)) * Σ_{k=1}^{12} P(k) * rel(k)
MAP@12 = (1/|U|) * Σ AP@12(u)

其中:
  m = 客户实际购买数
  P(k) = top-k的精确率
  rel(k) = 第k位预测是否命中
```

### 数据文件
| 文件 | 行数 | 大小 | 说明 |
|------|------|------|------|
| transactions_train.csv | ~3180万 | ~2.7GB | 2年购买历史 |
| articles.csv | ~105,542 | 小 | 25列商品元数据 |
| customers.csv | ~1,371,980 | 中 | 7列客户信息 |
| images/ | ~105K张 | 大 | 商品图片 |

### articles.csv 关键列 (25列)
- article_id, product_code, prod_name
- product_type_no/name, product_group_name
- graphical_appearance_no/name
- colour_group_code/name
- perceived_colour_value/master
- department_no/name, index_code/name
- index_group_no/name, section_no/name
- garment_group_no/name, detail_desc

### customers.csv 关键列
- customer_id, FN, Active (大量NaN)
- club_member_status, fashion_news_frequency
- age, postal_code

### 通用架构: 两阶段召回-排序管道

```
Stage 1: 候选生成 (召回)
  → 每个客户生成 50-200 个候选商品

Stage 2: 排序 (Learning-to-Rank)
  → 用 LGBMRanker/XGBoost/DNN 对候选打分
  → 取 top-12 作为最终预测
```

### 候选生成策略 (按重要性排序)
1. **复购 (Repurchase)** — 客户最近购买的商品 (最强信号)
2. **Item2Item协同过滤** — 买X的人也买Y
3. **近期热门** — 最近1-4周的畅销商品
4. **趋势商品** — 周环比增长的商品
5. **Word2Vec/Item2Vec** — 购买序列嵌入
6. **矩阵分解** — 用户-商品隐因子

### LGBMRanker 使用模式
```python
model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=500,
    learning_rate=0.05,
)
model.fit(X_train, y_train, group=group_sizes)
# group_sizes: 每个客户的候选数量
# y_train: 1=购买, 0=未购买
```

### Top 解决方案
| 排名 | 方法 | 核心技术 |
|------|------|---------|
| 1st | 多种召回 + GBDT排序 | 复购 + Item2Item + 热门 + 趋势 |
| 25th | LightGBM LambdaMART | 的人气/交互/趋势特征 |
| 45th (银牌) | 2召回 × 3排序 = 6模型集成 | LGBMRanker + LGBMClassifier + DNN |

---

## 三、ML Zoomcamp 2024 - Retail Demand Forecasting

### 基本信息
- **URL**: https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition
- **课程**: DataTalksClub ML Zoomcamp 2024
- **任务**: 预测4家门店28,182个商品的下月需求量
- **评估指标**: RMSE
- **数据大小**: ~872MB, 10个CSV文件
- **时间跨度**: 25个月 (2022年8月 - 2024年9月)

### 数据文件
| 文件 | 说明 | 关键列 |
|------|------|--------|
| sales.csv | 历史销售 | date, item_id, store_id, quantity, price |
| online.csv | 在线销售 | 同sales.csv |
| markdowns.csv | 降价/折扣数据 | date, item_id, store_id, markdown字段 |
| prices.csv | 商品定价 | date, item_id, price |
| stores.csv | 门店元数据 | store_id, 门店属性 |
| items.csv | 商品目录 | item_id, 商品属性 |
| test.csv | 测试集 | item_id, store_id, date |
| sample_submission.csv | 提交模板 | 行标识 + 目标列 |

### 关键特征
- **价格数据可用** — prices.csv 和 markdowns.csv 支持价格弹性分析
- **促销影响** — 促销数据分析
- **时序价格变化** — 25个月内的价格变化提供自然变异
- **商品目录可能不完整** — 部分销售数据中的item不在items.csv中

### Top 解决方案
| 排名 | 方法 | 分数 |
|------|------|------|
| 1st | AutoML | ~11.9 RMSE |
| 4th | 特征工程 + Boosting | ~11.95 RMSE |

### 典型特征工程
- Lag特征 (1, 7, 14, 28天的历史销量)
- 滚动统计 (mean, std)
- 价格变化特征
- 时间特征 (星期几, 月份, 年中第几周)
- 日粒度 → 月粒度聚合

### 价格弹性与因果推断
- 竞赛本身不要求因果推断
- 但数据包含价格和降价信息，适合自行分析价格弹性
- 需要额外引入 DML 等因果推断方法

---

## 四、三个竞赛的对比总结

| 维度 | Walmart | H&M | ML Zoomcamp |
|------|---------|-----|-------------|
| **任务** | 回归(周销量) | 推荐(12件商品) | 回归(月需求) |
| **指标** | WMAE | MAP@12 | RMSE |
| **数据量** | 中等 | 大(~3GB) | 大(~872MB) |
| **时序** | 是(周频) | 否(推荐) | 是(日频) |
| **核心挑战** | 缺失值+假期权重 | 大规模排序 | 价格弹性+需求预测 |
| **与本项目关联** | 季节性预测基础 | 推荐替代降价 | 因果推断实验室 |

## 五、学习路径建议

### 1. 先做 Walmart (1-2周)
- 理由: 数据量小，时序预测基础
- 重点: WMAE优化，缺失值处理，假期特征
- 技能: Lag特征，滚动统计，加权损失

### 2. 再做 H&M (2-3周)
- 理由: 推荐系统是"避免降价"的核心
- 重点: LGBMRanker，两阶段管道，大规模数据处理
- 技能: 候选生成，排序模型，MAP@12优化

### 3. 最后做 ML Zoomcamp (2-3周)
- 理由: 结合价格数据，引入因果推断
- 重点: 价格弹性估计，DML，降价效应分析
- 技能: 因果推断，价格-需求关系建模
