# ML Zoomcamp 2024 Retail Demand Forecast — 竞赛研究

## 基本信息

- **竞赛**: https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition
- **主办方**: DataTalks.Club (Alexey Grigorev)
- **评估指标**: RMSE
- **数据来源**: 俄罗斯零售商，4家门店，25个月（~2022.08 - 2024.09）
- **预测目标**: 预测 2024年9-10月 各门店各商品的销量 (quantity)
- **状态**: 已结束（2025.02）

## 数据概览

### 9个数据文件

| 文件 | 大小 | 说明 |
|------|------|------|
| sales.csv | 362 MB | 门店日销量（核心训练数据） |
| online.csv | 54 MB | 线上日销量 |
| discounts_history.csv | 326 MB | 促销历史 |
| price_history.csv | 28 MB | 价格变动记录 |
| catalog.csv | 24 MB | 商品目录（dept/class/subclass） |
| test.csv | 29 MB | 测试集（分号分隔） |
| sample_submission.csv | 8 MB | 提交格式 |
| actual_matrix.csv | 1 MB | 门店在售商品列表 |
| stores.csv | 158 B | 4家门店信息 |
| markdowns.csv | 416 KB | 降价商品 |

### 关键列

- **sales.csv**: date, item_id, quantity, price_base, sum_total, store_id
- **catalog.csv**: item_id, dept_name, class_name, subclass_name, item_type, weight_volume, weight_netto, fatness
- **test.csv**: row_id, item_id, store_id, date（分号分隔）
- **13,636 unique items** in test set, **28,182** in training

### 数据质量问题
- quantity 有负值（退货/数据错误）
- price_base 有 inf（sum_total/0）
- sum_total 有负值
- 所有 CSV 有多余的 `Unnamed: 0` 列
- test.csv 用分号分隔

## Top Solutions 分析

### 排行榜

| 排名 | 队伍 | RMSE | 方法 |
|------|------|------|------|
| 1 | Alvaro | 8.9580 | - |
| 2 | ArturG | 9.2348 | AutoGluon + DuckDB 特征工程 |
| 3 | Adi Kusuma | 9.2519 | - |
| 4 | AleTBM | 9.4650 | CatBoost + 滚动特征 + 假期 |
| 5 | Edison Marcelo | 9.5624 | - |
| 6 | KABIR | 9.8714 | RandomForest |

### 关键技术

1. **滚动窗口特征**: 7/14/30 天均值（最重要的特征类型）
2. **假期特征**: 俄罗斯假期（holidays 库）
3. **异常值处理**: 负数量删除，价格 inf 处理，分位数裁剪
4. **多源数据整合**: 线上+线下销售合并
5. **日期填充**: date × store × item cross-join 填充缺失日期
6. **CatBoost 主导**: Top solutions 多数用 CatBoost
7. **AutoGluon**: 2nd place 用 AutoML，说明特征工程比模型选择更重要
8. **内存优化**: float64→float32，分块处理

## 与我们经验的关系

| Store Sales 经验 | 本竞赛对应 |
|-----------------|-----------|
| WMAE评估 | RMSE（更简单，无权重） |
| 5-fold 验证 | 时序验证（按时间切分） |
| 滚动统计 | 核心特征（7/14/30天） |
| LightGBM | CatBoost 可能更优 |
| 零膨胀处理 | 负数量处理 |
| 特征工程7模式 | 直接适用 |

## 注意事项

- 数据来自俄罗斯零售商，假期用俄罗斯日历
- 只有4家门店，门店特征有限
- 商品层级结构（dept→class→subclass）是重要分类特征
- 稀疏时序：不是每个商品每天都在每家店卖
- 促销和降价对销量影响大
