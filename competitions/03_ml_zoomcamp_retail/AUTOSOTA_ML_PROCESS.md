# AutoSOTA-Inspired ML Experiment Framework

> 基于 AutoSOTA (清华 FIB Lab) 的方法论，优化我们的 ML 竞赛实验流程。
> AutoSOTA 用 LLM agent 自动优化 105 篇论文代码，我们将其核心机制内化。

## 核心机制映射

| AutoSOTA 机制 | 我们的 ML 流程 | 实现 |
|--------------|--------------|------|
| 三层想法系统 (ALGO/CODE/PARAM) | 实验分类 | 每个 experiment 标记类型 |
| Red Lines (6条硬约束) | 实验 Red Lines | 防止过拟合/泄露等 |
| Leap + Honeymoon | 跳出局部最优 | 连续3轮 PARAM → 强制 ALGO |
| 结构化记忆 (code_analysis) | 竞赛数据分析 | `DATA_ANALYSIS.md` |
| 结构化记忆 (research_report) | 竞赛研究 | `RESEARCH.md` |
| 结构化记忆 (idea_library) | 实验想法库 | `IDEA_LIBRARY.md` |
| 迭代追踪 (scores.jsonl) | 实验追踪 | `EXPERIMENT_LOG.md` |
| Git 回滚 | 脚本版本管理 | 每个 R 轮次独立脚本 |

## 三层实验分类

### ALGO — 算法/架构变更（最高优先级）
改变模型架构或核心方法。风险高但潜在收益大。

**示例（ML Zoomcamp）**：
- R07: 两阶段模型（零膨胀）
- 使用 Transformer/TSF 模型替代 GBDT
- 引入因果推断方法估计价格弹性
- 跨门店迁移学习

### CODE — 特征工程/数据处理变更
改变数据处理流程或特征创建逻辑。

**示例**：
- R02: 滚动窗口特征 (7/14/30 天)
- R03: 多源数据整合 (discounts/markdowns)
- R04: YoY 364天特征
- 新的编码策略

### PARAM — 超参数调优（最低优先级）
只调参数，不改算法或特征。

**示例**：
- R05: Optuna 调参
- learning_rate/depth 微调
- 正则化参数调整

## Red Lines（实验硬约束）

| # | Red Line | 说明 | 检查方法 |
|---|----------|------|---------|
| R1 | 无特征泄露 | 不使用测试期数据计算特征 | 检查 lag 特征的 cutoff |
| R2 | CV 策略不变 | 时序验证，不做随机 K-fold | 验证集日期 < 测试集日期 |
| R3 | 不修改评估指标 | RMSE，不换 metric | 提交前确认 metric |
| R4 | 提交不超过10次/竞赛 | 避免过拟合 LB | 追踪 kaggle 提交次数 |
| R5 | 不删除数据子集 | 不因"看着异常"就删除 | 异常值处理需有统计依据 |
| R6 | 每个 R 轮次独立脚本 | 可复现，可回滚 | 文件命名 run_r{N}_*.py |

## Leap + Honeymoon 机制

### Leap 触发条件
当最近 **3 个连续实验** 都是 PARAM 类型时，强制执行 Leap：
- 必须提出一个 ALGO 级别的想法
- 参考其他竞赛/论文的方法（跨域迁移）
- 在 idea_library 中标记为 LEAP

### Honeymoon 规则
Leap 实施后，如果首轮未改进：
- 保留 Leap 变更，继续 5 轮迭代
- 这 5 轮中可以用 PARAM 微调来充分挖掘 Leap 潜力
- 如果 5 轮内出现新 best → Leap 验证成功
- 如果 5 轮全部失败 → 回滚到 Leap 前

## 实验追踪格式

每轮实验记录：

```markdown
### R{N}: {标题}
- **类型**: ALGO / CODE / PARAM / LEAP
- **状态**: SUCCESS / FAIL / REGRESS
- **CV RMSE**: {val} (prev: {prev_val}, delta: {delta})
- **LB RMSE**: {val} (optional)
- **关键变更**: {描述}
- **提取的 Skills**: {skill_name} or None
- **教训**: {什么有效/什么无效}
- **Red Line 检查**: PASS/FAIL ({原因})
```

## 竞赛模板

每个新竞赛开始时创建：

```
competitions/{N}_{name}/
├── DATA_ANALYSIS.md     ← AutoSOTA code_analysis.md 等价物
├── RESEARCH.md          ← AutoSOTA research_report.md 等价物
├── IDEA_LIBRARY.md      ← AutoSOTA idea_library.md 等价物
├── EXPERIMENT_PLAN.md   ← 实验计划
├── EXPERIMENT_LOG.md    ← 实验追踪日志
├── RED_LINES.md         ← 本竞赛的 Red Lines
├── scripts/             ← 每轮独立脚本
├── notebooks/           ← EDA notebook
├── data/raw/            ← 原始数据
└── outputs/             ← submissions + models
```

## IDEA_LIBRARY.md 格式

```markdown
# Idea Library — {竞赛名}

## ALGO 级想法

### A1: {想法标题}
- **来源**: {论文/竞赛/经验}
- **描述**: {详细描述}
- **预期收益**: {高/中/低}
- **风险**: {描述}
- **状态**: TODO / IN_PROGRESS / SUCCESS / FAIL
- **实现轮次**: R{N}

### A2: ...

## CODE 级想法

### C1: {想法标题}
- ...（同上格式）

## PARAM 级想法

### P1: {想法标题}
- ...（同上格式）
```

## 与 Reverse Scaffolding 论文的关系

这个框架本身就是 Reverse Scaffolding 的一个应用案例：
- **L1 Core**: Red Lines, 实验分类, 追踪格式 → 跨竞赛通用
- **L2 Domain**: 特征工程想法, 模型选择 → 按竞赛类型积累
- **L3 Contextual**: 具体竞赛的数据分析, top solution 研究 → 单次使用

成功案例：AutoSOTA 证明了结构化 agent 优化有效（105篇论文平均提升10%+）。
我们的假说：将同样的结构化方法应用于 ML 竞赛，可以系统地提高实验效率。
