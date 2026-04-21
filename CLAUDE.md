# Fashion Lifecycle Pricing — Claude Code 工作指引

## 项目概述

机器学习驱动的服饰全生命周期决策优化研究。
覆盖 7 个竞赛/数据集，从需求预测到动态定价的完整技术栈。
详见 DESIGN.md 和 PLAN.md。

## 项目结构

- `DESIGN.md` — 完整技术设计文档
- `PLAN.md` — 执行计划和里程碑
- `src/` — 共享模块 (跨竞赛复用)
- `competitions/` — 7个竞赛独立目录
- `research/` — 研究笔记

## 工作规则

### 竞赛工作流

1. 每个竞赛独立目录: scripts/, notebooks/, outputs/, README.md
2. 遵循 ML Pipeline: EDA → Baseline → 特征工程 → 模型优化 → 集成 → 提交
3. 复用 `src/` 中的共享模块，避免重复代码
4. 每次实验记录: 版本号, CV分数, LB分数, 关键技术

### Skills 提取

- 每个阶段完成后执行 `/claudeception`
- 提取可复用的技术和模式为 skills
- 记录什么有效、什么无效

### 代码规范

- Python 3.10+
- 类型注解
- 每个脚本可独立运行
- 使用 argparse 支持命令行参数
- 日志输出到 stdout 和文件

## 与 S6E4 经验的关联

S6E4 项目积累的 skills 和经验直接适用:
- `sigmoid-smoothing-target-encoding` — 类别编码
- `tabular-feature-engineering-patterns` — 特征工程7大模式
- `ml-pipeline-unit-testing` — 数据质量验证
- `iterative-pseudo-labeling-backfire` — 避免过度迭代
- `kaggle-optimal-blending` — 集成策略
- `pairwise-target-encoding-strategy` — 交互编码

## Memory Management

- 参考全局 CLAUDE.md 中的内存管理规则
- 避免并行读取大文件
- 使用 tail/grep 替代读取完整日志
