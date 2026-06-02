---
description: Run comprehensive EDA pipeline on the current dataset using best-in-class libraries
---

# /eda — Exploratory Data Analysis

> Run the 5-stage EDA pipeline using ydata-profiling, missingno, sweetviz, AutoViz, and (optionally) tslumen. Generates HTML reports + PNG visualizations.

## Usage

```
/eda [data-path]
/eda [data-path] --segment=column_name
/eda [data-path] --time-series=date_column
```

Default data path: `data/train.csv`

## What This Does

Runs 5 stages per `memory/skills/retail-eda-framework.md`:

### Stage 1: Data Quality Audit
```bash
# Uses ydata-profiling + missingno
python eda/01_data_quality.py
```
**Outputs**:
- `eda/data_quality.html` — comprehensive profile
- `eda/missing_matrix.png`, `eda/missing_heatmap.png`, `eda/missing_dendrogram.png`

### Stage 2: Statistical Profiling
```bash
# Train vs Test comparison
python eda/02_train_vs_test.py
```
**Outputs**:
- `eda/train_vs_test.html` — side-by-side comparison
- `eda/segment_<name>.html` (if --segment specified)

### Stage 3: Domain-Specific EDA
For retail/fashion: RFM, transaction patterns, co-occurrence, seasonality.
```bash
python eda/03_domain_specific.py
```
**Outputs**:
- `eda/rfm_segments.html`
- `eda/transaction_patterns.png`
- `eda/cooccurrence_matrix.png` (if applicable)

### Stage 4: Time-Series EDA (if --time-series)
```bash
python eda/04_time_series.py
```
**Outputs**:
- `eda/decomposition.png`
- `eda/acf_pacf.png`
- `eda/stationarity_tests.html`

### Stage 5: EDA Summary
```bash
# Synthesizes all findings into a markdown report
python eda/05_summarize.py
```
**Outputs**:
- `eda/EDA_SUMMARY.md` — key findings + recommendations

## When to Use

**Always** at the start of any new competition/dataset, BEFORE feature engineering.

## Dependencies

Installs required (some are heavy):
```bash
pip install ydata-profiling missingno sweetviz autoviz
# Optional (for time-series):
pip install tslumen
```

If any package is not installed, that stage is skipped with a warning.

## Anti-Patterns

- ❌ Skipping EDA to "save time" → always leads to feature engineering errors
- ❌ Skipping Stage 2 (train vs test) → catastrophic in production
- ❌ Not checking missingness correlation → misses systematic missingness
- ❌ Running EDA on test set only → information leakage
