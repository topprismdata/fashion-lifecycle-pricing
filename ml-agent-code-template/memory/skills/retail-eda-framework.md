---
name: retail-eda-framework
description: Comprehensive EDA approach for retail/fashion/tabular ML using best-in-class libraries. Use during stage 1 (data understanding) of any ML pipeline.
type: learned
---

# Retail/Fashion EDA Framework

> Comprehensive EDA approach for retail/fashion/tabular data, validated through Walmart, H&M, and 7+ fashion-lifecycle-pricing competitions.

## When to Use

**Always** at the start of any new dataset/competition:
- After `data_preparation/loading`
- Before any feature engineering
- Before model selection

## The 5-Stage EDA Pipeline

### Stage 1: Data Quality Audit (FIRST)

**Tools**: `great_expectations` (11.5k★), `missingno` (4.2k★), `ydata-profiling` (13.5k★)

```python
# Quick data quality report
import ydata_profiling
profile = ydata_profiling.ProfileReport(df, title="Data Quality Report")
profile.to_file("eda/data_quality.html")

# Missing data visualization
import missingno as msno
msno.matrix(df)
msno.heatmap(df)  # Correlation of missingness
msno.dendrogram(df)  # Hierarchical missingness
```

**Look for**:
- Missing value patterns (random vs systematic)
- High-cardinality categoricals
- Skewed numerical features
- Constant/quasi-constant features
- Duplicate rows
- Outliers (use IQR or z-score, not just visual)

### Stage 2: Statistical Profiling

**Tools**: `pandas-profiling` (now ydata-profiling), `sweetviz` (3.1k★)

```python
# Compare train vs test
import sweetviz as sv
report = sv.compare([train_df, "Train"], [test_df, "Test"], target_feat="target")
report.show_html("eda/train_vs_test.html")

# Per-segment analysis
for segment in df['segment'].unique():
    subset = df[df['segment'] == segment]
    profile = ydata_profiling.ProfileReport(subset, minimal=True)
    profile.to_file(f"eda/segment_{segment}.html")
```

**Look for**:
- Train/test distribution shift
- Per-segment behavior differences
- Data leakage (test features in train)
- Time-based trends

### Stage 3: Domain-Specific EDA

For **retail/fashion**, focus on:

| Pattern | Tools | What to look for |
|---------|-------|-------------------|
| **Customer segmentation** | `RFM analysis` (sonwanesuresh95/rfm) | Recency, Frequency, Monetary quintiles |
| **Transaction patterns** | pandas, seaborn | Basket size, frequency, time-of-day, day-of-week |
| **Article co-occurrence** | custom ItemCF | Items bought together, substitute/companion |
| **Fashion seasonality** | statsmodels, tslumen | Year-over-year, holiday effects |
| **Cold start analysis** | pandas | % test customers with no history, % new articles |
| **Inventory/availability** | groupby, time series | Stockouts, restock patterns |

### Stage 4: Time-Series EDA (if applicable)

**Tools**: `tslumen` (72★, HSBC-maintained), `statsmodels`

```python
# For sales forecasting competitions
import tslumen
tslumen.from_ts(df.set_index('date')['sales']).plot()
# Decomposition, ACF/PACF, stationarity tests
```

**Look for**:
- Trend / seasonality / residual decomposition
- Stationarity (ADF test)
- Autocorrelation structure (ACF/PACF)
- Holiday/special-event effects
- Data frequency consistency (missing weeks, etc.)

### Stage 5: Image EDA (if applicable)

**Tools**: `PIL`, `torchvision`, custom grids

```python
# Sample images by class, check aspect ratios
from torchvision.utils import make_grid
grid = make_grid(samples, nrow=8)
# Save and inspect for: blurry images, wrong labels, lighting issues
```

**Look for**:
- Mislabeled examples
- Out-of-distribution images
- Class imbalance
- Aspect ratio consistency
- Color distribution shifts

## Top EDA Library Recommendations (2026-06-02)

| Library | Stars | Use for | Install |
|---------|------|---------|---------|
| `ydata-profiling` | 13.5k | Comprehensive 1-line EDA report | `pip install ydata-profiling` |
| `great_expectations` | 11.5k | Data quality + unit tests for data | `pip install great_expectations` |
| `visidata` | 9.1k | Terminal-based interactive exploration | `pip install visidata` |
| `lux` | 5.4k | Auto-viz on dataframe print | `pip install lux-api` |
| `missingno` | 4.2k | Missing data visualization | `pip install missingno` |
| `sweetviz` | 3.1k | Compare train vs test | `pip install sweetviz` |
| `dataprep` | 2.2k | Low-code data prep + EDA | `pip install dataprep` |
| `AutoViz` | 1.9k | 1-line automatic viz | `pip install autoviz` |
| `tslumen` | 72 | Time-series specific EDA | `pip install tslumen` |
| RFM analysis | 41 | Customer segmentation | `pip install rfm-analysis` (or custom) |

## Common EDA Mistakes (Anti-Patterns)

1. ❌ **Skipping EDA** to save time → always leads to feature engineering errors
2. ❌ **Only looking at summary statistics** → miss distribution shape
3. ❌ **Ignoring train/test distribution shift** → catastrophic in production
4. ❌ **Not checking missingness correlation** → indicates systematic missingness
5. ❌ **Assuming all categorical = independent** → categories may be related
6. ❌ **Forgetting to check time-based leakage** → future data in training
7. ❌ **Not visualizing outliers** → they may be valid (e.g., luxury items)
8. ❌ **Treating all features equally** → ID-like features need special handling

## Empirical Evidence

**MLE-Bench experiments**:
- **Spaceship Titanic Gold**: 0.8506 — EDA revealed `Cabin` split into deck/side/num was key
- **Jigsaw Toxic Gold**: 0.98829 — EDA on per-label distribution identified imbalance
- **TPS May Silver**: 0.99754 — EDA on f_27 string structure drove feature breakthrough

**fashion-lifecycle-pricing**:
- **H&M R27 (best, 0.02314)**: 27 experiments preceded by proper EDA of:
  - Customer behavior (recency, frequency, basket size)
  - Article attributes (color, type, price, age)
  - Co-occurrence patterns (item-item)
  - Train/test customer overlap (~80% cold start)
- **Walmart R08 (best, LB=2720)**: EDA on MarkDown missing patterns + seasonality

## How to Apply

```bash
# In a new competition
/eda
```

The `/eda` slash command runs the 5-stage pipeline above and generates:
1. `eda/data_quality.html` (ydata-profiling)
2. `eda/missing_pattern.png` (missingno)
3. `eda/train_vs_test.html` (sweetviz)
4. `eda/segment_breakdown.html` (ydata-profiling per segment)
5. `eda/time_series.html` (tslumen, if applicable)
6. `eda/image_grid.png` (if applicable)
7. `eda/EDA_SUMMARY.md` — written observations

## Related

- `cv-strategy` — CV strategy depends on EDA findings
- `feature-engineering-roi` — EDA findings drive feature engineering
- `external-data-fusion` — EDA may reveal need for external data
