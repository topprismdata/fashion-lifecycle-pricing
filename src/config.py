"""
Global configuration for Fashion Lifecycle Pricing project.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data_raw"
SUBMISSIONS = ROOT / "outputs" / "submissions"
MODELS = ROOT / "outputs" / "models"


@dataclass
class CompetitionConfig:
    """Per-competition configuration."""
    name: str = ""
    n_folds: int = 5
    random_state: int = 42
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    })
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
    })
