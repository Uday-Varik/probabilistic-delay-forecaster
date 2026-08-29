"""Train and save the production XGBoost model used by the live API.

Lean-dependency version of the full baseline comparison (src/baseline/train.py)
— this only needs pandas/xgboost (the deployed runtime dependencies), not
Prophet, so it's what the Dockerfile runs at build time. The full comparison
against Prophet and the accuracy/calibration report live in train.py, which
needs the dev dependency set.

Run:
    python -m src.baseline.train_production
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baseline.features import build_features, feature_columns
from src.baseline.xgboost_model import save_models, train_xgboost_quantiles

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "delay_timeseries.csv"
MODELS_DIR = ROOT / "models"


def train_and_save() -> None:
    feat_cols = feature_columns()
    df = build_features(pd.read_csv(DATA_PATH)).dropna(subset=feat_cols).reset_index(drop=True)
    models = train_xgboost_quantiles(df, feat_cols)
    save_models(models, MODELS_DIR)
    print(f"Trained on {len(df)} rows, saved production models to {MODELS_DIR}")


if __name__ == "__main__":
    train_and_save()
