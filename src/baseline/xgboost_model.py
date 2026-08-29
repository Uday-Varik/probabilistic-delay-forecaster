"""XGBoost quantile-regression baseline (p10/p50/p90) for delay-rate
forecasting, using the native `reg:quantileerror` objective (XGBoost >= 2.0)
so the baseline produces real calibrated quantiles directly comparable to
the deep-learning model's quantile outputs, not just a point estimate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

QUANTILES = [0.1, 0.5, 0.9]


def _prep_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    return X


def train_xgboost_quantiles(
    train_df: pd.DataFrame, feature_cols: list[str], target_col: str = "delay_rate"
) -> dict[float, xgb.XGBRegressor]:
    X = _prep_features(train_df, feature_cols)
    y = train_df[target_col]

    models: dict[float, xgb.XGBRegressor] = {}
    for q in QUANTILES:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            enable_categorical=True,
            random_state=42,
        )
        model.fit(X, y)
        models[q] = model
    return models


def predict_xgboost_quantiles(
    models: dict[float, xgb.XGBRegressor], df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    X = _prep_features(df, feature_cols)
    preds = pd.DataFrame(index=df.index)
    for q, model in models.items():
        preds[f"xgb_p{int(q * 100)}"] = model.predict(X)
    return preds


def save_models(models: dict[float, xgb.XGBRegressor], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for q, model in models.items():
        model.save_model(out_dir / f"xgb_p{int(q * 100)}.json")


def load_models(in_dir: Path) -> dict[float, xgb.XGBRegressor]:
    models: dict[float, xgb.XGBRegressor] = {}
    for q in QUANTILES:
        model = xgb.XGBRegressor(enable_categorical=True)
        model.load_model(in_dir / f"xgb_p{int(q * 100)}.json")
        models[q] = model
    return models
