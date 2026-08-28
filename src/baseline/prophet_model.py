"""Prophet per-lane baseline for delay-rate forecasting.

Prophet doesn't natively support multiple related series in one model, so
this trains one model per lane. Uses Prophet's native uncertainty interval
(interval_width=0.8) as the p10/p90 baseline — directly comparable to the
XGBoost quantile baseline and, later, the deep-learning model's quantiles.
"""

from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

REGRESSORS = ["weather_risk_score", "is_holiday", "is_peak_season", "is_weekend"]

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def train_prophet_per_lane(train_df: pd.DataFrame) -> dict[str, Prophet]:
    models: dict[str, Prophet] = {}
    for lane_id, lane_df in train_df.groupby("lane_id"):
        prophet_df = lane_df.rename(columns={"date": "ds", "delay_rate": "y"})[
            ["ds", "y"] + REGRESSORS
        ]
        model = Prophet(interval_width=0.8)
        for reg in REGRESSORS:
            model.add_regressor(reg)
        model.fit(prophet_df)
        models[lane_id] = model
    return models


def predict_prophet(models: dict[str, Prophet], df: pd.DataFrame) -> pd.DataFrame:
    all_preds = []
    for lane_id, lane_df in df.groupby("lane_id"):
        model = models[lane_id]
        future = lane_df.rename(columns={"date": "ds"})[["ds"] + REGRESSORS]
        forecast = model.predict(future)
        forecast["lane_id"] = lane_id
        forecast = forecast.rename(columns={
            "yhat": "prophet_p50", "yhat_lower": "prophet_p10", "yhat_upper": "prophet_p90",
        })
        all_preds.append(forecast[["ds", "lane_id", "prophet_p10", "prophet_p50", "prophet_p90"]])
    return pd.concat(all_preds, ignore_index=True).rename(columns={"ds": "date"})
