"""Serving-time forecasting: loads the saved production XGBoost model and
produces recursive multi-day-ahead quantile forecasts by lane.

XGBoost was chosen to serve live requests because it beat the TFT
deep-learning model on both accuracy and calibration (see case_study.md) —
serving the more complex model anyway, after it lost the comparison it was
built to win, would be complexity without justification.

Recursive forecasting: each day beyond day 1 uses the previous day's p50
prediction as its lag-1 input (lag-7/14 and rolling means still draw from
real history until the horizon exceeds those windows). This is standard
practice for tree-based multi-step forecasting — uncertainty compounds
honestly this way, since a 7-day-out forecast is built on 6 days of the
model's own predictions, not on information it doesn't actually have.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.baseline.features import LAG_DAYS, ROLLING_WINDOWS, feature_columns
from src.baseline.xgboost_model import load_models, predict_xgboost_quantiles

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "delay_timeseries.csv"
MODELS_DIR = ROOT / "models"

FIXED_HOLIDAYS_MMDD = [(1, 1), (7, 4), (11, 11), (12, 25)]
PEAK_SEASON_START_MMDD = (11, 20)
PEAK_SEASON_END_MMDD = (12, 26)

DEFAULT_WEATHER_RISK = 0.15  # historical non-shock average


def _is_holiday(d: date) -> bool:
    return (d.month, d.day) in FIXED_HOLIDAYS_MMDD


def _is_peak_season(d: date) -> bool:
    start = date(d.year, *PEAK_SEASON_START_MMDD)
    end = date(d.year, *PEAK_SEASON_END_MMDD)
    return start <= d <= end


class ForecastService:
    def __init__(self) -> None:
        self.models = load_models(MODELS_DIR)
        df = pd.read_csv(DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        self.history = df.sort_values(["lane_id", "date"]).reset_index(drop=True)
        self.lane_meta = (
            self.history.drop_duplicates("lane_id").set_index("lane_id")["distance_tier"].to_dict()
        )
        self.last_date = self.history["date"].max()

    @property
    def lanes(self) -> list[str]:
        return sorted(self.lane_meta.keys())

    def forecast(
        self, lane_id: str, horizon_days: int, weather_risk_override: float | None
    ) -> list[dict]:
        if lane_id not in self.lane_meta:
            raise ValueError(f"Unknown lane_id: {lane_id!r}. Known lanes: {self.lanes}")

        lane_history = self.history[self.history["lane_id"] == lane_id]
        delay_series: dict[pd.Timestamp, float] = (
            lane_history.set_index("date")["delay_rate"].to_dict()
        )

        results = []
        for step in range(1, horizon_days + 1):
            forecast_date = self.last_date + timedelta(days=step)
            row = {
                "lane_id": lane_id,
                "distance_tier": self.lane_meta[lane_id],
                "day_of_week": float(forecast_date.weekday()),
                "is_weekend": float(forecast_date.weekday() >= 5),
                "is_holiday": float(_is_holiday(forecast_date.date())),
                "is_peak_season": float(_is_peak_season(forecast_date.date())),
                "weather_risk_score": (
                    weather_risk_override if weather_risk_override is not None else DEFAULT_WEATHER_RISK
                ),
                "month": float(forecast_date.month),
            }
            for lag in LAG_DAYS:
                lag_date = forecast_date - timedelta(days=lag)
                row[f"delay_rate_lag_{lag}"] = delay_series.get(
                    lag_date, next(reversed(delay_series.values()))
                )
            for window in ROLLING_WINDOWS:
                recent = [
                    delay_series[forecast_date - timedelta(days=d)]
                    for d in range(1, window + 1)
                    if (forecast_date - timedelta(days=d)) in delay_series
                ]
                row[f"delay_rate_roll_mean_{window}"] = sum(recent) / len(recent) if recent else 0.0

            X = pd.DataFrame([row])
            preds = predict_xgboost_quantiles(self.models, X, feature_columns())
            p10, p50, p90 = preds.iloc[0][["xgb_p10", "xgb_p50", "xgb_p90"]]

            results.append({
                "date": forecast_date.date().isoformat(),
                "p10": round(float(p10), 4),
                "p50": round(float(p50), 4),
                "p90": round(float(p90), 4),
            })

            # Feed this day's point prediction back in so later steps' lag
            # features can see it.
            delay_series[forecast_date] = float(p50)

        return results
