"""Feature engineering for the XGBoost baseline.

Lag and rolling-mean features are computed per lane (grouped), and always
built from data shifted by at least 1 day so the current day's own delay
rate never leaks into its own features.
"""

from __future__ import annotations

import pandas as pd

LAG_DAYS = [1, 7, 14]
ROLLING_WINDOWS = [7, 14]

CATEGORICAL_FEATURES = ["lane_id", "distance_tier"]
NUMERIC_FEATURES = [
    "day_of_week", "is_weekend", "is_holiday", "is_peak_season",
    "weather_risk_score", "month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["lane_id", "date"]).reset_index(drop=True)
    df["month"] = df["date"].dt.month

    for lag in LAG_DAYS:
        df[f"delay_rate_lag_{lag}"] = df.groupby("lane_id")["delay_rate"].shift(lag)

    df["_delay_rate_shift1"] = df.groupby("lane_id")["delay_rate"].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"delay_rate_roll_mean_{window}"] = df.groupby("lane_id")["_delay_rate_shift1"].transform(
            lambda s: s.rolling(window).mean()
        )
    df = df.drop(columns=["_delay_rate_shift1"])

    return df


def feature_columns() -> list[str]:
    lag_cols = [f"delay_rate_lag_{lag}" for lag in LAG_DAYS]
    roll_cols = [f"delay_rate_roll_mean_{w}" for w in ROLLING_WINDOWS]
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES + lag_cols + roll_cols
