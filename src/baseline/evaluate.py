"""Accuracy + calibration metrics for comparing baseline forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: pd.Series, y_pred: pd.Series, epsilon: float = 1e-4) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))))


def interval_coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    return float(((y_true >= lower) & (y_true <= upper)).mean())
