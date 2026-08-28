"""Feature-distribution drift monitor.

Uses the Population Stability Index (PSI) per numeric feature, comparing a
new batch against the training distribution — a standard technique for
flagging when a model is operating outside the distribution it was trained
on. Thresholds follow the common industry rule of thumb:
  PSI < 0.10            no significant shift
  0.10 <= PSI < 0.25     moderate shift, worth watching
  PSI >= 0.25            significant shift, retraining likely warranted
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_BINS = 10
PSI_WATCH_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

MONITORED_FEATURES = ["weather_risk_score", "total_shipments", "delay_rate"]


def _psi_for_feature(reference: pd.Series, current: pd.Series, bins: int = PSI_BINS) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    bin_edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _status(psi: float) -> str:
    if psi >= PSI_ALERT_THRESHOLD:
        return "ALERT"
    if psi >= PSI_WATCH_THRESHOLD:
        return "WATCH"
    return "OK"


def compute_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame,
    features: list[str] = MONITORED_FEATURES,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        psi = _psi_for_feature(reference_df[feature], current_df[feature])
        rows.append({"feature": feature, "psi": round(psi, 4), "status": _status(psi)})
    return pd.DataFrame(rows)
