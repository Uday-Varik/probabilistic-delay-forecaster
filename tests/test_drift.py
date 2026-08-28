import numpy as np
import pandas as pd

from src.monitoring.drift import compute_drift_report

RNG = np.random.default_rng(7)


def _reference_df(n: int = 2000) -> pd.DataFrame:
    return pd.DataFrame({
        "weather_risk_score": np.clip(RNG.normal(0.2, 0.12, n), 0, 1),
        "total_shipments": RNG.normal(1000, 150, n),
        "delay_rate": np.clip(RNG.normal(0.05, 0.015, n), 0, 1),
    })


def test_identical_distribution_reports_ok():
    reference = _reference_df()
    current = _reference_df()  # same generating distribution, different draw

    report = compute_drift_report(reference, current)

    assert (report["status"] == "OK").all()
    assert (report["psi"] < 0.10).all()


def test_shifted_distribution_is_flagged():
    reference = _reference_df()
    current = _reference_df()
    # Inject a large, sustained shift — like a new lane class or a sensor
    # recalibration — into one feature only.
    current["weather_risk_score"] = np.clip(current["weather_risk_score"] + 0.5, 0, 1)

    report = compute_drift_report(reference, current)

    shifted = report.set_index("feature").loc["weather_risk_score"]
    assert shifted["status"] == "ALERT"
    assert shifted["psi"] >= 0.25

    # Untouched features should still look fine.
    unshifted = report.set_index("feature").loc["total_shipments"]
    assert unshifted["status"] == "OK"
