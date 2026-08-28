"""Generate the synthetic delivery-delay daily time series.

Produces one row per (lane, date): shipment volume, scanned/missing-scan
split, delayed count, and delay rate, driven by day-of-week effects,
holidays, a peak-shipping season, and randomly-placed multi-day "weather
shock" events per lane. A `weather_risk_score` feature is included as an
imperfect, forecastable leading indicator of shocks (elevated on shock days
but with both false positives and false negatives) — informative without
being a perfect predictor, which is what makes the uncertainty-quantification
comparison in later weeks meaningful rather than trivial.

Run:
    python data/scripts/generate_shipments.py
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from domain import (
    END_DATE,
    FIXED_HOLIDAYS_MMDD,
    LANES,
    PEAK_SEASON_END_MMDD,
    PEAK_SEASON_START_MMDD,
    START_DATE,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PATH = ROOT / "data" / "raw" / "delay_timeseries.csv"

MISSING_SCAN_RATE = 0.025
SHOCKS_PER_LANE = (15, 25)
SHOCK_DURATION_DAYS = (1, 3)
SHOCK_DELAY_MULTIPLIER = (2.5, 4.5)
WEEKEND_VOLUME_MULTIPLIER = 0.55
WEEKEND_DELAY_MULTIPLIER = 1.3
HOLIDAY_VOLUME_MULTIPLIER = 0.25
PEAK_SEASON_VOLUME_MULTIPLIER = 1.6
PEAK_SEASON_DELAY_MULTIPLIER = 1.7


def _all_dates() -> list[date]:
    n_days = (END_DATE - START_DATE).days + 1
    return [START_DATE + timedelta(days=i) for i in range(n_days)]


def _is_holiday(d: date) -> bool:
    return (d.month, d.day) in FIXED_HOLIDAYS_MMDD


def _is_peak_season(d: date) -> bool:
    start = date(d.year, *PEAK_SEASON_START_MMDD)
    end = date(d.year, *PEAK_SEASON_END_MMDD)
    return start <= d <= end


def _sample_shock_days(rng: np.random.Generator, all_days: list[date]) -> set[date]:
    n_shocks = rng.integers(SHOCKS_PER_LANE[0], SHOCKS_PER_LANE[1] + 1)
    shock_days: set[date] = set()
    start_indices = rng.integers(0, len(all_days), size=n_shocks)
    for idx in start_indices:
        duration = int(rng.integers(SHOCK_DURATION_DAYS[0], SHOCK_DURATION_DAYS[1] + 1))
        for offset in range(duration):
            day_idx = idx + offset
            if day_idx < len(all_days):
                shock_days.add(all_days[day_idx])
    return shock_days


def generate(seed: int, out_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_days = _all_dates()
    n_days = len(all_days)

    rows = []
    for lane_id, (base_volume, base_delay_rate, distance_tier) in LANES.items():
        shock_days = _sample_shock_days(rng, all_days)

        for d in all_days:
            is_weekend = d.weekday() >= 5
            is_holiday = _is_holiday(d)
            is_peak = _is_peak_season(d)
            is_shock = d in shock_days

            volume_mult = 1.0
            delay_mult = 1.0
            if is_weekend:
                volume_mult *= WEEKEND_VOLUME_MULTIPLIER
                delay_mult *= WEEKEND_DELAY_MULTIPLIER
            if is_holiday:
                volume_mult *= HOLIDAY_VOLUME_MULTIPLIER
            if is_peak:
                volume_mult *= PEAK_SEASON_VOLUME_MULTIPLIER
                delay_mult *= PEAK_SEASON_DELAY_MULTIPLIER
            if is_shock:
                delay_mult *= rng.uniform(*SHOCK_DELAY_MULTIPLIER)

            expected_volume = max(1.0, base_volume * volume_mult * rng.uniform(0.9, 1.1))
            total_shipments = max(1, int(round(rng.normal(expected_volume, expected_volume * 0.08))))

            if is_shock:
                weather_risk_score = float(np.clip(rng.normal(0.72, 0.15), 0.0, 1.0))
            else:
                weather_risk_score = float(np.clip(rng.normal(0.15, 0.12), 0.0, 1.0))

            effective_delay_rate = min(0.95, base_delay_rate * delay_mult)

            n_missing_scan = int(round(total_shipments * MISSING_SCAN_RATE * rng.uniform(0.5, 1.5)))
            n_missing_scan = min(n_missing_scan, total_shipments)
            scanned_shipments = total_shipments - n_missing_scan

            n_delayed = int(rng.binomial(scanned_shipments, effective_delay_rate)) if scanned_shipments else 0

            rows.append({
                "lane_id": lane_id,
                "distance_tier": distance_tier,
                "date": d.isoformat(),
                "day_of_week": d.weekday(),
                "is_weekend": int(is_weekend),
                "is_holiday": int(is_holiday),
                "is_peak_season": int(is_peak),
                "weather_risk_score": round(weather_risk_score, 4),
                "total_shipments": total_shipments,
                "scanned_shipments": scanned_shipments,
                "delayed_shipments": n_delayed,
                "delay_rate": round(n_delayed / scanned_shipments, 4) if scanned_shipments else 0.0,
                # Ground truth only — NOT a valid model input (would leak the label).
                # Kept for analysis of what the model can/can't see.
                "_shock_active_ground_truth": int(is_shock),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    df = generate(args.seed, args.out_path)

    overall_rate = df["delayed_shipments"].sum() / df["scanned_shipments"].sum()
    print(f"Generated {len(df)} lane-day rows across {len(LANES)} lanes "
          f"({START_DATE.isoformat()} to {END_DATE.isoformat()})")
    print(f"Overall delay rate: {overall_rate:.3%}")
    print(f"Shock days flagged: {int(df['_shock_active_ground_truth'].sum())} lane-days")
    print(f"Missing-scan shipments: {(df['total_shipments'] - df['scanned_shipments']).sum()} "
          f"of {df['total_shipments'].sum()} total")
    print(f"Wrote to {args.out_path}")


if __name__ == "__main__":
    main()
