"""Shared domain model for the synthetic delivery-delay time series.

Lane definitions, base rates, and the holiday/peak-season calendar are all
invented for this project — not derived from real FedEx operational data.
Generation (generate_shipments.py) is deterministic given a fixed seed.

Design note: rather than materializing individual shipment rows, each
lane-day's shipment volume and delay count are sampled from
Poisson/Binomial-style distributions parameterized by seasonality,
holidays, and shock state — statistically equivalent to shipment-level
simulation without the overhead of generating millions of individual rows.
"""

from __future__ import annotations

from datetime import date

START_DATE = date(2023, 8, 25)
END_DATE = date(2026, 8, 23)  # "today" — 3 years of daily history

# lane_id: (base_daily_volume, base_delay_rate, distance_tier)
# A synthetic hub-and-spoke structure loosely modeled on a Memphis-based air
# network for narrative plausibility — lane volumes/rates are invented, not
# real FedEx figures. Base delay rates are picked so the realized overall
# rate lands in the ~5-10% target range once seasonality/shocks are applied.
LANES: dict[str, tuple[int, float, str]] = {
    "MEM-ATL": (1800, 0.035, "short"),
    "MEM-DFW": (1500, 0.038, "medium"),
    "MEM-ORD": (1600, 0.042, "medium"),
    "MEM-DEN": (900, 0.045, "medium"),
    "MEM-LAX": (1200, 0.050, "long"),
    "MEM-SEA": (700, 0.055, "long"),
    "MEM-JFK": (1300, 0.044, "medium"),
    "MEM-MIA": (1000, 0.040, "medium"),
    "MEM-PHX": (650, 0.037, "medium"),
    "MEM-BOS": (850, 0.043, "medium"),
}

# Fixed-date federal holidays, approximated by fixed MM-DD (not exact
# floating-holiday math) — good enough for synthetic seasonality.
FIXED_HOLIDAYS_MMDD: list[tuple[int, int]] = [
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (11, 11),  # Veterans Day
    (12, 25),  # Christmas Day
]

# The Thanksgiving-to-Christmas peak season is the dominant volume+delay
# driver in real parcel networks, modeled explicitly as a window rather than
# folded into a generic "holiday" flag.
PEAK_SEASON_START_MMDD = (11, 20)
PEAK_SEASON_END_MMDD = (12, 26)
