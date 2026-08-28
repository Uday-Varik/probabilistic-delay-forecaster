# Case Study: Probabilistic Deep-Learning Delivery-Risk Forecasting

> Filled in progressively as the project develops. Structure mirrors what hiring managers
> actually ask about, so this doubles as interview prep.

## 1. Business problem
_TODO_

## 2. Why this project
_TODO_

## 3. Technical approach & key decisions
_TODO — document TFT vs. N-BEATS choice and why, based on actual Week 6 compute reality_

## 4. Research inspiration & adaptation
_TODO — Temporal Fusion Transformer / N-BEATS papers; document what was adapted (bursty,
imbalanced delay-event distribution vs. smooth retail/electricity benchmarks)_

## 5. What makes this not a tutorial clone
_TODO_

## 6. Results

**Baseline comparison** (`src/baseline/train.py`, 90-day time-based holdout, no shuffling):

| model | MAE | MAPE | p10-p90 coverage (nominal 80%) |
|---|---|---|---|
| XGBoost (quantile) | 0.0072 | 14.11% | 78.78% |
| Prophet (per-lane) | 0.0130 | 27.29% | 89.89% |

XGBoost wins clearly on point-forecast accuracy (roughly half Prophet's error on both MAE and
MAPE) and is well-calibrated (78.78% actual vs. 80% nominal coverage — close to ideal). Prophet
is over-conservative — 89.89% actual coverage means its intervals are wider than they need to
be, which combined with the higher error suggests the interval width is compensating for
inaccuracy rather than reflecting genuine uncertainty.

**Why XGBoost likely wins here:** it has direct access to lag (1/7/14-day) and rolling-mean
delay-rate features that Prophet's additive trend+seasonality model doesn't use in the same
way, and it implicitly pools statistical strength across all 10 lanes via the categorical
`lane_id` feature — each Prophet model, by contrast, is fit independently per lane on a smaller
slice of data. This is a real, measured result, not an assumption going in; it also sets the
bar the deep-learning model needs to clear to justify its added complexity per this project's
own non-goal (G2: not claiming DL wins by default).

**TFT/N-BEATS deep-learning results:** not yet run — Week 2 of this project.

## 7. Skills demonstrated
_TODO_

## 8. Data strategy

Deterministically generated (`data/scripts/domain.py` + `generate_shipments.py`, seeded via
`numpy.random.default_rng`) at daily lane-level granularity — each lane-day's shipment volume
and delayed count are sampled from normal/binomial distributions parameterized by seasonality,
holidays, and shock state, which is statistically equivalent to shipment-level simulation
without materializing millions of individual rows.

- **Scale:** 10,950 rows — 10 synthetic lanes (a Memphis hub-and-spoke structure, for narrative
  plausibility; lane names/volumes/rates are invented, not real FedEx data) × 1,095 days
  (2023-08-25 to 2026-08-23).
- **Realized delay rate:** 5.42% overall, within the ~5-10% target.
- **Seasonality:** peak shipping season (Nov 20 - Dec 26) carries a 1.6x volume and 1.7x delay
  multiplier, verified in the generated data (5.1% → 8.9% delay rate). Weekends carry a 0.55x
  volume and 1.3x delay multiplier (verified: 5.1% → 6.4%).
- **Shock events:** 15-25 randomly-placed 1-3 day "weather shock" windows per lane (427
  lane-days total), each with a 2.5-4.5x delay multiplier (verified: 5.0% → 17.5%).
- **Imperfect leading indicator:** `weather_risk_score` is elevated on shock days (mean 0.71)
  vs. non-shock days (mean 0.16) but with real noise on both sides — deliberately not a perfect
  predictor, so the deep-learning model has a genuinely informative but imperfect signal to
  work with rather than a solved problem.
- **Data quality issue:** ~2.5% of shipments per lane-day have no scan event (missing data),
  tracked via `total_shipments` vs. `scanned_shipments`, so delay rate is computed only over
  scanned shipments — same "trust but verify the denominator" issue real operational data has.
- **Label leakage guard:** `_shock_active_ground_truth` is included in the raw file for
  analysis only and is explicitly excluded from model features — the model only ever sees the
  imperfect `weather_risk_score`, never the ground-truth shock flag.

## 9. Deployment
_TODO — live demo link once deployed_

## 10. Interview talking points
_TODO — top 3 likely questions (incl. "why not just use Prophet/XGBoost"), 2-minute pitch,
"how did you adapt the research" answer, calibration plot as evidence_
