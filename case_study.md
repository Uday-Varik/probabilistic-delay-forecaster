# Case Study: Probabilistic Deep-Learning Delivery-Risk Forecasting

> Filled in progressively as the project develops. Structure mirrors what hiring managers
> actually ask about, so this doubles as interview prep.

## 1. Business problem

Hub capacity planners get delivery-delay forecasts as single-point estimates that give no sense
of how much to trust them — a forecast of "5% delayed" could mean a stable, well-understood
pattern or a coin flip, and a planner has no way to tell which without an uncertainty range.
Most portfolio-grade forecasting projects skip uncertainty quantification entirely and treat
"I built a model" as the whole story. This project answers: **"Show me something you built with
deep learning"** and **"Walk me through your most technically complex project."**

## 2. Why this project

This directly extends real work — package demand forecasting with XGBoost/LightGBM/Prophet at
FedEx — into the two places that work was thinnest on a resume: an actual PyTorch deep-learning
project (previously a resume-only claim, never demonstrated) and calibrated uncertainty
(previously absent — the FedEx work produced point forecasts). Framing it as a rigorous
baseline-vs-DL comparison rather than "I used a fancy model" gives it a real before/after growth
story, and specifically closes the credibility gap of PyTorch/TensorFlow being listed on a
resume with zero backing evidence.

## 3. Technical approach & key decisions

- **TFT over N-BEATS**, decided empirically rather than assumed in advance. The PRD flagged
  this as a real risk (no GPU access guaranteed) with N-BEATS as the documented fallback. A
  1-epoch smoke test on this CPU-only machine measured ~87s/epoch for the full
  `pytorch-forecasting` `TemporalFusionTransformer` — well within a usable local session budget
  — so TFT was kept rather than falling back, and the fallback plan itself is recorded here
  rather than silently discarded.
- **Quantile loss via XGBoost's native `reg:quantileerror`** (not three separate point models
  bolted together) for the baseline, so its p10/p50/p90 output is a real calibrated quantile
  forecast directly comparable to TFT's quantile heads and Prophet's native interval — an
  apples-to-apples uncertainty comparison was a design goal from the start, not an afterthought.
- **TFT consumes raw delay-rate history through its encoder — no hand-engineered lag features**,
  unlike the XGBoost baseline. This is a deliberate modeling-philosophy difference worth stating
  plainly: XGBoost needs lag/rolling features engineered by hand, TFT is designed to learn
  temporal patterns directly from the sequence. Giving both models genuinely different degrees
  of "hand-holding" respects how each is actually meant to be used.
- **Time-based (not random) train/test splits everywhere** — the test period is strictly after
  the training period for both the baseline and TFT, matching how the model would actually be
  used, and avoiding the classic time-series leakage bug of shuffling before splitting.
- **The result that actually came out of this:** XGBoost beat TFT on both accuracy and
  calibration (see §6). That wasn't the anticipated outcome going in, but it's the real one, and
  changing the write-up to match the result rather than adjusting the result to match a
  narrative was a deliberate choice.

## 4. Research inspiration & adaptation

Inspired by the Temporal Fusion Transformer paper (Lim et al.) and, as the documented fallback,
N-BEATS. Both are typically benchmarked on smooth, high-series-count retail/electricity/traffic
datasets. The adaptation here: a much smaller series count (10 lanes) with a bursty, imbalanced
target (a ~5.4% baseline delay rate punctuated by 2.5-4.5x shock multipliers) rather than a
smooth continuous signal, and an explicit test of whether the architecture's usual advantage
(learning cross-series and temporal patterns from raw sequences at scale) holds up when "at
scale" isn't actually true. It didn't, on this dataset — which is itself the useful finding: TFT
is not a drop-in upgrade that wins regardless of data volume, and this is direct evidence of
where the crossover point isn't yet met, not just a claim about it.

## 5. What makes this not a tutorial clone

- The headline result — XGBoost beats a Temporal Fusion Transformer on both accuracy and
  calibration — is reported as-is, including the part that undercuts the "look, I did deep
  learning" framing a portfolio project is tempted to lead with. A tutorial writeup shows the
  fancy model winning; this shows a measured result and reasons about *why* it came out that way
  (series count, engineered vs. learned features) instead of stopping at the number.
- The uncertainty comparison is apples-to-apples by construction (XGBoost's native quantile
  objective, not an ad-hoc interval bolted onto a point forecast), so the calibration numbers
  (78.8% vs. 71.4% vs. 89.9% against an 80% nominal target) are a real, comparable measurement
  across three genuinely different modeling approaches, not three different metrics dressed up
  to look comparable.
- The synthetic data has a specific, checkable design: seasonality/shock/weekend multipliers
  were verified against the actual generated data (not just assumed from the generator's
  parameters), and an imperfect (not solved, not impossible) leading indicator was built in
  deliberately so the uncertainty-quantification comparison has a genuine signal to work with.

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

**TFT deep-learning results** (`src/deep/train_tft.py`, quantile loss, CPU-only — see §3 for the
TFT-vs-N-BEATS decision):

| model | eval window | MAE | MAPE | coverage (nominal 80%) |
|---|---|---|---|---|
| XGBoost (quantile) | 90-day rolling | 0.0072 | 14.11% | 78.78% |
| Prophet (per-lane) | 90-day rolling | 0.0130 | 27.29% | 89.89% |
| **TFT** | 14-day, single window | **0.0084** | **15.19%** | **71.43%** |

**The honest headline result: deep learning did not win.** TFT is worse than XGBoost on both
accuracy (MAE 0.0084 vs 0.0072) and calibration (71.4% actual coverage vs. the 80% nominal
target — its intervals are too narrow, the opposite failure mode from Prophet's too-wide
intervals). Training converged early (early stopping at epoch 7 of a 20-epoch budget, ~75s/epoch
on CPU, ~9 minutes total) — plausibly because with only 10 series and ~991 days of history each,
this dataset is genuinely small for a transformer-style architecture that normally benefits from
many more related series to learn cross-series patterns from. XGBoost's engineered lag/rolling
features plus its ability to pool across all 10 lanes via a single model likely give it an
information advantage a "learn it from raw sequences" architecture doesn't get at this scale.

**Caveat on the comparison:** the TFT evaluation window (14 days x 10 lanes = 140 predictions,
pytorch-forecasting's standard single held-out window) is smaller and structurally different
from the baseline's 90-day rolling holdout (900 predictions) — they are not perfectly
apples-to-apples yet. Aligning both to an identical window is a documented next step, not
hidden; the direction of the result (XGBoost ahead on both axes) is unlikely to flip from that
alignment alone given the size of the gap, but the exact numbers could shift.

This is exactly the kind of result the project's own non-goal anticipated (G2: not claiming DL
wins by default) — and it's a more interesting, defensible portfolio finding than a predetermined
"deep learning wins" narrative would have been.

## 7. Skills demonstrated

**New:** end-to-end PyTorch model development (`pytorch-forecasting`'s `TemporalFusionTransformer`
via `TimeSeriesDataSet`, PyTorch Lightning training loop), probabilistic/quantile forecasting
with a real calibration metric (interval coverage, not just point accuracy), population-stability-index
drift monitoring built from first principles rather than an imported black box.

**Deepened:** time-series methodology (from tree-based/Prophet to neural, with an honest
head-to-head instead of assuming either wins), synthetic time-series data engineering (verified
seasonality/shock effects, not just generated-and-hoped), production API-serving patterns
carried over from the RAG project (lean vs. dev dependency separation, Docker builds that bake
in reproducible setup steps).

**Confirmed rather than assumed:** that deep learning requires enough data/series to earn its
complexity — a genuinely useful piece of judgment for deciding when *not* to reach for a neural
architecture, which is arguably as valuable an interview signal as knowing how to build one.

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

**Top 3 likely questions:**
1. *"Why not just use XGBoost/Prophet — did the deep-learning complexity pay off?"* — Honestly,
   no, not on this dataset: XGBoost beat the TFT model on both MAE (0.0072 vs 0.0084) and
   calibration (78.8% vs 71.4% actual coverage against an 80% target). The likely reason is
   series count — 10 lanes is small for an architecture that learns cross-series patterns from
   raw sequences, versus XGBoost's engineered lag features and ability to pool across all lanes
   in one model.
2. *"How do you know your uncertainty estimates are calibrated?"* — Interval coverage: what
   fraction of actual values fell inside the model's stated p10-p90 range, checked against the
   nominal 80%. XGBoost came in at 78.8% (close to ideal), Prophet at 89.9% (intervals too wide,
   masking inaccuracy), TFT at 71.4% (intervals too narrow, overconfident).
3. *"What would you do differently, or next?"* — Align the TFT and baseline evaluation windows
   exactly (currently 14-day single-window vs. 90-day rolling) before trusting the exact gap
   size, and try more training epochs / a smaller architecture bias before concluding the
   dataset scale is the whole explanation rather than under-tuning.

**2-minute pitch:** *"At FedEx I built demand forecasts with XGBoost, Prophet, and LightGBM —
all point estimates. I wanted to know two things: can I actually build and train a PyTorch deep
learning model end-to-end, and does it produce calibrated uncertainty that's useful for capacity
planning, not just a number. I built a Temporal Fusion Transformer with quantile loss, benchmarked
head-to-head against an XGBoost quantile-regression baseline and per-lane Prophet models on
identical synthetic shipment-delay data, using a real calibration metric — interval coverage —
not just accuracy. The honest result was that XGBoost won on both axes. I dug into why: with
only 10 lanes of history, there wasn't enough series diversity for the transformer's
raw-sequence learning to out-info XGBoost's hand-engineered lag features and cross-lane pooling.
That's a real, specific finding about when deep learning complexity is and isn't worth it — not
a predetermined 'deep learning wins' conclusion."*

**"How did you adapt the research?"** — *"TFT and N-BEATS are typically benchmarked on
high-series-count retail or electricity datasets. I deliberately tested the architecture at a
much smaller series count — 10 lanes — with a bursty, shock-driven delay signal instead of a
smooth one, specifically to see whether the usual 'more sophisticated model wins' assumption
holds at a realistic small-business data scale. It didn't, which is itself evidence about where
the crossover point actually is, not just a claim about it."*

**A fair criticism, and the honest response:** *"You only trained TFT for 7 epochs before early
stopping — is this actually TFT's ceiling, or just an undertrained model?"* Fair — early
stopping on validation loss is a reasonable stopping rule, but it's not proof the architecture
is capped there; more epochs, a different hidden size, or more series would need to be tried
before concluding the gap is structural rather than a tuning artifact. I'd want to see the
learning curve continue to flatten under a larger training budget before trusting the size of
the gap, even though the direction (XGBoost ahead) is unlikely to flip from tuning alone given
how the loss had already plateaued.
