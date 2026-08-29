# Probabilistic Deep-Learning Delivery-Risk Forecasting

PyTorch deep-learning forecaster predicting delivery-delay risk with calibrated uncertainty
bounds (p10/p50/p90), benchmarked against a classical XGBoost/Prophet baseline on identical
data, served behind a monitored API.

## Problem statement

Delivery-delay forecasts used for hub capacity planning are usually single-point estimates
that hide how confident the model actually is. This project builds a probabilistic
deep-learning forecaster (Temporal Fusion Transformer) that predicts delay risk with
calibrated uncertainty (p10/p50/p90), benchmarks it against a classical XGBoost/Prophet
baseline trained on identical data, and serves both behind a monitored API — quantifying
whether, and where, the added model complexity is actually worth it.

## Architecture

```
Synthetic shipment-level time series (seasonality, holidays, weather shocks, ~5-10% delay rate)
  │
  ├─► XGBoost / Prophet baseline ──────────────┐
  │                                             │
  └─► Temporal Fusion Transformer (quantile loss) ─┤
                                                  ▼
                                    Calibration comparison (p10/p50/p90
                                    coverage vs. baseline intervals)
                                                  │
                                                  ▼
                                 FastAPI serving (point + interval forecast)
                                                  │
                                                  ▼
                                     Drift monitor (feature distribution shift)
```

## Tech stack

- **Deep learning:** PyTorch + `pytorch-forecasting` (Temporal Fusion Transformer; N-BEATS as
  a lighter fallback if TFT proves too heavy for available compute)
- **Baseline:** XGBoost, Prophet (same libraries already used in production at FedEx)
- **Training compute:** Colab / Kaggle free-tier GPU
- **Serving:** FastAPI, returning point + quantile predictions
- **Monitoring:** custom drift detector (feature distribution shift)
- **Deployment:** Docker → Hugging Face Spaces or Render free tier

## Repo structure

See the parent [`../README.md`](../README.md) for the shared convention. Notable specifics:

- `data/scripts/` — synthetic shipment time-series generator (the portfolio artifact; raw/
  processed outputs are gitignored)
- `src/baseline/` — XGBoost (served live) + Prophet (evaluation-only) models
- `src/deep/` — TFT model, training loop, quantile loss (offline evaluation, not served live —
  see case_study.md for why)
- `src/monitoring/` — drift detector
- `src/serving.py` — recursive multi-day quantile forecasting used by the API
- `api/` — FastAPI app (`/forecast`, `/lanes`, `/health`)
- `models/` — saved production XGBoost models (gitignored; regenerate with
  `python -m src.baseline.train_production`)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate on cmd/PowerShell
pip install -r requirements-dev.txt   # full local/training/eval deps; requirements.txt alone
                                       # is the lean runtime set actually installed in the container

python data/scripts/generate_shipments.py
python -m src.baseline.train_production   # trains + saves the model the API serves
uvicorn api.main:app --reload
```

### Docker

The image bakes in data generation + production-model training at build time (both
deterministic/seeded and lean-dependency — no Prophet/torch needed), so the container is
self-contained. Build context must be the repo root:

```bash
docker build -f docker/Dockerfile -t probabilistic-delay-forecaster .
docker run -p 8000:8000 probabilistic-delay-forecaster
```

GPU-heavy training steps (TFT) ran locally on CPU in this project (see case_study.md) rather
than needing Colab/Kaggle — the model turned out small enough not to require it.

## Roadmap

- **Week 5:** synthetic shipment-delay dataset generated; XGBoost/Prophet baseline re-established
- **Week 6:** TFT/N-BEATS model built and trained (Colab/Kaggle GPU), quantile outputs working
- **Week 7:** baseline-vs-DL comparison, calibration analysis, FastAPI serving + drift monitor
- **Week 8:** Docker, deploy, tests, case-study writeup, demo polish

## Status

🚧 Week 3 — baseline (XGBoost, Prophet) and TFT deep-learning model both trained and evaluated.
Headline result — **deep learning did not win**: XGBoost beat TFT on both accuracy and
calibration, so XGBoost is what actually serves the API. Full honest writeup in case_study.md.

| model | eval window | MAE | MAPE | coverage (nominal 80%) |
|---|---|---|---|---|
| XGBoost (quantile) | 90-day rolling | 0.0072 | 14.11% | 78.78% |
| Prophet (per-lane) | 90-day rolling | 0.0130 | 27.29% | 89.89% |
| TFT | 14-day window | 0.0084 | 15.19% | 71.43% |

PSI-based drift monitor (`src/monitoring/drift.py`) built and tested. `/forecast` API built and
verified end-to-end — both locally and from a real running Docker container (correct lane list,
sensible weekday/weekend forecast shape, and a `weather_risk_override` "what-if" parameter that
demonstrably moves the forecast: 0.054 → 0.194 predicted delay rate under a simulated storm
warning on the same lane/day). Next: live deployment (HF Spaces/Render), Streamlit dashboard.

Regenerate/retrain locally with:
```bash
python data/scripts/generate_shipments.py
python -m src.baseline.train
python -m src.deep.train_tft --max-epochs 20
```
