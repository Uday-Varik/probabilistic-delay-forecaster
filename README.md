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
- `src/baseline/` — XGBoost + Prophet models
- `src/deep/` — TFT/N-BEATS model, training loop, quantile loss
- `src/monitoring/` — drift detector
- `api/` — FastAPI app (`/forecast`, `/health`, `/metrics`)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate on cmd/PowerShell
pip install -r requirements.txt
```

GPU-heavy training steps (TFT training) run on Colab/Kaggle notebooks, not locally — see
`notebooks/` once added.

## Roadmap

- **Week 5:** synthetic shipment-delay dataset generated; XGBoost/Prophet baseline re-established
- **Week 6:** TFT/N-BEATS model built and trained (Colab/Kaggle GPU), quantile outputs working
- **Week 7:** baseline-vs-DL comparison, calibration analysis, FastAPI serving + drift monitor
- **Week 8:** Docker, deploy, tests, case-study writeup, demo polish

## Status

🚧 Not started — scaffold only. Begins after `lending-compliance-rag` (Project A) is complete.
