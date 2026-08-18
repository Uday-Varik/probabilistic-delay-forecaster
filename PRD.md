# PRD: Probabilistic Delivery-Risk Forecaster

**Repo:** `probabilistic-delay-forecaster` | **Owner:** Uday Varikuppala | **Status:** Draft | **Last updated:** 2026-08-18

## 1. Summary

A PyTorch deep-learning forecasting system that predicts delivery-delay risk with calibrated
uncertainty bounds (p10/p50/p90), benchmarked head-to-head against a classical XGBoost/Prophet
baseline on identical data, served behind a monitored API.

## 2. Problem statement

Delay forecasts used for hub capacity planning are typically single-point estimates that hide
model confidence. A planner can't tell "this forecast is solid" from "this forecast is a
coin flip" without an uncertainty range — and most portfolio-grade forecasting projects skip
uncertainty quantification entirely, using it as the differentiator here.

## 3. Goals

- G1: Produce delay-risk forecasts with calibrated quantile bounds (p10/p50/p90), not just a
  point estimate.
- G2: Benchmark the deep-learning model against a classical baseline (XGBoost/Prophet) on the
  same data, and report the comparison honestly — including if the baseline wins on some axis.
- G3: Detect input distribution drift and flag when the model is operating outside its training
  distribution.
- G4: Ship as a deployed, queryable forecast API with a comparison dashboard.

## 4. Non-goals

- Not a claim that deep learning strictly beats gradient boosting — the point is measuring
  where/whether it helps, not proving a predetermined conclusion.
- Not using real FedEx or any employer data — synthetic shipment data only.
- Not building a full-scale production forecasting platform — single-model serving, not
  multi-model orchestration.

## 5. Primary persona

**Marcus, hub capacity planner.** Needs a delay-risk forecast per lane/date to decide staffing
and routing, and needs to know how much to trust a given forecast — wide uncertainty bands
should change his decision differently than a tight, confident one.

## 6. Functional requirements

| ID | Requirement |
|---|---|
| FR1 | Generate synthetic shipment-level time series with seasonality, holiday effects, weather-shock events, and a realistic (~5-10%) delay rate. |
| FR2 | Train baseline models (XGBoost, Prophet) producing point forecasts + baseline intervals. |
| FR3 | Train a deep-learning model (Temporal Fusion Transformer, or N-BEATS if compute-constrained) producing quantile forecasts (p10/p50/p90) via quantile loss. |
| FR4 | Compare baseline vs. DL model on both accuracy (MAE/MAPE) and calibration (empirical interval coverage). |
| FR5 | Serve forecasts via API: given a shipment/lane and horizon, return point + quantile predictions. |
| FR6 | Monitor incoming feature distributions and flag drift relative to the training distribution. |
| FR7 | Provide a dashboard visualizing forecast bands vs. actuals and the baseline-vs-DL comparison side by side. |

## 7. Non-functional requirements

- **Cost:** $0 — Colab/Kaggle free-tier GPU for training, free-tier hosting for serving.
- **Compute constraint:** training must complete within a single free-tier GPU session
  (documented time budget; N-BEATS as fallback if TFT doesn't fit).
- **Latency:** forecast serving target under ~1s per request.
- **Deployability:** Dockerized; runs on Hugging Face Spaces or Render free tier.

## 8. Success metrics

- Calibration: empirical coverage of the p10/p90 interval within a reasonable margin of the
  nominal 80% (exact bar set after baseline calibration is measured).
- Accuracy comparison (MAE/MAPE) vs. baseline reported for both, with an honest verdict on
  where each model wins.
- Drift monitor correctly flags an injected distribution-shift test scenario.
- Deployed demo + completed `case_study.md` by end of Week 8.

## 9. Milestones

| Week | Deliverable |
|---|---|
| 5 | Synthetic shipment-delay dataset generated; XGBoost/Prophet baseline re-established |
| 6 | TFT/N-BEATS model trained (Colab/Kaggle GPU), quantile outputs working |
| 7 | Baseline-vs-DL comparison, calibration analysis, FastAPI serving + drift monitor |
| 8 | Docker, deploy, tests, case-study writeup, demo polish |

## 10. Risks & assumptions

- **Risk:** TFT training may exceed free-tier GPU session limits. *Mitigation:* fall back to
  N-BEATS or a subsampled dataset; decide in Week 6 based on actual runtime.
- **Risk:** synthetic data may be too clean, understating real operational noise.
  *Mitigation:* explicitly inject missing scan events and shock events during generation.
- **Assumption:** free-tier GPU availability is sufficient for one full training run per week
  during Weeks 6-7.

## 11. Open questions

- TFT vs. N-BEATS final choice — decided in Week 6 based on compute reality, documented either
  way in `case_study.md`.
- Final hosting choice (HF Spaces vs. Render) — decide in Week 8 based on cold-start latency.
