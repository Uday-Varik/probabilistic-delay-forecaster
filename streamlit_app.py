"""Comparison dashboard: baseline vs. deep-learning forecast quality (FR7),
plus a live "what-if" forecast using the model actually served by the API.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.baseline.evaluate import interval_coverage, mae, mape
from src.serving import ForecastService

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"

st.set_page_config(page_title="Delivery-Risk Forecasting", layout="wide")
st.title("Probabilistic Delivery-Risk Forecasting")
st.caption(
    "Baseline (XGBoost, Prophet) vs. deep learning (TFT) comparison, plus a live forecast "
    "from the model actually served by the API."
)


@st.cache_data
def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "baseline_predictions.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_tft() -> pd.DataFrame | None:
    path = DATA_DIR / "tft_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_service() -> ForecastService:
    return ForecastService()


baseline_df = load_baseline()
tft_df = load_tft()

# --- Section 1: accuracy/calibration comparison, computed live from the
# saved prediction files, not hardcoded ---
st.header("1. Model comparison")

rows = []
y_true = baseline_df["delay_rate"]
rows.append({
    "model": "XGBoost (quantile)",
    "eval window": "90-day rolling",
    "MAE": mae(y_true, baseline_df["xgb_p50"]),
    "MAPE": mape(y_true, baseline_df["xgb_p50"]),
    "coverage (nominal 80%)": interval_coverage(y_true, baseline_df["xgb_p10"], baseline_df["xgb_p90"]),
})
rows.append({
    "model": "Prophet (per-lane)",
    "eval window": "90-day rolling",
    "MAE": mae(y_true, baseline_df["prophet_p50"]),
    "MAPE": mape(y_true, baseline_df["prophet_p50"]),
    "coverage (nominal 80%)": interval_coverage(
        y_true, baseline_df["prophet_p10"], baseline_df["prophet_p90"]
    ),
})
if tft_df is not None:
    tft_true = tft_df["delay_rate"]
    rows.append({
        "model": "TFT (deep learning)",
        "eval window": "14-day window",
        "MAE": mae(tft_true, tft_df["tft_p50"]),
        "MAPE": mape(tft_true, tft_df["tft_p50"]),
        "coverage (nominal 80%)": interval_coverage(tft_true, tft_df["tft_p10"], tft_df["tft_p90"]),
    })

comparison = pd.DataFrame(rows)
st.dataframe(
    comparison.style.format({"MAE": "{:.4f}", "MAPE": "{:.2%}", "coverage (nominal 80%)": "{:.2%}"}),
    hide_index=True,
    width="stretch",
)
st.info(
    "**Honest headline result:** XGBoost wins on both accuracy and calibration. It's the model "
    "actually served by the API below — see case_study.md for the full analysis of why."
)

# --- Section 2: forecast bands vs actuals, by lane ---
st.header("2. Forecast bands vs. actuals (90-day holdout)")

lane_options = sorted(baseline_df["lane_id"].unique())
selected_lane = st.selectbox("Lane", lane_options, key="eval_lane")

lane_df = baseline_df[baseline_df["lane_id"] == selected_lane].sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=lane_df["date"], y=lane_df["delay_rate"], name="Actual", mode="lines",
    line=dict(color="black", width=2),
))
fig.add_trace(go.Scatter(
    x=lane_df["date"], y=lane_df["xgb_p90"], name="XGBoost p90", mode="lines",
    line=dict(width=0), showlegend=False,
))
fig.add_trace(go.Scatter(
    x=lane_df["date"], y=lane_df["xgb_p10"], name="XGBoost p10-p90", mode="lines",
    line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.25)",
))
fig.add_trace(go.Scatter(
    x=lane_df["date"], y=lane_df["xgb_p50"], name="XGBoost p50", mode="lines",
    line=dict(color="rgb(31,119,180)", dash="dash"),
))
fig.add_trace(go.Scatter(
    x=lane_df["date"], y=lane_df["prophet_p50"], name="Prophet p50", mode="lines",
    line=dict(color="rgb(214,39,40)", dash="dot"),
))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Delay rate", height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# --- Section 3: live forecast from the model actually served by the API ---
st.header("3. Live forecast — try it")
st.caption(
    "This calls the same ForecastService the API uses. Push weather risk up to see the "
    "uncertainty band widen and shift — the model was never told 'shock' directly, only this "
    "imperfect leading indicator."
)

service = load_service()
col1, col2, col3 = st.columns(3)
with col1:
    live_lane = st.selectbox("Lane", service.lanes, key="live_lane")
with col2:
    horizon = st.slider("Horizon (days)", 1, 14, 7)
with col3:
    weather = st.slider("Weather risk override", 0.0, 1.0, 0.15, step=0.05)

forecast = service.forecast(live_lane, horizon, weather)
forecast_df = pd.DataFrame(forecast)
forecast_df["date"] = pd.to_datetime(forecast_df["date"])

live_fig = go.Figure()
live_fig.add_trace(go.Scatter(
    x=forecast_df["date"], y=forecast_df["p90"], name="p90", mode="lines",
    line=dict(width=0), showlegend=False,
))
live_fig.add_trace(go.Scatter(
    x=forecast_df["date"], y=forecast_df["p10"], name="p10-p90", mode="lines",
    line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.25)",
))
live_fig.add_trace(go.Scatter(
    x=forecast_df["date"], y=forecast_df["p50"], name="p50 (point forecast)", mode="lines+markers",
    line=dict(color="rgb(31,119,180)"),
))
live_fig.update_layout(xaxis_title="Date", yaxis_title="Predicted delay rate", height=400)
st.plotly_chart(live_fig, use_container_width=True)
st.dataframe(forecast_df, hide_index=True, width="stretch")
