"""FastAPI service for delivery-risk forecasting.

Serves the XGBoost quantile model — it beat the TFT deep-learning model on
both accuracy and calibration (see case_study.md), so it's the model that
actually serves live requests; the TFT comparison lives in the offline
evaluation, not the API. Requires `models/` to exist (run
`python -m src.baseline.train` first).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving import ForecastService

app = FastAPI(title="Delivery-Risk Forecasting")
service = ForecastService()


class ForecastRequest(BaseModel):
    lane_id: str
    horizon_days: int = Field(default=7, ge=1, le=30)
    weather_risk_override: float | None = Field(default=None, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/lanes")
def lanes() -> dict:
    return {"lanes": service.lanes}


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict:
    try:
        predictions = service.forecast(
            request.lane_id, request.horizon_days, request.weather_risk_override
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"lane_id": request.lane_id, "forecast": predictions}
