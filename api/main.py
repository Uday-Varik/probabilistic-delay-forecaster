"""FastAPI service for delivery-risk forecasting.

Week 5 stub: health check only. /forecast is added once the baseline and
TFT models (src/baseline/, src/deep/) exist.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Delivery-Risk Forecasting")


class ForecastRequest(BaseModel):
    shipment_id: str
    horizon_days: int = 7


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast")
def forecast(_: ForecastRequest) -> dict:
    return {"detail": "not implemented yet - models pending"}
