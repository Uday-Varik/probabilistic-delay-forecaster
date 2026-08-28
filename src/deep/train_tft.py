"""Temporal Fusion Transformer for probabilistic delay-rate forecasting.

Trains on raw delay-rate history via the encoder — no hand-engineered lag
features, unlike the XGBoost baseline. TFT is designed to learn temporal
patterns directly from the sequence; quantile loss gives calibrated
p10/p50/p90 output directly comparable to the baseline's quantiles.

Compute note: this machine reports torch.cuda.is_available() == False (no
GPU). The PRD anticipated exactly this (Week 6 risk: "TFT training may
exceed free-tier GPU session limits ... fall back to N-BEATS") — the
TFT-vs-N-BEATS choice here is made based on measured CPU runtime, recorded
in case_study.md, not assumed in advance.

Run:
    python -m src.deep.train_tft
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Known noisy/harmless: pytorch-forecasting passes plain arrays through a
# fitted sklearn StandardScaler on every batch, which warns about missing
# feature names each time.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "delay_timeseries.csv"

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 14
QUANTILES = [0.1, 0.5, 0.9]

KNOWN_REALS = [
    "time_idx", "day_of_week", "is_weekend", "is_holiday",
    "is_peak_season", "weather_risk_score", "month",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["lane_id", "date"]).reset_index(drop=True)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["month"] = df["date"].dt.month
    for col in ["day_of_week", "is_weekend", "is_holiday", "is_peak_season", "month"]:
        df[col] = df[col].astype(float)
    return df


def build_datasets(df: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    training_cutoff = df["time_idx"].max() - MAX_PREDICTION_LENGTH

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="delay_rate",
        group_ids=["lane_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["distance_tier"],
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=["delay_rate"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    return training, validation


def train(max_epochs: int, batch_size: int, hidden_size: int
          ) -> tuple[TemporalFusionTransformer, TimeSeriesDataSet, TimeSeriesDataSet]:
    df = load_data()
    training, validation = build_datasets(df)

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=max(hidden_size // 2, 4),
        loss=QuantileLoss(QUANTILES),
        learning_rate=0.03,
    )
    n_params = sum(p.numel() for p in tft.parameters())
    print(f"TFT parameter count: {n_params:,}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
    )

    start = time.time()
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    elapsed = time.time() - start
    epochs_run = trainer.current_epoch + 1
    print(f"Training took {elapsed:.1f}s for {epochs_run} epoch(s) "
          f"({elapsed / epochs_run:.1f}s/epoch)")

    return tft, training, validation


def evaluate(tft: TemporalFusionTransformer, validation: TimeSeriesDataSet,
             raw_df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Predicts quantiles on the validation window (last MAX_PREDICTION_LENGTH
    days per lane — a smaller, differently-shaped window than the baseline's
    90-day rolling holdout; see case_study.md for why they aren't directly
    apples-to-apples yet) and merges with actuals for metric computation."""
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    result = tft.predict(val_dataloader, mode="quantiles", return_index=True)

    quantile_idx = {q: i for i, q in enumerate(QUANTILES)}
    preds = result.output
    index_df = result.index

    rows = []
    for sample_i in range(preds.shape[0]):
        lane_id = index_df.iloc[sample_i]["lane_id"]
        start_time_idx = int(index_df.iloc[sample_i]["time_idx"])
        for step in range(preds.shape[1]):
            rows.append({
                "lane_id": lane_id,
                "time_idx": start_time_idx + step,
                "tft_p10": preds[sample_i, step, quantile_idx[0.1]].item(),
                "tft_p50": preds[sample_i, step, quantile_idx[0.5]].item(),
                "tft_p90": preds[sample_i, step, quantile_idx[0.9]].item(),
            })
    pred_df = pd.DataFrame(rows)

    actuals = raw_df[["lane_id", "time_idx", "delay_rate"]]
    return pred_df.merge(actuals, on=["lane_id", "time_idx"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    args = parser.parse_args()

    tft, training, validation = train(args.max_epochs, args.batch_size, args.hidden_size)
    raw_df = load_data()
    eval_df = evaluate(tft, validation, raw_df, args.batch_size)

    from src.baseline.evaluate import interval_coverage, mae, mape

    y_true = eval_df["delay_rate"]
    print(f"\nTFT evaluation ({eval_df['lane_id'].nunique()} lanes x "
          f"{MAX_PREDICTION_LENGTH}-day window, nominal 80% interval):")
    print(f"  MAE:      {mae(y_true, eval_df['tft_p50']):.4f}")
    print(f"  MAPE:     {mape(y_true, eval_df['tft_p50']):.2%}")
    print(f"  coverage: {interval_coverage(y_true, eval_df['tft_p10'], eval_df['tft_p90']):.2%}")

    out_path = ROOT / "data" / "raw" / "tft_predictions.csv"
    eval_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
