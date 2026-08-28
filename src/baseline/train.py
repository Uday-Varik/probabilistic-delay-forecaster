"""Train and evaluate the XGBoost and Prophet baselines on the synthetic
delay-rate time series, with a time-based train/test split (no shuffling —
the test period is strictly after the train period, as it would be in
production).

Run:
    python -m src.baseline.train
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baseline.evaluate import interval_coverage, mae, mape
from src.baseline.features import build_features, feature_columns
from src.baseline.prophet_model import predict_prophet, train_prophet_per_lane
from src.baseline.xgboost_model import predict_xgboost_quantiles, train_xgboost_quantiles

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "delay_timeseries.csv"
TEST_DAYS = 90


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)
    df = df.dropna(subset=feature_columns()).reset_index(drop=True)

    cutoff = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train_df = df[df["date"] <= cutoff].reset_index(drop=True)
    test_df = df[df["date"] > cutoff].reset_index(drop=True)
    return train_df, test_df


def main() -> None:
    train_df, test_df = load_and_split()
    print(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} to "
          f"{train_df['date'].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min().date()} to "
          f"{test_df['date'].max().date()})")

    feat_cols = feature_columns()

    print("\nTraining XGBoost quantile models...")
    xgb_models = train_xgboost_quantiles(train_df, feat_cols)
    xgb_preds = predict_xgboost_quantiles(xgb_models, test_df, feat_cols)

    print("Training Prophet per-lane models (10 lanes)...")
    prophet_models = train_prophet_per_lane(train_df)
    prophet_preds = predict_prophet(prophet_models, test_df)

    # Explicit key-based merge rather than positional concat — Prophet's
    # per-lane groupby+concat does not preserve test_df's original row order.
    eval_df = pd.concat(
        [test_df[["date", "lane_id", "delay_rate"]], xgb_preds], axis=1
    )
    eval_df = eval_df.merge(prophet_preds, on=["date", "lane_id"], how="left")

    y_true = eval_df["delay_rate"]
    results = {
        "XGBoost": {
            "MAE": mae(y_true, eval_df["xgb_p50"]),
            "MAPE": mape(y_true, eval_df["xgb_p50"]),
            "p10-p90 coverage": interval_coverage(y_true, eval_df["xgb_p10"], eval_df["xgb_p90"]),
        },
        "Prophet": {
            "MAE": mae(y_true, eval_df["prophet_p50"]),
            "MAPE": mape(y_true, eval_df["prophet_p50"]),
            "p10-p90 coverage": interval_coverage(
                y_true, eval_df["prophet_p10"], eval_df["prophet_p90"]
            ),
        },
    }

    print(f"\nBaseline comparison (test set, {TEST_DAYS}-day holdout, nominal 80% interval):")
    print(f"{'model':<10} {'MAE':>8} {'MAPE':>8} {'coverage':>10}")
    for name, m in results.items():
        print(f"{name:<10} {m['MAE']:>8.4f} {m['MAPE']:>8.2%} {m['p10-p90 coverage']:>10.2%}")

    eval_df.to_csv(ROOT / "data" / "raw" / "baseline_predictions.csv", index=False)
    print(f"\nWrote per-row predictions to data/raw/baseline_predictions.csv")


if __name__ == "__main__":
    main()
