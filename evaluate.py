"""
Backtest the Prophet forecaster against naive baselines.

Holds out the last HORIZON months, fits on the remainder, and reports error
metrics for Prophet and two baselines. A forecast that cannot beat "repeat the
last value" is not worth deploying, so the comparison is the point.

Usage: python evaluate.py
"""

import logging

import pandas as pd

from src.forecasting import Forecaster
from src.loader import fetch_data
from src.processing import prepare_for_prophet

# Prophet/cmdstanpy are chatty on stdout; keep the report readable.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

HORIZON = 12
SEASON = 12


def metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """MAPE (%), MAE and RMSE for one forecast against the held-out truth."""
    error = actual.to_numpy() - predicted.to_numpy()
    absolute = abs(error)
    return {
        "MAPE": float((absolute / actual.to_numpy()).mean() * 100),
        "MAE": float(absolute.mean()),
        "RMSE": float(((error ** 2).mean()) ** 0.5),
    }


def main() -> None:
    df = prepare_for_prophet(fetch_data())
    train, test = df.iloc[:-HORIZON], df.iloc[-HORIZON:]

    print(f"Observations: {len(df)}  |  train: {len(train)}  |  test: {len(test)}")
    print(f"Test window: {test['ds'].min():%Y-%m} to {test['ds'].max():%Y-%m}\n")

    forecaster = Forecaster()
    forecaster.train(train)
    forecast = forecaster.predict(periods=HORIZON)
    prophet_pred = forecast.iloc[-HORIZON:]["yhat"]

    results = {
        "Prophet": metrics(test["y"], prophet_pred),
        "Naive (last value)": metrics(
            test["y"], pd.Series([train["y"].iloc[-1]] * HORIZON)
        ),
        "Seasonal naive (t-12)": metrics(
            test["y"], train["y"].iloc[-SEASON:].reset_index(drop=True)
        ),
    }

    header = f"{'Model':<24}{'MAPE %':>10}{'MAE':>12}{'RMSE':>12}"
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        print(
            f"{name:<24}{scores['MAPE']:>10.2f}"
            f"{scores['MAE']:>12.1f}{scores['RMSE']:>12.1f}"
        )


if __name__ == "__main__":
    main()
