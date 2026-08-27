"""
Rolling-origin backtesting.

A single train/test split measures one window. Sliding the origin forward
measures several, which exposes the variance a single number hides.
"""

import numpy as np
import pandas as pd

from src.config import BACKTEST_FOLDS, SeriesSource
from src.forecasters import build_all


def rolling_origin_cutoffs(
    observations: int, horizon: int, folds: int = BACKTEST_FOLDS
) -> list[int]:
    """Training-set end index for each fold, oldest first."""
    cutoffs = [observations - horizon * (folds - i) for i in range(folds)]
    # Every fold needs enough history left to fit on.
    return [c for c in cutoffs if c >= horizon * 2]


def mase_denominator(train: pd.Series, seasonal_period: int) -> float:
    """In-sample mean absolute error of the seasonal naive."""
    values = train.to_numpy()
    m = seasonal_period if 1 <= seasonal_period < len(values) else 1
    differences = np.abs(values[m:] - values[:-m])
    scale = float(np.mean(differences)) if differences.size else 0.0
    # A constant series would divide by zero.
    return scale or 1.0


def score(actual: np.ndarray, forecast: pd.DataFrame, denominator: float) -> dict:
    """Point accuracy and interval calibration for one fold."""
    predicted = forecast["yhat"].to_numpy()
    errors = actual - predicted

    inside = (actual >= forecast["yhat_lower"].to_numpy()) & (
        actual <= forecast["yhat_upper"].to_numpy()
    )

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mase": float(np.mean(np.abs(errors)) / denominator),
        "coverage": float(np.mean(inside)),
    }


def backtest(
    history: pd.DataFrame, source: SeriesSource, folds: int = BACKTEST_FOLDS
) -> list[dict]:
    """Every model family against every fold. One row per model and fold."""
    history = history.reset_index(drop=True)
    cutoffs = rolling_origin_cutoffs(len(history), source.horizon, folds)

    results = []
    for fold, cutoff in enumerate(cutoffs):
        train = history.iloc[:cutoff]
        test = history.iloc[cutoff : cutoff + source.horizon]
        if len(test) < source.horizon:
            continue

        denominator = mase_denominator(train["y"], source.seasonal_period)
        actual = test["y"].to_numpy()

        for model in build_all(source.seasonal_period):
            forecast = model.fit(train).predict(source.horizon, source.frequency)
            results.append({
                "model": model.name,
                "fold": fold,
                "cutoff": pd.Timestamp(train["ds"].iloc[-1]).to_pydatetime(),
                "horizon": source.horizon,
                **score(actual, forecast, denominator),
            })

    return results
