"""
Forecasting model families behind one interface.

Every implementation returns ds, yhat, yhat_lower and yhat_upper so the
backtest can score point accuracy and interval calibration the same way.
"""

import logging
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import INTERVAL_WIDTH, MAX_SARIMA_SEASONAL_PERIOD

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

ALPHA = 1 - INTERVAL_WIDTH


def future_index(history: pd.DataFrame, periods: int, frequency: str) -> pd.DatetimeIndex:
    """Timestamps continuing the history at its own frequency."""
    last = pd.Timestamp(history["ds"].iloc[-1])
    return pd.date_range(start=last, periods=periods + 1, freq=frequency)[1:]


class SeasonalNaive:
    """Repeats the observation one seasonal period back."""

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int):
        self.seasonal_period = max(1, seasonal_period)
        self.history: pd.DataFrame | None = None
        self.spread = 0.0

    def fit(self, history: pd.DataFrame) -> "SeasonalNaive":
        self.history = history.reset_index(drop=True)
        values = self.history["y"].to_numpy()
        m = self.seasonal_period
        residuals = values[m:] - values[:-m] if len(values) > m else np.array([0.0])
        self.spread = float(np.std(residuals)) or 1.0
        return self

    def predict(self, periods: int, frequency: str) -> pd.DataFrame:
        values = self.history["y"].to_numpy()
        m = self.seasonal_period
        season = values[-m:]
        point = np.array([season[i % m] for i in range(periods)])

        # Uncertainty grows with the square root of the horizon, as in a random walk.
        widening = self.spread * np.sqrt(np.arange(1, periods + 1))
        z = 1.2816  # 80% two-sided normal quantile
        return pd.DataFrame({
            "ds": future_index(self.history, periods, frequency),
            "yhat": point,
            "yhat_lower": point - z * widening,
            "yhat_upper": point + z * widening,
        })


class ProphetForecaster:
    """Additive decomposition into trend, seasonality and holidays."""

    name = "prophet"

    def __init__(self, seasonal_period: int):
        self.seasonal_period = seasonal_period
        self.model: Prophet | None = None
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "ProphetForecaster":
        self.history = history.reset_index(drop=True)
        self.model = Prophet(interval_width=INTERVAL_WIDTH)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.history[["ds", "y"]])
        return self

    def predict(self, periods: int, frequency: str) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = self.model.predict(future).tail(periods)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


class SarimaForecaster:
    """Seasonal ARIMA with a fixed (1,1,1)(1,1,1,m) specification."""

    name = "sarima"

    def __init__(self, seasonal_period: int):
        # Fitting cost grows with the seasonal period, and a 365-day cycle is out
        # of reach, so long cycles fall back to a non-seasonal specification.
        self.seasonal_period = (
            seasonal_period
            if 1 < seasonal_period <= MAX_SARIMA_SEASONAL_PERIOD
            else 0
        )
        self.result = None
        self.history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "SarimaForecaster":
        self.history = history.reset_index(drop=True)
        seasonal_order = (
            (1, 1, 1, self.seasonal_period) if self.seasonal_period else (0, 0, 0, 0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = SARIMAX(
                self.history["y"].to_numpy(),
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        return self

    def predict(self, periods: int, frequency: str) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.result.get_forecast(steps=periods)
            interval = forecast.conf_int(alpha=ALPHA)

        interval = np.asarray(interval)
        return pd.DataFrame({
            "ds": future_index(self.history, periods, frequency),
            "yhat": np.asarray(forecast.predicted_mean),
            "yhat_lower": interval[:, 0],
            "yhat_upper": interval[:, 1],
        })


class LightGBMForecaster:
    """Gradient boosting over lag and calendar features, forecast recursively."""

    name = "lightgbm"

    def __init__(self, seasonal_period: int):
        self.seasonal_period = max(1, seasonal_period)
        self.lags = self._lags()
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.history: pd.DataFrame | None = None

    def _lags(self) -> list[int]:
        base = [1, 2, 3]
        if self.seasonal_period > 1:
            base += [self.seasonal_period, 2 * self.seasonal_period]
        return sorted(set(base))

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for lag in self.lags:
            out[f"lag_{lag}"] = frame["y"].shift(lag)
        out["month"] = frame["ds"].dt.month
        out["dayofweek"] = frame["ds"].dt.dayofweek
        out["dayofyear"] = frame["ds"].dt.dayofyear
        return out

    def fit(self, history: pd.DataFrame) -> "LightGBMForecaster":
        self.history = history.reset_index(drop=True)
        features = self._features(self.history)
        usable = features.dropna().index

        X, y = features.loc[usable], self.history.loc[usable, "y"]
        for quantile in (ALPHA / 2, 0.5, 1 - ALPHA / 2):
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                verbose=-1,
            )
            model.fit(X, y)
            self.models[quantile] = model
        return self

    def predict(self, periods: int, frequency: str) -> pd.DataFrame:
        index = future_index(self.history, periods, frequency)
        working = self.history[["ds", "y"]].copy()

        rows = []
        for timestamp in index:
            working = pd.concat(
                [working, pd.DataFrame({"ds": [timestamp], "y": [np.nan]})],
                ignore_index=True,
            )
            features = self._features(working).iloc[[-1]]
            predictions = {q: float(m.predict(features)[0]) for q, m in self.models.items()}

            # The median path feeds the lags for every subsequent step.
            working.loc[working.index[-1], "y"] = predictions[0.5]
            rows.append(predictions)

        lower, upper = ALPHA / 2, 1 - ALPHA / 2
        return pd.DataFrame({
            "ds": index,
            "yhat": [r[0.5] for r in rows],
            "yhat_lower": [min(r[lower], r[0.5]) for r in rows],
            "yhat_upper": [max(r[upper], r[0.5]) for r in rows],
        })


FAMILIES = (SeasonalNaive, ProphetForecaster, SarimaForecaster, LightGBMForecaster)


def build_all(seasonal_period: int) -> list:
    """One instance of every family, configured for a series."""
    return [family(seasonal_period) for family in FAMILIES]
