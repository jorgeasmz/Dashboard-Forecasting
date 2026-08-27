import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, mase_denominator, rolling_origin_cutoffs, score


def test_cutoffs_step_forward_by_one_horizon():
    assert rolling_origin_cutoffs(144, horizon=12, folds=4) == [96, 108, 120, 132]


def test_cutoffs_drop_folds_without_enough_history():
    """A fold that would train on almost nothing is not evaluated."""
    assert rolling_origin_cutoffs(40, horizon=12, folds=4) == [28]


def test_mase_denominator_uses_the_seasonal_lag():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Every observation is three larger than the one three steps back.
    assert mase_denominator(values, seasonal_period=3) == pytest.approx(3.0)


def test_mase_denominator_survives_a_constant_series():
    """A flat series has zero seasonal error, which would divide by zero."""
    assert mase_denominator(pd.Series([5.0] * 10), seasonal_period=1) == 1.0


def test_mase_denominator_falls_back_when_the_period_exceeds_the_history():
    assert mase_denominator(pd.Series([1.0, 3.0, 5.0]), seasonal_period=99) > 0


def test_a_perfect_forecast_scores_zero():
    actual = np.array([10.0, 12.0, 14.0])
    forecast = pd.DataFrame({
        "yhat": actual, "yhat_lower": actual - 1, "yhat_upper": actual + 1,
    })

    result = score(actual, forecast, denominator=2.0)

    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["mase"] == 0.0
    assert result["coverage"] == 1.0


def test_coverage_counts_actuals_inside_the_interval():
    actual = np.array([10.0, 50.0])
    forecast = pd.DataFrame({
        "yhat": [10.0, 10.0], "yhat_lower": [9.0, 9.0], "yhat_upper": [11.0, 11.0],
    })

    assert score(actual, forecast, denominator=1.0)["coverage"] == 0.5


def test_mase_is_relative_to_the_denominator():
    actual = np.array([10.0, 10.0])
    forecast = pd.DataFrame({
        "yhat": [12.0, 12.0], "yhat_lower": [0.0, 0.0], "yhat_upper": [20.0, 20.0],
    })

    # Mean absolute error of 2 against a seasonal naive that errs by 4.
    assert score(actual, forecast, denominator=4.0)["mase"] == pytest.approx(0.5)


def test_backtest_scores_every_family_on_every_fold(history, source):
    results = backtest(history, source, folds=2)

    assert len(results) == 2 * 4
    assert {row["model"] for row in results} == {
        "seasonal_naive", "prophet", "sarima", "lightgbm"
    }
    assert {row["fold"] for row in results} == {0, 1}


def test_backtest_reports_every_metric(history, source):
    results = backtest(history, source, folds=2)

    assert set(results[0]) == {
        "model", "fold", "cutoff", "horizon", "mae", "rmse", "mase", "coverage"
    }
