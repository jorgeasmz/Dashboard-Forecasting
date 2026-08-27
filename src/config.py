"""
Series registry and service configuration.
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'forecasting.db'}")

APP_TITLE = "Time-Series Forecasting Service"

# Width of the prediction interval every model must produce.
INTERVAL_WIDTH = 0.80

# Rolling-origin folds evaluated per series and model.
BACKTEST_FOLDS = 4

# SARIMA fitting time grows with the seasonal period, so longer cycles are
# modelled without a seasonal term.
MAX_SARIMA_SEASONAL_PERIOD = 24


def _normalise_database_url(url: str) -> str:
    """Points bare postgres:// and postgresql:// URLs at psycopg 3."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = _normalise_database_url(DATABASE_URL)

BASE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"


@dataclass(frozen=True)
class SeriesSource:
    """Where a series comes from and how it should be modelled."""

    slug: str
    name: str
    filename: str
    date_column: str
    value_column: str
    frequency: str
    # Cycle length in observations, used by the seasonal naive and by MASE.
    seasonal_period: int
    horizon: int
    unit: str
    max_observations: int | None = None

    @property
    def url(self) -> str:
        return BASE_URL + self.filename


SERIES = [
    SeriesSource(
        slug="airline-passengers",
        name="Airline passengers",
        filename="airline-passengers.csv",
        date_column="Month",
        value_column="Passengers",
        frequency="MS",
        seasonal_period=12,
        horizon=12,
        unit="passengers (thousands)",
    ),
    SeriesSource(
        slug="car-sales",
        name="Monthly car sales",
        filename="monthly-car-sales.csv",
        date_column="Month",
        value_column="Sales",
        frequency="MS",
        seasonal_period=12,
        horizon=12,
        unit="vehicles",
    ),
    SeriesSource(
        slug="sunspots",
        name="Monthly sunspots",
        filename="monthly-sunspots.csv",
        date_column="Month",
        value_column="Sunspots",
        frequency="MS",
        # The solar cycle runs about 132 months, which no seasonal term here can
        # capture, so the series is treated as non-seasonal.
        seasonal_period=1,
        horizon=12,
        unit="sunspot count",
        max_observations=600,
    ),
    SeriesSource(
        slug="min-temperatures",
        name="Daily minimum temperatures",
        filename="daily-min-temperatures.csv",
        date_column="Date",
        value_column="Temp",
        frequency="D",
        seasonal_period=365,
        horizon=28,
        unit="degrees Celsius",
        max_observations=1460,
    ),
    SeriesSource(
        slug="female-births",
        name="Daily female births",
        filename="daily-total-female-births.csv",
        date_column="Date",
        value_column="Births",
        frequency="D",
        seasonal_period=7,
        horizon=14,
        unit="births",
    ),
]

SERIES_BY_SLUG = {source.slug: source for source in SERIES}
