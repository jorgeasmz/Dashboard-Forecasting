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

# Observations the dashboard draws behind a forecast. Every series but the
# hourly one is shorter than this and is drawn whole.
DISPLAY_OBSERVATIONS = 2_000

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


def warehouse_url() -> str:
    """Connection to the energy platform's warehouse.

    Read at ingest time only, and read on each call rather than at import, since
    ingestion is the one command that needs it and nothing else does.

    This one goes to psycopg directly rather than through SQLAlchemy, so the
    driver suffix that DATABASE_URL carries is stripped if it is pasted here.
    """
    return os.getenv("WAREHOUSE_URL", "").replace("postgresql+psycopg://", "postgresql://", 1)


@dataclass(frozen=True)
class PublishedCsv:
    """A series published as a CSV and read over HTTP."""

    filename: str
    date_column: str
    value_column: str

    @property
    def location(self) -> str:
        return BASE_URL + self.filename


@dataclass(frozen=True)
class WarehouseColumn:
    """A measurement column of the energy platform's hourly mart.

    Aggregation to a coarser grain runs in the warehouse, so a daily series
    transfers 3,896 rows rather than 93,504 to compute the same means.
    """

    column: str
    # None reads the hourly grain; "day" averages the hours of each complete day.
    grain: str | None = None


@dataclass(frozen=True)
class SeriesSource:
    """Where a series comes from and how it should be modelled."""

    slug: str
    name: str
    origin: PublishedCsv | WarehouseColumn
    frequency: str
    # Cycle length in observations, used by the seasonal naive and by MASE.
    seasonal_period: int
    horizon: int
    unit: str
    max_observations: int | None = None
    # Observations each fold may train on. None trains on everything before the
    # cutoff, which is what the short series want and what the long ones cannot
    # afford.
    max_train: int | None = None


SERIES = [
    SeriesSource(
        slug="airline-passengers",
        name="Airline passengers",
        origin=PublishedCsv("airline-passengers.csv", "Month", "Passengers"),
        frequency="MS",
        seasonal_period=12,
        horizon=12,
        unit="passengers (thousands)",
    ),
    SeriesSource(
        slug="car-sales",
        name="Monthly car sales",
        origin=PublishedCsv("monthly-car-sales.csv", "Month", "Sales"),
        frequency="MS",
        seasonal_period=12,
        horizon=12,
        unit="vehicles",
    ),
    SeriesSource(
        slug="sunspots",
        name="Monthly sunspots",
        origin=PublishedCsv("monthly-sunspots.csv", "Month", "Sunspots"),
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
        origin=PublishedCsv("daily-min-temperatures.csv", "Date", "Temp"),
        frequency="D",
        seasonal_period=365,
        horizon=28,
        unit="degrees Celsius",
        max_observations=1460,
    ),
    SeriesSource(
        slug="female-births",
        name="Daily female births",
        origin=PublishedCsv("daily-total-female-births.csv", "Date", "Births"),
        frequency="D",
        seasonal_period=7,
        horizon=14,
        unit="births",
    ),
    SeriesSource(
        slug="spot-price-daily",
        name="Colombian spot price, daily",
        origin=WarehouseColumn("spot_price", grain="day"),
        frequency="D",
        # The weekly cycle. The annual one is left to the families that model a
        # trend, since a 365-period seasonal term is out of reach here.
        seasonal_period=7,
        horizon=14,
        unit="COP/kWh",
        # Three years. The level is not stationary over the decade the warehouse
        # holds: yearly means run from 106 to 676. Fitted through all of it,
        # Prophet extrapolates a trend that scores 9.679 at 0% coverage; over
        # three years it scores 0.955 at 96%, and over one 1.235 at 57%.
        max_train=1_095,
    ),
    SeriesSource(
        slug="spot-price-hourly",
        name="Colombian spot price, hourly",
        origin=WarehouseColumn("spot_price"),
        frequency="h",
        seasonal_period=24,
        # Day ahead, which is the horizon the market itself settles on.
        horizon=24,
        unit="COP/kWh",
        # One year of hours. Measured: a SARIMA fit at this seasonal period costs
        # 97 s over 8,760 observations and over 420 s over 26,280, so an expanding
        # window would price the family out of the comparison entirely.
        max_train=8_760,
    ),
]

SERIES_BY_SLUG = {source.slug: source for source in SERIES}
