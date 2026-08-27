import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.config import SERIES, SeriesSource
from src.schema import Observation, Series


def fetch_series(source: SeriesSource, location: str | None = None) -> pd.DataFrame:
    """Reads one series and returns it with ds and y columns."""
    frame = pd.read_csv(location or source.url)
    frame = frame.rename(columns={source.date_column: "ds", source.value_column: "y"})

    frame["ds"] = pd.to_datetime(frame["ds"])
    # Some published series carry sentinel characters on a few values.
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")

    frame = frame.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
    if source.max_observations:
        frame = frame.tail(source.max_observations).reset_index(drop=True)

    return frame[["ds", "y"]]


def upsert_series(session: Session, source: SeriesSource) -> Series:
    """Creates or updates the series row and returns it."""
    series = session.scalar(select(Series).where(Series.slug == source.slug))
    if series is None:
        series = Series(slug=source.slug)
        session.add(series)

    series.name = source.name
    series.frequency = source.frequency
    series.seasonal_period = source.seasonal_period
    series.horizon = source.horizon
    series.unit = source.unit

    session.commit()
    session.refresh(series)
    return series


def replace_observations(session: Session, series: Series, frame: pd.DataFrame) -> int:
    """Replaces the stored observations for one series. Idempotent by design."""
    session.execute(delete(Observation).where(Observation.series_id == series.id))
    session.add_all(
        Observation(series_id=series.id, ts=row.ds.to_pydatetime(), value=float(row.y))
        for row in frame.itertuples()
    )
    session.commit()
    return len(frame)


def load_observations(session: Session, slug: str) -> pd.DataFrame:
    """Reads one series back as a ds/y frame ordered by time."""
    statement = (
        select(Observation.ts, Observation.value)
        .join(Series)
        .where(Series.slug == slug)
        .order_by(Observation.ts)
    )
    rows = session.execute(statement).all()
    return pd.DataFrame(rows, columns=["ds", "y"])


def ingest_all(session: Session) -> dict[str, int]:
    """Downloads every registered series and stores it. Returns row counts."""
    counts = {}
    for source in SERIES:
        frame = fetch_series(source)
        series = upsert_series(session, source)
        counts[source.slug] = replace_observations(session, series, frame)
    return counts
