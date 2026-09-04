import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src import warehouse
from src.config import SERIES, PublishedCsv, SeriesSource, WarehouseColumn, warehouse_url
from src.schema import Observation, Series


def read_csv(origin: PublishedCsv, location: str | None = None) -> pd.DataFrame:
    """A published CSV as a ds/y frame, before cleaning."""
    frame = pd.read_csv(location or origin.location)
    return frame.rename(columns={origin.date_column: "ds", origin.value_column: "y"})


def fetch_series(source: SeriesSource, location: str | None = None) -> pd.DataFrame:
    """Reads one series from wherever it comes from, with ds and y columns."""
    if isinstance(source.origin, WarehouseColumn):
        frame = warehouse.read(source.origin, warehouse_url())
    else:
        frame = read_csv(source.origin, location)

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
    series.max_train = source.max_train

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


def load_observations(
    session: Session, slug: str, limit: int | None = None
) -> pd.DataFrame:
    """Reads one series back as a ds/y frame ordered by time.

    A limit takes the most recent observations rather than the first, since every
    caller that sets one wants the end of the series.
    """
    statement = select(Observation.ts, Observation.value).join(Series).where(
        Series.slug == slug
    )
    if limit is None:
        rows = session.execute(statement.order_by(Observation.ts)).all()
    else:
        recent = session.execute(
            statement.order_by(Observation.ts.desc()).limit(limit)
        ).all()
        rows = list(reversed(recent))

    return pd.DataFrame(rows, columns=["ds", "y"])


def ingest_all(
    session: Session, slugs: list[str] | None = None
) -> dict[str, int | None]:
    """Reads every registered series and stores it. Returns row counts.

    A warehouse-backed series is skipped rather than fatal when no warehouse is
    configured, so the published series still ingest with no infrastructure
    beyond this project's own database. Skipped series map to None.
    """
    counts: dict[str, int | None] = {}
    for source in SERIES:
        if slugs is not None and source.slug not in slugs:
            continue

        if isinstance(source.origin, WarehouseColumn) and not warehouse_url():
            counts[source.slug] = None
            continue

        frame = fetch_series(source)
        series = upsert_series(session, source)
        counts[source.slug] = replace_observations(session, series, frame)
    return counts
