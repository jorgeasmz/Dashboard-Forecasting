"""
Forecasting with the model the backtest selected for each series.
"""

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.forecasters import build_all
from src.ingest import load_observations
from src.schema import Series
from src.selection import best_model


def forecast(
    session: Session, slug: str, horizon: int | None = None
) -> tuple[str, pd.DataFrame]:
    """Fits the selected model on the full history and returns its forecast."""
    series = session.scalar(select(Series).where(Series.slug == slug))
    if series is None:
        raise LookupError(f"Series '{slug}' is not registered.")

    history = load_observations(session, slug)
    if history.empty:
        raise LookupError(f"Series '{slug}' has no observations.")

    name = best_model(session, slug) or "seasonal_naive"
    chosen = next(m for m in build_all(series.seasonal_period) if m.name == name)

    steps = horizon or series.horizon
    return name, chosen.fit(history).predict(steps, series.frequency)
