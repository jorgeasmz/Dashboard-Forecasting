import logging
import os
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.schemas import (
    ForecastOut,
    ForecastPoint,
    ModelEvaluation,
    ObservationOut,
    SeriesSummary,
)
from src.config import APP_TITLE
from src.database import get_session
from src.ingest import load_observations
from src.schema import Observation, Series
from src.selection import aggregate, best_model
from src.serving import forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=APP_TITLE,
    description=(
        "Serves several time series, the rolling-origin backtest of four model "
        "families against each, and forecasts from the model that scored best."
    ),
    version="2.0.0",
)

SessionDep = Annotated[Session, Depends(get_session)]


def _require_series(session: Session, slug: str) -> Series:
    series = session.scalar(select(Series).where(Series.slug == slug))
    if series is None:
        raise HTTPException(status_code=404, detail=f"Unknown series '{slug}'.")
    return series


@app.get("/")
def root(session: SessionDep):
    """Health check. Reports how many series have been ingested."""
    return {
        "message": "Forecasting API is operational. Visit /docs.",
        "series": session.scalar(select(func.count(Series.id))) or 0,
    }


@app.get("/series", response_model=list[SeriesSummary])
def list_series(session: SessionDep):
    """Every registered series with its selected model."""
    counts = dict(
        session.execute(
            select(Observation.series_id, func.count(Observation.id)).group_by(
                Observation.series_id
            )
        ).all()
    )

    return [
        SeriesSummary(
            slug=series.slug,
            name=series.name,
            frequency=series.frequency,
            seasonal_period=series.seasonal_period,
            horizon=series.horizon,
            unit=series.unit,
            observations=counts.get(series.id, 0),
            best_model=best_model(session, series.slug),
        )
        for series in session.scalars(select(Series).order_by(Series.slug))
    ]


@app.get("/series/{slug}/observations", response_model=list[ObservationOut])
def series_observations(
    slug: str,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=20_000)] = 2_000,
):
    """The most recent observations, oldest first.

    The hourly series holds 93,504 of them, which is a response no caller wants by
    accident, so the tail is bounded. Every other series is shorter than the
    default and comes back whole.
    """
    _require_series(session, slug)
    history = load_observations(session, slug, limit=limit)
    return [ObservationOut(ts=row.ds, value=row.y) for row in history.itertuples()]


@app.get("/series/{slug}/evaluation", response_model=list[ModelEvaluation])
def series_evaluation(slug: str, session: SessionDep):
    """Mean backtest metrics per model, best MASE first."""
    _require_series(session, slug)
    summary = aggregate(session, slug)
    return [ModelEvaluation(**row) for row in summary.to_dict("records")]


@app.get("/series/{slug}/forecast", response_model=ForecastOut)
def series_forecast(
    slug: str,
    session: SessionDep,
    horizon: Annotated[int | None, Query(ge=1, le=365)] = None,
):
    """Forecast from the selected model, fitted on the full history."""
    _require_series(session, slug)
    try:
        model, frame = forecast(session, slug, horizon)
    except LookupError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None

    return ForecastOut(
        slug=slug,
        model=model,
        horizon=len(frame),
        points=[ForecastPoint(**row) for row in frame.to_dict("records")],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
