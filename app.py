import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.config import APP_TITLE, DISPLAY_OBSERVATIONS, INTERVAL_WIDTH
from src.database import SessionLocal
from src.ingest import load_observations
from src.plotting import plot_accuracy, plot_coverage, plot_forecast
from src.schema import Series
from src.selection import aggregate
from src.serving import forecast

st.set_page_config(page_title=APP_TITLE, layout="wide")


@st.cache_resource
def session_factory():
    """One session factory per Streamlit process."""
    return SessionLocal


@st.cache_data(ttl=300)
def registered_series() -> list[dict] | None:
    """Registered series, or None when the schema has not been created yet."""
    try:
        with session_factory()() as session:
            return [
                {
                    "slug": s.slug, "name": s.name, "unit": s.unit,
                    "frequency": s.frequency, "horizon": s.horizon,
                }
                for s in session.query(Series).order_by(Series.slug)
            ]
    except SQLAlchemyError:
        return None


@st.cache_data(ttl=300)
def series_data(slug: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with session_factory()() as session:
        history = load_observations(session, slug, limit=DISPLAY_OBSERVATIONS)
        return history, aggregate(session, slug)


@st.cache_data(ttl=300)
def series_forecast(slug: str, horizon: int) -> tuple[str, pd.DataFrame]:
    with session_factory()() as session:
        return forecast(session, slug, horizon)


st.title(APP_TITLE)

available = registered_series()
if available is None:
    st.error("Could not read the database. Has `alembic upgrade head` been run?")
    st.stop()
if not available:
    st.warning("No series ingested yet. Run `python -m src.pipeline` first.")
    st.stop()

labels = {item["slug"]: item["name"] for item in available}
slug = st.sidebar.selectbox(
    "Series", list(labels), format_func=lambda key: labels[key]
)
source = next(item for item in available if item["slug"] == slug)
horizon = st.sidebar.slider(
    "Forecast horizon", 1, source["horizon"] * 3, source["horizon"],
    help="Steps ahead, in the series' own frequency",
)

history, summary = series_data(slug)
unit = source["unit"]

if summary.empty:
    st.warning("This series has no backtest results. Run `python -m src.pipeline`.")
    st.stop()

selected = str(summary.iloc[0]["model"])

top = st.columns(4)
top[0].metric("Observations", len(history))
top[1].metric("Frequency", source["frequency"])
top[2].metric("Selected model", selected)
top[3].metric("MASE", f"{summary.iloc[0]['mase']:.3f}")

with st.spinner(f"Forecasting with {selected}..."):
    model, prediction = series_forecast(slug, horizon)

st.plotly_chart(plot_forecast(history, prediction, unit), width="stretch")

st.subheader("Rolling-origin backtest")
st.caption(
    f"Four folds, horizon {source['horizon']}. MASE scales the error against the "
    "seasonal naive, so 1.0 means the model matched it and above 1.0 means it lost."
)

left, right = st.columns(2)
left.plotly_chart(plot_accuracy(summary), width="stretch")
right.plotly_chart(plot_coverage(summary), width="stretch")

st.dataframe(
    summary.rename(columns={
        "model": "Model", "folds": "Folds", "mae": "MAE",
        "rmse": "RMSE", "mase": "MASE", "coverage": "Coverage",
    }).style.format({
        "MAE": "{:.2f}", "RMSE": "{:.2f}", "MASE": "{:.3f}", "Coverage": "{:.0%}",
    }),
    width="stretch",
    hide_index=True,
)

worst = summary["coverage"].min()
if worst < INTERVAL_WIDTH - 0.15:
    st.info(
        f"Nominal interval width is {INTERVAL_WIDTH:.0%}, and the weakest model here "
        f"captures {worst:.0%} of actual values. A point forecast without a calibrated "
        "interval overstates how much is known about the future."
    )
