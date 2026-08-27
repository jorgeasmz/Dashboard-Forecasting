import pandas as pd
import plotly.graph_objs as go

from src.config import INTERVAL_WIDTH

OBSERVED = "rgb(70, 90, 110)"
FORECAST = "rgb(0, 114, 178)"
INTERVAL = "rgba(0, 114, 178, 0.2)"


def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, unit: str) -> go.Figure:
    """History, prediction and the uncertainty band around it."""
    figure = go.Figure()

    # Interval first, so the lines are drawn on top of the shaded band.
    figure.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
        line={"width": 0}, showlegend=False, hoverinfo="skip", name="Upper bound",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
        line={"width": 0}, fill="tonexty", fillcolor=INTERVAL,
        hoverinfo="skip", name=f"{INTERVAL_WIDTH:.0%} interval",
    ))
    figure.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], mode="lines",
        line={"color": FORECAST}, name="Forecast",
    ))
    figure.add_trace(go.Scatter(
        x=history["ds"], y=history["y"], mode="lines",
        line={"color": OBSERVED, "width": 1.4}, name="Observed",
    ))

    figure.update_layout(
        xaxis_title="Date", yaxis_title=unit, height=430,
        margin={"t": 30, "b": 40}, hovermode="x unified",
    )
    return figure


def plot_accuracy(summary: pd.DataFrame) -> go.Figure:
    """Mean MASE per model against the seasonal naive baseline at 1.0."""
    figure = go.Figure(go.Bar(
        x=summary["model"], y=summary["mase"],
        marker={"color": ["#1e8449" if v < 1 else "#c0392b" for v in summary["mase"]]},
        hovertemplate="MASE %{y:.3f}<extra></extra>",
    ))
    figure.add_hline(
        y=1.0, line={"color": "#555", "dash": "dash"},
        annotation_text="seasonal naive", annotation_position="top left",
    )
    figure.update_layout(
        title="Forecast error, scaled", yaxis_title="MASE",
        showlegend=False, height=340, margin={"t": 50, "b": 40},
    )
    return figure


def plot_coverage(summary: pd.DataFrame) -> go.Figure:
    """Observed interval coverage against the nominal level."""
    figure = go.Figure(go.Bar(
        x=summary["model"], y=summary["coverage"],
        marker={"color": "rgb(0, 114, 178)"},
        hovertemplate="%{y:.0%}<extra></extra>",
    ))
    figure.add_hline(
        y=INTERVAL_WIDTH, line={"color": "#555", "dash": "dash"},
        annotation_text=f"nominal {INTERVAL_WIDTH:.0%}", annotation_position="top left",
    )
    figure.update_layout(
        title="Interval coverage", yaxis_title="Share of actuals inside",
        yaxis={"tickformat": ".0%", "range": [0, 1.05]},
        showlegend=False, height=340, margin={"t": 50, "b": 40},
    )
    return figure
