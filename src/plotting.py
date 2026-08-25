import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Okabe-Ito blue: colourblind-safe and the tone Prophet itself uses.
FORECAST_COLOUR = "rgb(0, 114, 178)"
INTERVAL_COLOUR = "rgba(0, 114, 178, 0.2)"


def plot_raw_data(df: pd.DataFrame, date_col: str, value_col: str):
    """
    Plots the historical raw data using Plotly Express.
    """
    fig = px.line(df, x=date_col, y=value_col, title='Historical Sales Data')
    fig.update_layout(xaxis_title='Date', yaxis_title='Sales')
    return fig


def plot_forecast(model, forecast: pd.DataFrame):
    """
    Plots observed history, prediction and uncertainty interval.

    Built with graph_objects rather than prophet.plot.plot_plotly, which fails
    on a fitted model: it runs `assert m.history` on a DataFrame, and pandas
    raises ValueError instead of evaluating truthiness.
    """
    fig = go.Figure()

    # Interval first, so the lines are drawn on top of the shaded band.
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line={'width': 0},
        showlegend=False, hoverinfo='skip', name='Upper bound',
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', line={'width': 0},
        fill='tonexty', fillcolor=INTERVAL_COLOUR,
        hoverinfo='skip', name='Uncertainty interval',
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', line={'color': FORECAST_COLOUR}, name='Forecast',
    ))
    fig.add_trace(go.Scatter(
        x=model.history['ds'], y=model.history['y'],
        mode='markers', marker={'color': 'black', 'size': 4}, name='Observed',
    ))

    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales Prediction",
    )
    return fig


def plot_components(forecast: pd.DataFrame):
    """
    Manually creates component plots (trend) using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'))
    fig.update_layout(title="Forecast Trend Component", xaxis_title="Date", yaxis_title="Trend")
    return fig
