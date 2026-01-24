import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from prophet.plot import plot_plotly

def plot_raw_data(df: pd.DataFrame, date_col: str, value_col: str):
    """
    Plots the historical raw data using Plotly Express.
    """
    fig = px.line(df, x=date_col, y=value_col, title='Historical Sales Data')
    fig.update_layout(xaxis_title='Date', yaxis_title='Sales')
    return fig

def plot_forecast(model, forecast):
    """
    Uses Prophet's built-in Plotly integration to plot the forecast.
    """
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales Prediction"
    )
    return fig

def plot_components(forecast):
    """
    Manually creates component plots (trend) using Plotly.

    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'))
    fig.update_layout(title="Forecast Trend Component", xaxis_title="Date", yaxis_title="Trend")
    return fig
