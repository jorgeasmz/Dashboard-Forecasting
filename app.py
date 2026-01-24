import streamlit as st
import pandas as pd
from src.config import APP_TITLE, DATE_COLUMN_RAW, SALES_COLUMN_RAW
from src.loader import load_data
from src.processing import prepare_for_prophet
from src.forecasting import Forecaster
from src.plotting import plot_raw_data, plot_forecast, plot_components

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    
    # 1. Load Data
    with st.spinner('Loading data...'):
        df = load_data()
        
    if df.empty:
        st.warning("No data found or error loading data.")
        return

    # Sidebar Controls
    st.sidebar.header("Configuration")
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (Months)", 
        min_value=1, 
        max_value=36, 
        value=12
    )

    # 2. EDA Section
    st.subheader("1. Historical Data Explorer")
    
    # Metrics
    total_sales = df[SALES_COLUMN_RAW].sum()
    avg_sales = df[SALES_COLUMN_RAW].mean()
    col1, col2 = st.columns(2)
    col1.metric("Total Historical Sales", f"{total_sales:,.0f}")
    col2.metric("Average Monthly Sales", f"{avg_sales:,.0f}")

    # Plot Raw Data
    st.plotly_chart(plot_raw_data(df, DATE_COLUMN_RAW, SALES_COLUMN_RAW), width='stretch')

    if show_raw_data:
        st.dataframe(df)

    # 3. Forecasting Section
    st.subheader("2. Future Forecast")
    
    if st.button("Generate Forecast"):
        with st.spinner('Training model and generating forecast...'):
            # Prepare data
            prophet_df = prepare_for_prophet(df)
            
            # Train model
            forecaster = Forecaster()
            forecaster.train(prophet_df)
            
            # Predict
            forecast = forecaster.predict(periods=forecast_horizon)
            
            # Visualizations
            st.markdown(f"### Forecast for the next {forecast_horizon} months")
            
            # Main Forecast Plot
            fig_forecast = plot_forecast(forecaster.model, forecast)
            st.plotly_chart(fig_forecast, width='stretch')
            
            # Components Plot
            st.markdown("### Trend Analysis")
            fig_components = plot_components(forecast)
            st.plotly_chart(fig_components, width='stretch')
            
            # Download Data
            st.markdown("### Download Results")
            # Filter only future dates for the download
            last_hist_date = prophet_df['ds'].max()
            future_df = forecast[forecast['ds'] > last_hist_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Forecast CSV",
                csv,
                "forecast.csv",
                "text/csv",
                key='download-csv'
            )

    st.markdown("---")

if __name__ == "__main__":
    main()
