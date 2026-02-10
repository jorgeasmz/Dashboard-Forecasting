# Dashboard-Forecasting

Technical dashboard for time-series analysis and forecasting.

## Overview

This application implements a forecasting pipeline using the [Prophet](https://facebook.github.io/prophet/) library. The interface is built with [Streamlit](https://streamlit.io/).

**Live Demo:** [Streamlit App](https://jorgeasmz-dashboard-forecasting.streamlit.app/)

The system performs the following operations:
1.  **Data Ingestion**: Loads time-series data from CSV sources.
2.  **Preprocessing**: Standardizes date formats and prepares data frames for the Prophet model.
3.  **Modelling**: Trains a Prophet model on historical data.
4.  **Forecasting**: Generates predictions for a user-specified horizon.
5.  **Visualization**: Renders historical data, forecast trends, and seasonality components using Plotly.

## Project Structure

```
├── app.py                # Application entry point (Streamlit interface)
├── requirements.txt      # Project dependencies
└── src/
    ├── config.py         # Configuration constants
    ├── forecasting.py    # Prophet model wrapper class
    ├── loader.py         # Data loading and caching
    ├── plotting.py       # Plotly visualization functions
    └── processing.py     # Data transformation utilities
```

## Setup and Usage

### Prerequisites

*   Python 3.8+

### Installation

1.  Clone the repository and navigate to the project root.
2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Execution

Run the Streamlit application:

```bash
streamlit run app.py
```
