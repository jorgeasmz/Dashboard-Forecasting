from prophet import Prophet
import pandas as pd

class Forecaster:
    def __init__(self):
        self.model = None
        self.forecast = None

    def train(self, df: pd.DataFrame):
        """
        Trains the Prophet model on the provided historical data.
        """
        self.model = Prophet()
        self.model.fit(df)

    def predict(self, periods: int, freq: str = 'MS') -> pd.DataFrame:
        """
        Generates future predictions.
        
        Args:
            periods (int): Number of periods to forecast forward.
            freq (str): Frequency of the data (e.g., 'D' for daily, 'MS' for Month Start).
        """
        if not self.model:
            raise ValueError("Model has not been trained yet.")
            
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)
        return self.forecast
    
    def get_forecast(self) -> pd.DataFrame:
        return self.forecast
