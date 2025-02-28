import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def load_data(filepath="data/processed/cleaned_data.csv"):
    """Load the cleaned financial dataset."""
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def train_arima(df, asset="TSLA_Close"):
    """Train an ARIMA model and make future predictions."""
    series = df[asset].dropna()

    # Train ARIMA model
    model = ARIMA(series, order=(5,1,0))
    arima_fit = model.fit()

    # Forecast next 30 days
    forecast = arima_fit.forecast(steps=30)

    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(series, label="Actual Data")
    plt.plot(pd.date_range(series.index[-1], periods=30, freq="D"), forecast, label="ARIMA Forecast", linestyle="dashed")
    plt.title(f"ARIMA Forecast for {asset}")
    plt.legend()
    plt.grid()
    plt.savefig(f"reports/{asset}_arima_forecast.png")
    plt.show()

    return arima_fit, forecast

def train_prophet(df, asset="TSLA_Close"):
    """Train a Facebook Prophet model for time series forecasting."""
    series = df[[asset]].dropna().reset_index()
    series.columns = ["ds", "y"]  # Rename columns for Prophet compatibility

    # Train model
    model = Prophet()
    model.fit(series)

    # Create future dataframe
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot results
    fig = model.plot(forecast)
    fig.savefig(f"reports/{asset}_prophet_forecast.png")
    plt.show()

    return model, forecast

def evaluate_forecast(actual, predicted):
    """Compute RMSE and MAPE for model evaluation."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    print(f"RMSE: {rmse}, MAPE: {mape}")
    return rmse, mape

if __name__ == "__main__":
    df = load_data()

    # ARIMA Forecasting
    arima_model, arima_forecast = train_arima(df, "TSLA_Close")

    # Prophet Forecasting
    prophet_model, prophet_forecast = train_prophet(df, "TSLA_Close")

    # Evaluate on last known data
    actual_values = df["TSLA_Close"].iloc[-30:]  # Last 30 days of known data
    predicted_values = arima_forecast[:30]
    evaluate_forecast(actual_values, predicted_values)
