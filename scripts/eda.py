import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose

def load_data(filepath="data/processed/cleaned_data.csv"):
    """Load the cleaned financial dataset."""
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

def plot_closing_prices(df):
    """Plot closing prices of TSLA, BND, and SPY."""
    plt.figure(figsize=(12,6))
    for asset in ["TSLA_Close", "BND_Close", "SPY_Close"]:
        plt.plot(df.index, df[asset], label=asset)
    
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("reports/closing_prices.png")
    plt.show()

def compute_daily_returns(df):
    """Compute and plot daily returns to analyze volatility."""
    daily_returns = df.filter(like="Close").pct_change().dropna()
    
    plt.figure(figsize=(12,6))
    for col in daily_returns.columns:
        plt.plot(daily_returns.index, daily_returns[col], label=col)
    
    plt.title("Daily Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid()
    plt.savefig("reports/daily_returns.png")
    plt.show()

    return daily_returns

def plot_rolling_statistics(df, window=30):
    """Compute rolling mean and standard deviation."""
    plt.figure(figsize=(12,6))
    for asset in ["TSLA_Close", "BND_Close", "SPY_Close"]:
        df[asset].rolling(window).mean().plot(label=f"{asset} {window}-day MA")
    
    plt.title("Rolling Mean (30-day) for Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("reports/rolling_mean.png")
    plt.show()

def detect_outliers(daily_returns, threshold=0.05):
    """Identify days with extreme returns."""
    outliers = daily_returns[(daily_returns > threshold) | (daily_returns < -threshold)]
    
    plt.figure(figsize=(12,6))
    for col in daily_returns.columns:
        plt.scatter(outliers.index, outliers[col], label=col, marker='o')
    
    plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    plt.axhline(y=-threshold, color='r', linestyle='--')
    plt.title("Outlier Detection: Extreme Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid()
    plt.savefig("reports/outliers.png")
    plt.show()

def decompose_time_series(df, asset="TSLA_Close"):
    """Decompose time series into trend, seasonality, and residual components."""
    result = seasonal_decompose(df[asset], model="additive", period=252)  # Approx. 1-year period

    plt.figure(figsize=(10,8))
    result.plot()
    plt.savefig(f"reports/{asset}_decomposition.png")
    plt.show()

def calculate_risk_metrics(daily_returns):
    """Compute Value at Risk (VaR) and Sharpe Ratio."""
    var_95 = daily_returns.quantile(0.05)  # 5% worst-case loss
    sharpe_ratio = daily_returns.mean() / daily_returns.std()
    
    print("\n**Risk Metrics**")
    print(f"Value at Risk (95% confidence level):\n{var_95}")
    print(f"Sharpe Ratio:\n{sharpe_ratio}")

    return var_95, sharpe_ratio

if __name__ == "__main__":
    df = load_data()

    plot_closing_prices(df)
    daily_returns = compute_daily_returns(df)
    plot_rolling_statistics(df)
    detect_outliers(daily_returns)
    decompose_time_series(df, "TSLA_Close")
    calculate_risk_metrics(daily_returns)
