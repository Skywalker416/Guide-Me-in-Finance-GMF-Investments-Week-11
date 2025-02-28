import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date="2015-01-01", end_date="2025-01-31"):
    """Fetch historical stock data from YFinance."""
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Save raw data
    raw_path = "data/raw/financial_data.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    data.to_csv(raw_path)
    
    print(f"Data saved to {raw_path}")
    return data

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    fetch_data(tickers)
