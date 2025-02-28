import pandas as pd
import os

def preprocess_data(input_path="data/raw/financial_data.csv", output_path="data/processed/cleaned_data.csv"):
    """Load, clean, and preprocess financial data."""
    
    # Load data
    df = pd.read_csv(input_path, header=[0,1], index_col=0, parse_dates=True)
    
    # Flatten multi-index columns (YFinance stores data in hierarchical format)
    df.columns = ['_'.join(col).strip() for col in df.columns]
    
    # Handle missing values
    df = df.fillna(method='ffill')  # Forward fill missing values
    df = df.dropna()  # Drop any remaining missing values

    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data()
