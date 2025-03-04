# Time Series Forecasting for Portfolio Management Optimization

## Challenge Overview
This project focuses on applying time series forecasting techniques to enhance portfolio management strategies for GMF Investments. The goal is to predict market trends, optimize asset allocation, and minimize risks.

## Business Objective
GMF Investments aims to leverage financial data and machine learning models to:
- Forecast stock prices
- Optimize investment portfolios
- Minimize risks while maximizing returns

## Data Source
- **YFinance**: Extracts historical stock prices for TSLA, BND, and SPY.

---

## Tasks

### Task 1: Preprocess and Explore the Data
1. **Extract Data**: Download historical financial data for TSLA, BND, and SPY.
2. **Data Cleaning**:
   - Check for missing values and handle them.
   - Normalize or scale data if required.
3. **Exploratory Data Analysis (EDA)**:
   - Visualize closing prices and percentage changes.
   - Compute rolling means, standard deviations, and outliers.
   - Perform time series decomposition to analyze trends and seasonality.
4. **Risk Analysis**:
   - Compute VaR (Value at Risk) and Sharpe Ratio.

### Task 2: Develop Time Series Forecasting Models
1. Choose a forecasting model:
   - **ARIMA**: Suitable for univariate time series without seasonality.
   - **SARIMA**: Extends ARIMA by incorporating seasonality.
   - **LSTM**: Deep learning approach for long-term dependencies.
2. Split dataset into training and testing sets.
3. Train and optimize the selected model.
4. Evaluate model performance using:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)

### Task 3: Forecast Future Market Trends
1. **Generate forecasts** for TSLA stock price (6-12 months ahead).
2. **Analyze the forecast**:
   - Visualize future prices alongside historical data.
   - Include confidence intervals.
3. **Interpret results**:
   - Identify long-term trends.
   - Discuss potential risks and volatility.
   - Highlight market opportunities.

### Task 4: Optimize Portfolio Based on Forecast
1. **Forecast BND and SPY** prices using the selected model.
2. **Portfolio Optimization**:
   - Compute annual return and volatility.
   - Analyze covariance between assets.
   - Adjust weights to maximize the Sharpe Ratio.
3. **Risk & Return Analysis**:
   - Calculate potential losses (VaR).
   - Measure portfolio standard deviation and expected returns.
   - Adjust asset allocation based on forecasted volatility.
4. **Visualize portfolio performance**:
   - Plot cumulative returns.
   - Compare optimized vs. original portfolio.

---

## Expected Deliverables
- Cleaned dataset
- Forecasting model and predictions
- Portfolio optimization strategy
- Visualizations and analysis reports

## Tools & Libraries
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `statsmodels`, `pmdarima` for ARIMA/SARIMA modeling
- `tensorflow`, `keras` for LSTM
- `scipy.optimize` for portfolio optimization

---

## How to Run the Project
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels pmdarima tensorflow keras yfinance
   ```
2. Run the preprocessing and EDA script.
3. Train and evaluate forecasting models.
4. Generate forecasts and optimize the portfolio.
5. Analyze the results and generate reports.

---

## Contributors
- Amanuel Legesse
- 10 Academy AI Mastery Program Team
