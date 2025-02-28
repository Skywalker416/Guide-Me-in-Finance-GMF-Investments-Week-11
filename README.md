# Time Series Forecasting for Portfolio Management Optimization

## 📌 Project Overview
This project applies **time series forecasting** to financial data for **Guide Me in Finance (GMF) Investments** to enhance portfolio management strategies. Using **machine learning models**, we predict future market trends to optimize asset allocation and minimize risks.

### **📊 Assets Analyzed**
- **Tesla (TSLA):** High-growth, high-risk stock.
- **Vanguard Total Bond Market ETF (BND):** Low-risk, stable bond ETF.
- **S&P 500 ETF (SPY):** Diversified U.S. market exposure.

---

## 📂 Project Structure
```
├── data
│   ├── raw/              # Raw dataset from YFinance
│   ├── processed/        # Cleaned and preprocessed dataset
│
├── scripts
│   ├── data_preprocessing.py  # Data extraction and cleaning
│   ├── eda.py                 # Exploratory Data Analysis (EDA)
│   ├── forecasting.py         # Time series forecasting (ARIMA & Prophet)
│
├── reports/            # Visualizations & model outputs
├── README.md           # Project documentation
```

---

## 🔧 Setup & Installation

### **1️⃣ Install Dependencies**
Run the following command to install required libraries:

```bash
pip install pandas numpy matplotlib seaborn statsmodels yfinance pmdarima prophet
```

### **2️⃣ Run the Scripts**
To execute the data pipeline, run:

```bash
python scripts/data_preprocessing.py
python scripts/eda.py
python scripts/forecasting.py
```

---

## 📈 Key Features & Analysis

### **✅ Task 1: Data Preprocessing**
✔ Extracted **10 years** of financial data (2015-2025).  
✔ Cleaned missing values & normalized data.  

### **✅ Task 2: Exploratory Data Analysis (EDA)**
✔ Analyzed trends, volatility, and market fluctuations.  
✔ Identified **outliers and extreme price movements**.  
✔ Computed **risk metrics (VaR & Sharpe Ratio)**.  

### **✅ Task 3: Time Series Forecasting**
✔ Implemented **ARIMA & Facebook Prophet** models.  
✔ Forecasted **30 days of future stock prices**.  
✔ Evaluated predictions using **RMSE & MAPE**.  

---

## 🚀 Next Steps
📌 Integrate **forecast results** into **portfolio optimization**.  
📌 Experiment with **LSTMs (Deep Learning for Time Series)**.  
📌 Optimize model hyperparameters for **better accuracy**.  

---

## ✨ Contributors
- **Amanuel Legesse**  
  - [LinkedIn](https://www.linkedin.com/in/amanuel-legesse-041949205/)  

---

## 📝 License
This project is for educational purposes under **10 Academy AI Mastery**.
