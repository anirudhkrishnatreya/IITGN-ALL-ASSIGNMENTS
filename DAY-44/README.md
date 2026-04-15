# Day 44: Time Series Forecasting \u0026 Sensor Failure Prediction

## 🎯 Overview
This assignment explores time-series analysis for forecasting daily e-commerce sales and predicting 24-hour equipment failure risk from warehouse sensor data. It requires handling non-stationary data, building ARIMA and SARIMA models, and executing a cost matrix analysis to justify a machine learning approach over a simpler rule-based predictive system.

## 📋 Task Description
The notebook (`week08_monday_timeseries.ipynb`) contains the following key components focusing on time-series analysis and forecasting:

1. **Sub-step 1 (Easy)** — E-Commerce Sales: Characterise \u0026 Stationarity
2. **Sub-step 2 (Easy)** — Sensor Data: Identify \u0026 Fix All Quality Issues
3. **Sub-step 3 (Medium)** — ARIMA Model: Fit, Evaluate, Justify
4. **Sub-step 4 (Medium)** — SARIMA: Capture Seasonal Patterns
5. **Sub-step 5 (Medium)** — Sensor Failure Prediction (24-hour horizon)
6. **Sub-step 6 (Hard)** — Rule vs ML: Cost Matrix Analysis

## 🛠️ Solution Implementation

### 1 \u0026 2. Data Characterisation \u0026 Quality Issues
- Analysed synthetically generated datasets mapping to e-commerce sales and warehouse sensor networks.
- Identified and fixed data quality issues commonly found in raw sensor data (e.g., missing values, anomalies).
- Assessed the stationarity of the e-commerce sales time-series.

### 3. ARIMA Model Construction
- Implemented an Auto-Regressive Integrated Moving Average (ARIMA) model for sales forecasting.
- Instead of relying solely on `auto_arima`, performed manual selection of order parameters ($p, d, q$) based on the initial inspection of Auto-Correlation Function (ACF) and Partial Auto-Correlation Function (PACF) plots.
- Validated stationarity requirements using the Augmented Dickey-Fuller (ADF) test, verifying that first-order differencing ($d=1$) was sufficient to render the time-series stationary prior to modeling.

### 4. SARIMA \u0026 Seasonality
- Extended the baseline ARIMA model to Seasonal ARIMA (SARIMA) to reliably capture and model the distinct weekly seasonal patterns in e-commerce purchasing behaviors shown in the dataset.

### 5. Predictive Maintenance (Sensor Failure)
- Engineered time-series specific features from the raw sensor stream to predict equipment failure over a 24-hour horizon.
- Strictly maintained chronological ordering without random train/test splits, averting data leakage.

### 6. Rule vs ML: Cost Matrix Analysis
- Compared the performance of a naive, rule-based threshold approach against the trained ML model.
- Evaluated these models using a highly targeted business cost matrix: Rs 50,000 for a missed failure vs Rs 5,000 for a false alarm.
- Derived the final justification for the ML model based directly on these domain-derived cost dynamics.

## 💡 Key Takeaways
- **Chronological Integrity is Crucial:** Never use random train/test split on time-series or sequential data. It breaches causality and causes data leakage.
- **Manual Parameter Tuning vs. AutoML:** Manually verifying ARIMA parameters ($p, d, q$) with ACF/PACF plots and unit root tests ensures a much deeper statistical understanding of the data's underlying generating process, despite the availability of automated tools.
- **Business Costs > Pure Accuracy:** Standard metrics like accuracy fall short in predictive maintenance. Domain-derived cost matrices reflecting the real-world impact of false negatives versus false positives are paramount to successful model deployment.
