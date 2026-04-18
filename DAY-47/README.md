# Day 47: Time Series Forecasting and Sensor Failure Prediction

## Overview
This assignment is implemented in `week08_monday_timeseries.ipynb` and focuses on two practical time-series problems:

- forecasting daily e-commerce sales using ARIMA and SARIMA
- predicting 24-hour equipment failure risk from warehouse sensor data

The notebook emphasizes correct time-series practice, especially avoiding random train/test splits and preserving temporal order during evaluation.

## Task Breakdown
The notebook is organized into six main sub-steps:

1. **E-Commerce Sales: Characterise and Test Stationarity**
   - explores a 2-year daily sales series
   - checks stationarity using ADF and KPSS tests
   - studies trend and seasonality before modeling

2. **Sensor Data: Identify and Fix Quality Issues**
   - works with hourly warehouse sensor readings
   - handles missing values, duplicate timestamps, and out-of-range spikes
   - prepares a clean time-indexed dataset for downstream modeling

3. **ARIMA Model: Fit, Evaluate, and Justify**
   - differences the sales series to achieve stationarity
   - uses ACF and PACF patterns to justify the ARIMA order
   - evaluates forecast quality on a time-based split

4. **SARIMA Model: Capture Seasonal Patterns**
   - extends ARIMA with weekly seasonality
   - fits a SARIMA model to improve seasonal forecasting performance
   - compares seasonal and non-seasonal approaches

5. **Sensor Failure Prediction (24-Hour Horizon)**
   - engineers rolling-window and cross-sensor features
   - trains a `RandomForestClassifier` with a temporal split
   - prioritizes recall because missed failures are more expensive than false alarms

6. **Rule vs ML: Cost Matrix Analysis**
   - compares a simple threshold rule against the ML model
   - converts false negatives and false positives into business cost
   - finds an alert threshold that minimizes expected fleet-wide daily loss

## Solution Highlights

### Sales Forecasting
- Uses `adfuller`, `kpss`, `acf`, and `pacf` from `statsmodels` for stationarity and lag analysis.
- Fits an `ARIMA(1,1,1)` model after first differencing.
- Fits a seasonal `SARIMA(1,1,1)(1,1,1,7)` model to capture weekly patterns.
- Evaluates forecasts with regression-style error metrics such as MAE, RMSE, and MAPE.

### Sensor Pipeline
- Generates synthetic sensor data matching the assignment scenario.
- Cleans quality issues with sorting, deduplication, forward fill, interpolation, and backfill.
- Builds rolling statistics and interaction features for failure prediction.
- Trains and evaluates a `RandomForestClassifier` for next-24-hour failure risk.

### Decision Support
- Compares rule-based monitoring with ML-based prediction.
- Uses an asymmetric cost setup:
  - missed failure: `Rs 50,000`
  - false alarm: `Rs 5,000`
- Optimizes the operating threshold using business cost instead of only F1 score.

## Key Takeaways
- Time-series data must be split chronologically, not randomly.
- Stationarity checks are essential before fitting ARIMA-style models.
- Seasonal effects can materially improve forecasts when modeled explicitly with SARIMA.
- Real sensor pipelines require strong data cleaning before modeling.
- In operations problems, the best threshold is often the one that minimizes business cost, not the one with the best generic ML metric.

## Notes
- The notebook creates synthetic datasets, so it can run without external LMS files.
- Main libraries used include `pandas`, `numpy`, `statsmodels`, `scikit-learn`, and `matplotlib`.
- The implementation also includes an AI usage documentation section inside the notebook.
