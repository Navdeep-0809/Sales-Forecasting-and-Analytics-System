# Sales Forecasting and Analytics System

An interactive sales analytics dashboard and forecasting application built with Python, Streamlit, and XGBoost. It analyzes historical sales data, displays model predictions, and supports scenario planning and future revenue projections.

## Features

- **Sales data ingestion:** Use the included cleaned dataset or upload a custom CSV.
- **Flexible CSV mapping:** Normalize column names; detect common sales/revenue columns or calculate sales from `units_sold` and `unit_price`; parse mixed date formats.
- **Sales analytics dashboard:** View total revenue, average daily revenue, predicted revenue, and variance for the selected date range.
- **Category and date filtering:** Filter dashboard results by product category and historical date range.
- **Time-series features:** Aggregate sales by day and generate calendar fields, 1/7/30-day sales lags, and 7/30-day rolling means.
- **Revenue forecasting:** Apply the pre-trained XGBoost model to historical feature data and compare actual sales with model predictions.
- **30-day forecast:** Generate an interactive projection for the 30 days after the latest available sales date.
- **What-if scenarios:** Adjust the promotional boost slider to see its effect on predicted sales.
- **Inventory demand alerts:** Compare projected sales with the historical average and show high-demand, low-demand, or stable-demand guidance.
- **Model training workflow:** Prepare data, engineer time-series features, train an XGBoost regressor, and evaluate it with RMSE, MAE, and MAPE.

## Tech Stack

- **Python** — application and data-processing language
- **Streamlit** — interactive web dashboard
- **Pandas and NumPy** — CSV handling, aggregation, feature engineering, and numerical operations
- **XGBoost** — sales regression and forecasting model
- **scikit-learn** — model evaluation metrics during training
- **Joblib** — serialization and loading of the trained model
- **Plotly Express and Plotly Graph Objects** — interactive charts

## Run the application

Python 3.11 is used by the included development-container configuration.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (by default, `http://localhost:8501`).

The dashboard loads `Data/cleaned_sales.csv` and `models/xgboost_sales_model.pkl` by default. To use a different dataset, upload a CSV from the sidebar. It must include an `order_date` column and a sales/revenue column, or both `units_sold` and `unit_price`.

## Data and model workflow

The scripts under `src/` provide the data preparation and training workflow:

1. `preprocess.py` standardizes column names, parses dates, sorts records, and fills missing numeric values.
2. `feature_engineering.py` aggregates daily sales and creates calendar, lag, and rolling-average features.
3. `train_model.py` trains and evaluates an XGBoost regression model and saves it to `models/xgboost_sales_model.pkl`.

The training script expects the corresponding cleaned and featured CSV files to be present under `Data/`.

## Repository structure

```text
Data/
  train.csv
  cleaned_sales.csv
  featured_sales.csv
models/
  xgboost_sales_model.pkl
src/
  preprocess.py
  feature_engineering.py
  train_model.py
app.py
requirements.txt
README.md
```
