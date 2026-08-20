# 📈 Enterprise AI Sales Forecasting & Analytics System

An end-to-end Machine Learning web application built with **Python**, **Streamlit**, and **XGBoost**. This system converts raw enterprise transaction logs into real-time predictive revenue insights, future demand projections, interactive promotional "What-If" simulations, and automated inventory reorder alerts.

---

## 🌟 Key Features

* **Dynamic Data Ingestion Engine:** Upload any custom sales CSV file. The system automatically standardizes column headers, handles mixed date formats on the fly, and computes lag and rolling temporal features dynamically.
* **Interactive Filtering & Analytics:** Filter revenue metrics, average daily sales, and line trends dynamically by date range and product categories.
* **XGBoost Forecasting Model:** Uses pre-engineered lag features (`sales_lag_1`, `sales_lag_7`, `sales_lag_30`) and rolling averages (`rolling_mean_7`, `rolling_mean_30`) to forecast future demand accurately.
* **🔮 30-Day Automated Future Projection Engine:** Generates interactive multi-day future trendlines and computes projected total revenue for the upcoming month.
* **🎯 What-If Scenario Planner:** Simulate promotional lifts and marketing boosts in real time using interactive UI sliders to project expected revenue variance.
* **⚠️ Automated Inventory Alert System:** Evaluates predicted revenue against baseline averages to flag inventory stockout warnings or low-demand promotion windows.

---

## 📁 Repository Structure

text
├── data/
│   ├── cleaned_sales.csv            # Default dataset fallback
│   └── train.csv                    # Training dataset
├── models/
│   └── xgboost_sales_model.pkl      # Pre-trained XGBoost regression model
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Application dependencies
├── .gitignore                       # System and environment file exclusions
└── README.md                        # Project documentation
