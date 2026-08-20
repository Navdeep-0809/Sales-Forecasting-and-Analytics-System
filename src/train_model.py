import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

def train_forecasting_model(data_path):
    # 1. Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # 2. Load Feature Dataset
    df = pd.read_csv(data_path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # 3. Define Features and Target
    feature_cols = [
        'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
        'rolling_mean_7', 'rolling_mean_30'
    ]
    target_col = 'sales'

    X = df[feature_cols]
    y = df[target_col]

    # 4. Time-based Train-Test Split (Last 20% for testing)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

    # 5. Train XGBoost Model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 6. Predict & Evaluate
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, a_min=0, a_max=None) # Ensure no negative sales predictions

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    # Avoid division by zero in MAPE by ignoring zero/near-zero sales days
    non_zero_mask = y_test > 1.0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
    else:
        mape = 0.0

    print("\n--- Model Evaluation ---")
    print(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
    print(f"MAE (Mean Absolute Error): ${mae:.2f}")
    print(f"Adjusted MAPE: {mape:.2f}%")

    # 7. Save Model
    model_path = "models/xgboost_sales_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to '{model_path}'!")

if __name__ == "__main__":
    train_forecasting_model("data/featured_sales.csv")