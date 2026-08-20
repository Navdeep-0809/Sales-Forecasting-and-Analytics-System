import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Aggregate daily sales and create temporal, lag, and rolling window features.
    """
    # 1. Aggregate total sales by date
    daily_sales = df.groupby('order_date')['sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values('order_date').reset_index(drop=True)

    # 2. Extract Calendar Features
    daily_sales['year'] = daily_sales['order_date'].dt.year
    daily_sales['month'] = daily_sales['order_date'].dt.month
    daily_sales['day'] = daily_sales['order_date'].dt.day
    daily_sales['dayofweek'] = daily_sales['order_date'].dt.dayofweek
    daily_sales['quarter'] = daily_sales['order_date'].dt.quarter
    daily_sales['is_weekend'] = daily_sales['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Create Lag Features (Sales from previous days)
    daily_sales['sales_lag_1'] = daily_sales['sales'].shift(1)
    daily_sales['sales_lag_7'] = daily_sales['sales'].shift(7)
    daily_sales['sales_lag_30'] = daily_sales['sales'].shift(30)

    # 4. Create Rolling Statistics (Moving Averages)
    daily_sales['rolling_mean_7'] = daily_sales['sales'].shift(1).rolling(window=7).mean()
    daily_sales['rolling_mean_30'] = daily_sales['sales'].shift(1).rolling(window=30).mean()

    # Drop missing rows created by shifts/rolling windows
    daily_sales = daily_sales.dropna().reset_index(drop=True)

    return daily_sales

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv("data/cleaned_sales.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Build features
    features_df = create_time_features(df)

    # Save engineered dataset
    features_df.to_csv("data/featured_sales.csv", index=False)
    print("--- Feature Engineering Complete! ---")
    print(f"Dataset Shape: {features_df.shape}")
    print("\nFeature Columns Generated:")
    print(features_df.columns.tolist())
    print("\nFirst 3 rows:")
    print(features_df.head(3))