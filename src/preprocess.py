import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    # 1. Load Data
    df = pd.read_csv(file_path)
    print("Initial Data Shape:", df.shape)

    # 2. Clean Column Names (Strip spaces & convert to lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # 3. Convert Order Date to Datetime
    # Note: Adjust format if your CSV uses DD/MM/YYYY or YYYY-MM-DD
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    
    # 4. Sort chronologically by date (Crucial for time-series!)
    df = df.sort_values('order_date').reset_index(drop=True)

    # 5. Handle Missing Values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Fill missing numeric values with 0 or median if any exist
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print("\nCleaned Data Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    
    return df

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("data/train.csv")
    
    # Save cleaned version to data folder
    cleaned_df.to_csv("data/cleaned_sales.csv", index=False)
    print("\nCleaned data saved to 'data/cleaned_sales.csv'!")