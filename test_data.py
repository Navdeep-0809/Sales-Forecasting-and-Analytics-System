import pandas as pd

# Load the dataset
data_path = "data/train.csv"
df = pd.read_csv(data_path)

print("--- Data Successfully Loaded! ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}\n")

print("--- Column Names ---")
print(df.columns.tolist())

print("\n--- First 3 Rows ---")
print(df.head(3))