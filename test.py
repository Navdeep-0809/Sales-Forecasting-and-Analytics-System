import pandas as pd
df = pd.read_csv("Data/train.csv")
print(df.shape)
print(df.duplicated().sum())
print(df.isnull().sum())