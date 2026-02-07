import pandas as pd

df = pd.read_csv('accessibility.csv')

print(df.head())
print(df.columns)
print(df.isna().sum())

