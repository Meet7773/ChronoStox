import pandas as pd

# df = pd.read_parquet("macro_features.parquet")
# print(df.head())
# print(df.tail())
# print(df.columns)
# print(df.index)
# df[df["Date"] > "2025-11-25"][["Date"]].head()
# print(df.to_string())

df = pd.read_parquet("sentiment_clean.parquet")
print(df.head())
print(df.columns)
print(df.index)
print(df.tail())
df[df["Date"] > "2025-11-25"][["Date"]].head()
print(df.to_string())