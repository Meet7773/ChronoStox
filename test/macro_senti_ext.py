import pandas as pd

# df = pd.read_parquet("macro_features.parquet")
# print(df.head())
# print(df.tail())
# print(df.columns)
# print(df.index)

df = pd.read_parquet("ticker_sentiment_scores (1).parquet")
print(df.head())
print(df.columns)
print(df.index)
print(df.tail())
