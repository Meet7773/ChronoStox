import pandas as pd

df = pd.read_parquet('all_ticker_data.parquet')
# print(df.sort_index(by='date', ascending=False))
df.index = df.index.tz_localize(None)
df = df.sort_index(ascending=True)
print(df)