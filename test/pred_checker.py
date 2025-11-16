import pandas as pd
import yfinance as yf

ticker = "RELIANCE.NS"
as_of = pd.Timestamp("2025-11-07")
horizons = [5, 21, 63, 126, 252]

# Fetch enough future data
df = yf.download(ticker, start=as_of - pd.Timedelta(days=3),
                 end=as_of + pd.Timedelta(days=450))

df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Find AS_OF index
idx = df.index[df['Date'] == as_of]
if len(idx) == 0:
    raise ValueError("AS_OF not a trading day. Use previous trading day.")

start_idx = idx[0]

actuals = {}

for h in horizons:
    target_idx = start_idx + h  # FIX: correct trading-day alignment
    if target_idx < len(df):
        actuals[h] = (df.loc[target_idx, 'Date'], df.loc[target_idx, 'Close'])
    else:
        actuals[h] = ("Not available (future)", None)

print("Actual future close prices:")
for h in horizons:
    print(f"{h}-day → Date: {actuals[h][0]}, Close: {actuals[h][1]}")
