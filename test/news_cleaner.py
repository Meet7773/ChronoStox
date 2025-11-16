import pandas as pd

df = pd.read_csv("news_dataset.csv")

# 1) Normalize tickers (remove .NS/.BO)
df["TickerNorm"] = df["Ticker"].str.replace(".NS", "", regex=False)\
                                .str.replace(".BO", "", regex=False)\
                                .str.upper()

# 2) Remove exact duplicates (common for NSE/BSE)
df = df.drop_duplicates(subset=["TickerNorm", "PublishedUTC", "Title"])

# 3) Standardize datetime
df["PublishedUTC"] = pd.to_datetime(df["PublishedUTC"], errors="coerce")
df = df.dropna(subset=["PublishedUTC"])

# 4) Combine Title + Summary
df["text"] = df["Title"].fillna("") + " " + df["Summary"].fillna("")

# 5) Save cleaned file
df.to_csv("news_dataset_clean.csv", index=False)
print("Saved → news_dataset_clean.csv")
print(df.head())
print("Total rows:", len(df))
