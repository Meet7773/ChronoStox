import pandas as pd
import numpy as np

INPUT = "ticker_sentiment_scores (1).parquet"
OUTPUT = "sentiment_clean.parquet"

def fix_sentiment_parquet():
    df = pd.read_parquet(INPUT)

    print("=== BEFORE CLEANING ===")
    print(df.head())
    print(df.dtypes)

    # ---------------------------
    # 1) Fix Ticker column
    # ---------------------------
    ticker_col = None
    for c in df.columns:
        if "ticker" in c.lower():
            ticker_col = c
            break

    if ticker_col is None:
        raise ValueError("No ticker column found")

    df = df.rename(columns={ticker_col: "Ticker_YF"})

    # drop missing tickers
    df = df[df["Ticker_YF"].notna()]

    # ensure uppercase + strip
    df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip().str.upper()

    # ensure .NS / .BO suffix
    df["Ticker_YF"] = df["Ticker_YF"].apply(
        lambda t: t if t.endswith((".NS", ".BO")) else t + ".NS"
    )

    # ---------------------------
    # 2) Fix Date column
    # ---------------------------
    if "Date" not in df.columns:
        # maybe index is date
        df = df.reset_index().rename(columns={"index": "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # remove timezone
    df["Date"] = df["Date"].dt.tz_localize(None)

    # ---------------------------
    # 3) Fix sentiment column
    # ---------------------------
    if "sentiment_score" not in df.columns:
        # attempt fallback
        for c in df.columns:
            if "sentiment" in c.lower():
                df["sentiment_score"] = df[c]
                break

    if "sentiment_score" not in df.columns:
        # create neutral
        df["sentiment_score"] = 0.0

    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

    # ---------------------------
    # 4) Keep final usable columns
    # ---------------------------
    df = df[["Date", "Ticker_YF", "sentiment_score"]]

    # drop duplicates
    df = df.drop_duplicates()

    df = df.sort_values(["Ticker_YF", "Date"]).reset_index(drop=True)

    print("\n=== AFTER CLEANING ===")
    print(df.head())
    print(df.dtypes)

    df.to_parquet(OUTPUT, index=False)
    print(f"\nSaved CLEANED parquet → {OUTPUT}")

fix_sentiment_parquet()
