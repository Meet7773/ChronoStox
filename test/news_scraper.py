import time

import yfinance as yf
from datetime import datetime, timezone
import pandas as pd
import os

DEBUG = False
DAYS = 14
DEBUG_TICKERS = ["RELIANCE.NS", "TCS.NS", "ONGC.NS"]
OUTPUT_CSV = "news_dataset.csv"

def extract_timestamp(item):
    """
    Extracts timestamp from Yahoo Finance news objects.
    Supports both:
      - providerPublishTime (UNIX)
      - content.pubDate (ISO string)
    Returns a datetime in UTC or None.
    """
    # Case 1: UNIX timestamp
    ts = item.get("providerPublishTime")
    if ts:
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass

    # Case 2: ISO timestamp inside content
    content = item.get("content", {})
    pub_date = content.get("pubDate")
    if pub_date:
        try:
            return datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        except Exception:
            pass

    return None  # No valid timestamp found


def run_scraper():
    tickers = DEBUG_TICKERS if DEBUG else pd.read_csv("tickers.csv")["Ticker"].tolist()

    final_rows = []

    for ticker in tickers:
        print(f"\nFetching news for: {ticker}")

        news_items = yf.Ticker(ticker).news or []
        print(f"  Found {len(news_items)} items")

        for item in news_items:
            ts = extract_timestamp(item)

            if ts is None:
                print("    ❌ No timestamp (skipped)")
                continue

            # Filter only last N days
            age_days = (datetime.now(timezone.utc) - ts).days
            if age_days > DAYS:
                continue

            content = item.get("content", {})

            # Relevance filter: skip generic articles Yahoo attaches to small tickers
            ticker_name = ticker.replace(".NS", "").replace(".BO", "").lower()
            article_text = (content.get("title", "") + " " + content.get("summary", "")).lower()
            if ticker_name not in article_text and len(ticker_name) > 2:
                continue

            row = {
                "Ticker": ticker,
                "PublishedUTC": ts.isoformat(),
                "Title": content.get("title", ""),
                "Summary": content.get("summary", ""),
                "URL": content.get("canonicalUrl", {}).get("url", "")
            }

            if DEBUG:
                print(f"    📰 {row['PublishedUTC']}  |  {row['Title']}")

            final_rows.append(row)
        time.sleep(1.6)

    df = pd.DataFrame(final_rows)
    print(f"\nNew articles scraped: {len(df)}")
    print(df.head(10))

    # Append to existing CSV instead of overwriting
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([df_existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["Ticker", "PublishedUTC", "Title"], keep="last")
        print(f"Merged with existing data. Total rows: {len(df)}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")

    return df


if __name__ == "__main__":
    run_scraper()
