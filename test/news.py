import yfinance as yf
from datetime import datetime
import pandas as pd

# df = pd.read_csv('ticker.csv')
# cat_name = df['Category Name'].dropna().unique()
# print(cat_name, len(cat_name))


# ticker_lst = ["ONGC.NS"]
# ticker = yf.download(ticker_lst)
ticker = yf.Ticker('RELIANCE.NS')
# dt = ticker.financials
# print(dt)
news = ticker.news

print(news)
for i, item in enumerate(news, 1):
    content = item.get("content", {})

    print(f"\n📰 News {i}")
    print(f"Title      : {content.get('title')}")
    print(f"Summary    : {content.get('summary')}")
    print(f"Publisher  : {item.get('publisher')}")
    print(f"Type       : {content.get('contentType')}")
    print(f"Link       : {content.get('canonicalUrl', {}).get('url')}")

    # ---- HANDLE TIMESTAMPS ----
    pub_time = None

    # 1) providerPublishTime (UNIX timestamp)
    if "providerPublishTime" in item:
        try:
            pub_time = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
        except Exception:
            pass

    # 2) content.pubDate (ISO-string)
    if pub_time is None:
        pub_date_str = content.get("pubDate")
        if pub_date_str:
            try:
                pub_time = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except:
                pass

    # 3) Print result
    if pub_time:
        print("Published  :", pub_time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    else:
        print("Published  : Not available")
