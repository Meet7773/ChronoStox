import yfinance as yf
from datetime import datetime

ticker = yf.Ticker("RELIANCE.NS")
news = ticker.news

print(news)
# for i, item in enumerate(news, 1):
#     content = item.get("content", {})
#     print(f"\n📰 News {i}")
#     print(f"Title      : {content.get('title')}")
#     print(f"Summary    : {content.get('summary')}")
#     print(f"Publisher  : {item.get('publisher')}")
#     print(f"Type       : {content.get('contentType')}")
#     print(f"Link       : {content.get('canonicalUrl', {}).get('url')}")
#
#     # If image available
#     thumbnail = content.get("thumbnail", {}).get("resolutions", [])
#     if thumbnail:
#         print(f"Image      : {thumbnail[0].get('url')}")
#
#     # Convert publish time
#     if "providerPublishTime" in item:
#         ts = datetime.fromtimestamp(item["providerPublishTime"])
#         print(f"Published  : {ts.strftime('%Y-%m-%d %H:%M:%S')}")
