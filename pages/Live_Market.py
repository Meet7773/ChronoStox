# FILE: pages/Live_Market.py
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from utils.sidebar import render_sidebar

# Page config & style
st.set_page_config(page_title="Live Market", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Render centralized sidebar (always render)
render_sidebar()

# ---------------- Helpers & caching (no UI inside) ---------------------------
DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
TICKER_CSV_PATH = "data/ticker.csv"

@st.cache_data(show_spinner="Loading stock list...")
def load_tickers_from_csv(path: str = TICKER_CSV_PATH):
    """
    Load tickers from disk. NO STREAMLIT UI inside this function.
    Returns list of tickers or DEFAULT_TICKERS if file missing/invalid.
    """
    import pandas as pd
    import os

    if not os.path.exists(path):
        return DEFAULT_TICKERS

    try:
        df = pd.read_csv(path)
    except Exception:
        return DEFAULT_TICKERS

    if "Ticker" not in df.columns:
        return DEFAULT_TICKERS

    tickers = df["Ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    return tickers or DEFAULT_TICKERS

@st.cache_data(ttl=60 * 60)
def get_stock_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=60 * 10)
def fetch_history(ticker: str, days: int = 365) -> pd.DataFrame:
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if not df.empty and getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60 * 5)
def fetch_news_yf(ticker: str):
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []

# ---------------- Shared session-state defaults -----------------------------
if "username" not in st.session_state:
    st.session_state.username = "D3sTr0"
if "virtual_cash" not in st.session_state:
    st.session_state.virtual_cash = 100_000.00
if "holdings" not in st.session_state:
    st.session_state.holdings = {}
if "trades" not in st.session_state:
    st.session_state.trades = []
if "ticker" not in st.session_state:
    st.session_state.ticker = "RELIANCE.NS"
if "ticker_data" not in st.session_state:
    st.session_state.ticker_data = pd.DataFrame()

# ---------------- Sidebar: load tickers (no uploader) ------------------------
tickers = load_tickers_from_csv()

# Show warning if file missing or invalid (only informative)
if not os.path.exists(TICKER_CSV_PATH):
    st.sidebar.warning(f"Using default tickers because '{TICKER_CSV_PATH}' was not found. Place your ticker.csv at that path to use custom list.")

with st.sidebar:
    # pick ticker
    try:
        default_index = tickers.index(st.session_state.ticker) if st.session_state.ticker in tickers else 0
    except Exception:
        default_index = 0
    st.session_state.ticker = st.selectbox("Search for a Stock", options=tickers, index=default_index)

    if st.button("Fetch Data", use_container_width=True, type="primary"):
        with st.spinner(f"Fetching data for {st.session_state.ticker}..."):
            df = fetch_history(st.session_state.ticker, days=730)
            if df.empty:
                st.error("No historical data found for the selected ticker.")
            else:
                st.session_state.ticker_data = df
                st.success(f"Loaded data for {st.session_state.ticker}")

with st.sidebar:
    st.divider()
    st.metric(label="Virtual Cash", value=f"₹{st.session_state.virtual_cash:,.2f}")
    pos_total = sum([h.get('quantity', 0) for h in st.session_state.holdings.values()])
    st.caption(f"Total Positions: {pos_total}")

# ---------------- Main ------------------------------------------------------
if st.session_state.ticker_data.empty:
    st.info("Search a stock in the sidebar and click 'Fetch Data' to begin.")
else:
    ticker = st.session_state.ticker
    df = st.session_state.ticker_data
    info = get_stock_info(ticker)

    st.header(f"{info.get('longName', ticker)} — {ticker}")

    tab1, tab2, tab3 = st.tabs(["Price Chart & Trading", "Key Information", "Recent News"])

    # Tab 1: Chart & trading
    with tab1:
        candle = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
        vol = go.Bar(x=df.index, y=df['Volume'], yaxis='y2', opacity=0.3)
        fig = go.Figure(data=[candle, vol])
        fig.update_layout(xaxis_rangeslider_visible=False, height=520,
                          yaxis=dict(domain=[0.25, 1.0]), yaxis2=dict(domain=[0.0, 0.2], anchor='x'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trade Execution")
        current_price = float(df['Close'].iloc[-1])
        st.metric("Current Market Price", f"₹{current_price:,.2f}")

        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        est_cost = qty * current_price
        st.info(f"Estimated Cost: ₹{est_cost:,.2f}")

        buy_col, sell_col = st.columns(2)

        # BUY
        if buy_col.button("BUY", use_container_width=True):
            cost = est_cost
            if st.session_state.virtual_cash >= cost:
                h = st.session_state.holdings.get(ticker, {"quantity": 0, "avg_price": 0.0})
                new_qty = h['quantity'] + qty
                new_avg = ((h['avg_price'] * h['quantity']) + cost) / new_qty if new_qty else 0.0
                st.session_state.holdings[ticker] = {"quantity": new_qty, "avg_price": new_avg}
                st.session_state.virtual_cash -= cost
                st.session_state.trades.append({
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "action": "BUY",
                    "quantity": int(qty),
                    "price": round(current_price, 4),
                    "value": round(cost, 2),
                })
                st.success(f"Bought {qty} × {ticker} @ ₹{current_price:,.2f}")
            else:
                st.error("Not enough virtual cash.")

        # SELL
        if sell_col.button("SELL", use_container_width=True):
            h = st.session_state.holdings.get(ticker, {"quantity": 0, "avg_price": 0.0})
            if qty > h.get('quantity', 0):
                st.error("Not enough holdings to sell.")
            else:
                proceeds = qty * current_price
                h['quantity'] -= qty
                if h['quantity'] == 0:
                    h['avg_price'] = 0.0
                st.session_state.holdings[ticker] = h
                st.session_state.virtual_cash += proceeds
                st.session_state.trades.append({
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "action": "SELL",
                    "quantity": int(qty),
                    "price": round(current_price, 4),
                    "value": round(proceeds, 2),
                })
                st.success(f"Sold {qty} × {ticker} @ ₹{current_price:,.2f}")

        st.divider()
        h = st.session_state.holdings.get(ticker, {"quantity": 0, "avg_price": 0.0})
        st.subheader("Holdings Summary")
        holdings_df = pd.DataFrame([{
            "Ticker": ticker,
            "Quantity": h.get('quantity', 0),
            "Avg Price": round(h.get('avg_price', 0.0), 2),
            "Last Price": round(current_price, 2),
            "Unrealized P&L": round((current_price - h.get('avg_price', 0.0)) * h.get('quantity', 0), 2)
        }])
        st.dataframe(holdings_df, use_container_width=True)

    # Tab 2: Key Info
    with tab2:
        st.subheader("Company Profile")
        key_info = {
            "Market Cap": info.get('marketCap', 'N/A'),
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A'),
            "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
            "Forward P/E": info.get('forwardPE', 'N/A')
        }
        st.json(key_info)
        if info.get('longBusinessSummary'):
            st.markdown('#### Business Summary')
            st.write(info.get('longBusinessSummary'))

        st.divider()
        st.markdown("### My Trades")
        trades_df = pd.DataFrame(st.session_state.trades)
        if trades_df.empty:
            st.info("No trades yet.")
        else:
            st.dataframe(trades_df[::-1], use_container_width=True)
            csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Trades CSV", data=csv, file_name="live_trades.csv", mime='text/csv')

    # Tab 3: News
    with tab3:
        st.caption("Note: Yahoo finance news can be older or incomplete. Consider RSS for live feeds.")
        st.subheader("Financial News")

        news_list = fetch_news_yf(ticker)
        if news_list:
            max_articles = st.slider("Max articles", min_value=3, max_value=15, value=7)
            shown = 0
            for article in news_list:
                if shown >= max_articles:
                    break
                try:
                    content = article.get('content') or {}
                    title = content.get('title') or article.get('title') or 'No Title'
                    summary = content.get('summary') or article.get('summary') or ''
                    link = content.get('canonicalUrl', {}).get('url') or article.get('link') or '#'

                    thumbnail = content.get('thumbnail') if isinstance(content.get('thumbnail'), dict) else None
                    image_url = None
                    if thumbnail and 'resolutions' in thumbnail and len(thumbnail['resolutions']) > 0:
                        image_url = thumbnail['resolutions'][0].get('url')

                    published = article.get('providerPublishTime')
                    if published:
                        try:
                            published_str = datetime.fromtimestamp(published).strftime('%Y-%m-%d %H:%M')
                        except Exception:
                            published_str = 'Unknown Date'
                    else:
                        published_str = 'Unknown Date'

                    cols = st.columns([1, 4])
                    if image_url:
                        cols[0].image(image_url, use_container_width=True)
                    with cols[1]:
                        st.markdown(f"### [{title}]({link})")
                        st.caption(f"📰 {article.get('publisher', 'Unknown')} | 📅 {published_str}")
                        if summary:
                            st.write(summary)
                    st.divider()
                    shown += 1
                except Exception as e:
                    st.warning(f"Skipping malformed article: {e}")
        else:
            st.info("No recent news found for this ticker.")
