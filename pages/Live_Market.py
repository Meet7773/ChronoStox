import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Live Market", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            div[data-testid="stSidebarNav"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading stock list...")
def load_tickers():
    """UPDATED: Loads tickers from your local ticker.csv file."""
    try:
        df = pd.read_csv('data/ticker.csv')
        # Ensure the 'Ticker' column exists
        if 'Ticker' in df.columns:
            return df['Ticker'].dropna().tolist()
        else:
            st.error("The 'Ticker' column was not found in your ticker.csv file.")
            return []
    except FileNotFoundError:
        st.error("ticker.csv not found. Please place it in the root project directory.")
        return []


@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


@st.cache_data(ttl=600)
def get_stock_news(ticker):
    try:
        return yf.Ticker(ticker).news
    except Exception:
        return []


tickers = load_tickers()

st.title("📈 Live Market Trading")
with st.sidebar:
    st.title("ChronoStox")
    st.divider()
    st.page_link("Dashboard.py", label="Market Overview", icon="🌎")
    st.page_link("pages/My_Portfolio.py", label="My Portfolio", icon="💼")
    st.page_link("pages/ChronoTrade.py", label="ChronoTrade", icon="⏳")
    st.page_link("pages/Stock_Screener.py", label="Stock Screener", icon="🔍")
    st.divider()

if 'ticker' not in st.session_state: st.session_state.ticker = "RELIANCE.NS"
if 'ticker_data' not in st.session_state: st.session_state.ticker_data = pd.DataFrame()

with st.sidebar:
    st.header("Stock Selection")

    # UPDATED: Use a selectbox for search with autocomplete
    st.session_state.ticker = st.selectbox(
        "Search for a Stock",
        options=tickers,
        index=tickers.index(st.session_state.ticker) if st.session_state.ticker in tickers else 0
    )

    if st.button("Fetch Data", use_container_width=True, type="primary"):
        try:
            stock = yf.Ticker(st.session_state.ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            st.session_state.ticker_data = stock.history(start=start_date, end=end_date)
            if st.session_state.ticker_data.empty:
                st.error("No data found.")
            else:
                st.success(f"Data loaded for {st.session_state.ticker}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if not st.session_state.ticker_data.empty:
    info = get_stock_info(st.session_state.ticker)
    st.header(f"{info.get('longName', st.session_state.ticker)}")

    tab1, tab2, tab3 = st.tabs(["Price Chart & Trading", "Key Information", "Recent News"])

    with tab1:
        fig = go.Figure(data=[
            go.Candlestick(x=st.session_state.ticker_data.index, open=st.session_state.ticker_data['Open'],
                           high=st.session_state.ticker_data['High'], low=st.session_state.ticker_data['Low'],
                           close=st.session_state.ticker_data['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False, height=500,
                          title=f"{info.get('longName', st.session_state.ticker)} Candlestick Chart")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Trade Execution")
        current_price = st.session_state.ticker_data['Close'].iloc[-1]
        st.metric("Current Market Price", f"₹{current_price:,.2f}")
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        estimated_cost = current_price * quantity
        st.info(f"Estimated Cost: ₹{estimated_cost:,.2f}")
        buy_col, sell_col = st.columns(2)
        if buy_col.button("BUY", use_container_width=True):
            if st.session_state.virtual_cash >= estimated_cost:
                st.session_state.virtual_cash -= estimated_cost
                st.success(f"Successfully bought {quantity} shares of {st.session_state.ticker}!")
                st.rerun()
            else:
                st.error("Not enough virtual cash.")
        if sell_col.button("SELL", use_container_width=True):
            st.success(f"Successfully sold {quantity} shares of {st.session_state.ticker}!")

    with tab2:
        st.subheader("Company Profile")
        st.json({"Market Cap": info.get('marketCap', 'N/A'), "Sector": info.get('sector', 'N/A'),
                 "Industry": info.get('industry', 'N/A'), "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
                 "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'), "Forward P/E": info.get('forwardPE', 'N/A')})
        if info.get('longBusinessSummary'):
            st.markdown("#### Business Summary")
            st.write(info.get('longBusinessSummary'))

    with tab3:
        st.subheader("Latest News")
        try:
            news_list = get_stock_news(st.session_state.ticker)
        except Exception as e:
            st.error(f"Could not fetch news: {e}")
            news_list = []

        if news_list:
            for article in news_list:
                try:
                    content = article.get("content") or {}
                    title = content.get("title") or "No Title"
                    summary = content.get("summary") or "No summary available."
                    link = (
                            content.get("canonicalUrl", {}).get("url")
                            or article.get("link")
                            or "#"
                    )
                    publisher = article.get("publisher") or "Unknown"
                    published_time = article.get("providerPublishTime")
                    if published_time:
                        try:
                            published_str = datetime.fromtimestamp(
                                published_time
                            ).strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            published_str = "Unknown Date"
                    else:
                        published_str = "Unknown Date"

                    # Handle thumbnail safely
                    image_url = None
                    thumbnail = content.get("thumbnail")
                    if (
                            thumbnail
                            and isinstance(thumbnail, dict)
                            and "resolutions" in thumbnail
                            and len(thumbnail["resolutions"]) > 0
                    ):
                        image_url = thumbnail["resolutions"][0].get("url")

                    with st.container():
                        cols = st.columns([1, 4])
                        if image_url:
                            cols[0].image(image_url, use_container_width=True)
                        with cols[1]:
                            st.markdown(f"### [{title}]({link})")
                            st.caption(f"📰 {publisher} | 📅 {published_str}")
                            if summary:
                                st.write(summary)
                        st.divider()
                except Exception as inner_e:
                    st.warning(f"Skipping an article due to error: {inner_e}")
        else:
            st.info("No recent news found for this ticker.")
else:
    st.info("Search for a stock in the sidebar and click 'Fetch Data' to begin.")
