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
            header {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading stock list...")
def load_tickers():
    """Loads the list of Yahoo Finance tickers from a CSV file."""
    url = 'https://raw.githubusercontent.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis/refs/heads/master/Yahoo-Finance-Ticker-Symbols.csv'
    df = pd.read_csv(url)
    return df['Ticker'].tolist()


tickers = load_tickers()

st.title("📈 Live Market Trading")

if 'ticker' not in st.session_state: st.session_state.ticker = "RELIANCE.NS"
if 'ticker_data' not in st.session_state: st.session_state.ticker_data = pd.DataFrame()

with st.sidebar:
    st.title("ChronoStox")
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
    st.divider()
    st.page_link("Dashboard.py", label="Market Overview", icon="🌎")
    st.page_link("pages/My_Portfolio.py", label="My Portfolio", icon="💼")
    st.page_link("pages/ChronoTrade.py", label="ChronoTrade", icon="⏳")

if not st.session_state.ticker_data.empty:
    info = yf.Ticker(st.session_state.ticker).info
    st.header(f"{info.get('longName', st.session_state.ticker)}")
    st.subheader("Price Chart")
    fig = go.Figure(data=[
        go.Candlestick(x=st.session_state.ticker_data.index, open=st.session_state.ticker_data['Open'],
                       high=st.session_state.ticker_data['High'], low=st.session_state.ticker_data['Low'],
                       close=st.session_state.ticker_data['Close'])])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500,
                      title=f"{info.get('longName', st.session_state.ticker)} Candlestick Chart")
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Trade Execution")
    col1, col2 = st.columns([1, 2])
    with col1:
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
    with col2:
        st.markdown("#### Key Information")
        st.json({"Market Cap": info.get('marketCap', 'N/A'), "Sector": info.get('sector', 'N/A'),
                 "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
                 "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A')})
else:
    st.info("Search for a stock in the sidebar and click 'Fetch Data' to begin.")