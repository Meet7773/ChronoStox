# pages/Market_Overview.py

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="ChronoStox | Market Overview",
    page_icon="🌎",
    layout="wide"
)

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- Session State Initialization ---
if "username" not in st.session_state:
    st.session_state.username = "D3sTr0"
if "virtual_cash" not in st.session_state:
    st.session_state.virtual_cash = 100000.00
if "holdings" not in st.session_state:
    st.session_state.holdings = []

# --- Sidebar ---
with st.sidebar:
    st.header(f"Welcome, {st.session_state.username}! 👋")
    st.divider()
    st.page_link("Dashboard.py", label="Market Overview", icon="🌎")
    st.page_link("pages/Live_Market.py", label="Live Market", icon="📈")
    st.page_link("pages/ChronoTrade.py", label="ChronoTrade", icon="⏳")
    st.page_link("pages/My_Portfolio.py", label="My Portfolio", icon="💼")
    st.page_link("pages/Stock_Screener.py", label="Stock Screener", icon="🔍")
    st.divider()
    st.metric("💰 Virtual Cash", f"₹{st.session_state.virtual_cash:,.2f}")

# --- Title ---
st.title("🌎 Market Overview")
st.write("Stay updated with live snapshots of key Indian market indices.")

# --- Define Indices ---
indices = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
}

# --- Fetch Index Data ---
@st.cache_data(ttl=600)
def get_index_data(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1mo")
        if data.empty or len(data) < 2:
            return None, None, None, None
        last_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2]
        change = last_close - prev_close
        change_pct = (change / prev_close) * 100
        return data, last_close, change, change_pct
    except Exception:
        return None, None, None, None

# --- Display Indices in Grid (2 per row) ---
rows = list(indices.items())
for i in range(0, len(rows), 2):
    cols = st.columns(2, gap="large")
    for col, (name, ticker) in zip(cols, rows[i:i+2]):
        with col:
            data, last_close, change, change_pct = get_index_data(ticker)

            if last_close is not None:
                st.metric(
                    label=name,
                    value=f"{last_close:,.2f}",
                    delta=f"{change:,.2f} ({change_pct:.2f}%)",
                )

                fig = go.Figure(
                    go.Candlestick(
                        x=data.index,
                        open=data["Open"],
                        high=data["High"],
                        low=data["Low"],
                        close=data["Close"],
                        increasing_line_color="green",
                        decreasing_line_color="red",
                    )
                )
                fig.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=20),
                    xaxis_rangeslider_visible=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.metric(label=name, value="Data unavailable")

st.caption("📊 Index data is fetched from Yahoo Finance and may be delayed.")
