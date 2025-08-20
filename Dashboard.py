import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="ChronoStox | Market Overview",
    page_icon="🌎",
    layout="wide"
)

# --- HIDE STREAMLIT STYLE ---
# This injects CSS to hide the default Streamlit main menu and footer
# and the auto-generated page navigation list.
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            div[data-testid="stSidebarNav"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🌎 Market Overview")
st.write("A real-time snapshot of key market indices.")

# --- Session State Initialization (same as before) ---
if 'username' not in st.session_state:
    st.session_state.username = "D3sTr0"
if 'virtual_cash' not in st.session_state:
    st.session_state.virtual_cash = 100000.00
if 'holdings' not in st.session_state:
    st.session_state.holdings = [
        {"ticker": "RELIANCE.NS", "quantity": 20, "avg_price": 2800.00},
        {"ticker": "TCS.NS", "quantity": 10, "avg_price": 3850.00},
    ]

# --- Sidebar ---
with st.sidebar:
    st.header(f"Welcome, {st.session_state.username}!")
    st.divider()
    # Note: These links will work once you rename the other files
    st.page_link("pages/Live_Market.py", label="Live Market", icon="📈")
    st.page_link("pages/ChronoTrade.py", label="ChronoTrade", icon="⏳")
    st.page_link("pages/My_Portfolio.py", label="My Folio", icon="💼")
    st.page_link("pages/Stock_Screener.py", label="Stock Screener", icon="🔍")
    st.divider()
    st.metric(label="Virtual Cash", value=f"₹{st.session_state.virtual_cash:,.2f}")


# --- Define Indices ---
indices = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT"
}


# --- Function to fetch and display index data ---
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def get_index_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1mo")  # Fetch 1 month of data for the chart
        info = yf.Ticker(ticker).info

        if data.empty:
            return None, None, None, None

        last_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = last_close - prev_close
        change_pct = (change / prev_close) * 100
        return data, last_close, change, change_pct
    except Exception:
        return None, None, None, None


# --- Display Indices in a Grid ---
cols = st.columns(2)
col_idx = 0

for name, ticker in indices.items():
    with cols[col_idx]:
        data, last_close, change, change_pct = get_index_data(ticker)

        if last_close is not None:
            st.metric(
                label=name,
                value=f"{last_close:,.2f}",
                delta=f"{change:,.2f} ({change_pct:.2f}%)"
            )

            # --- Charting ---
            fig = go.Figure(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )
            )
            # Style it to be compact
            fig.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=0, b=20),
                xaxis_rangeslider_visible=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.metric(label=name, value="Data unavailable")

    col_idx = (col_idx + 1) % 2

st.caption("Index data is fetched from Yahoo Finance and may be delayed.")

