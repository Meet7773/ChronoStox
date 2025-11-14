import streamlit as st
import requests
import pandas as pd
from datetime import datetime

from utils.sidebar import render_sidebar
from utils.auth import require_login

# --- Config ---
st.set_page_config(
    page_title="ChronoStox | My Portfolio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Render sidebar
user_id = require_login()
render_sidebar(show_cash=False)  # Show cash on Portfolio page

API_URL = "http://127.0.0.1:8000"


# --- API Fetching ---
@st.cache_data(ttl=60)
def fetch_portfolio(user_id: str):
    try:
        res = requests.get(f"{API_URL}/portfolio/{user_id}")
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        st.error("Could not fetch portfolio. Is the backend API running?")
        return None


@st.cache_data(ttl=120)
def fetch_stock_quote(ticker):
    """Fetches full quote for portfolio enrichment."""
    try:
        res = requests.get(f"{API_URL}/stock/{ticker}")
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=300)
def fetch_history(ticker, period="1mo"):
    """Fetches history for sparkline."""
    try:
        res = requests.get(f"{API_URL}/history/{ticker}", params={"period": period})
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return []


def execute_trade(ticker, quantity, action):
    """Execute trade using current market price from API."""
    try:
        # Price will be fetched automatically by API
        payload = {
            "userId": user_id,
            "ticker": ticker.upper(),
            "quantity": quantity,
            "action": action.upper()
            # Price not provided - API will fetch current market price
        }
        res = requests.post(f"{API_URL}/trade", json=payload)
        res.raise_for_status()
        result = res.json()
        # Clear caches after a successful trade
        st.cache_data.clear()
        st.success(result.get("message", "Trade executed!"))
        # Force a rerun to show new data immediately
        st.rerun()
    except requests.exceptions.RequestException as e:
        error_detail = "Unknown error"
        try:
            if hasattr(e, 'response') and e.response is not None:
                error_detail = e.response.json().get("detail", "Unknown error")
        except:
            pass
        st.error(f"Trade Failed: {error_detail}")


# --- Helpers ---
def format_inr(value):
    return f"₹{value:,.2f}"


# --- Main Page ---
st.title("My Portfolio")

portfolio_data = fetch_portfolio(user_id)

if not portfolio_data:
    st.stop()

# --- Enrich Portfolio Data ---
enriched_rows = []
total_invested = 0
total_current_value = 0

with st.spinner("Updating holdings..."):
    for holding in portfolio_data.get("holdings", []):
        quote = fetch_stock_quote(holding["ticker"])

        if not quote or not quote.get("currentPrice"):
            # Fallback if API fails for one ticker
            current_price = holding["avgPrice"]
            history = []
        else:
            current_price = quote["currentPrice"]
            history_data = fetch_history(holding["ticker"])
            history = [h['close'] for h in history_data]  # Just the close prices for sparkline

        invested_value = holding["avgPrice"] * holding["quantity"]
        current_value = current_price * holding["quantity"]
        pl_value = current_value - invested_value
        pl_percent = (pl_value / invested_value) * 100 if invested_value > 0 else 0

        enriched_rows.append({
            "Ticker": holding["ticker"],
            "Qty": holding["quantity"],
            "Avg": holding["avgPrice"],
            "LTP": current_price,
            "Value": current_value,
            "P&L": pl_value,
            "P&L %": pl_percent,
            "Trend": history
        })

        total_invested += invested_value
        total_current_value += current_value

total_pl = total_current_value - total_invested
total_equity = total_current_value + portfolio_data.get("virtualCash", 0)

# --- Display KPI Cards ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Equity", format_inr(total_equity))
col2.metric("Holdings Value", format_inr(total_current_value))
col3.metric("Virtual Cash", format_inr(portfolio_data.get("virtualCash", 0)))
st.divider()

# --- Display Trade Widget & Holdings ---
grid_col1, grid_col2 = st.columns([1, 2])  # 1/3 for trade, 2/3 for table

with grid_col1:
    st.subheader("Execute Trade")
    with st.form("trade_form"):
        trade_ticker = st.text_input("Ticker Symbol (e.g., RELIANCE.NS)")
        trade_qty = st.number_input("Quantity", min_value=1, step=1)

        form_col1, form_col2 = st.columns(2)
        with form_col1:
            buy_button = st.form_submit_button("BUY", use_container_width=True)
        with form_col2:
            sell_button = st.form_submit_button("SELL", use_container_width=True)

        if buy_button:
            if trade_ticker and trade_qty > 0:
                execute_trade(trade_ticker.upper(), trade_qty, "BUY")
            else:
                st.warning("Please enter a valid ticker and quantity.")

        if sell_button:
            if trade_ticker and trade_qty > 0:
                execute_trade(trade_ticker.upper(), trade_qty, "SELL")
            else:
                st.warning("Please enter a valid ticker and quantity.")

with grid_col2:
    st.subheader(f"Holdings ({len(enriched_rows)})")

    if not enriched_rows:
        st.info("You have no holdings. Buy assets using the trade widget.")
    else:
        df = pd.DataFrame(enriched_rows)

        # Format for display
        df_display = df.style.format({
            "Avg": format_inr,
            "LTP": format_inr,
            "Value": format_inr,
            "P&L": format_inr,
            "P&L %": "{:,.2f}%".format
        }).map(
            lambda v: "color: #22c55e;" if v > 0 else "color: #ef4444;", subset=["P&L", "P&L %"]
        )

        st.dataframe(
            df_display,
            column_config={
                "Trend": st.column_config.LineChartColumn(
                    "Trend (1mo)",
                    width="medium"
                )
            },
            use_container_width=True,
            hide_index=True
        )