# FILE: pages/2_⏳_ChronoTrade.py
# DESC: ChronoTrade — simulate trades inside historical market scenarios with a timeline, P&L, and exportable trade log.
# NOTE: Uses data/ticker.csv (no uploader). Sidebar is centralized via utils.sidebar.render_sidebar()

import os
from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from utils.sidebar import render_sidebar

# ----------------------------- Page Configuration -----------------------------
st.set_page_config(
    page_title="ChronoStox | ChronoTrade",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit chrome
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Render centralized sidebar
render_sidebar()

# ----------------------------- Helpers & Cache --------------------------------
DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
TICKER_CSV_PATH = "data/ticker.csv"

@st.cache_data(show_spinner="Loading stock list...")
def load_tickers_from_csv(path: str = TICKER_CSV_PATH):
    import pandas as pd
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

@st.cache_data(ttl=60*60)
def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if not df.empty and getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# ----------------------------- Scenarios --------------------------------------
SCENARIOS = {
    "2008 Financial Crisis": {
        "start": "2007-10-01",
        "end": "2009-04-01",
        "default_ticker": "ICICIBANK.NS",
        "description": "Trade through the credit crunch and global deleveraging shock.",
    },
    "COVID-19 Crash": {
        "start": "2020-01-01",
        "end": "2020-06-01",
        "default_ticker": "RELIANCE.NS",
        "description": "Navigate the volatility during the early pandemic months.",
    },
    "Dot-Com Bubble Aftermath": {
        "start": "1999-01-01",
        "end": "2001-12-31",
        "default_ticker": "INFY.NS",
        "description": "Experience the rollercoaster of early IT giants.",
    },
}

# ----------------------------- Session State ----------------------------------
if "username" not in st.session_state:
    st.session_state.username = "D3sTr0"
if "virtual_cash" not in st.session_state:
    st.session_state.virtual_cash = 100_000.00
if "holdings" not in st.session_state:
    # dict: {ticker: {quantity, avg_price}}
    st.session_state.holdings = {}
if "trades" not in st.session_state:
    st.session_state.trades = []  # list of dicts
if "chrono_ticker_data" not in st.session_state:
    st.session_state.chrono_ticker_data = pd.DataFrame()
if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = None
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None
# store last-known prices per ticker to value full portfolio
if "prices" not in st.session_state:
    st.session_state.prices = {}  # e.g. { "RELIANCE.NS": 2893.50 }

# ----------------------------- Sidebar ----------------------------------------
tickers = load_tickers_from_csv()
if not os.path.exists(TICKER_CSV_PATH):
    st.sidebar.warning(f"Using default tickers because '{TICKER_CSV_PATH}' was not found. Place your ticker.csv at that path to use custom list.")

with st.sidebar:
    st.header("Scenario Selection")

with st.sidebar:
    scenario_name = st.selectbox("Choose a Historical Event", options=list(SCENARIOS.keys()))
    scenario = SCENARIOS[scenario_name]
    st.info(scenario["description"])

    use_picker = st.checkbox("Pick from list", value=True)
    if use_picker:
        try:
            idx = tickers.index(scenario["default_ticker"]) if scenario["default_ticker"] in tickers else 0
        except Exception:
            idx = 0
        ticker = st.selectbox("Select a Ticker", options=tickers, index=idx)
    else:
        ticker = st.text_input("Enter a Stock Ticker", value=scenario["default_ticker"]).upper().strip()

    if st.button("Load Scenario Data", use_container_width=True, type="primary"):
        with st.spinner(f"Fetching {ticker} from {scenario['start']} to {scenario['end']}..."):
            data = fetch_history(ticker, scenario["start"], scenario["end"])
            if data.empty:
                st.error("No data found for this ticker in the selected period.")
            else:
                st.session_state.chrono_ticker_data = data
                st.session_state.sim_idx = len(data) - 1
                st.session_state.current_ticker = ticker
                st.session_state.current_scenario = scenario_name
                # store last-known price for this ticker
                try:
                    last_px = float(data["Close"].iloc[-1])
                    st.session_state.prices[ticker] = last_px
                except Exception:
                    pass
                st.success(f"Scenario loaded for {ticker}!")

with st.sidebar:
    st.divider()
    st.metric(label="Virtual Cash", value=f"₹{st.session_state.virtual_cash:,.2f}")

# ----------------------------- Trading Engine ---------------------------------
def execute_trade_record(ticker: str, action: str, quantity: int, price: float, dt: pd.Timestamp, scenario: str = None):
    """
    Record trade and update holdings & virtual cash.
    """
    if quantity <= 0:
        st.error("Quantity must be positive.")
        return

    h = st.session_state.holdings.get(ticker, {"quantity": 0, "avg_price": 0.0})

    if action == "BUY":
        cost = price * quantity
        if st.session_state.virtual_cash < cost:
            st.error("Not enough virtual cash.")
            return
        new_qty = h["quantity"] + quantity
        new_avg = ((h["avg_price"] * h["quantity"]) + cost) / new_qty if new_qty > 0 else 0.0
        h.update({"quantity": new_qty, "avg_price": new_avg})
        st.session_state.holdings[ticker] = h
        st.session_state.virtual_cash -= cost
        st.success(f"Bought {quantity} × {ticker} @ ₹{price:,.2f}")

    elif action == "SELL":
        if quantity > h["quantity"]:
            st.error("You don't have enough shares to sell.")
            return
        proceeds = price * quantity
        h.update({"quantity": h["quantity"] - quantity})
        if h["quantity"] == 0:
            h["avg_price"] = 0.0
        st.session_state.holdings[ticker] = h
        st.session_state.virtual_cash += proceeds
        st.success(f"Sold {quantity} × {ticker} @ ₹{price:,.2f}")

    else:
        st.error("Unknown action.")
        return

    st.session_state.trades.append({
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": action,
        "quantity": int(quantity),
        "price": round(price, 4),
        "value": round(price * quantity, 2),
        "scenario": scenario or st.session_state.current_scenario or "-",
    })

# ----------------------------- Main Content -----------------------------------
df = st.session_state.chrono_ticker_data
if not df.empty and st.session_state.current_ticker:
    ticker = st.session_state.current_ticker
    scenario_name = st.session_state.current_scenario

    st.header(f"Trading: {ticker}")
    st.subheader(f"Scenario: {scenario_name}")

    dates = df.index.to_pydatetime().tolist()
    if st.session_state.sim_idx is None:
        st.session_state.sim_idx = len(dates) - 1

    sim_idx = st.slider(
        "Simulation Date",
        min_value=0,
        max_value=len(dates) - 1,
        value=st.session_state.sim_idx,
        step=1,
        key="sim_idx_slider",
        help="Drag to move through the scenario timeline. Trades execute at the selected day's close.",
    )
    st.session_state.sim_idx = sim_idx
    sim_dt = dates[sim_idx]

    # Use the close of sim day as execution price
    current_price = float(df.iloc[sim_idx]["Close"])
    # update last-known price for this ticker (keeps portfolio valuation meaningful)
    st.session_state.prices[ticker] = current_price

    # Chart (candlestick + volume + marker)
    candle = go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")
    vol_bar = go.Bar(x=df.index, y=df["Volume"], name="Volume", yaxis="y2", opacity=0.3)
    vline = go.Scatter(x=[sim_dt, sim_dt], y=[df["Low"].min(), df["High"].max()], mode="lines",
                       line=dict(width=2, dash="dot"), showlegend=False)
    fig = go.Figure(data=[candle, vol_bar, vline])
    fig.update_layout(xaxis_rangeslider_visible=False, height=520,
                      yaxis=dict(domain=[0.25, 1.0], title="Price"), yaxis2=dict(domain=[0.0, 0.2], title="Volume", anchor="x"))
    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics ---
    colA, colB, colC, colD, colE = st.columns(5)
    start_price = float(df.iloc[0]["Close"]) if not df.empty else 0.0
    change_abs = current_price - start_price
    change_pct = (change_abs / start_price * 100.0) if start_price else 0.0

    with colA:
        st.metric("Sim Date", sim_dt.strftime("%Y-%m-%d"))
    with colB:
        st.metric("Price (Close)", f"₹{current_price:,.2f}")
    with colC:
        st.metric("From Start", f"₹{change_abs:,.2f}", f"{change_pct:.2f}%")
    with colD:
        st.metric("Holdings (qty)", f"{st.session_state.holdings.get(ticker, {}).get('quantity', 0)}")
    with colE:
        st.metric("Virtual Cash", f"₹{st.session_state.virtual_cash:,.2f}")

    st.divider()

    # ------------------------- Trading Panel ---------------------------------
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Trade Execution")
        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        est_cost = qty * current_price
        st.info(f"Est. Trade Value: ₹{est_cost:,.2f}")
        bcol, scol = st.columns(2)
        if bcol.button("BUY", use_container_width=True):
            execute_trade_record(ticker, "BUY", int(qty), current_price, pd.Timestamp(sim_dt), scenario=scenario_name)

        if scol.button("SELL", use_container_width=True):
            execute_trade_record(ticker, "SELL", int(qty), current_price, pd.Timestamp(sim_dt), scenario=scenario_name)

        # FULL-portfolio valuation using last-known prices stored in session_state.prices
        holdings_total = 0.0
        for t, h in st.session_state.holdings.items():
            qty_h = h.get("quantity", 0)
            if qty_h <= 0:
                continue
            last_px = st.session_state.prices.get(t)
            # if we don't have last_px, skip valuation for that ticker (could fetch, but avoid network calls here)
            if last_px is None:
                continue
            holdings_total += qty_h * last_px

        port_val = st.session_state.virtual_cash + holdings_total
        st.metric("Portfolio Value (estimated)", f"₹{port_val:,.2f}")

    with right:
        st.markdown("#### Scenario Data (up to sim date)")
        clipped = df.iloc[: sim_idx + 1].tail(200)
        st.dataframe(clipped, use_container_width=True)

    st.divider()

    # ------------------------- Holdings & Trades -----------------------------
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Holdings")
        holding = st.session_state.holdings.get(ticker, {"quantity": 0, "avg_price": 0.0})
        unrealized = (current_price - holding.get("avg_price", 0.0)) * holding.get("quantity", 0)
        holdings_df = pd.DataFrame([
            {
                "Ticker": t,
                "Quantity": h.get("quantity", 0),
                "Avg Price": round(h.get("avg_price", 0.0), 2),
                "Last Price (known)": round(st.session_state.prices.get(t, 0.0), 2) if st.session_state.prices.get(t) is not None else "N/A",
                "Unrealized P&L": round((st.session_state.prices.get(t, current_price if t == ticker else 0.0) - h.get("avg_price", 0.0)) * h.get("quantity", 0), 2)
            }
            for t, h in st.session_state.holdings.items()
        ]) if st.session_state.holdings else pd.DataFrame(columns=["Ticker", "Quantity", "Avg Price", "Last Price (known)", "Unrealized P&L"])
        st.dataframe(holdings_df, use_container_width=True)

    with c2:
        st.markdown("### Trades Log")
        log_df = pd.DataFrame(st.session_state.trades)
        if not log_df.empty:
            st.dataframe(log_df[::-1], use_container_width=True)
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Trades CSV", data=csv, file_name="chronotrade_trades.csv", mime="text/csv")
        else:
            st.info("No trades yet. Use BUY/SELL to simulate.")

    st.divider()

    # ------------------------- Benchmark -------------------------------------
    st.markdown("#### Benchmark vs. Buy & Hold")
    if start_price > 0:
        bh_ret = (current_price / start_price - 1.0) * 100.0
        st.write(f"If bought at the start (₹{start_price:,.2f}) and held, return to sim date = **{bh_ret:.2f}%**.")
    else:
        st.write("Not enough data to compute benchmark.")

else:
    st.info("Select a scenario and ticker in the sidebar and click 'Load Scenario Data' to begin.")
