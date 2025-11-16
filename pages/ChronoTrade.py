# FILE: pages/2_⏳_ChronoTrade.py
# DESC: ChronoTrade — simulate trades inside historical market scenarios with a timeline, P&L, and exportable trade log.
# NOTE: Uses data/ticker.csv (no uploader). Sidebar is centralized via utils.sidebar.render_sidebar()

import os
from datetime import datetime
import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go

from utils.sidebar import render_sidebar
from utils.auth import require_login

# API endpoint
API_URL = "http://127.0.0.1:8000"

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
user_id = require_login()
render_sidebar(show_cash=False)

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
    """Fetch historical data from API."""
    try:
        res = requests.get(
            f"{API_URL}/history/{ticker}",
            params={"start": start, "end": end}
        )
        res.raise_for_status()
        data = res.json()
        
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        if "time" in df.columns:
            df["Date"] = pd.to_datetime(df["time"])
            df.set_index("Date", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]]
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_simulation_portfolio(user_id: str):
    """Fetch simulation portfolio for the given user."""
    try:
        res = requests.get(f"{API_URL}/simulation/portfolio/{user_id}")
        res.raise_for_status()
        return res.json()
    except Exception:
        return {"virtualCash": 100000.0, "holdings": []}


@st.cache_data(ttl=30)
def fetch_simulation_trades(user_id: str):
    """Fetch simulation trades."""
    try:
        res = requests.get(f"{API_URL}/simulation/trades/{user_id}")
        res.raise_for_status()
        return res.json()
    except Exception:
        return []


def perform_simulation_trade(
    user_id: str,
    ticker: str,
    quantity: int,
    price: float,
    action: str,
    scenario: str,
):
    """Execute a simulation trade via API."""
    try:
        payload = {
            "userId": user_id,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "action": action.upper(),
            "scenario": scenario,
        }
        res = requests.post(f"{API_URL}/simulation/trade", json=payload)
        res.raise_for_status()
        st.cache_data.clear()
        st.success(res.json().get("message", f"{action.upper()} executed."))
        st.rerun()
    except requests.exceptions.RequestException as e:
        error_detail = "Unknown error"
        try:
            if e.response is not None:
                error_detail = e.response.json().get("detail", "Unknown error")
        except Exception:
            pass
        st.error(f"Simulation trade failed: {error_detail}")


def reset_simulation(user_id: str):
    """Reset the simulation state for the user."""
    try:
        res = requests.post(f"{API_URL}/simulation/reset/{user_id}")
        res.raise_for_status()
        st.cache_data.clear()
        st.success("Simulation reset successfully.")
        st.rerun()
    except requests.exceptions.RequestException as e:
        error_detail = "Unknown error"
        try:
            if e.response is not None:
                error_detail = e.response.json().get("detail", "Unknown error")
        except Exception:
            pass
        st.error(f"Failed to reset simulation: {error_detail}")

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

simulation_portfolio = fetch_simulation_portfolio(user_id)
simulation_cash = simulation_portfolio.get("virtualCash", 0.0)
simulation_holdings = simulation_portfolio.get("holdings", [])
simulation_trades = fetch_simulation_trades(user_id)


def get_simulation_holding(ticker: str):
    return next((h for h in simulation_holdings if h.get("ticker") == ticker.upper()), None)


# ----------------------------- Session State ----------------------------------
if "chrono_ticker_data" not in st.session_state:
    st.session_state.chrono_ticker_data = pd.DataFrame()
if "sim_idx" not in st.session_state:
    st.session_state.sim_idx = None
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None

# ----------------------------- Sidebar ----------------------------------------
tickers = load_tickers_from_csv()
if not os.path.exists(TICKER_CSV_PATH):
    st.sidebar.warning(f"Using default tickers because '{TICKER_CSV_PATH}' was not found. Place your ticker.csv at that path to use custom list.")

with st.sidebar:
    st.divider()
    st.metric(label="Simulation Cash", value=f"₹{simulation_cash:,.2f}")
    if st.sidebar.button("Reset Simulation", use_container_width=True):
        reset_simulation(user_id)

# ----------------------------- Main Content -----------------------------------
st.header("ChronoTrade — Historical Scenario Trading")

# Scenario Selection and Search on main page
st.subheader("Scenario Selection")

scenario_col1, scenario_col2 = st.columns([2, 1])

with scenario_col1:
    scenario_name = st.selectbox("Choose a Historical Event", options=list(SCENARIOS.keys()), key="scenario_select_main")
    scenario = SCENARIOS[scenario_name]
    st.info(scenario["description"])

with scenario_col2:
    st.write("")  # Spacer

# Ticker selection
ticker_col1, ticker_col2, ticker_col3 = st.columns([2, 2, 1])

with ticker_col1:
    use_picker = st.checkbox("Pick from list", value=True, key="use_picker_main")
    if use_picker:
        try:
            idx = tickers.index(scenario["default_ticker"]) if scenario["default_ticker"] in tickers else 0
        except Exception:
            idx = 0
        ticker = st.selectbox("Select a Ticker", options=tickers, index=idx, key="ticker_select_main")
    else:
        ticker = st.text_input("Enter a Stock Ticker", value=scenario["default_ticker"], key="ticker_input_main").upper().strip()

with ticker_col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    if st.button("Load Scenario Data", use_container_width=True, type="primary", key="load_scenario_main"):
        with st.spinner(f"Fetching {ticker} from {scenario['start']} to {scenario['end']}..."):
            data = fetch_history(ticker, scenario["start"], scenario["end"])
            if data.empty:
                st.error("No data found for this ticker in the selected period.")
            else:
                st.session_state.chrono_ticker_data = data
                st.session_state.sim_idx = len(data) - 1
                st.session_state.current_ticker = ticker
                st.session_state.current_scenario = scenario_name
                st.success(f"Scenario loaded for {ticker}!")
                st.rerun()

with ticker_col3:
    st.write("")  # Spacer

st.divider()

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
    current_holding = get_simulation_holding(ticker)
    holding_quantity = current_holding.get("quantity", 0) if current_holding else 0
    with colD:
        st.metric("Holdings (qty)", f"{holding_quantity}")
    with colE:
        st.metric("Simulation Cash", f"₹{simulation_cash:,.2f}")

    st.divider()

    # ------------------------- Trading Panel ---------------------------------
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Trade Execution")
        
        # Display company info if available
        @st.cache_data(ttl=3600)
        def get_company_name(t):
            try:
                res = requests.get(f"{API_URL}/stock/{t}")
                res.raise_for_status()
                data = res.json()
                return data.get('name', t)
            except:
                return t
        
        company_name = get_company_name(ticker)
        st.caption(f"**{company_name}** ({ticker})")
        
        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        est_cost = qty * current_price
        st.info(f"Est. Trade Value: ₹{est_cost:,.2f}")
        st.caption(f"⚠️ Trade will execute at simulation price: ₹{current_price:,.2f}")
        
        bcol, scol = st.columns(2)
        if bcol.button("BUY", use_container_width=True):
            perform_simulation_trade(user_id, ticker, int(qty), current_price, "BUY", scenario_name)

        if scol.button("SELL", use_container_width=True):
            perform_simulation_trade(user_id, ticker, int(qty), current_price, "SELL", scenario_name)

        current_prices = {ticker.upper(): current_price}
        holdings_total = 0.0
        for holding_item in simulation_holdings:
            qty_h = holding_item.get("quantity", 0)
            if qty_h <= 0:
                continue
            price_estimate = current_prices.get(holding_item["ticker"], holding_item.get("avgPrice", 0.0))
            holdings_total += qty_h * price_estimate

        port_val = simulation_cash + holdings_total
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
        if simulation_holdings:
            rows = []
            for holding_item in simulation_holdings:
                price_estimate = current_prices.get(holding_item["ticker"], holding_item.get("avgPrice", 0.0))
                rows.append({
                    "Ticker": holding_item["ticker"],
                    "Quantity": holding_item.get("quantity", 0),
                    "Avg Price": round(holding_item.get("avgPrice", 0.0), 2),
                    "Est. Price": round(price_estimate, 2),
                    "Est. Value": round(price_estimate * holding_item.get("quantity", 0), 2),
                    "Scenario": holding_item.get("scenario", "-"),
                })
            holdings_df = pd.DataFrame(rows)
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        else:
            st.info("No simulation holdings yet. Use BUY/SELL to simulate.")

    with c2:
        st.markdown("### Trades Log")
        if simulation_trades:
            trades_df = pd.DataFrame(simulation_trades)
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            trades_df = trades_df.sort_values(by="timestamp", ascending=False)
            st.dataframe(
                trades_df[["timestamp", "ticker", "action", "quantity", "price", "tradeValue", "scenario"]],
                use_container_width=True,
                hide_index=True,
            )
            csv = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Trades CSV", data=csv, file_name="chronotrade_trades.csv", mime="text/csv")
        else:
            st.info("No simulation trades yet. Use BUY/SELL to simulate.")

    st.divider()

    # ------------------------- Benchmark -------------------------------------
    st.markdown("#### Benchmark vs. Buy & Hold")
    if start_price > 0:
        bh_ret = (current_price / start_price - 1.0) * 100.0
        st.write(f"If bought at the start (₹{start_price:,.2f}) and held, return to sim date = **{bh_ret:.2f}%**.")
    else:
        st.write("Not enough data to compute benchmark.")

else:
    st.info("Select a scenario and ticker above and click 'Load Scenario Data' to begin.")
