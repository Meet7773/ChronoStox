# FILE: pages/2_⏳_ChronoTrade.py
# DESC: The ChronoTrade page for simulating trades in historical market scenarios.

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="ChronoStox | ChronoTrade",
    page_icon="⏳",
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


st.title("⏳ ChronoTrade Simulation")

# --- Sidebar ---
with st.sidebar:
    st.header(f"Welcome, {st.session_state.username}!")
    st.divider()
    # Note: These links will work once you rename the other files
    st.page_link("Dashboard.py", label="Indices", icon="🌎")
    st.page_link("pages/Live_Market.py", label="Live Market", icon="📈")
    st.page_link("pages/My_Portfolio.py", label="My Folio", icon="💼")
    st.divider()


# --- Define Historical Scenarios ---
# In a real app, this could come from a database.
scenarios = {
    "2008 Financial Crisis": {
        "start_date": "2007-10-01",
        "end_date": "2009-04-01",
        "default_ticker": "C",  # Citigroup, as an example
        "description": "Trade during the collapse of the housing market and the subsequent global recession."
    },
    "COVID-19 Crash": {
        "start_date": "2020-01-01",
        "end_date": "2020-06-01",
        "default_ticker": "MRNA",  # Moderna
        "description": "Navigate the extreme volatility at the onset of the global pandemic."
    },
    "Dot-Com Bubble Burst": {
        "start_date": "1999-01-01",
        "end_date": "2001-01-01",
        "default_ticker": "CSCO",  # Cisco
        "description": "Experience the rise and fall of internet stocks at the turn of the millennium."
    }
}

# --- Session State Initialization ---
if 'chrono_ticker_data' not in st.session_state:
    st.session_state.chrono_ticker_data = pd.DataFrame()

# --- Main Layout ---
# Sidebar for Scenario and Ticker Selection
with st.sidebar:
    st.header("Scenario Selection")

    selected_scenario_name = st.selectbox(
        "Choose a Historical Event",
        options=list(scenarios.keys())
    )

    selected_scenario = scenarios[selected_scenario_name]
    st.info(selected_scenario["description"])

    ticker = st.text_input(
        "Enter a Stock Ticker",
        value=selected_scenario["default_ticker"]
    ).upper()

    if st.button("Load Scenario Data", use_container_width=True, type="primary"):
        with st.spinner(
                f"Fetching data for {ticker} from {selected_scenario['start_date']} to {selected_scenario['end_date']}..."):
            try:
                stock = yf.Ticker(ticker)
                st.session_state.chrono_ticker_data = stock.history(
                    start=selected_scenario["start_date"],
                    end=selected_scenario["end_date"]
                )

                if st.session_state.chrono_ticker_data.empty:
                    st.error("No data found for this ticker in the selected period.")
                else:
                    st.success(f"Scenario loaded for {ticker}!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Main content area
if not st.session_state.chrono_ticker_data.empty:
    st.header(f"Trading: {ticker}")
    st.subheader(f"Scenario: {selected_scenario_name}")

    # --- Charting ---
    fig = go.Figure(data=[go.Candlestick(
        x=st.session_state.chrono_ticker_data.index,
        open=st.session_state.chrono_ticker_data['Open'],
        high=st.session_state.chrono_ticker_data['High'],
        low=st.session_state.chrono_ticker_data['Low'],
        close=st.session_state.chrono_ticker_data['Close']
    )])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        title=f"{ticker} Historical Price Chart"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # --- Trading Panel ---
    st.subheader("Trade Execution")
    col1, col2 = st.columns([1, 2])

    with col1:
        # For simulation, let's just use the last available price in the dataset
        current_price = st.session_state.chrono_ticker_data['Close'].iloc[-1]
        st.metric("Last Available Price", f"₹{current_price:,.2f}")

        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="chrono_quantity")

        estimated_cost = current_price * quantity
        st.info(f"Estimated Cost: ₹{estimated_cost:,.2f}")

        buy_col, sell_col = st.columns(2)
        if buy_col.button("BUY", use_container_width=True, key="chrono_buy"):
            st.success(f"Simulated BUY of {quantity} shares of {ticker}.")

        if sell_col.button("SELL", use_container_width=True, key="chrono_sell"):
            st.warning(f"Simulated SELL of {quantity} shares of {ticker}.")

    with col2:
        st.markdown("#### Scenario Data")
        st.dataframe(st.session_state.chrono_ticker_data.tail(), use_container_width=True)

else:
    st.info("Select a scenario and ticker in the sidebar and click 'Load Scenario Data' to begin.")

with st.sidebar:
    st.divider()
    st.metric(label="Virtual Cash", value=f"₹{st.session_state.virtual_cash:,.2f}")