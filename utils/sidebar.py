# utils/sidebar.py
import streamlit as st

def render_sidebar():
    st.sidebar.title("ChronoStox")

    # Virtual cash
    virtual_cash = st.session_state.get("virtual_cash", 100000.0)
    st.sidebar.metric("Virtual Cash", f"₹{virtual_cash:,.2f}")

    # Holdings (dictionary of ticker -> {quantity, avg_price})
    holdings = st.session_state.get("holdings", {})

    if isinstance(holdings, list):  # fallback if old format was used
        st.session_state.holdings = {str(i): h for i, h in enumerate(holdings)}
        holdings = st.session_state.holdings

    total_positions = sum(h.get("quantity", 0) for h in holdings.values())
    st.sidebar.write(f"Total Positions: {total_positions}")

    st.sidebar.markdown("---")

    # Navigation Links
    st.sidebar.page_link("Dashboard.py", label="🌎 Market Overview")
    st.sidebar.page_link("pages/Live_Market.py", label="📈 Live Market")
    st.sidebar.page_link("pages/ChronoTrade.py", label="ChronoTrade", icon="⏳")
    st.sidebar.page_link("pages/My_Portfolio.py", label="My Portfolio", icon="💼")
    st.sidebar.page_link("pages/Stock_Screener.py", label="🔍 Stock Screener")

