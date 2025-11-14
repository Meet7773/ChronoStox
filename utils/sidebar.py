# utils/sidebar.py
import streamlit as st
import requests

# API endpoint
API_URL = "http://127.0.0.1:8000"


@st.cache_data(ttl=60)
def fetch_portfolio(user_id: str):
    """Fetch portfolio from API."""
    try:
        res = requests.get(f"{API_URL}/portfolio/{user_id}")
        res.raise_for_status()
        return res.json()
    except Exception:
        return None


def render_sidebar(show_cash: bool = False):
    """
    Render sidebar with navigation.
    show_cash: If True, display cash and positions (for Portfolio and Overview pages)
    """
    st.sidebar.title("ChronoStox")
    st.sidebar.markdown("---")

    user_id = st.session_state.get("auth_user")
    if user_id:
        st.sidebar.caption(f"Signed in as **{user_id}**")
    else:
        st.sidebar.caption("Not signed in")

    # Only show cash on Portfolio and Overview pages
    if show_cash and user_id:
        portfolio = fetch_portfolio(user_id)

        if portfolio:
            virtual_cash = portfolio.get("virtualCash", 0.0)
            st.sidebar.metric("Virtual Cash", f"₹{virtual_cash:,.2f}")

            holdings = portfolio.get("holdings", [])
            total_positions = sum(h.get("quantity", 0) for h in holdings)
            st.sidebar.caption(f"Total Positions: {total_positions}")
        else:
            st.sidebar.metric("Virtual Cash", "N/A")
            st.sidebar.caption("Total Positions: N/A")

        st.sidebar.markdown("---")

    # Navigation Links
    st.sidebar.page_link("pages/Dashboard.py", label="🌎 Market Overview")
    st.sidebar.page_link("pages/Live_Market.py", label="📈 Live Market")
    st.sidebar.page_link("pages/ChronoTrade.py", label="⏳ ChronoTrade")
    st.sidebar.page_link("pages/My_Portfolio.py", label="💼 My Portfolio")
    st.sidebar.page_link("pages/Stock_Screener.py", label="🔍 Stock Screener")
    st.sidebar.page_link("pages/Insights.py", label="📰 Market Insights")

    st.sidebar.markdown("---")
    if user_id:
        if st.sidebar.button("Log out", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("cache") or key in (
                    "auth_user",
                    "portfolio_cache",
                    "simulation_cache",
                ):
                    st.session_state.pop(key, None)
            st.cache_data.clear()
            st.switch_page("Login.py")

