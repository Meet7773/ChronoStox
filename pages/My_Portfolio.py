import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ChronoStox | My Portfolio",
    page_icon="💼",
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



# --- Page Content ---
st.title("💼 My Portfolio")

portfolio_value = 125430.50 # This would be calculated in a real app
st.metric(label="Total Portfolio Value", value=f"₹{portfolio_value:,.2f}", delta="₹1,250.00 (1.0%)")
st.divider()

st.subheader("My Holdings")
if st.session_state.holdings:
    holdings_df = pd.DataFrame(st.session_state.holdings)
    st.dataframe(holdings_df, use_container_width=True)
else:
    st.write("You do not have any holdings yet.")

# --- Sidebar ---
with st.sidebar:
    st.header(f"Welcome, {st.session_state.username}!")
    st.divider()
    # Note: These links will work once you rename the other files
    st.page_link("Dashboard.py", label="Indices", icon="🌎")
    st.page_link("pages/Live_Market.py", label="Live Market", icon="📈")
    st.page_link("pages/My_Portfolio.py", label="My Folio", icon="💼")
    st.page_link("pages/Stock_Screener.py", label="Stock Screener", icon="🔍")
    st.divider()
    st.metric(label="Virtual Cash", value=f"₹{st.session_state.virtual_cash:,.2f}")