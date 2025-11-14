import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- Config ---
st.set_page_config(
    page_title="ChronoStox Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import sidebar
from utils.sidebar import render_sidebar
from utils.auth import require_login

user_id = require_login()
render_sidebar(show_cash=False)  # Show cash on Overview page

# API endpoint
API_URL = "http://127.0.0.1:8000"

# --- Custom Styling ---
st.markdown("""
<style>
    /* Main app background */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Hide Streamlit-specific UI elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Set background to match our theme */
    body {
        background: #05080f;
    }

    /* Style the metric cards */
    div[data-testid="stMetric"] {
        background-color: #0f172a; /* slate-900 */
        border: 1px solid #334155; /* slate-700 */
        border-radius: 0.5rem;
        padding: 1.5rem;
    }

    div[data-testid="stMetric"] > label {
        color: #94a3b8; /* slate-400 */
    }

    /* Style the header */
    h1.dashboard-title {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 2.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    p.dashboard-subtitle {
        font-size: 1.125rem;
        color: #94a3b8; /* slate-400 */
    }
</style>
""", unsafe_allow_html=True)


# --- API Fetching Functions ---
@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_indices():
    try:
        res = requests.get(f"{API_URL}/indices")
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend API: {e}")
        return []


@st.cache_data(ttl=60)
def fetch_portfolio(user_id: str):
    try:
        res = requests.get(f"{API_URL}/portfolio/{user_id}")
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=300)
def fetch_stock_price(ticker):
    """Fetches just the current price for portfolio calculation."""
    try:
        res = requests.get(f"{API_URL}/stock/{ticker}")
        res.raise_for_status()
        return res.json().get("currentPrice")
    except requests.exceptions.RequestException:
        return None


# --- Helper Functions ---
def format_large_number(value):
    if not isinstance(value, (int, float)):
        return "$0.00"
    if value >= 1_000_000_000_000:
        return f"${(value / 1_000_000_000_000):.2f}T"
    if value >= 1_000_000_000:
        return f"${(value / 1_000_000_000):.2f}B"
    if value >= 1_000_000:
        return f"${(value / 1_000_000):.2f}M"
    return f"${value:,.2f}"


# --- Main App Logic ---
indices_data = fetch_indices()
portfolio_data = fetch_portfolio(user_id)

if not portfolio_data:
    st.error("Unable to load portfolio data. Please try again later.")
    st.stop()

# --- Calculate Top-Row Stats ---
total_equity = 0
if portfolio_data:
    invested_value = 0
    for holding in portfolio_data.get("holdings", []):
        current_price = fetch_stock_price(holding["ticker"]) or holding["avgPrice"]
        invested_value += current_price * holding["quantity"]
    total_equity = invested_value + portfolio_data.get("virtualCash", 0)

avg_change = 0
positive_count = 0
if indices_data:
    total_change = sum(i["changePct"] for i in indices_data)
    positive_count = sum(1 for i in indices_data if i["changePct"] >= 0)
    if indices_data:
        avg_change = total_change / len(indices_data)

# --- Render The Dashboard UI ---

st.markdown("""
<h1 class="dashboard-title">
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="#06b6d4" class="w-8 h-8" style="width: 32px; height: 32px;">
      <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-3.996-3.996M21 12l-3.996 3.996" />
    </svg>
    Stock Market Dashboard
</h1>
<p class="dashboard-subtitle">Real-time market insights and global indices tracking.</p>
""", unsafe_allow_html=True)

st.write("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Portfolio Value",
        value=format_large_number(total_equity),
        delta="+2.4% vs last week (static)",  # Static demo
    )

with col2:
    st.metric(
        label="Average Index Change",
        value=f"{avg_change:.2f}%",
        delta=f"{positive_count}/{len(indices_data)} positive",
        delta_color="normal" if avg_change >= 0 else "inverse"
    )

with col3:
    st.metric(
        label="Active Indices",
        value=len(indices_data),
        delta="All markets",
        delta_color="off"
    )

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Market Indices")
st.markdown(
    '<p class="dashboard-subtitle" style="font-size: 1rem; margin-top: -10px;">Live data from global exchanges</p>',
    unsafe_allow_html=True)

if not indices_data:
    st.warning("Could not fetch index data from backend. Is the API running?")
else:
    columns = st.columns(3)
    for i, index in enumerate(indices_data):
        col = columns[i % 3]

        if "history" in index and index["history"]:
            chart_data = pd.DataFrame(index["history"])
            chart_data['time'] = pd.to_datetime(chart_data['time'])
            chart_data = chart_data.set_index("time")["close"]
        else:
            chart_data = None

        with col:
            with st.container(border=True):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.caption(index.get("region", "Global").upper())
                    st.markdown(f"**{index['name']}**")
                with col_b:
                    change_pct = index['changePct']
                    color = "#22c55e" if change_pct >= 0 else "#ef4444"
                    st.markdown(
                        f'<div style="text-align: right; color: {color}; font-weight: 600;">{change_pct:+.2f}%</div>',
                        unsafe_allow_html=True)

                abs_change = index['lastClose'] * (index['changePct'] / 100)
                price_color = "#f1f5f9"

                st.markdown(f"""
                <div style="font-size: 2rem; font-weight: 700; margin-top: 10px; color: {price_color};">{index['lastClose']:,.2f}</div>
                <div style="color: {color}; font-weight: 500;">{abs_change:+.2f}</div>
                """, unsafe_allow_html=True)

                if chart_data is not None:
                    # Create interactive Plotly chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data.values,
                        mode='lines',
                        line=dict(color=color, width=2),
                        hovertemplate='<b>%{x}</b><br>Price: ₹%{y:,.2f}<extra></extra>',
                        name=index['name']
                    ))
                    fig.update_layout(
                        height=100,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False, showticklabels=False),
                        hovermode='x unified',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                # --- THIS IS THE CHANGE ---
                # Make the card clickable to navigate to the Live Market page
                if st.button(f"View Details", key=f"btn_{i}", use_container_width=True):
                    st.session_state.prefilled_ticker = index['ticker']
                    st.switch_page("pages/Live_Market.py")

    # --- THIS BLOCK IS REMOVED ---
    # All the "if st.session_state.selected_index:" logic is no longer
    # needed on this page, as it will be handled by Live_Market.py