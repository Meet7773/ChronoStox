# FILE: pages/Stock_Screener.py
# DESC: Stock Screener — loads local CSV, robust filters, sorting, pagination,
#       company quick-view (yfinance), and filtered CSV export.

import os
from typing import List

import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------- Page config & styles --------------------------
st.set_page_config(
    page_title="ChronoStox | Stock Screener",
    page_icon="🔍",
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

# ----------------------------- Shared sidebar (graceful fallback) -----------
def _fallback_sidebar():
    with st.sidebar:
        st.title("ChronoStox")
        st.divider()
        # Adjust these paths to match your app's filenames if needed
        st.page_link("Dashboard.py", label="🌎 Market Overview")
        st.page_link("pages/Live_Market.py", label="📈 Live Market")
        st.page_link("pages/ChronoTrade.py", label="⏳ ChronoTrade")
        st.page_link("pages/My_Portfolio.py", label="💼 My Portfolio")
        st.page_link("pages/Stock_Screener.py", label="🔍 Stock Screener")
        st.divider()
        st.metric("Virtual Cash", f"₹{st.session_state.get('virtual_cash', 100000.0):,.2f}")
        holdings = st.session_state.get("holdings", [])
        if isinstance(holdings, dict):
            total_positions = sum(h.get("quantity", 0) for h in holdings.values())
        else:
            total_positions = sum((h or {}).get("quantity", 0) for h in holdings) if isinstance(holdings, list) else 0
        st.caption(f"Total Positions: {total_positions}")

try:
    from utils.sidebar import render_sidebar
    render_sidebar()
except Exception:
    _fallback_sidebar()

# ----------------------------- Helpers --------------------------------------
DEFAULT_TICKER_CSV = "data/ticker.csv"

@st.cache_data(show_spinner="Loading stock data...")
def load_full_ticker_data(path: str = DEFAULT_TICKER_CSV) -> pd.DataFrame:
    """Load and clean the local ticker CSV. Requires: Ticker, Name, Category Name."""
    if not os.path.exists(path):
        st.error(f"Required file not found: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()

    # Normalize column names and values
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Ticker", "Name", "Category Name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in {path}: {', '.join(missing)}")
        return pd.DataFrame()

    df = df.dropna(subset=["Ticker"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Category Name"] = df["Category Name"].astype(str).str.strip()

    # Try to coerce a market-cap-like column if present
    # Supports "Market Cap", "MarketCap", "market_cap"
    mcap_cols = [c for c in df.columns if c.lower().replace(" ", "") in ("marketcap", "market_cap")]
    if mcap_cols:
        # Use the first match as canonical "Market Cap"
        src = mcap_cols[0]
        df["Market Cap"] = pd.to_numeric(df[src], errors="coerce")

    # Coerce optional numeric columns if they exist (safe no-ops if missing)
    for opt_num in ["Price", "PE", "P/E", "Dividend Yield", "DividendYield"]:
        if opt_num in df.columns:
            df[opt_num] = pd.to_numeric(df[opt_num], errors="coerce")

    return df

@st.cache_data(ttl=60 * 5)
def yf_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

# ----------------------------- Load data -----------------------------------
full_df = load_full_ticker_data()

st.title("🔍 Stock Screener")
st.write("Filter and discover stocks using search, categories, sorting, and export.")

if full_df.empty:
    st.info("No data available. Ensure 'data/ticker.csv' exists with columns: Ticker, Name, Category Name.")
    st.stop()

# ----------------------------- Filters UI ---------------------------------
left, right = st.columns([3, 1])

with left:
    # Quick text search
    q = st.text_input(
        "Search by ticker or company name",
        value="",
        placeholder="e.g. RELIANCE or RELIANCE.NS",
        key="screener_search",
    )

    # Category / sector filters
    sectors: List[str] = sorted(full_df["Category Name"].dropna().unique().tolist())
    sel_sectors = st.multiselect(
        "Sectors / Categories",
        options=sectors,
        default=sectors[:4] if len(sectors) >= 4 else sectors,
        key="screener_sectors",
    )

    # Optional market cap range
    has_mcap = "Market Cap" in full_df.columns and full_df["Market Cap"].notna().any()
    if has_mcap:
        mcap_min = int(full_df["Market Cap"].min(skipna=True))
        mcap_max = int(full_df["Market Cap"].max(skipna=True))
        c1, c2 = st.columns(2)
        with c1:
            min_val = st.number_input("Min Market Cap", value=mcap_min, step=1, key="screener_mcap_min")
        with c2:
            max_val = st.number_input("Max Market Cap", value=mcap_max, step=1, key="screener_mcap_max")
    else:
        min_val, max_val = None, None

    # Sorting
    numeric_cols = [c for c in full_df.columns if pd.api.types.is_numeric_dtype(full_df[c])]
    sort_col = st.selectbox(
        "Sort by",
        options=["None"] + numeric_cols,
        index=0,
        key="screener_sort_col",
    )
    sort_asc = st.checkbox("Ascending", value=False, key="screener_sort_asc")

# Results options (pagination + download)
with right:
    st.header("Results")
    per_page = st.selectbox("Rows per page", options=[10, 25, 50, 100], index=1, key="screener_rpp")
    st.caption("Download is always the **filtered** set below.")
    # (We render the actual download button after computing the filtered DF.)

# ----------------------------- Apply filters -------------------------------
filtered = full_df.copy()

if q:
    ql = q.strip().lower()
    mask = filtered["Ticker"].str.lower().str.contains(ql) | filtered["Name"].str.lower().str.contains(ql)
    filtered = filtered[mask]

if sel_sectors:
    filtered = filtered[filtered["Category Name"].isin(sel_sectors)]

if has_mcap and min_val is not None and max_val is not None:
    filtered = filtered[(filtered["Market Cap"] >= min_val) & (filtered["Market Cap"] <= max_val)]

if sort_col != "None":
    filtered = filtered.sort_values(by=sort_col, ascending=sort_asc, na_position="last")

total = len(filtered)
st.subheader(f"{total:,} results")

# ----------------------------- Pagination ----------------------------------
pages = max(1, (total + per_page - 1) // per_page)
page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="screener_page")
start = (page - 1) * per_page
end = start + per_page

display_cols = ["Ticker", "Name", "Category Name"]
# If Market Cap exists, show it too (nice to see in the table)
if "Market Cap" in filtered.columns:
    display_cols.append("Market Cap")

display_df = filtered.iloc[start:end][display_cols].reset_index(drop=True)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Single-click download of the full filtered set (not just the current page)
csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Filtered CSV",
    data=csv_bytes,
    file_name="screener_filtered.csv",
    mime="text/csv",
    key="screener_dl",
)

st.divider()

# ----------------------------- Quick company detail ------------------------
st.markdown("### Company Quick View")
if display_df.empty:
    st.info("No rows on this page. Adjust filters or change page to view company details.")
else:
    sel_row = st.selectbox(
        "Select a row to inspect",
        options=list(display_df.index),
        format_func=lambda i: f"{display_df.loc[i, 'Ticker']} — {display_df.loc[i, 'Name']}",
        key="screener_row",
    )
    sel_ticker = display_df.loc[sel_row, "Ticker"]

    with st.expander(f"Details for {sel_ticker}", expanded=False):
        info = yf_info(sel_ticker)
        if info:
            price = info.get("regularMarketPrice", None)
            if price is not None:
                st.metric("Current Price", f"₹{price:,.2f}")
            meta = {
                "Market Cap": info.get("marketCap", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Website": info.get("website", "N/A"),
            }
            st.write(meta)
            if info.get("longBusinessSummary"):
                st.markdown("#### Business Summary")
                st.write(info.get("longBusinessSummary"))
        else:
            st.info("No extra info available from yfinance for this ticker.")

st.caption("Tip: Only local CSV is used here. Ensure 'data/ticker.csv' is kept up-to-date.")
