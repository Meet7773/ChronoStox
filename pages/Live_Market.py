# FILE: pages/Live_Market.py
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go

from utils.sidebar import render_sidebar
from utils.auth import require_login

# API endpoint
API_URL = "http://127.0.0.1:8000"

# Page config & style
st.set_page_config(page_title="Live Market", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div[data-testid="stSidebarNav"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Render centralized sidebar (always render)
user_id = require_login()
render_sidebar(show_cash=False)

# ---------------- Helpers & caching (no UI inside) ---------------------------
DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
TICKER_CSV_PATH = "data/ticker.csv"

@st.cache_data(show_spinner="Loading stock list...")
def load_tickers_from_csv(path: str = TICKER_CSV_PATH):
    """
    Load tickers from disk. NO STREAMLIT UI inside this function.
    Returns list of tickers or DEFAULT_TICKERS if file missing/invalid.
    """
    import pandas as pd
    import os

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

@st.cache_data(ttl=60)
def get_stock_info(ticker: str) -> dict:
    """Fetch stock info from API."""
    try:
        res = requests.get(f"{API_URL}/stock/{ticker}")
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}

@st.cache_data(ttl=60 * 10)
def fetch_history(ticker: str, days: int = 365) -> pd.DataFrame:
    """Fetch history from API and convert to DataFrame."""
    try:
        # Calculate start date
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Fetch from API
        res = requests.get(
            f"{API_URL}/history/{ticker}",
            params={"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}
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

# Handle prefilled ticker from Dashboard (after helpers defined)
if st.session_state.get("prefilled_ticker"):
    prefilled_ticker = st.session_state.prefilled_ticker
    st.session_state.ticker = prefilled_ticker
    with st.spinner(f"Loading data for {prefilled_ticker}..."):
        df_prefilled = fetch_history(prefilled_ticker, days=730)
        if not df_prefilled.empty:
            st.session_state.ticker_data = df_prefilled
    st.session_state.prefilled_ticker = None

@st.cache_data(ttl=60 * 5)
def fetch_news_yf(ticker: str):
    """Fetch news from yfinance using the logic from test/news.py"""
    try:
        import yfinance as yf
        from datetime import timezone
        
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        
        if not news:
            return []
        
        # Process news items similar to test/news.py
        processed_news = []
        for item in news:
            content = item.get("content", {})
            
            # Extract title
            title = content.get('title') or item.get('title', 'No Title')
            
            # Extract summary
            summary = content.get('summary') or item.get('summary', '')
            
            # Extract link
            link = None
            if isinstance(content.get('canonicalUrl'), dict):
                link = content.get('canonicalUrl', {}).get('url')
            if not link:
                link = item.get('link', '#')
            
            # Extract publisher
            publisher = item.get('publisher', 'Unknown')
            
            # Extract publish time (similar to news.py logic)
            pub_time = None
            if "providerPublishTime" in item:
                try:
                    pub_time = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
                except Exception:
                    pass
            
            if pub_time is None:
                pub_date_str = content.get("pubDate")
                if pub_date_str:
                    try:
                        pub_time = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    except:
                        pass
            
            # Extract thumbnail
            thumbnail = content.get('thumbnail') if isinstance(content.get('thumbnail'), dict) else None
            image_url = None
            if thumbnail and 'resolutions' in thumbnail and len(thumbnail['resolutions']) > 0:
                image_url = thumbnail['resolutions'][0].get('url')
            
            processed_news.append({
                'title': title,
                'summary': summary,
                'link': link,
                'publisher': publisher,
                'providerPublishTime': item.get('providerPublishTime'),
                'pub_time': pub_time,
                'image_url': image_url,
                'content': content,
                'raw': item
            })
        
        return processed_news
    except Exception as e:
        st.warning(f"Error fetching news: {e}")
        return []

# ---------------- Shared session-state defaults -----------------------------
if "ticker" not in st.session_state:
    st.session_state.ticker = "RELIANCE.NS"
if "ticker_data" not in st.session_state:
    st.session_state.ticker_data = pd.DataFrame()

# Fetch portfolio for sidebar display
@st.cache_data(ttl=60)
def fetch_portfolio(user_id: str):
    try:
        res = requests.get(f"{API_URL}/portfolio/{user_id}")
        res.raise_for_status()
        return res.json()
    except Exception:
        return None

# ---------------- Sidebar: load tickers (no uploader) ------------------------
tickers = load_tickers_from_csv()

# Ensure the currently selected ticker (possibly set via Dashboard) is in the list
current_selected_ticker = st.session_state.get("ticker", DEFAULT_TICKERS[0])
if current_selected_ticker not in tickers:
    tickers = [current_selected_ticker] + [t for t in tickers if t != current_selected_ticker]

# Show warning if file missing or invalid (only informative)
if not os.path.exists(TICKER_CSV_PATH):
    st.sidebar.warning(f"Using default tickers because '{TICKER_CSV_PATH}' was not found. Place your ticker.csv at that path to use custom list.")

with st.sidebar:
    st.divider()
    portfolio = fetch_portfolio(user_id)
    if portfolio:
        st.metric(label="Virtual Cash", value=f"₹{portfolio.get('virtualCash', 0):,.2f}")
        pos_total = sum([h.get('quantity', 0) for h in portfolio.get('holdings', [])])
        st.caption(f"Total Positions: {pos_total}")

# ---------------- Main ------------------------------------------------------
# Search bar on main page
st.header("Live Market Analysis")

# Search section
search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    # Checkbox to choose between list selection or manual input
    use_ticker_list = st.checkbox("Select ticker from list", value=True, key="use_ticker_list")
    
    # pick ticker - either from list or manual input
    if use_ticker_list:
        try:
            default_index = tickers.index(st.session_state.ticker) if st.session_state.ticker in tickers else 0
        except Exception:
            default_index = 0
        st.session_state.ticker = st.selectbox("Search for a Stock", options=tickers, index=default_index)
    else:
        # Manual text input
        manual_ticker = st.text_input(
            "Enter Ticker Symbol",
            value=st.session_state.get("ticker", ""),
            placeholder="e.g., RELIANCE.NS",
            key="manual_ticker_input"
        )
        if manual_ticker:
            st.session_state.ticker = manual_ticker.strip().upper()

with search_col2:
    st.write("")  # Spacer for alignment
    st.write("")  # Spacer for alignment
    if st.button("Fetch Data", use_container_width=True, type="primary", key="fetch_data_main"):
        with st.spinner(f"Fetching data for {st.session_state.ticker}..."):
            df = fetch_history(st.session_state.ticker, days=730)
            if df.empty:
                st.error("No historical data found for the selected ticker.")
            else:
                st.session_state.ticker_data = df
                st.success(f"Loaded data for {st.session_state.ticker}")
                st.rerun()

st.divider()

if st.session_state.ticker_data.empty:
    st.info("Search a stock above and click 'Fetch Data' to begin.")
else:
    ticker = st.session_state.ticker
    df = st.session_state.ticker_data
    info = get_stock_info(ticker)

    st.header(f"{info.get('name', ticker)} — {ticker}")

    # Initialize active tab in session state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Chart & Trading", "Key Information", "Recent News", "Trade Engine"])
    
    # JavaScript to maintain active tab after rerun
    # Only inject if we need to switch to a tab other than the first one
    if st.session_state.get("active_tab", 0) > 0:
        tab_index = st.session_state.active_tab
        st.markdown(f"""
        <script>
        (function() {{
            function switchToTab() {{
                // Find tab buttons - Streamlit uses specific structure
                // Try multiple selectors to find tabs
                let tabButtons = null;
                
                // Method 1: Standard Streamlit tabs
                const tabContainer = document.querySelector('[data-testid="stTabs"]');
                if (tabContainer) {{
                    tabButtons = tabContainer.querySelectorAll('button[role="tab"]');
                }}
                
                // Method 2: Alternative selector
                if (!tabButtons || tabButtons.length === 0) {{
                    tabButtons = document.querySelectorAll('button[data-baseweb="tab"]');
                }}
                
                // Method 3: Find by button text or structure
                if (!tabButtons || tabButtons.length === 0) {{
                    const allButtons = document.querySelectorAll('button');
                    tabButtons = Array.from(allButtons).filter(btn => 
                        btn.getAttribute('role') === 'tab' || 
                        btn.closest('[data-testid="stTabs"]') !== null
                    );
                }}
                
                if (tabButtons && tabButtons.length > {tab_index}) {{
                    // Click the tab button
                    tabButtons[{tab_index}].click();
                    return true;
                }}
                return false;
            }}
            
            // Wait for DOM to be ready
            function initTabSwitch() {{
                // Use MutationObserver to wait for tabs to be ready
                const observer = new MutationObserver(function(mutations, obs) {{
                    if (switchToTab()) {{
                        obs.disconnect();
                    }}
                }});
                
                // Start observing
                observer.observe(document.body, {{
                    childList: true,
                    subtree: true
                }});
                
                // Also try immediately and with delays as fallback
                if (switchToTab()) {{
                    observer.disconnect();
                }} else {{
                    setTimeout(function() {{
                        if (switchToTab()) observer.disconnect();
                    }}, 100);
                    setTimeout(function() {{
                        if (switchToTab()) observer.disconnect();
                        observer.disconnect(); // Clean up after 1 second
                    }}, 1000);
                }}
            }}
            
            // Run when DOM is ready
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', initTabSwitch);
            }} else {{
                initTabSwitch();
            }}
        }})();
        </script>
        """, unsafe_allow_html=True)

    # Tab 1: Chart & trading
    with tab1:
        candle = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
        vol = go.Bar(x=df.index, y=df['Volume'], yaxis='y2', opacity=0.3)
        fig = go.Figure(data=[candle, vol])
        fig.update_layout(xaxis_rangeslider_visible=False, height=520,
                          yaxis=dict(domain=[0.25, 1.0]), yaxis2=dict(domain=[0.0, 0.2], anchor='x'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trade Execution")
        current_price = float(df['Close'].iloc[-1])
        st.metric("Current Market Price", f"₹{current_price:,.2f}")

        qty = st.number_input("Quantity", min_value=1, value=1, step=1)
        est_cost = qty * current_price
        st.info(f"Estimated Cost: ₹{est_cost:,.2f}")

        buy_col, sell_col = st.columns(2)

        # BUY
        if buy_col.button("BUY", use_container_width=True):
            try:
                payload = {
                    "userId": user_id,
                    "ticker": ticker.upper(),
                    "quantity": int(qty),
                    "action": "BUY",
                    "price": current_price
                }
                res = requests.post(f"{API_URL}/trade", json=payload)
                res.raise_for_status()
                result = res.json()
                st.success(result.get("message", f"Bought {qty} × {ticker} @ ₹{current_price:,.2f}"))
                st.cache_data.clear()
                st.rerun()
            except requests.exceptions.RequestException as e:
                error_detail = "Unknown error"
                try:
                    error_detail = e.response.json().get("detail", "Unknown error")
                except:
                    pass
                st.error(f"Trade Failed: {error_detail}")

        # SELL
        if sell_col.button("SELL", use_container_width=True):
            try:
                payload = {
                    "userId": user_id,
                    "ticker": ticker.upper(),
                    "quantity": int(qty),
                    "action": "SELL",
                    "price": current_price
                }
                res = requests.post(f"{API_URL}/trade", json=payload)
                res.raise_for_status()
                result = res.json()
                st.success(result.get("message", f"Sold {qty} × {ticker} @ ₹{current_price:,.2f}"))
                st.cache_data.clear()
                st.rerun()
            except requests.exceptions.RequestException as e:
                error_detail = "Unknown error"
                try:
                    error_detail = e.response.json().get("detail", "Unknown error")
                except:
                    pass
                st.error(f"Trade Failed: {error_detail}")

        st.divider()
        portfolio = fetch_portfolio(user_id)
        holdings_list = portfolio.get('holdings', []) if portfolio else []
        ticker_holding = next((h for h in holdings_list if h['ticker'] == ticker.upper()), None)
        if ticker_holding:
            st.subheader("Holdings Summary")
            holdings_df = pd.DataFrame([{
                "Ticker": ticker,
                "Quantity": ticker_holding.get('quantity', 0),
                "Avg Price": round(ticker_holding.get('avgPrice', 0.0), 2),
                "Last Price": round(current_price, 2),
                "Unrealized P&L": round((current_price - ticker_holding.get('avgPrice', 0.0)) * ticker_holding.get('quantity', 0), 2)
            }])
            st.dataframe(holdings_df, use_container_width=True)

    # Tab 2: Key Info
    with tab2:
        st.subheader("Company Profile")
        key_info = {
            "Market Cap": info.get('marketCap', 'N/A'),
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A'),
            "Current Price": info.get('currentPrice', 'N/A'),
            "Change": info.get('change', 'N/A'),
            "Change %": f"{info.get('changePct', 0):.2f}%" if info.get('changePct') else 'N/A',
            "Volume": info.get('volume', 'N/A'),
        }
        st.json(key_info)

        st.divider()
        st.markdown("### Company Information")
        st.info("Trade history is stored in the database. View your portfolio page for full trade history.")

    # Tab 3: News
    with tab3:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Financial News")
        with col2:
            max_articles = st.slider("Max articles", min_value=3, max_value=15, value=7, key="news_slider")
        
        st.caption("Note: Yahoo finance news can be older or incomplete. Consider RSS for live feeds.")
        
        news_list = fetch_news_yf(ticker)
        if news_list:
            shown = 0
            for article in news_list:
                if shown >= max_articles:
                    break
                try:
                    title = article.get('title', 'No Title')
                    summary = article.get('summary', '')
                    link = article.get('link', '#')
                    publisher = article.get('publisher', 'Unknown')
                    image_url = article.get('image_url')
                    
                    pub_time = article.get('pub_time')
                    if pub_time:
                        published_str = pub_time.strftime('%Y-%m-%d %H:%M')
                    else:
                        published_str = 'Unknown Date'

                    cols = st.columns([1, 4])
                    if image_url:
                        cols[0].image(image_url, use_container_width=True)
                    with cols[1]:
                        st.markdown(f"### [{title}]({link})")
                        st.caption(f"📰 {publisher} | 📅 {published_str}")
                        if summary:
                            st.write(summary)
                    st.divider()
                    shown += 1
                except Exception as e:
                    st.warning(f"Skipping malformed article: {e}")
        else:
            st.info("No recent news found for this ticker.")
    
    # Tab 4: Trade Engine
    with tab4:
        st.subheader("ChronoStox Trade Engine")
        st.caption("AI-powered trading signals and price predictions")
        
        if st.button("Run Analysis", use_container_width=True, type="primary", key="run_analysis_btn"):
            # Set active tab to Trade Engine (index 3) before running analysis
            # This ensures we stay on this tab after rerun
            st.session_state.active_tab = 3
            with st.spinner("Running trade engine analysis... This may take a moment."):
                try:
                    # Import the trade engine functions
                    import sys
                    import os
                    # Get the test directory path (one level up from pages/)
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(current_dir)
                    test_dir = os.path.join(project_root, 'test')
                    
                    # Add test directory to path for imports
                    if test_dir not in sys.path:
                        sys.path.insert(0, test_dir)
                    
                    # Import necessary components from local_cliv2
                    # Note: We import the module and access functions to avoid sys.exit issues
                    import local_cliv2 as te
                    
                    timer = te.Timer()
                    
                    # Determine data directory (test folder)
                    data_dir = test_dir
                    model_dir = data_dir
                    
                    # Load models and data
                    # Wrap in try-except to handle sys.exit calls gracefully
                    try:
                        models = te.load_models(model_dir, timer)
                        df_macro = te.load_macro(data_dir, timer)
                        df_senti = te.load_sentiment(data_dir, timer)
                        df_sectors = te.load_sectors(data_dir)
                        df_raw = te.load_price(ticker, timer)
                        close_price = float(df_raw["Close"].iloc[-1])
                        
                        # Feature engineering
                        df_feat, trend_score, macro_score, vol_regime, df_full_merged = te.engineer_features(
                            df_raw, df_macro, df_senti, df_sectors, models["features"], timer
                        )
                        
                        # Prediction
                        preds = te.predict_all(models, df_feat, df_full_merged, timer)
                        
                        # Risk flags
                        sentiment_val = df_full_merged["sentiment_score"].iloc[-1] if "sentiment_score" in df_full_merged.columns else 0
                        warnings = te.compute_risk_flags(trend_score, macro_score, vol_regime, sentiment_val, df_full_merged)
                    except SystemExit:
                        st.error("Trade engine encountered a fatal error. Please check that all required model files and data files are present in the test/ directory.")
                        st.stop()
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trend Score", f"{trend_score}/100")
                    with col2:
                        st.metric("Macro Score", f"{macro_score}/100")
                    with col3:
                        st.metric("Volatility Regime", vol_regime)
                    
                    st.divider()
                    
                    # Price targets and signals
                    st.subheader("Price Targets & Signals")
                    targets_data = []
                    for i, h in enumerate(te.HORIZONS):
                        pred_ret = float(preds["raw"][i])
                        signal = te.classify_signal(pred_ret)
                        conf = te.compute_confidence(pred_ret, trend_score, macro_score, vol_regime)
                        target = float(preds["hybrid"][i])
                        
                        targets_data.append({
                            "Horizon (days)": h,
                            "Signal": signal,
                            "Confidence": f"{conf}%",
                            "Price Target": f"₹{target:,.2f}",
                            "Expected Return": f"{pred_ret*100:.2f}%"
                        })
                    
                    import pandas as pd
                    targets_df = pd.DataFrame(targets_data)
                    st.dataframe(targets_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # Risk warnings
                    st.subheader("Risk Flags")
                    for warning in warnings:
                        if "⚠" in warning:
                            st.warning(warning)
                        else:
                            st.info(warning)
                    
                    st.divider()
                    
                    # Current price
                    st.metric("Current Price", f"₹{close_price:,.2f}")
                    
                except Exception as e:
                    st.error(f"Error running trade engine: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
