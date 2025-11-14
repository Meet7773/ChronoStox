# FILE: pages/Insights.py
# DESC: Market Insights — News, market analysis, and financial insights

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import yfinance as yf

from utils.sidebar import render_sidebar
from utils.auth import require_login

# API endpoint
API_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(
    page_title="ChronoStox | Market Insights",
    page_icon="📰",
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
render_sidebar(show_cash=False)

# --- API Functions ---
@st.cache_data(ttl=60)
def fetch_indices():
    """Fetch market indices."""
    try:
        res = requests.get(f"{API_URL}/indices")
        res.raise_for_status()
        return res.json()
    except Exception:
        return []

@st.cache_data(ttl=60 * 5)
def fetch_stock_news(ticker: str):
    """Fetch news for a specific ticker."""
    try:
        ticker_obj = yf.Ticker(ticker)
        news = ticker_obj.news
        return news or []
    except Exception:
        return []

@st.cache_data(ttl=60)
def get_stock_info(ticker: str):
    """Fetch stock info from API."""
    try:
        res = requests.get(f"{API_URL}/stock/{ticker}")
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}

# --- Main Content ---
st.title("📰 Market Insights")
st.markdown("Stay updated with the latest market news and financial insights.")

# Tabs for different insights
tab1, tab2, tab3 = st.tabs(["Market Overview", "Stock News", "Market Analysis"])

# Tab 1: Market Overview
with tab1:
    st.subheader("Global Market Indices")
    indices = fetch_indices()
    
    if indices:
        # Display indices in a grid
        cols = st.columns(3)
        for i, idx in enumerate(indices[:9]):
            col = cols[i % 3]
            with col:
                change_pct = idx.get('changePct', 0)
                color = "#22c55e" if change_pct >= 0 else "#ef4444"
                st.metric(
                    label=idx.get('name', 'N/A'),
                    value=f"₹{idx.get('lastClose', 0):,.2f}",
                    delta=f"{change_pct:+.2f}%",
                    delta_color="normal" if change_pct >= 0 else "inverse"
                )
    else:
        st.info("No market data available. Ensure the API is running.")

# Tab 2: Market News
with tab2:
    st.subheader("Latest Market News")
    st.caption("Stay updated with the latest financial news from major markets")
    
    # Get popular tickers for news aggregation
    popular_tickers = ["^GSPC", "^DJI", "^IXIC", "^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("📰 Showing latest news from major markets and popular stocks")
    with col2:
        max_articles = st.number_input("Max Articles", min_value=10, max_value=50, value=20, step=5)
    
    # Fetch news from multiple sources
    all_news = []
    with st.spinner("Fetching latest market news..."):
        for ticker in popular_tickers:
            try:
                news_list = fetch_stock_news(ticker)
                if news_list:
                    all_news.extend(news_list)
            except Exception as e:
                continue
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_news = []
    for article in all_news:
        title = article.get('title', '')
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_news.append(article)
    
    # Sort by publish time (newest first)
    unique_news.sort(key=lambda x: x.get('providerPublishTime', 0), reverse=True)
    
    if unique_news:
        st.markdown(f"#### Found {len(unique_news)} unique articles")
        shown = 0
        
        for article in unique_news:
            if shown >= max_articles:
                break
            
            try:
                # Extract article data
                title = article.get('title', 'No Title')
                link = article.get('link', '#')
                publisher = article.get('publisher', 'Unknown')
                
                # Get publish time
                pub_time = article.get('providerPublishTime')
                if pub_time:
                    try:
                        pub_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                    except:
                        pub_str = 'Unknown Date'
                else:
                    pub_str = 'Unknown Date'
                
                # Display article
                with st.container(border=True):
                    st.markdown(f"### [{title}]({link})")
                    st.caption(f"📰 {publisher} | 📅 {pub_str}")
                    
                    # Try to get summary/description
                    if 'summary' in article:
                        st.write(article['summary'])
                    elif 'longSummary' in article:
                        st.write(article['longSummary'])
                
                shown += 1
            except Exception as e:
                continue
        
        if shown == 0:
            st.info("No articles could be displayed.")
    else:
        st.info("No recent news found. Please check your internet connection and try again.")

# Tab 3: Market Analysis
with tab3:
    st.subheader("Market Analysis & Insights")
    
    indices = fetch_indices()
    
    if indices:
        # Calculate market statistics
        changes = [idx.get('changePct', 0) for idx in indices]
        positive_count = sum(1 for c in changes if c >= 0)
        negative_count = len(changes) - positive_count
        avg_change = sum(changes) / len(changes) if changes else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Change", f"{avg_change:+.2f}%")
        with col2:
            st.metric("Positive Indices", positive_count)
        with col3:
            st.metric("Negative Indices", negative_count)
        
        st.divider()
        
        # Market sentiment
        if avg_change > 0:
            st.success("📈 **Market Sentiment: Bullish** - Overall market is trending upward.")
        elif avg_change < 0:
            st.error("📉 **Market Sentiment: Bearish** - Overall market is trending downward.")
        else:
            st.info("➡️ **Market Sentiment: Neutral** - Market is relatively stable.")
        
        st.divider()
        
        # Top movers
        st.subheader("Top Movers")
        sorted_indices = sorted(indices, key=lambda x: x.get('changePct', 0), reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🟢 Top Gainers")
            for idx in sorted_indices[:5]:
                if idx.get('changePct', 0) > 0:
                    st.write(f"**{idx.get('name')}**: {idx.get('changePct', 0):+.2f}%")
        
        with col2:
            st.markdown("##### 🔴 Top Losers")
            for idx in sorted(sorted_indices, key=lambda x: x.get('changePct', 0))[:5]:
                if idx.get('changePct', 0) < 0:
                    st.write(f"**{idx.get('name')}**: {idx.get('changePct', 0):+.2f}%")
    else:
        st.info("No market data available for analysis.")

st.divider()
st.caption("💡 News data is sourced from Yahoo Finance. For real-time updates, ensure your API is running and connected.")

