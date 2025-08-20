import streamlit as st
import pandas as pd

st.set_page_config(page_title="Stock Screener", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            div[data-testid="stSidebarNav"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading stock data...")
def load_full_ticker_data():
    """Loads the full ticker CSV into a pandas DataFrame."""
    try:
        df = pd.read_csv('data/ticker.csv')
        # Drop rows where 'Category Name' is missing as we can't filter them
        df.dropna(subset=['Category Name'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("data/ticker.csv not found. Please place it in the 'data' folder.")
        return pd.DataFrame()


ticker_df = load_full_ticker_data()

st.title("🔍 Stock Screener")
st.write("Filter and discover stocks based on their sector or industry.")

with st.sidebar:
    st.title("ChronoStox")
    st.divider()
    st.page_link("Dashboard.py", label="Market Overview", icon="🌎")
    st.page_link("pages/My_Portfolio.py", label="My Portfolio", icon="💼")
    st.page_link("pages/Live_Market.py", label="Live Market", icon="📈")
    st.page_link("pages/ChronoTrade.py", label="ChronoTrade", icon="⏳")
    st.divider()

with st.sidebar:

    st.header("Screener Filters")

    if not ticker_df.empty:
        # Get unique categories (sectors) and sort them
        sectors = sorted(ticker_df['Category Name'].unique())

        selected_sectors = st.multiselect(
            "Select Sectors/Industries",
            options=sectors,
            default=sectors[:3]  # Default to the first 3 sectors
        )
    else:
        selected_sectors = []




# --- Filter and Display Data ---
if not ticker_df.empty and selected_sectors:
    # Filter the DataFrame based on the user's selection
    filtered_df = ticker_df[ticker_df['Category Name'].isin(selected_sectors)]

    st.subheader(f"Showing {len(filtered_df)} stocks in selected sectors")

    # Display the filtered data in a table
    st.dataframe(
        filtered_df[['Ticker', 'Name', 'Category Name']],
        use_container_width=True,
        hide_index=True
    )
elif not ticker_df.empty:
    st.info("Select one or more sectors from the sidebar to see results.")