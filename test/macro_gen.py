import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os

MACRO_FILE = "macro_features.parquet"

def get_macro_features(start_date="2003-01-01", end_date=None):
    import numpy as np
    import pandas as pd
    import pandas_ta as ta
    import yfinance as yf

    if end_date is None:
        end_date = pd.Timestamp.today().date()

    print(f"Fetching macro data {start_date} → {end_date}")

    macro_tickers = {
        "^NSEI": "NIFTY",
        "^INDIAVIX": "VIX",
        "INR=X": "USD_INR",
        "SPY": "SP500",
        "DX-Y.NYB": "USD_IDX",
        "^TNX": "US_10Y_YIELD",
        "CL=F": "OIL",
        "GC=F": "GOLD",
        "HG=F": "COPPER",
        "^N225": "NIKKEI",
        "^HSI": "HANG_SENG",
        "^FTSE": "FTSE100"
    }

    df = yf.download(list(macro_tickers.keys()), start=start_date, end=end_date)

    # Close prices
    df_close = df["Close"].rename(columns=macro_tickers).ffill()

    # Log returns
    df_ret = np.log(df_close).diff().add_suffix("_log_ret")

    # EMAs
    df_ema = pd.DataFrame(index=df_close.index)
    for col in df_close.columns:
        df_ema[f"{col}_ema200"] = ta.ema(df_close[col], length=200)

    # Trend ratio
    df_trend = df_close.div(df_ema.values)

    # ---- FIX: SAFE COLUMN RENAMING ----
    df_trend.columns = [f"{col}_vs_ema200" for col in df_close.columns[:df_trend.shape[1]]]

    # Combine
    df_macro = pd.concat([df_ret, df_trend], axis=1)

    # Raw values
    if "VIX" in df_close.columns:
        df_macro["VIX_value"] = df_close["VIX"]

    if "US_10Y_YIELD" in df_close.columns:
        df_macro["US_10Y_YIELD_value"] = df_close["US_10Y_YIELD"]

    df_macro = df_macro.dropna()

    return df_macro.reset_index().rename(columns={"index": "Date"})



def update_macro():
    MACRO_FILE = "macro_features.parquet"

    if not os.path.exists(MACRO_FILE):
        print("No macro file found. Creating a new one…")
        df_new = get_macro_features("2003-01-01", pd.Timestamp.today().date())
        df_new.to_parquet(MACRO_FILE)
        print("Macro created fresh.")
        return

    print("Existing macro file found. Updating…")

    df_old = pd.read_parquet(MACRO_FILE)

    # --- FIX 1: Ensure Date column exists ---
    if "Date" not in df_old.columns:
        if isinstance(df_old.index, pd.DatetimeIndex):
            df_old = df_old.reset_index().rename(columns={"index": "Date"})
        else:
            raise ValueError("Macro parquet has no Date column and index is not datetime!")

    df_old["Date"] = pd.to_datetime(df_old["Date"])
    df_old = df_old.sort_values("Date")

    last_date = df_old["Date"].max().date()
    today = pd.Timestamp.today().date()

    if last_date >= today:
        print("Macro already up-to-date.")
        return

    start_date = last_date + pd.Timedelta(days=1)
    end_date = today

    print(f"Fetching new macro rows: {start_date} → {end_date}")

    df_new = get_macro_features(start_date, end_date)

    # --- FIX 2: Ensure new block also has Date ---
    if "Date" not in df_new.columns:
        df_new = df_new.reset_index().rename(columns={"index": "Date"})

    # --- Append & Clean ---
    df_final = pd.concat([df_old, df_new]).drop_duplicates("Date").sort_values("Date")

    df_final.to_parquet(MACRO_FILE)

    print(f"Macro updated. Total rows: {len(df_final)}")


if __name__ == "__main__":
    update_macro()
