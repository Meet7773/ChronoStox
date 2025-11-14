import os
import sys

# --- PRELOAD pandas & pandas_ta BEFORE tensorflow ---
import pandas as pd
import pandas_ta as ta

import numpy as np
import argparse
import joblib
import yfinance as yf
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime


# ============================================================
#                     CONFIGURATION
# ============================================================

MODEL_PATH = "."
JOBLIB_FILE = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
LSTM_FILE = "final_lstm_20251113_124137.keras"

MACRO_FILE = "macro_features.parquet"
SENTI_FILE = "ticker_sentiment_scores (1).parquet"
TICKER_MAP_FILE = "ticker.csv"

HISTORY_DAYS = 300
DEBUG_MODE = False


# ============================================================
#                     UTILITIES
# ============================================================

class SimpleProgress:
    def __init__(self):
        self.t = datetime.now()

    def mark(self, msg):
        now = datetime.now()
        dt = (now - self.t).total_seconds()
        print(f"[TIMER] {msg} finished in {dt:.4f}s")
        self.t = now


# ============================================================
#                   MODEL LOADING
# ============================================================

def load_models(timer):
    try:
        bundle = joblib.load(os.path.join(MODEL_PATH, JOBLIB_FILE))
        lstm = load_model(os.path.join(MODEL_PATH, LSTM_FILE), compile=False)

        models = {
            "scaler": bundle["scaler"],
            "lgbm": bundle["model_lgbm"],
            "meta_models": bundle["meta_models"],
            "features": bundle["features"],
            "horizons": bundle["horizons"],
            "seq_len": bundle["lstm_sequence_length"],
            "lstm": lstm,
        }

        timer.mark("Model Loading")

        if DEBUG_MODE:
            print("\n===== DEBUG: MODEL INFO =====")
            print(f"Expected Feature Count: {len(models['features'])}")
            print(f"LSTM Sequence Length: {models['seq_len']}")
            print(f"LSTM Input Shape: {models['lstm'].input_shape}")
            print(f"Scaler Mean Size: {len(models['scaler'].mean_)}")

        return models

    except Exception as e:
        print(f"FATAL: Failed loading model: {e}")
        sys.exit(1)


# ============================================================
#            SENTIMENT + MACRO + SECTOR LOADING
# ============================================================

def load_macro(timer):
    try:
        df = pd.read_parquet(MACRO_FILE)
        df = df.reset_index().rename(columns={"index": "Date"}) if "Date" not in df.columns else df
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values("Date")

        timer.mark("Macro Loading")

        if DEBUG_MODE:
            print("\n===== DEBUG: MACRO FEATURES =====")
            print(df.head(3))
            print(df.tail(3))
            print("Macro columns:", list(df.columns))

        return df

    except Exception as e:
        print(f"FATAL: Failed loading macro_features.parquet: {e}")
        sys.exit(1)


def load_sentiment(timer):
    try:
        df = pd.read_parquet(SENTI_FILE)

        # Normalize column names
        rename_map = {}

        # PRIORITY: Use “sentiment_score” if exists in file
        if "sentiment_score" in df.columns:
            rename_map["sentiment_score"] = "sentiment_score"
        elif "Smoothed" in df.columns:
            rename_map["Smoothed"] = "sentiment_score"
        elif "AvgSentiment" in df.columns:
            rename_map["AvgSentiment"] = "sentiment_score"

        # Also normalize ticker column
        if "Ticker" in df.columns:
            rename_map["Ticker"] = "Ticker_YF"

        df = df.rename(columns=rename_map)

        # Keep only needed columns
        df = df[["Date", "Ticker_YF", "sentiment_score"]]

        df = df.drop_duplicates(subset=["Date", "Ticker_YF"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values(["Ticker_YF", "Date"])

        timer.mark("Sentiment Loading")

        if DEBUG_MODE:
            print("\n===== DEBUG: SENTIMENT FEATURES =====")
            print(df.head(5))
            print(df.tail(5))

        return df

    except Exception as e:
        print(f"FATAL: Failed loading sentiment file: {e}")
        sys.exit(1)



def load_sector_map():
    if not os.path.exists(TICKER_MAP_FILE):
        print("FATAL: ticker.csv not found.")
        sys.exit(1)

    df = pd.read_csv(TICKER_MAP_FILE, header=None, names=["Ticker_YF", "Name", "Exchange", "Sector", "Country"])
    df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()

    return df[["Ticker_YF", "Sector"]].drop_duplicates()


# ============================================================
#                PRICE DATA FETCHING
# ============================================================

def fetch_price(ticker, timer):
    try:
        df = yf.Ticker(ticker).history(period=f"{HISTORY_DAYS}d", interval="1d")

        if df.empty:
            raise Exception("Empty price data")

        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Ticker_YF"] = ticker

        timer.mark("Price Fetching")

        if DEBUG_MODE:
            print("\n===== DEBUG: PRICE DATA =====")
            print(df.head(3))

        return df

    except Exception as e:
        print(f"FATAL: Failed fetching price data: {e}")
        sys.exit(1)


# ============================================================
#                FEATURE ENGINEERING
# ============================================================

def engineer_features(df_raw, df_macro, df_sent, df_sector, expected_features, timer):

    df = df_raw.copy()

    # === 1: Technical Indicators ===
    df.ta.adx(append=True)
    df.ta.atr(append=True)

    bb = df.ta.bbands(length=5, append=False)
    if bb is not None and not bb.empty:
        df["BBB_5_2.0"] = bb["BBB_5_2.0"]
        df["BBP_5_2.0"] = bb["BBP_5_2.0"]

    mac = df.ta.macd(append=False)
    if mac is not None and not mac.empty:
        df["MACDh_12_26_9"] = mac.iloc[:, 1]
        df["MACDs_12_26_9"] = mac.iloc[:, 2]

    df.ta.rsi(append=True)
    df.ta.ema(length=200, append=True)

    if "EMA_200" in df.columns:
        df["close_to_ema200"] = df["Close"] / (df["EMA_200"] + 1e-9)

    # === 2: Macro merge (global as-of) ===
    df_merged = pd.merge_asof(
        df.sort_values("Date"),
        df_macro.sort_values("Date"),
        on="Date",
        direction="backward"
    )

    # === 3: Sentiment merge (per ticker as-of) ===
    sent_sub = df_sent[df_sent["Ticker_YF"] == df["Ticker_YF"].iloc[0]]
    df_merged = pd.merge_asof(
        df_merged.sort_values("Date"),
        sent_sub[["Date", "sentiment_score"]].sort_values("Date"),
        on="Date",
        direction="backward"
    )

    # === 4: Sector merge ===
    sec = df_sector[df_sector["Ticker_YF"] == df["Ticker_YF"].iloc[0]]
    if sec.empty:
        sector = "UNKNOWN"
    else:
        sector = sec["Sector"].iloc[0]

    df_merged["Sector"] = sector
    df_merged = pd.get_dummies(df_merged, columns=["Sector"], dummy_na=True)

    # === 5: Sentiment × Sector interactions ===
    for col in df_merged.columns:
        if col.startswith("Sector_"):
            df_merged[f"sentiment_x_{col}"] = df_merged[col] * df_merged["sentiment_score"]

    # === 6: Fill missing engineered columns ===
    for col in expected_features:
        if col not in df_merged.columns:
            df_merged[col] = 0.0

    df_merged = df_merged.fillna(0.0)

    df_final = df_merged[expected_features]

    timer.mark("Feature Engineering")

    if DEBUG_MODE:
        print("\n===== DEBUG: ENGINEERED FEATURES =====")
        print(df_final.head(3))
        print("Feature count:", df_final.shape[1])
        missing = set(expected_features) - set(df_final.columns)
        if missing:
            print("MISSING FEATURES:", missing)

    return df_final


# ============================================================
#                    PREDICTION
# ============================================================

def predict(df_features, models, timer):
    seq_len = models["seq_len"]
    scaler = models["scaler"]
    lgbm = models["lgbm"]
    lstm = models["lstm"]
    meta = models["meta_models"]
    horizons = models["horizons"]

    if len(df_features) < seq_len:
        print("FATAL: Not enough rows for LSTM sequence.")
        sys.exit(1)

    X_scaled = scaler.transform(df_features)

    # LSTM
    lstm_input = X_scaled[-seq_len:].reshape(1, seq_len, X_scaled.shape[1])
    lstm_pred = lstm.predict(lstm_input, verbose=0)

    # LGBM
    lgbm_input = X_scaled[-1].reshape(1, -1)
    lgbm_pred = lgbm.predict(lgbm_input)

    if DEBUG_MODE:
        print("\n===== DEBUG: RAW MODEL OUTPUTS =====")
        print("LSTM raw:", lstm_pred)
        print("LGBM raw:", lgbm_pred)

    # Meta-learning
    labels = ["SELL", "HOLD", "BUY"]
    signals = []

    for i, h in enumerate(horizons):
        meta_input = np.array([[lgbm_pred[0][i], lstm_pred[0][i]]])
        probs = meta[h].predict_proba(meta_input)[0]
        idx = np.argmax(probs)

        signals.append({
            "Horizon": f"{h}d",
            "Signal": labels[idx],
            "Confidence": f"{probs[idx] * 100:.2f}%"
        })

    timer.mark("Prediction")
    return signals


# ============================================================
#                        MAIN
# ============================================================

def main():
    global DEBUG_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DEBUG_MODE = args.debug
    ticker = args.ticker.upper()

    print(f"\n[ChronoStox v7.1] Initializing for: {ticker}")
    print("=" * 40)

    timer = SimpleProgress()

    models = load_models(timer)
    df_macro = load_macro(timer)
    df_sent = load_sentiment(timer)
    df_sector = load_sector_map()
    df_raw = fetch_price(ticker, timer)

    df_feat = engineer_features(
        df_raw, df_macro, df_sent, df_sector,
        models["features"], timer
    )

    signals = predict(df_feat, models, timer)

    print("=" * 40)
    print(f"ChronoStox Signal Report for: {ticker}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Last Close Price: {df_raw['Close'].iloc[-1]:.2f}")
    print("-" * 40)
    print(f"{'HORIZON':<10} | {'SIGNAL':<8} | {'CONFIDENCE':<10}")
    print("-" * 40)

    for s in signals:
        print(f"{s['Horizon']:<10} | {s['Signal']:<8} | {s['Confidence']:<10}")

    print("=" * 40)


if __name__ == "__main__":
    main()
