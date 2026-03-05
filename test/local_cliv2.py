# ===============================================================
# ChronoStox v8.4 - Hybrid Quant Engine (ATR + ML + Monte Carlo)
# Final Polish: Explicit Return % Calculation in Report
# ===============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import joblib
import time
from datetime import datetime
import argparse

# TF IMPORT
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    tf.get_logger().setLevel("ERROR")
except Exception as e:
    print("FATAL: TensorFlow load error:", e)
    sys.exit(1)

warnings.filterwarnings("ignore")

# ===============================================================
# CONFIG
# ===============================================================

HISTORY_DAYS = 400  # fetch extra to stabilize indicators
MACRO_FILE = "macro_features.parquet"
# Use the exact local filename you mentioned earlier
SENTIMENT_FILE = "sentiment_clean.parquet"
SECTOR_FILE = "ticker.csv"
MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"

DEBUG_MODE = False  # toggled by --debug

# Horizons
HORIZONS = [5, 21, 63, 126, 252]

# ATR multipliers for targets
ATR_MULT = {
    5: 1.0,
    21: 1.8,
    63: 3.0,
    126: 4.8,
    252: 7.2
}

# ===============================================================
# EXPERIMENTAL WEIGHTS / CONSTANTS (Editable)
# ===============================================================
GLOBAL_END_DATE = "" # "2025-11-08"   # e.g., "2025-11-05"

# ---- Hybrid Price Target Weights ----
WEIGHT_ML_H = {
    5: 0.62,
    21: 0.55,
    63: 0.50,  # ML model output weight
    126: 0.42,
    252: 0.35
}

WEIGHT_ATR_H = {
    h: 1 - WEIGHT_ML_H[h] for h in HORIZONS  # ATR-based band expansion weight
}

# ---- Signal Classification Thresholds ----
# These are now checked *after* confidence is confirmed
SELL_THRESHOLD = -0.010  # raw predicted return < threshold → SELL
BUY_THRESHOLD = 0.012  # raw predicted return > threshold → BUY

# ---- Confidence Score Weights ----
CONF_WEIGHT_ML = 0.50  # ML model contributes this much to confidence
CONF_WEIGHT_TREND = 0.20  # Trend score weighting
CONF_WEIGHT_MACRO = 0.20  # Macro score weighting

# Volatility penalties applied AFTER confidence:
VOL_PENALTY_HIGH = 0.75  # if "High-Risk"
VOL_PENALTY_VOLATILE = 0.88  # if "Volatile"

# ---- Trend Score Rules ----
TREND_EMA_BULL = 15
TREND_EMA_BEAR = 3

TREND_MACD_POS = 25
TREND_MACD_NEG = 0

TREND_RSI_GOOD = 12
TREND_RSI_OVER = 10
TREND_RSI_BAD = 5

TREND_CLOSE_STRONG = 15
TREND_CLOSE_OK = 10
TREND_CLOSE_WEAK = 5

# ---- Macro Score Rules ----
MACRO_VIX_LOW = 20
MACRO_VIX_MEDIUM = 12
MACRO_VIX_HIGH = 5

MACRO_USD_GOOD = 12
MACRO_USD_BAD = 5

MACRO_YIELD_GOOD = 12
MACRO_YIELD_BAD = 10

MACRO_INDX_POS = 10  # per positive index

# ---- Sentiment thresholds ----
SENTIMENT_NEUTRAL = 0.02
SENTIMENT_NEG = -0.15

# --- THIS IS THE NEW LOGIC'S CONTROL KNOB ---
# Any signal with confidence < this value will be forced to HOLD.
CONFIDENCE_THRESHOLD = 35  # <---- TUNE THIS VALUE


# ===============================================================
# SIMPLE TIMER (with delta tracking)
# ===============================================================
class Timer:
    def __init__(self):
        self._t0 = time.time()
        self._last = self._t0
        self.times = {}

    def mark(self, name):
        now = time.time()
        elapsed = now - self._last
        self.times[name] = elapsed
        print(f"[TIMER] {name} finished in {elapsed:.4f}s")
        self._last = now

    def delta(self, name):
        return float(self.times.get(name, 0.0))


# ===============================================================
# Helpers
# ===============================================================
def safe_read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"FATAL: Failed to read parquet {path}: {e}")
        sys.exit(1)


# ===============================================================
# LOAD MODELS (Joblib + Keras)
# ===============================================================
def load_models(model_dir, timer):
    global DEBUG_MODE

    try:
        joblib_path = os.path.join(model_dir, MODEL_JOBLIB)
        keras_path = os.path.join(model_dir, MODEL_KERAS)
        bundle = joblib.load(joblib_path)
        lstm_model = load_model(keras_path, compile=False)
    except Exception as e:
        print("FATAL: Error loading model files:", e)
        sys.exit(1)

    models = {
        "scaler": bundle["scaler"],
        "lgbm": bundle["model_lgbm"],
        "meta_models": bundle.get("meta_models", {}),
        "features": bundle["features"],
        "horizons": bundle["horizons"],
        "seq_len": bundle["lstm_sequence_length"],
        "lstm": lstm_model
    }

    timer.mark("Model Loading")

    if DEBUG_MODE:
        print("\n===== DEBUG: MODEL INFO =====")
        print("Feature count (model bundle):", len(models["features"]))
        print("LSTM Sequence Length:", models["seq_len"])
        print("LSTM Input Shape:", lstm_model.input_shape)
        print("Scaler mean size:", getattr(models["scaler"], "mean_", None))

    return models


# ===============================================================
# LOAD MACRO FEATURES
# ===============================================================
def load_macro(data_dir, timer):
    global DEBUG_MODE

    path = os.path.join(data_dir, MACRO_FILE)
    if not os.path.exists(path):
        print(f"FATAL: Macro file '{path}' not found.")
        sys.exit(1)

    df_macro = safe_read_parquet(path)

    # If Date missing but index is datetime -> reset index
    if "Date" not in df_macro.columns:
        if isinstance(df_macro.index, pd.DatetimeIndex):
            df_macro = df_macro.reset_index().rename(columns={"index": "Date"})
        else:
            print("FATAL: Macro file missing a datetime index or 'Date' column.")
            sys.exit(1)

    df_macro["Date"] = pd.to_datetime(df_macro["Date"], errors="coerce")
    df_macro["Date"] = df_macro["Date"].dt.tz_localize(None)
    df_macro = df_macro.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    timer.mark("Macro Loading")
    if DEBUG_MODE:
        print("\n===== DEBUG: MACRO FEATURES =====")
        print(df_macro.head(3))
        print(df_macro.tail(3))
        print("Macro columns:", df_macro.columns.tolist())

    return df_macro


# ===============================================================
# LOAD SENTIMENT
# ===============================================================
def load_sentiment(data_dir, timer):
    global DEBUG_MODE

    path = os.path.join(data_dir, SENTIMENT_FILE)
    if not os.path.exists(path):
        print(f"FATAL: Sentiment file '{path}' not found.")
        sys.exit(1)

    df_s = safe_read_parquet(path)

    # ---- FIX: Force Date to datetime BEFORE anything else ----
    if "Date" in df_s.columns:
        df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce")
    else:
        # maybe index holds the date
        if isinstance(df_s.index, pd.DatetimeIndex):
            df_s = df_s.reset_index().rename(columns={"index": "Date"})
            df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce")
        else:
            print("FATAL: Sentiment parquet missing 'Date' column.")
            print("Columns found:", df_s.columns)
            sys.exit(1)

    # ---- REMOVE timezone ----
    df_s["Date"] = df_s["Date"].dt.tz_localize(None)

    df_s = df_s.dropna(subset=["Date"])
    df_s = df_s.sort_values("Date").reset_index(drop=True)

    # ---- FIX: Standardise ticker column ----
    ticker_col = None
    for c in df_s.columns:
        if c.lower() in ["ticker", "tickeryf", "ticker_yf", "symbol"]:
            ticker_col = c
            break

    if ticker_col is None:
        for c in df_s.columns:
            if "ticker" in c.lower():
                ticker_col = c
                break

    if ticker_col is None:
        print("FATAL: Sentiment file missing ticker column.")
        print("Columns available:", df_s.columns)
        sys.exit(1)

    df_s = df_s.rename(columns={ticker_col: "Ticker_YF"})

    # ---- FIX: Standardise sentiment column ----
    if "sentiment_score" not in df_s.columns:
        for c in df_s.columns:
            if "sentiment" in c.lower():
                df_s = df_s.rename(columns={c: "sentiment_score"})
                break

    if "sentiment_score" not in df_s.columns:
        print("FATAL: Sentiment file missing sentiment_score column.")
        print("Columns available:", df_s.columns)
        sys.exit(1)

    timer.mark("Sentiment Loading")

    if DEBUG_MODE:
        print("\n===== DEBUG: SENTIMENT FEATURES =====")
        print(df_s.info())
        print(df_s.head(5))
        print(df_s.tail(5))

    return df_s


# ===============================================================
# LOAD TICKER SECTOR METADATA
# ===============================================================
def load_sectors(data_dir=None):
    path = SECTOR_FILE if data_dir is None else os.path.join(data_dir, SECTOR_FILE)
    if not os.path.exists(path):
        print("FATAL: ticker.csv not found at", path)
        sys.exit(1)

    df = pd.read_csv(path, header=None, low_memory=False)

    # Attempt to discover columns from the CSV sample header (if present)
    # If file has header row, read again with header=0
    try:
        # re-try reading with header=0 to catch real headers
        df0 = pd.read_csv(path, low_memory=False)
        if df0.shape[1] >= 3 and any("Sector" in c or "sector" in c for c in df0.columns):
            df = df0
    except Exception:
        pass

    cols = df.columns.tolist()
    # heuristics: find ticker col & sector col
    ticker_col = None
    sector_col = None
    for c in cols:
        if str(c).lower() in ["ticker", "ticker_yf", "symbol"]:
            ticker_col = c
        if "sector" in str(c).lower():
            sector_col = c

    # fallback to positional guesses
    if ticker_col is None:
        ticker_col = cols[0]
    if sector_col is None:
        # try third column if csv like sample
        sector_col = cols[2] if len(cols) > 2 else cols[-1]

    df = df.rename(columns={ticker_col: "Ticker_YF", sector_col: "Sector"})
    df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip().str.upper()
    return df[["Ticker_YF", "Sector"]].drop_duplicates()


# ===============================================================
# LOAD / FETCH PRICE DATA
# ===============================================================
def load_price(ticker, timer):
    global DEBUG_MODE
    try:
        yf_obj = yf.Ticker(ticker)

        # --- If GLOBAL_END_DATE is set, fetch HISTORY_DAYS before that date ---
        if GLOBAL_END_DATE:
            # Convert end date to datetime
            end_dt = pd.to_datetime(GLOBAL_END_DATE)
            start_dt = end_dt - pd.Timedelta(days=HISTORY_DAYS)

            df = yf_obj.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d"
            )

        else:
            df = yf_obj.history(period=f"{HISTORY_DAYS}d", interval="1d")
    except Exception as e:
        print(f"FATAL: Failed to fetch price data for {ticker}: {e}")
        sys.exit(1)

    if df.empty:
        print(f"FATAL: No price data returned for {ticker}.")
        sys.exit(1)

    df = df.reset_index()
    if "Date" not in df.columns:
        print("FATAL: Price dataframe missing 'Date' column.")
        print(df.columns)
        sys.exit(1)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Ticker_YF"] = ticker

    timer.mark("Price Fetching")
    if DEBUG_MODE:
        print("\n===== DEBUG: PRICE DATA =====")
        print(df.head(3))
    return df


# ===============================================================
# FEATURE ENGINEERING
# returns: df_final (features in expected order), trend_score, macro_score, vol_regime, df_merged_full
# ===============================================================
def engineer_features(df_price, df_macro, df_senti, df_sector, expected_cols, timer):
    global DEBUG_MODE

    df = df_price.copy()

    # TA: ADX, ATR, BBands, MACD, RSI, EMA50, EMA200
    try:
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)

        bb = df.ta.bbands(length=5, append=False)
        if bb is not None and not bb.empty:
            # pandas_ta naming convention: BBL, BBM, BBU, BBB, BBP sometimes differ; handle robustly
            if "BBB_5_2.0" in bb.columns:
                df["BBB_5_2.0"] = bb["BBB_5_2.0"]
            else:
                # fallback to positional
                df["BBB_5_2.0"] = bb.iloc[:, 3]

            if "BBP_5_2.0" in bb.columns:
                df["BBP_5_2.0"] = bb["BBP_5_2.0"]
            else:
                df["BBP_5_2.0"] = bb.iloc[:, 4]

        mac = df.ta.macd(append=False)
        if mac is not None and not mac.empty:
            # mac columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            # keep hist and signal
            if mac.shape[1] >= 3:
                df["MACDh_12_26_9"] = mac.iloc[:, 1]
                df["MACDs_12_26_9"] = mac.iloc[:, 2]

        df.ta.rsi(append=True)

        # close_to_ema200
        ema200_candidates = [c for c in df.columns if "EMA" in c and "200" in str(c)]
        if ema200_candidates:
            ema200 = ema200_candidates[0]
            df["close_to_ema200"] = df["Close"] / (df[ema200] + 1e-9)
        else:
            df["close_to_ema200"] = np.nan

    except Exception as e:
        print("FATAL: TA computation failed:", e)
        sys.exit(1)

    # Merge macro (asof)
    try:
        df = df.sort_values("Date").reset_index(drop=True)
        df_macro_sorted = df_macro.sort_values("Date").reset_index(drop=True)
        df = pd.merge_asof(df, df_macro_sorted, on="Date", direction="backward")
    except Exception as e:
        print("FATAL: Macro merge failed:", e)
        sys.exit(1)

    # Merge sentiment (per-ticker asof)
    # ================= SENTIMENT MERGE (FINAL FIX) =================
    try:
        df_senti_local = df_senti.copy()

        # --- FIX 1: Ensure dtype(Date) = datetime64 ---
        df_senti_local["Date"] = pd.to_datetime(
            df_senti_local["Date"], errors="coerce"
        )
        df_senti_local["Date"] = df_senti_local["Date"].dt.tz_localize(None)
        df_senti_local = df_senti_local.dropna(subset=["Date"])

        # --- FIX 2: Normalize ticker format ---
        df_senti_local["Ticker_YF"] = (
            df_senti_local["Ticker_YF"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        ticker = df["Ticker_YF"].iloc[0].upper().strip()

        # filter sentiment for this ticker only
        ssub = df_senti_local[df_senti_local["Ticker_YF"] == ticker]

        # --- FIX 3: fallback: no entries for ticker → create neutral frame ---
        if ssub.empty:
            ssub = pd.DataFrame({
                "Date": df["Date"],
                "sentiment_score": np.zeros(len(df))
            })

        # --- FIX 4: asof requires sorted & datetime64 only ---
        ssub = ssub.sort_values("Date")[["Date", "sentiment_score"]]
        df = df.sort_values("Date")

        df = pd.merge_asof(
            df,
            ssub,
            on="Date",
            direction="backward",
            allow_exact_matches=True
        )

    except Exception as e:
        print("FATAL: Sentiment merge failed:", e)
        print("== DEBUG INFO ==")
        print("Price dtypes:", df.dtypes)
        print("Senti dtypes:", df_senti_local.dtypes)
        sys.exit(1)

    # Sector merge
    try:
        ticker = df["Ticker_YF"].iloc[0]
        sec_row = df_sector[df_sector["Ticker_YF"] == ticker]
        sector = sec_row["Sector"].iloc[0] if not sec_row.empty else "nan"
        sector = str(sector).strip()
        df["Sector"] = sector

        unique_sectors = [
            "Communication Services",
            "Consumer Cyclical",
            "Consumer Defensive",
            "Energy",
            "Financial Services",
            "Healthcare",
            "Industrials",
            "Real Estate",
            "Technology",
            "Utilities",
            "nan"
        ]

        for s in unique_sectors:
            cname = f"Sector_{s}"
            df[cname] = 1.0 if sector == s else 0.0

    except Exception as e:
        print("FATAL: Sector merge failed:", e)
        sys.exit(1)

    # sentiment x sector interactions — names MUST match trained features:
    try:
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0

        for c in df.columns:
            if str(c).startswith("Sector_"):
                # feature name expected in training: sentiment_x_Sector_Technology (i.e., prefix 'sentiment_x_' + full sector col)
                interaction_name = f"sentiment_x_{c}"
                df[interaction_name] = df[c].astype(float) * df["sentiment_score"].astype(float)
    except Exception as e:
        print("FATAL: Interaction creation failed:", e)
        sys.exit(1)

    # Final cleanup - keep numeric, fill na
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    # ensure all expected cols exist
    missing_cols = set(expected_cols) - set(df.columns)
    for col in missing_cols:
        df[col] = 0.0

    # reorder to expected
    try:
        df_final = df[expected_cols].copy()
    except Exception as e:
        print("FATAL: Feature mismatch. Missing columns:", e)
        print("Expected count:", len(expected_cols))
        print("Available columns:", len(df.columns))
        sys.exit(1)

    timer.mark("Feature Engineering")

    # compute signal helpers on full merged df (not limited to features)
    trend_score = compute_trend_score(df)
    macro_score = compute_macro_score(df)
    vol_regime = compute_volatility_regime(df)

    if DEBUG_MODE:
        print("\n===== DEBUG: ENGINEERED FEATURES =====")
        print(df_final.head(3))
        print("Feature count:", df_final.shape[1])
        print("Trend score:", trend_score, "Macro score:", macro_score, "Vol regime:", vol_regime)

    return df_final, trend_score, macro_score, vol_regime, df


# ===============================================================
# MARKET INTELLIGENCE ENGINES (Trend, Macro, Volatility)
# (same as your functions with small robustness fixes)
# ===============================================================
def compute_trend_score(df):
    try:
        latest = df.iloc[-1]
        score = 0

        # EMA50 / EMA200
        ema50_col = next((c for c in df.columns if "EMA_50" in c or "EMA_50" in str(c)), None)
        ema200_col = next((c for c in df.columns if "EMA_200" in c or "EMA_200" in str(c)), None)
        if ema50_col and ema200_col:
            if latest[ema50_col] > latest[ema200_col]:
                score += TREND_EMA_BULL
            else:
                score += 5

        # MACD hist
        if "MACDh_12_26_9" in latest and latest["MACDh_12_26_9"] > 0:
            score += TREND_EMA_BULL

        # RSI
        if "RSI_14" in latest:
            rsi = latest["RSI_14"]
            if 50 < rsi < 70:
                score += TREND_EMA_BULL
            elif rsi >= 70:
                score += 10
            else:
                score += 5

        # close_to_ema200
        if "close_to_ema200" in latest:
            ratio = latest["close_to_ema200"]
            if ratio > 1.03:
                score += TREND_EMA_BULL
            elif ratio > 1.0:
                score += 10
            else:
                score += 5

        return min(100, int(score))
    except:
        return 50


def compute_macro_score(df):
    try:
        latest = df.iloc[-1]
        score = 0

        if "VIX_value" in latest:
            v = latest["VIX_value"]
            if v < 14:
                score += 30
            elif v < 18:
                score += 15
            else:
                score += 5

        if "USD_IDX_log_ret" in latest:
            if latest["USD_IDX_log_ret"] < 0:
                score += 20
            else:
                score += 5

        if "US_10Y_YIELD_log_ret" in latest:
            if latest["US_10Y_YIELD_log_ret"] < 0:
                score += 20
            else:
                score += 10

        rb = 0
        for col in ["SP500_log_ret", "NIKKEI_log_ret"]:
            if col in latest:
                rb += 1 if latest[col] > 0 else 0

        score += rb * 15
        return min(100, int(score))
    except:
        return 50


def compute_volatility_regime(df):
    """
    v8.2 PATCH: Robust Column Finder + Z-Score.
    """
    if DEBUG_MODE:
        print("DEBUG: *** VOLATILITY PATCH v8.2 ACTIVE ***")
    try:
        # 1. Smart Column Search
        # pandas_ta sometimes outputs 'ATR_14' or 'ATRr_14'
        atr_col = None
        for c in ["ATR_14", "ATRr_14", "ATR"]:
            if c in df.columns:
                atr_col = c
                break

        if atr_col is None:
            print("[WARN] No ATR column found! Defaulting to Normal.")
            if DEBUG_MODE:
                # Print first 10 cols to help debug
                print(f"[DEBUG] Available cols: {df.columns.tolist()[:10]}")
            return "Normal"

        # 2. Get the history of ATR% (Volatility)
        atr_series = df[atr_col]
        close_series = df["Close"]

        atr_pct_series = (atr_series / close_series) * 100
        atr_pct_series = atr_pct_series.dropna()

        if len(atr_pct_series) < 30:
            if DEBUG_MODE:
                print(f"[WARN] Not enough ATR data ({len(atr_pct_series)} rows).")
            return "Normal"

        # 3. Calculate Z-Score
        current_atrp = atr_pct_series.iloc[-1]
        mean_atrp = atr_pct_series.mean()
        std_atrp = atr_pct_series.std()

        if std_atrp == 0:
            z_score = 0
        else:
            z_score = (current_atrp - mean_atrp) / std_atrp

        if DEBUG_MODE:
            print(
                f"[DEBUG] Volatility Z-Score: {z_score:.2f} (Curr: {current_atrp:.2f}%, Mean: {mean_atrp:.2f}%) using col: {atr_col}")

        # 4. Dynamic Classification
        if z_score > 2.0:
            return "High-Risk"
        elif z_score > 1.0:
            return "Volatile"
        elif z_score < -1.0:
            return "Calm"
        else:
            return "Normal"

    except Exception as e:
        print(f"[Volatility ERROR] {e}")
        return "Normal"


# ===============================================================
# MONTE CARLO ENGINE (New in v8.4)
# ===============================================================
class MonteCarloEngine:
    def __init__(self, num_sims=5000):
        self.num_sims = num_sims

    def run_simulation(self, current_price, volatility_daily, drift_daily, horizon_days):
        """
        Runs Geometric Brownian Motion (GBM) simulations.
        Returns: (min_bound, max_bound) for the given confidence interval (1st/99th percentile).
        """
        if horizon_days <= 0:
            return current_price, current_price

        # Random component (Brownian Motion)
        Z = np.random.normal(0, 1, self.num_sims)

        # GBM Formula: S_t = S_0 * exp((mu - 0.5 * sigma^2)t + sigma * sqrt(t) * Z)
        # drift_daily is the expected daily return (mu)
        term1 = (drift_daily - 0.5 * volatility_daily ** 2) * horizon_days
        term2 = volatility_daily * np.sqrt(horizon_days) * Z

        simulated_prices = current_price * np.exp(term1 + term2)

        # Guardrails: 1st to 99th percentile (covers 98% of outcomes)
        lower_bound = np.percentile(simulated_prices, 1)
        upper_bound = np.percentile(simulated_prices, 99)

        return lower_bound, upper_bound

    def validate_prediction(self, lstm_target, current_price, vol_annual, drift_annual, horizon_days):
        """
        Checks if the LSTM target is statistically possible.
        """
        # Convert annual metrics to daily
        vol_daily = vol_annual / np.sqrt(252)

        # Conservative Drift: Use 50% of historical drift to avoid projecting bull runs forever
        # If historical drift is negative, keep it negative.
        drift_daily = (drift_annual / 252) * 0.5

        low, high = self.run_simulation(current_price, vol_daily, drift_daily, horizon_days)

        if lstm_target > high:
            return False, f"Hallucination (Target {lstm_target:.2f} > MC Max {high:.2f})", high
        elif lstm_target < low:
            return False, f"Crash Overshoot (Target {lstm_target:.2f} < MC Min {low:.2f})", low

        return True, "Valid", lstm_target


# ===============================================================
# PRICE TARGET ENGINE (ATR + ML Hybrid + Monte Carlo)
# ===============================================================
def compute_price_targets(df_full, model_lgb, model_lstm, scaler, seq_len, feature_order):
    """
    v8.4: Syncs Price Engine with ATRr_14 + Monte Carlo Guardrails
    """
    close = float(df_full["Close"].iloc[-1])

    # --- ML Prediction Block ---
    try:
        X_row = df_full[feature_order].iloc[-1].values.astype(np.float32).reshape(1, -1)
        Xs = scaler.transform(X_row)
        raw_lgb = model_lgb.predict(Xs)[0]

        if len(df_full) >= seq_len:
            seq_df = df_full[feature_order].tail(seq_len).values.astype(np.float32)
            lstm_in = seq_df.reshape(1, seq_len, -1)
            raw_lstm = model_lstm.predict(lstm_in, verbose=0)[0]
        else:
            raw_lstm = np.zeros_like(raw_lgb)

        raw_pred = (raw_lgb + raw_lstm) / 2.0
    except Exception as e:
        print("WARN: model prediction failed:", e)
        raw_pred = np.zeros(len(HORIZONS))

    # --- ATR Band Block ---
    atr_col = None
    for c in ["ATR_14", "ATRr_14", "ATR"]:
        if c in df_full.columns:
            atr_col = c
            break

    if atr_col:
        atr = float(df_full[atr_col].iloc[-1])
    else:
        atr = close * 0.01  # Fallback

    # Calculate Volatility Factor
    vol_factor = atr / close

    # Calculate Annualized Volatility and Drift for Monte Carlo
    # Approx annual volatility = daily_vol * sqrt(252)
    vol_annual = vol_factor * np.sqrt(252)

    # Calculate Drift (Log Return Mean over last 252 days)
    # This gives the "Trend Bias" for the Monte Carlo sim
    try:
        log_returns = np.log(df_full["Close"] / df_full["Close"].shift(1))
        drift_annual = log_returns.tail(252).mean() * 252
        if np.isnan(drift_annual): drift_annual = 0.0
    except:
        drift_annual = 0.0

    # --- MONTE CARLO ENGINE INIT ---
    mc_engine = MonteCarloEngine()
    mc_notes = []

    # --- HYBRID + CLAMPING BLOCK ---
    atr_mult = np.array([ATR_MULT[h] for h in HORIZONS])
    atr_expansion = close + (atr * atr_mult)

    ml_prices = close * (1.0 + raw_pred)
    hybrid = []

    for i, h in enumerate(HORIZONS):
        w_ml = WEIGHT_ML_H[h]
        w_atr = WEIGHT_ATR_H[h]
        raw_hybrid = ml_prices[i] * w_ml + atr_expansion[i] * w_atr

        # 1. Beta Clamp (Physics Check)
        max_move_pct = vol_factor * np.sqrt(h) * 2.0
        max_price = close * (1 + max_move_pct)
        min_price = close * (1 - max_move_pct)
        clamped_price = max(min_price, min(max_price, raw_hybrid))

        # 2. Monte Carlo Guardrail (Probability Check)
        is_valid, msg, safe_limit = mc_engine.validate_prediction(
            clamped_price, close, vol_annual, drift_annual, h
        )

        if not is_valid:
            # If invalid, we clamp to the Monte Carlo limit
            # This stops the model from predicting statistically impossible prices
            clamped_price = safe_limit
            mc_notes.append(msg)

        hybrid.append(clamped_price)

    hybrid = np.array(hybrid)

    # Return notes for reporting
    return {
        "raw": raw_pred,
        "atr": atr_expansion,
        "hybrid": hybrid,
        "mc_notes": mc_notes
    }


# ===============================================================
# PREDICTION WRAPPER (LGBM + LSTM -> price targets)
# ===============================================================
def predict_all(models, df_features, df_full_merged, timer):
    seq_len = models["seq_len"]
    scaler = models["scaler"]
    model_lgb = models["lgbm"]
    model_lstm = models["lstm"]
    feature_order = models["features"]

    if df_features.shape[1] != len(feature_order):
        try:
            df_features = df_features[feature_order]
        except Exception:
            print("FATAL: Feature vector shape mismatch vs model features.")
            sys.exit(1)

    timer.mark("Prediction - ScalePrepare")

    preds = compute_price_targets(df_full_merged, model_lgb, model_lstm, scaler, seq_len, feature_order)

    timer.mark("Prediction")
    if DEBUG_MODE:
        print("\n===== DEBUG: RAW MODEL OUTPUTS =====")
        print("Raw preds (returns):", preds["raw"])
        print("Hybrid:", preds["hybrid"])

    return preds


# ===============================================================
#   SIGNAL ENGINE / CONFIDENCE / RISK / REPORT (FIXED LOGIC)
# ===============================================================
# This section has been rewritten to fix the logic bug.
# 1. We compute confidence FIRST.
# 2. We use confidence to *determine* the final signal (overruling weak signals).
# ===============================================================

def compute_final_confidence(pred_ret, trend_score, macro_score, vol_regime):
    """
    This is your *exact* compute_confidence function.
    Its only job is to return the final confidence *number*.
    """
    # Calculate confidence component from the ML model's raw prediction
    # The more extreme the prediction (positive or negative), the higher the base confidence
    ml_conf = np.clip(abs(pred_ret) * 950, 0, 60)

    # Combine ML confidence with Trend and Macro scores
    base = (
            ml_conf * CONF_WEIGHT_ML
            + trend_score * CONF_WEIGHT_TREND
            + macro_score * CONF_WEIGHT_MACRO
    )

    # Apply penalties based on the volatility regime
    if vol_regime == "High-Risk":
        base *= VOL_PENALTY_HIGH
    elif vol_regime == "Volatile":
        base *= VOL_PENALTY_VOLATILE

    return int(min(100, base))


def classify_signal_from_confidence(pred_ret, confidence_score, conf_threshold=55):
    """
    This is the new, critical decision function.
    It checks confidence BEFORE it checks the signal direction.
    """

    # 1. THE MOST IMPORTANT CHECK:
    # If confidence is below our threshold, the *only* signal is HOLD.
    # This immediately kills all "BUY 26%" or "SELL 30%" signals.
    if confidence_score < conf_threshold:
        return "HOLD"

    # 2. ONLY IF CONFIDENCE IS HIGH...
    # ...do we then check the *direction* of the trade based on thresholds.
    if pred_ret < SELL_THRESHOLD:
        return "SELL"
    elif pred_ret > BUY_THRESHOLD:
        return "BUY"
    else:
        # If confidence is high but the return is in the neutral zone,
        # it's still a HOLD (a high-confidence HOLD).
        return "HOLD"


def compute_risk_flags(trend_score, macro_score, vol_regime, sentiment, df):
    # This function is unchanged, just moved here.
    warnings = []
    if vol_regime == "High-Risk":
        warnings.append("⚠ Market volatility extremely high")
    if macro_score < 40:
        warnings.append("⚠ Macro headwinds detected")
    if trend_score < 40:
        warnings.append("⚠ Weak price trend")
    if "sentiment_score" in df.columns:
        if abs(df["sentiment_score"].iloc[-1]) < SENTIMENT_NEUTRAL:
            warnings.append("⚠ Neutral sentiment (low conviction)")
        elif df["sentiment_score"].iloc[-1] < SENTIMENT_NEG:
            warnings.append("⚠ Strong negative sentiment")
    if len(warnings) == 0:
        warnings.append("No major risk flags")
    return warnings


def print_report(ticker, close, preds, trend, macro, vol, risk, timer):
    print("========================================")
    print(f"ChronoStox v8.4 Quant Signal Report")
    print(f"Ticker       : {ticker}")
    print(f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Last Close   : {close:.2f}")
    print("----------------------------------------")
    print(f"Trend Score  : {trend}/100")
    print(f"Macro Score  : {macro}/100")
    print(f"Vol Regime   : {vol}")
    print("----------------------------------------")
    print(f"Model Latency: {timer.delta('Prediction'):.4f}s")
    print("----------------------------------------")
    # Updated header to include RETURN %
    print(f"{'HORIZON':<8} | {'SIGNAL':<6} | {'CONF':<5} | {'TARGET':<10} | {'RETURN':<8}")
    print("----------------------------------------")


    # --- ------------------------------------ ---

    for i, h in enumerate(HORIZONS):
        target_price = preds["hybrid"][i]

        # Calculate Safe Return % based on the Final Clamped Target
        safe_return_pct = (target_price - close) / close

        # 1. Calculate confidence FIRST based on the SAFE return
        conf = compute_final_confidence(safe_return_pct, trend, macro, vol)

        # 2. Use confidence to DETERMINE the final signal
        signal = classify_signal_from_confidence(safe_return_pct, conf, conf_threshold=CONFIDENCE_THRESHOLD)

        # 3. Print results with Explicit Return %
        print(f"{str(h).ljust(8)} | {signal.ljust(6)} | {str(conf).ljust(5)} | {target_price:.2f}".ljust(
            33) + f" | {safe_return_pct * 100:+.2f}%")

    print("----------------------------------------")
    print("RISK FLAGS:")
    for r in risk:
        print(" -", r)

    # Print MC Notes if any
    if preds["mc_notes"]:
        print("MONTE CARLO ALERTS:")
        for note in preds["mc_notes"]:
            print(f" - {note}")

    print("========================================")


# ===============================================================
# MASTER PREDICTION PIPELINE
# ===============================================================
def run_prediction(ticker):
    global DEBUG_MODE
    timer = Timer()

    # 1) Load models
    # ASSUMING models are in a relative dir '../test'
    # You may need to change "../test" to "." if your models are in the same dir
    model_dir = "."
    try:
        # Check current dir
        if os.path.exists(os.path.join(".", MODEL_JOBLIB)):
            model_dir = "."
        # Check parent/test dir
        elif os.path.exists(os.path.join("../test", MODEL_JOBLIB)):
            model_dir = "../test"
        else:
            print(f"FATAL: Cannot find model files in '.' or '../test'")
            sys.exit(1)

        models = load_models(model_dir, timer)
    except Exception as e:
        print(f"FATAL: Error during model loading from '{model_dir}'. {e}")
        sys.exit(1)

    # 2) Load macro
    # ASSUMING data is in the same dir or '../test'
    data_dir = model_dir  # Use the same logic as models
    df_macro = load_macro(data_dir, timer)

    # 3) Load sentiment
    df_senti = load_sentiment(data_dir, timer)

    # 3.5) Load sectors
    df_sectors = load_sectors(data_dir)  # Use data_dir

    # 4) Load price
    df_raw = load_price(ticker, timer)
    close_price = float(df_raw["Close"].iloc[-1])

    # 5) Feature engineering
    df_feat, trend_score, macro_score, vol_regime, df_full_merged = engineer_features(
        df_raw,
        df_macro,
        df_senti,
        df_sectors,
        models["features"],
        timer
    )

    # 6) Prediction
    preds = predict_all(models, df_feat, df_full_merged, timer)

    # 7) Risk Flags
    warnings = compute_risk_flags(
        trend_score,
        macro_score,
        vol_regime,
        df_full_merged["sentiment_score"].iloc[-1] if "sentiment_score" in df_full_merged.columns else 0,
        df_full_merged
    )

    # 8) Output
    print_report(
        ticker,
        close_price,
        preds,
        trend_score,
        macro_score,
        vol_regime,
        warnings,
        timer
    )


# ===============================================================
# COMMAND LINE INTERFACE
# ===============================================================
def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser(description="ChronoStox v8.4 CLI (Monte Carlo Edition)")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., RELIANCE.NS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    DEBUG_MODE = args.debug

    print(f"\n[ChronoStox v8.4] Initializing for: {args.ticker.upper()}")
    print("========================================")

    try:
        run_prediction(args.ticker.upper())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print("\nFATAL ERROR:", str(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()