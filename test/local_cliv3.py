# ===============================================================
# ChronoStox v9.0 - Hybrid Quant Engine WITH META-LEARNER
# Based on local_cliv2.py (v8.4) — integrates the XGBoost
# meta-learner that was trained during CV stacking.
#
# KEY CHANGES vs v8.4:
#   1. Meta-learner (XGBoost per-horizon) now used for signal
#   2. LGBM + LSTM predictions fed to meta-learner separately
#   3. Meta-learner probabilities drive confidence & signal
#   4. Side-by-side report: old system vs meta-learner
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

HISTORY_DAYS = 400
MACRO_FILE = "macro_features.parquet"
SENTIMENT_FILE = "sentiment_clean.parquet"
SECTOR_FILE = "ticker.csv"
MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"

DEBUG_MODE = False

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
# WEIGHTS / CONSTANTS
# ===============================================================
GLOBAL_END_DATE = ""

# Hybrid Price Target Weights
WEIGHT_ML_H = {
    5: 0.62,
    21: 0.55,
    63: 0.50,
    126: 0.42,
    252: 0.35
}

WEIGHT_ATR_H = {
    h: 1 - WEIGHT_ML_H[h] for h in HORIZONS
}

# Signal Classification Thresholds (used by legacy system)
SELL_THRESHOLD = -0.010
BUY_THRESHOLD = 0.012

# Confidence Score Weights
CONF_WEIGHT_ML = 0.50
CONF_WEIGHT_TREND = 0.20
CONF_WEIGHT_MACRO = 0.20

# Volatility penalties
VOL_PENALTY_HIGH = 0.75
VOL_PENALTY_VOLATILE = 0.88

# Trend Score Rules
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

# Macro Score Rules
MACRO_VIX_LOW = 20
MACRO_VIX_MEDIUM = 12
MACRO_VIX_HIGH = 5
MACRO_USD_GOOD = 12
MACRO_USD_BAD = 5
MACRO_YIELD_GOOD = 12
MACRO_YIELD_BAD = 10
MACRO_INDX_POS = 10

# Sentiment thresholds
SENTIMENT_NEUTRAL = 0.02
SENTIMENT_NEG = -0.15

# Confidence threshold (legacy system)
CONFIDENCE_THRESHOLD = 35

# ---------------------------------------------------------------
# META-LEARNER CONFIG (NEW in v9.0)
# ---------------------------------------------------------------
# Meta-learner probability thresholds for signal classification
# P(Buy) > this → BUY, P(Sell) > this → SELL, else HOLD
META_BUY_PROB_THRESHOLD = 0.40
META_SELL_PROB_THRESHOLD = 0.40

# Minimum gap between winning class and second class to avoid
# ambiguous signals (e.g., P(Buy)=0.35, P(Hold)=0.33 → too close)
META_MIN_CONFIDENCE_GAP = 0.10

# Meta-learner weight in final signal decision
# 1.0 = fully trust meta-learner, 0.0 = fully trust legacy
META_SIGNAL_WEIGHT = 0.70


# ===============================================================
# TIMER
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
# LOAD MODELS (Joblib + Keras) — now also loads meta_models
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

    meta_models = bundle.get("meta_models", {})

    models = {
        "scaler": bundle["scaler"],
        "lgbm": bundle["model_lgbm"],
        "meta_models": meta_models,
        "features": bundle["features"],
        "horizons": bundle["horizons"],
        "seq_len": bundle["lstm_sequence_length"],
        "lstm": lstm_model
    }

    timer.mark("Model Loading")

    # Report meta-learner status
    if meta_models:
        print(f"[META] ✅ Loaded {len(meta_models)} meta-learner(s) for horizons: {list(meta_models.keys())}")
    else:
        print("[META] ⚠ No meta-learners found in bundle — falling back to legacy signals")

    if DEBUG_MODE:
        print("\n===== DEBUG: MODEL INFO =====")
        print("Feature count (model bundle):", len(models["features"]))
        print("LSTM Sequence Length:", models["seq_len"])
        print("LSTM Input Shape:", lstm_model.input_shape)
        print("Meta-models:", list(meta_models.keys()) if meta_models else "NONE")

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

    if "Date" in df_s.columns:
        df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce")
    else:
        if isinstance(df_s.index, pd.DatetimeIndex):
            df_s = df_s.reset_index().rename(columns={"index": "Date"})
            df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce")
        else:
            print("FATAL: Sentiment parquet missing 'Date' column.")
            sys.exit(1)

    df_s["Date"] = df_s["Date"].dt.tz_localize(None)
    df_s = df_s.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Standardise ticker column
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
        sys.exit(1)
    df_s = df_s.rename(columns={ticker_col: "Ticker_YF"})

    # Standardise sentiment column
    if "sentiment_score" not in df_s.columns:
        for c in df_s.columns:
            if "sentiment" in c.lower():
                df_s = df_s.rename(columns={c: "sentiment_score"})
                break
    if "sentiment_score" not in df_s.columns:
        print("FATAL: Sentiment file missing sentiment_score column.")
        sys.exit(1)

    timer.mark("Sentiment Loading")
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
    try:
        df0 = pd.read_csv(path, low_memory=False)
        if df0.shape[1] >= 3 and any("Sector" in c or "sector" in c for c in df0.columns):
            df = df0
    except Exception:
        pass

    cols = df.columns.tolist()
    ticker_col = None
    sector_col = None
    for c in cols:
        if str(c).lower() in ["ticker", "ticker_yf", "symbol"]:
            ticker_col = c
        if "sector" in str(c).lower():
            sector_col = c

    if ticker_col is None:
        ticker_col = cols[0]
    if sector_col is None:
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
        if GLOBAL_END_DATE:
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
        sys.exit(1)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Ticker_YF"] = ticker

    timer.mark("Price Fetching")
    return df


# ===============================================================
# FEATURE ENGINEERING
# ===============================================================
def engineer_features(df_price, df_macro, df_senti, df_sector, expected_cols, timer):
    global DEBUG_MODE

    df = df_price.copy()

    # TA indicators
    try:
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)

        bb = df.ta.bbands(length=5, append=False)
        if bb is not None and not bb.empty:
            if "BBB_5_2.0" in bb.columns:
                df["BBB_5_2.0"] = bb["BBB_5_2.0"]
            else:
                df["BBB_5_2.0"] = bb.iloc[:, 3]
            if "BBP_5_2.0" in bb.columns:
                df["BBP_5_2.0"] = bb["BBP_5_2.0"]
            else:
                df["BBP_5_2.0"] = bb.iloc[:, 4]

        mac = df.ta.macd(append=False)
        if mac is not None and not mac.empty:
            if mac.shape[1] >= 3:
                df["MACDh_12_26_9"] = mac.iloc[:, 1]
                df["MACDs_12_26_9"] = mac.iloc[:, 2]

        df.ta.rsi(append=True)

        ema200_candidates = [c for c in df.columns if "EMA" in c and "200" in str(c)]
        if ema200_candidates:
            df["close_to_ema200"] = df["Close"] / (df[ema200_candidates[0]] + 1e-9)
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
    try:
        df_senti_local = df_senti.copy()
        df_senti_local["Date"] = pd.to_datetime(df_senti_local["Date"], errors="coerce")
        df_senti_local["Date"] = df_senti_local["Date"].dt.tz_localize(None)
        df_senti_local = df_senti_local.dropna(subset=["Date"])
        df_senti_local["Ticker_YF"] = df_senti_local["Ticker_YF"].astype(str).str.upper().str.strip()

        ticker = df["Ticker_YF"].iloc[0].upper().strip()
        ssub = df_senti_local[df_senti_local["Ticker_YF"] == ticker]

        if ssub.empty:
            ssub = pd.DataFrame({
                "Date": df["Date"],
                "sentiment_score": np.zeros(len(df))
            })

        ssub = ssub.sort_values("Date")[["Date", "sentiment_score"]]
        df = df.sort_values("Date")
        df = pd.merge_asof(df, ssub, on="Date", direction="backward", allow_exact_matches=True)
    except Exception as e:
        print("FATAL: Sentiment merge failed:", e)
        sys.exit(1)

    # Sector merge
    try:
        ticker = df["Ticker_YF"].iloc[0]
        sec_row = df_sector[df_sector["Ticker_YF"] == ticker]
        sector = sec_row["Sector"].iloc[0] if not sec_row.empty else "nan"
        sector = str(sector).strip()
        df["Sector"] = sector

        unique_sectors = [
            "Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy",
            "Financial Services", "Healthcare", "Industrials", "Real Estate",
            "Technology", "Utilities", "nan"
        ]
        for s in unique_sectors:
            df[f"Sector_{s}"] = 1.0 if sector == s else 0.0
    except Exception as e:
        print("FATAL: Sector merge failed:", e)
        sys.exit(1)

    # Sentiment × sector interactions
    try:
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0
        for c in df.columns:
            if str(c).startswith("Sector_"):
                df[f"sentiment_x_{c}"] = df[c].astype(float) * df["sentiment_score"].astype(float)
    except Exception as e:
        print("FATAL: Interaction creation failed:", e)
        sys.exit(1)

    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in set(expected_cols) - set(df.columns):
        df[col] = 0.0

    try:
        df_final = df[expected_cols].copy()
    except Exception as e:
        print("FATAL: Feature mismatch:", e)
        sys.exit(1)

    timer.mark("Feature Engineering")

    trend_score = compute_trend_score(df)
    macro_score = compute_macro_score(df)
    vol_regime = compute_volatility_regime(df)

    return df_final, trend_score, macro_score, vol_regime, df


# ===============================================================
# MARKET INTELLIGENCE ENGINES
# ===============================================================
def compute_trend_score(df):
    try:
        latest = df.iloc[-1]
        score = 0

        ema50_col = next((c for c in df.columns if "EMA_50" in c or "EMA_50" in str(c)), None)
        ema200_col = next((c for c in df.columns if "EMA_200" in c or "EMA_200" in str(c)), None)
        if ema50_col and ema200_col:
            if latest[ema50_col] > latest[ema200_col]:
                score += TREND_EMA_BULL
            else:
                score += 5

        if "MACDh_12_26_9" in latest and latest["MACDh_12_26_9"] > 0:
            score += TREND_EMA_BULL

        if "RSI_14" in latest:
            rsi = latest["RSI_14"]
            if 50 < rsi < 70:
                score += TREND_EMA_BULL
            elif rsi >= 70:
                score += 10
            else:
                score += 5

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
    if DEBUG_MODE:
        print("DEBUG: *** VOLATILITY PATCH v8.2 ACTIVE ***")
    try:
        atr_col = None
        for c in ["ATR_14", "ATRr_14", "ATR"]:
            if c in df.columns:
                atr_col = c
                break
        if atr_col is None:
            return "Normal"

        atr_series = df[atr_col]
        close_series = df["Close"]
        atr_pct_series = (atr_series / close_series) * 100
        atr_pct_series = atr_pct_series.dropna()

        if len(atr_pct_series) < 30:
            return "Normal"

        current_atrp = atr_pct_series.iloc[-1]
        mean_atrp = atr_pct_series.mean()
        std_atrp = atr_pct_series.std()

        z_score = (current_atrp - mean_atrp) / std_atrp if std_atrp != 0 else 0

        if z_score > 2.0:
            return "High-Risk"
        elif z_score > 1.0:
            return "Volatile"
        elif z_score < -1.0:
            return "Calm"
        return "Normal"

    except Exception as e:
        return "Normal"


# ===============================================================
# MONTE CARLO ENGINE
# ===============================================================
class MonteCarloEngine:
    def __init__(self, num_sims=5000):
        self.num_sims = num_sims

    def run_simulation(self, current_price, volatility_daily, drift_daily, horizon_days):
        if horizon_days <= 0:
            return current_price, current_price

        Z = np.random.normal(0, 1, self.num_sims)
        term1 = (drift_daily - 0.5 * volatility_daily ** 2) * horizon_days
        term2 = volatility_daily * np.sqrt(horizon_days) * Z
        simulated_prices = current_price * np.exp(term1 + term2)

        return np.percentile(simulated_prices, 1), np.percentile(simulated_prices, 99)

    def validate_prediction(self, lstm_target, current_price, vol_annual, drift_annual, horizon_days):
        vol_daily = vol_annual / np.sqrt(252)
        drift_daily = (drift_annual / 252) * 0.5
        low, high = self.run_simulation(current_price, vol_daily, drift_daily, horizon_days)

        if lstm_target > high:
            return False, f"Hallucination (Target {lstm_target:.2f} > MC Max {high:.2f})", high
        elif lstm_target < low:
            return False, f"Crash Overshoot (Target {lstm_target:.2f} < MC Min {low:.2f})", low
        return True, "Valid", lstm_target


# ===============================================================
# PRICE TARGET ENGINE (v9.0 — returns raw preds separately)
# ===============================================================
def compute_price_targets(df_full, model_lgb, model_lstm, scaler, seq_len, feature_order):
    """
    v9.0: Returns raw LGBM and LSTM predictions SEPARATELY
    so the meta-learner can use them.
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

        # Legacy average (still used for price targets)
        raw_avg = (raw_lgb + raw_lstm) / 2.0
    except Exception as e:
        print("WARN: model prediction failed:", e)
        raw_lgb = np.zeros(len(HORIZONS))
        raw_lstm = np.zeros(len(HORIZONS))
        raw_avg = np.zeros(len(HORIZONS))

    # --- ATR Band Block ---
    atr_col = None
    for c in ["ATR_14", "ATRr_14", "ATR"]:
        if c in df_full.columns:
            atr_col = c
            break
    atr = float(df_full[atr_col].iloc[-1]) if atr_col else close * 0.01

    # Volatility
    vol_factor = atr / close
    vol_annual = vol_factor * np.sqrt(252)

    try:
        log_returns = np.log(df_full["Close"] / df_full["Close"].shift(1))
        drift_annual = log_returns.tail(252).mean() * 252
        if np.isnan(drift_annual):
            drift_annual = 0.0
    except:
        drift_annual = 0.0

    # Monte Carlo
    mc_engine = MonteCarloEngine()
    mc_notes = []

    # Hybrid + Clamping
    atr_mult = np.array([ATR_MULT[h] for h in HORIZONS])
    atr_expansion = close + (atr * atr_mult)

    ml_prices = close * (1.0 + raw_avg)
    hybrid = []

    for i, h in enumerate(HORIZONS):
        w_ml = WEIGHT_ML_H[h]
        w_atr = WEIGHT_ATR_H[h]
        raw_hybrid = ml_prices[i] * w_ml + atr_expansion[i] * w_atr

        # Beta Clamp
        max_move_pct = vol_factor * np.sqrt(h) * 2.0
        max_price = close * (1 + max_move_pct)
        min_price = close * (1 - max_move_pct)
        clamped_price = max(min_price, min(max_price, raw_hybrid))

        # Monte Carlo Guardrail
        is_valid, msg, safe_limit = mc_engine.validate_prediction(
            clamped_price, close, vol_annual, drift_annual, h
        )
        if not is_valid:
            clamped_price = safe_limit
            mc_notes.append(msg)

        hybrid.append(clamped_price)

    hybrid = np.array(hybrid)

    return {
        "raw_lgb": raw_lgb,       # NEW: separate LGBM predictions
        "raw_lstm": raw_lstm,     # NEW: separate LSTM predictions
        "raw_avg": raw_avg,       # legacy average
        "atr": atr_expansion,
        "hybrid": hybrid,
        "mc_notes": mc_notes
    }


# ===============================================================
# META-LEARNER SIGNAL ENGINE (NEW in v9.0)
# ===============================================================
def apply_meta_learner(meta_models, raw_lgb, raw_lstm, horizons):
    """
    Feeds raw LGBM + LSTM predictions to the per-horizon
    XGBoost meta-learner classifiers.

    Returns per-horizon:
      - meta_signal: "BUY" / "SELL" / "HOLD"
      - meta_confidence: 0-100
      - meta_proba: [P(Sell), P(Hold), P(Buy)]
    """
    results = {}

    for i, h in enumerate(horizons):
        if h not in meta_models:
            results[h] = {
                "signal": "HOLD",
                "confidence": 0,
                "proba": [0.0, 1.0, 0.0],
                "available": False
            }
            continue

        model = meta_models[h]

        # Build meta-feature vector: [lgbm_pred_h, lstm_pred_h]
        X_meta = np.array([[raw_lgb[i], raw_lstm[i]]], dtype=np.float32)

        try:
            proba = model.predict_proba(X_meta)[0]  # [P(Sell), P(Hold), P(Buy)]
            pred_class = model.predict(X_meta)[0]    # 0=Sell, 1=Hold, 2=Buy
        except Exception as e:
            if DEBUG_MODE:
                print(f"[META] Error for {h}d horizon: {e}")
            results[h] = {
                "signal": "HOLD",
                "confidence": 0,
                "proba": [0.0, 1.0, 0.0],
                "available": False
            }
            continue

        p_sell, p_hold, p_buy = proba[0], proba[1], proba[2]

        # Determine signal from probabilities
        sorted_proba = sorted(enumerate(proba), key=lambda x: -x[1])
        best_class, best_prob = sorted_proba[0]
        second_prob = sorted_proba[1][1]
        gap = best_prob - second_prob

        # Signal classification with confidence gap check
        if gap < META_MIN_CONFIDENCE_GAP:
            # Too ambiguous — default to HOLD
            signal = "HOLD"
        elif best_class == 2 and p_buy >= META_BUY_PROB_THRESHOLD:
            signal = "BUY"
        elif best_class == 0 and p_sell >= META_SELL_PROB_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence: based on how dominant the winning class is
        # Scale from 0-100: max confidence when one class has ~100%
        confidence = int(min(100, best_prob * 100))

        results[h] = {
            "signal": signal,
            "confidence": confidence,
            "proba": [round(p_sell, 3), round(p_hold, 3), round(p_buy, 3)],
            "available": True
        }

    return results


# ===============================================================
# PREDICTION WRAPPER
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

    # Apply meta-learner
    meta_results = {}
    if models["meta_models"]:
        meta_results = apply_meta_learner(
            models["meta_models"],
            preds["raw_lgb"],
            preds["raw_lstm"],
            HORIZONS
        )
        timer.mark("Meta-Learner")

    preds["meta"] = meta_results
    return preds


# ===============================================================
# LEGACY SIGNAL ENGINE (kept for comparison)
# ===============================================================
def compute_final_confidence(pred_ret, trend_score, macro_score, vol_regime):
    ml_conf = np.clip(abs(pred_ret) * 950, 0, 60)
    base = (
        ml_conf * CONF_WEIGHT_ML
        + trend_score * CONF_WEIGHT_TREND
        + macro_score * CONF_WEIGHT_MACRO
    )
    if vol_regime == "High-Risk":
        base *= VOL_PENALTY_HIGH
    elif vol_regime == "Volatile":
        base *= VOL_PENALTY_VOLATILE
    return int(min(100, base))


def classify_signal_legacy(pred_ret, confidence_score, conf_threshold=35):
    if confidence_score < conf_threshold:
        return "HOLD"
    if pred_ret < SELL_THRESHOLD:
        return "SELL"
    elif pred_ret > BUY_THRESHOLD:
        return "BUY"
    return "HOLD"


def compute_risk_flags(trend_score, macro_score, vol_regime, sentiment, df):
    warnings_list = []
    if vol_regime == "High-Risk":
        warnings_list.append("⚠ Market volatility extremely high")
    if macro_score < 40:
        warnings_list.append("⚠ Macro headwinds detected")
    if trend_score < 40:
        warnings_list.append("⚠ Weak price trend")
    if "sentiment_score" in df.columns:
        if abs(df["sentiment_score"].iloc[-1]) < SENTIMENT_NEUTRAL:
            warnings_list.append("⚠ Neutral sentiment (low conviction)")
        elif df["sentiment_score"].iloc[-1] < SENTIMENT_NEG:
            warnings_list.append("⚠ Strong negative sentiment")
    if len(warnings_list) == 0:
        warnings_list.append("No major risk flags")
    return warnings_list


# ===============================================================
# COMBINED SIGNAL (v9.0 — blends legacy + meta-learner)
# ===============================================================
def compute_combined_signal(legacy_signal, legacy_conf, meta_result, vol_regime):
    """
    Blends legacy rule-based signal with meta-learner output.

    Logic:
    - If meta-learner is available and confident, prefer it
    - If meta-learner agrees with legacy, boost confidence
    - If they disagree, use the one with higher confidence
    - Apply vol penalty to final confidence
    """
    if not meta_result.get("available", False):
        return legacy_signal, legacy_conf, "legacy-only"

    meta_signal = meta_result["signal"]
    meta_conf = meta_result["confidence"]

    # Agreement boost
    if legacy_signal == meta_signal:
        # Both agree → high confidence
        combined_conf = int(min(100, max(legacy_conf, meta_conf) * 1.15))
        return meta_signal, combined_conf, "agreed"

    # Disagreement → weighted decision
    meta_weight = META_SIGNAL_WEIGHT
    legacy_weight = 1.0 - meta_weight

    meta_score = meta_conf * meta_weight
    legacy_score = legacy_conf * legacy_weight

    if meta_score >= legacy_score:
        # Meta wins
        combined_conf = int(min(100, meta_conf * 0.9))  # slight penalty for disagreement
        return meta_signal, combined_conf, "meta-override"
    else:
        # Legacy wins
        combined_conf = int(min(100, legacy_conf * 0.9))
        return legacy_signal, combined_conf, "legacy-override"


# ===============================================================
# REPORT (v9.0 — shows meta-learner, legacy, and combined)
# ===============================================================
def print_report(ticker, close, preds, trend, macro, vol, risk, timer):
    has_meta = bool(preds.get("meta", {}))

    print("=" * 70)
    print(f"  ChronoStox v9.0 Quant Signal Report (Meta-Learner Edition)")
    print("=" * 70)
    print(f"  Ticker       : {ticker}")
    print(f"  Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Last Close   : ₹{close:.2f}")
    print("-" * 70)
    print(f"  Trend Score  : {trend}/100")
    print(f"  Macro Score  : {macro}/100")
    print(f"  Vol Regime   : {vol}")
    print(f"  Model Latency: {timer.delta('Prediction'):.4f}s")
    if has_meta:
        print(f"  Meta Latency : {timer.delta('Meta-Learner'):.4f}s")
    print("-" * 70)

    # Header
    if has_meta:
        print(f"{'HRZ':<5} | {'COMBINED':<8} | {'CONF':<5} | {'TARGET':<10} | {'RET':<8} | {'META':<6} | {'P(S/H/B)':<18} | {'LEGACY':<6} | {'SRC'}")
        print("-" * 100)
    else:
        print(f"{'HRZ':<5} | {'SIGNAL':<8} | {'CONF':<5} | {'TARGET':<10} | {'RET':<8}")
        print("-" * 55)

    for i, h in enumerate(HORIZONS):
        target_price = preds["hybrid"][i]
        safe_return_pct = (target_price - close) / close

        # Legacy signal
        legacy_conf = compute_final_confidence(safe_return_pct, trend, macro, vol)
        legacy_signal = classify_signal_legacy(safe_return_pct, legacy_conf, CONFIDENCE_THRESHOLD)

        if has_meta and h in preds["meta"]:
            meta_result = preds["meta"][h]

            # Combined signal
            combined_signal, combined_conf, source = compute_combined_signal(
                legacy_signal, legacy_conf, meta_result, vol
            )

            proba = meta_result["proba"]
            proba_str = f"{proba[0]:.2f}/{proba[1]:.2f}/{proba[2]:.2f}"
            meta_sig = meta_result["signal"]

            print(
                f"{str(h).ljust(5)} | "
                f"{combined_signal.ljust(8)} | "
                f"{str(combined_conf).ljust(5)} | "
                f"₹{target_price:<9.2f} | "
                f"{safe_return_pct * 100:+6.2f}% | "
                f"{meta_sig.ljust(6)} | "
                f"{proba_str.ljust(18)} | "
                f"{legacy_signal.ljust(6)} | "
                f"{source}"
            )
        else:
            print(
                f"{str(h).ljust(5)} | "
                f"{legacy_signal.ljust(8)} | "
                f"{str(legacy_conf).ljust(5)} | "
                f"₹{target_price:<9.2f} | "
                f"{safe_return_pct * 100:+6.2f}%"
            )

    print("-" * (100 if has_meta else 55))

    # Risk flags
    print("RISK FLAGS:")
    for r in risk:
        print(f"  - {r}")

    # MC Notes
    if preds["mc_notes"]:
        print("MONTE CARLO ALERTS:")
        for note in preds["mc_notes"]:
            print(f"  - {note}")

    # Meta-learner debug
    if has_meta and DEBUG_MODE:
        print("\n===== META-LEARNER RAW DEBUG =====")
        print(f"  Raw LGBM preds : {preds['raw_lgb']}")
        print(f"  Raw LSTM preds : {preds['raw_lstm']}")
        for h in HORIZONS:
            if h in preds["meta"]:
                m = preds["meta"][h]
                print(f"  {h}d: signal={m['signal']}, conf={m['confidence']}, proba={m['proba']}")

    print("=" * 70)


# ===============================================================
# MASTER PREDICTION PIPELINE
# ===============================================================
def run_prediction(ticker):
    global DEBUG_MODE
    timer = Timer()

    # Load models
    model_dir = "."
    if os.path.exists(os.path.join(".", MODEL_JOBLIB)):
        model_dir = "."
    elif os.path.exists(os.path.join("../test", MODEL_JOBLIB)):
        model_dir = "../test"
    else:
        print(f"FATAL: Cannot find model files in '.' or '../test'")
        sys.exit(1)

    models = load_models(model_dir, timer)

    # Load data
    data_dir = model_dir
    df_macro = load_macro(data_dir, timer)
    df_senti = load_sentiment(data_dir, timer)
    df_sectors = load_sectors(data_dir)
    df_raw = load_price(ticker, timer)
    close_price = float(df_raw["Close"].iloc[-1])

    # Feature engineering
    df_feat, trend_score, macro_score, vol_regime, df_full_merged = engineer_features(
        df_raw, df_macro, df_senti, df_sectors, models["features"], timer
    )

    # Prediction (includes meta-learner)
    preds = predict_all(models, df_feat, df_full_merged, timer)

    # Risk flags
    risk_warnings = compute_risk_flags(
        trend_score, macro_score, vol_regime,
        df_full_merged["sentiment_score"].iloc[-1] if "sentiment_score" in df_full_merged.columns else 0,
        df_full_merged
    )

    # Report
    print_report(
        ticker, close_price, preds,
        trend_score, macro_score, vol_regime,
        risk_warnings, timer
    )


# ===============================================================
# CLI
# ===============================================================
def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser(description="ChronoStox v9.0 CLI (Meta-Learner Edition)")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., RELIANCE.NS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    DEBUG_MODE = args.debug

    print(f"\n[ChronoStox v9.0] Initializing for: {args.ticker.upper()}")
    print("=" * 70)

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
