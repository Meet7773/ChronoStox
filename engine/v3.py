# ===============================================================
# ChronoStox v8.4 - Hybrid Quant Engine (ATR + ML + Monte Carlo)
# Fixed Config Section
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
# CONFIG (VERIFY THESE PATHS!)
# ===============================================================

HISTORY_DAYS = 400

# 1. MACRO DATA (Must be .parquet)
MACRO_FILE = "macro_features.parquet"

# 2. SENTIMENT DATA (Must be .parquet)
SENTIMENT_FILE = "sentiment_clean.parquet"

# 3. SECTOR MAP (Must be .csv)
SECTOR_FILE = "ticker.csv"

# 4. TRAINED MODELS
MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"  # <-- Ensure this is ONLY here

DEBUG_MODE = False

# Horizons
HORIZONS = [5, 21, 63, 126, 252]

# ATR multipliers
ATR_MULT = {
    5: 1.0,
    21: 1.8,
    63: 3.0,
    126: 4.8,
    252: 7.2
}

# ===============================================================
# WEIGHTS
# ===============================================================
GLOBAL_END_DATE = None

WEIGHT_ML_H = {
    5: 0.62,
    21: 0.55,
    63: 0.50,
    126: 0.42,
    252: 0.35
}

WEIGHT_ATR_H = {h: 1 - WEIGHT_ML_H[h] for h in HORIZONS}

# Thresholds
SELL_THRESHOLD = -0.010
BUY_THRESHOLD = 0.012

# Confidence Weights
CONF_WEIGHT_ML = 0.50
CONF_WEIGHT_TREND = 0.20
CONF_WEIGHT_MACRO = 0.20

VOL_PENALTY_HIGH = 0.75
VOL_PENALTY_VOLATILE = 0.88

# Rules
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

MACRO_VIX_LOW = 20
MACRO_VIX_MEDIUM = 12
MACRO_VIX_HIGH = 5
MACRO_USD_GOOD = 12
MACRO_USD_BAD = 5
MACRO_YIELD_GOOD = 12
MACRO_YIELD_BAD = 10
MACRO_INDX_POS = 10

SENTIMENT_NEUTRAL = 0.02
SENTIMENT_NEG = -0.15


# ===============================================================
# HELPERS
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


def safe_read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"FATAL: Failed to read parquet {path}: {e}")
        sys.exit(1)


# ===============================================================
# LOADERS
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
    return models


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
            print("FATAL: Macro file missing Date column.")
            sys.exit(1)

    df_macro["Date"] = pd.to_datetime(df_macro["Date"], errors="coerce").dt.tz_localize(None)
    df_macro = df_macro.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    timer.mark("Macro Loading")
    return df_macro


def load_sentiment(data_dir, timer):
    global DEBUG_MODE
    path = os.path.join(data_dir, SENTIMENT_FILE)
    if not os.path.exists(path):
        print(f"FATAL: Sentiment file '{path}' not found.")
        sys.exit(1)

    df_s = safe_read_parquet(path)
    if "Date" in df_s.columns:
        df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce")
    elif isinstance(df_s.index, pd.DatetimeIndex):
        df_s = df_s.reset_index().rename(columns={"index": "Date"})
        df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce")
    else:
        print("FATAL: Sentiment parquet missing 'Date' column.")
        sys.exit(1)

    df_s["Date"] = df_s["Date"].dt.tz_localize(None)
    df_s = df_s.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Standardise Ticker Col
    ticker_col = None
    for c in df_s.columns:
        if c.lower() in ["ticker", "tickeryf", "ticker_yf", "symbol"]:
            ticker_col = c
            break
    if not ticker_col:
        for c in df_s.columns:
            if "ticker" in c.lower():
                ticker_col = c
                break
    if not ticker_col:
        print("FATAL: Sentiment file missing ticker column.")
        sys.exit(1)
    df_s = df_s.rename(columns={ticker_col: "Ticker_YF"})

    # Standardise Sentiment Col
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


def load_sectors(data_dir=None):
    path = SECTOR_FILE if data_dir is None else os.path.join(data_dir, SECTOR_FILE)
    if not os.path.exists(path):
        print("FATAL: ticker.csv not found at", path)
        sys.exit(1)

    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        df = pd.read_csv(path, header=None, low_memory=False)

    cols = df.columns.tolist()
    ticker_col, sector_col = None, None
    for c in cols:
        if str(c).lower() in ["ticker", "ticker_yf", "symbol"]:
            ticker_col = c
        if "sector" in str(c).lower():
            sector_col = c

    if not ticker_col: ticker_col = cols[0]
    if not sector_col: sector_col = cols[2] if len(cols) > 2 else cols[-1]

    df = df.rename(columns={ticker_col: "Ticker_YF", sector_col: "Sector"})
    df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip().str.upper()
    return df[["Ticker_YF", "Sector"]].drop_duplicates()


def load_price(ticker, timer):
    global DEBUG_MODE
    try:
        yf_obj = yf.Ticker(ticker)
        if GLOBAL_END_DATE:
            end_dt = pd.to_datetime(GLOBAL_END_DATE)
            start_dt = end_dt - pd.Timedelta(days=HISTORY_DAYS)
            df = yf_obj.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d")
        else:
            df = yf_obj.history(period=f"{HISTORY_DAYS}d", interval="1d")
    except Exception as e:
        print(f"FATAL: Failed to fetch price: {e}")
        sys.exit(1)

    if df.empty:
        print(f"FATAL: No price data for {ticker}")
        sys.exit(1)

    df = df.reset_index()
    if "Date" not in df.columns:
        print("FATAL: Price missing Date column")
        sys.exit(1)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
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

    # TA
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

        ema200_c = [c for c in df.columns if "EMA" in c and "200" in str(c)]
        if ema200_c:
            df["close_to_ema200"] = df["Close"] / (df[ema200_c[0]] + 1e-9)
        else:
            df["close_to_ema200"] = np.nan

    except Exception as e:
        print("FATAL: TA failed:", e)
        sys.exit(1)

    # Macro Merge
    try:
        df = df.sort_values("Date").reset_index(drop=True)
        df_macro_sorted = df_macro.sort_values("Date").reset_index(drop=True)
        df = pd.merge_asof(df, df_macro_sorted, on="Date", direction="backward")
    except Exception as e:
        print("FATAL: Macro merge failed:", e)
        sys.exit(1)

    # Sentiment Merge
    try:
        df_sl = df_senti.copy()
        df_sl["Ticker_YF"] = df_sl["Ticker_YF"].astype(str).str.upper().str.strip()
        ticker = df["Ticker_YF"].iloc[0].upper().strip()
        ssub = df_sl[df_sl["Ticker_YF"] == ticker]

        if ssub.empty:
            ssub = pd.DataFrame({"Date": df["Date"], "sentiment_score": np.zeros(len(df))})

        ssub = ssub.sort_values("Date")[["Date", "sentiment_score"]]
        df = pd.merge_asof(df, ssub, on="Date", direction="backward", allow_exact_matches=True)
    except Exception as e:
        print("FATAL: Sentiment merge failed:", e)
        sys.exit(1)

    # Sector Merge
    try:
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

    # Interactions
    try:
        if "sentiment_score" not in df.columns: df["sentiment_score"] = 0.0
        for c in df.columns:
            if str(c).startswith("Sector_"):
                df[f"sentiment_x_{c}"] = df[c].astype(float) * df["sentiment_score"].astype(float)
    except Exception as e:
        print("FATAL: Interaction failed:", e)
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

    trend = compute_trend_score(df)
    macro = compute_macro_score(df)
    vol = compute_volatility_regime(df)

    return df_final, trend, macro, vol, df


# ===============================================================
# SCORING ENGINES
# ===============================================================
def compute_trend_score(df):
    try:
        latest = df.iloc[-1]
        score = 0
        ema50 = next((c for c in df.columns if "EMA_50" in str(c)), None)
        ema200 = next((c for c in df.columns if "EMA_200" in str(c)), None)
        if ema50 and ema200 and latest[ema50] > latest[ema200]:
            score += TREND_EMA_BULL
        else:
            score += 5

        if "MACDh_12_26_9" in latest and latest["MACDh_12_26_9"] > 0: score += TREND_EMA_BULL

        if "RSI_14" in latest:
            rsi = latest["RSI_14"]
            if 50 < rsi < 70:
                score += TREND_EMA_BULL
            elif rsi >= 70:
                score += 10
            else:
                score += 5

        if "close_to_ema200" in latest:
            r = latest["close_to_ema200"]
            if r > 1.03:
                score += TREND_EMA_BULL
            elif r > 1.0:
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
            score += 20 if latest["USD_IDX_log_ret"] < 0 else 5
        if "US_10Y_YIELD_log_ret" in latest:
            score += 20 if latest["US_10Y_YIELD_log_ret"] < 0 else 10

        rb = 0
        for c in ["SP500_log_ret", "NIKKEI_log_ret"]:
            if c in latest and latest[c] > 0: rb += 1
        score += rb * 15
        return min(100, int(score))
    except:
        return 50


def compute_volatility_regime(df):
    try:
        if DEBUG_MODE: print("DEBUG: *** VOLATILITY PATCH v8.2 ACTIVE ***")
        atr_col = None
        for c in ["ATR_14", "ATRr_14", "ATR"]:
            if c in df.columns:
                atr_col = c
                break
        if not atr_col: return "Normal"

        atr_series = df[atr_col]
        close_series = df["Close"]
        atr_pct = (atr_series / close_series) * 100
        atr_pct = atr_pct.dropna()

        if len(atr_pct) < 30: return "Normal"

        cur = atr_pct.iloc[-1]
        mean = atr_pct.mean()
        std = atr_pct.std()
        z = (cur - mean) / std if std != 0 else 0

        if DEBUG_MODE: print(f"[DEBUG] Z-Score: {z:.2f} (Curr: {cur:.2f}%)")

        if z > 2.0:
            return "High-Risk"
        elif z > 1.0:
            return "Volatile"
        elif z < -1.0:
            return "Calm"
        return "Normal"
    except:
        return "Normal"


# ===============================================================
# MONTE CARLO ENGINE
# ===============================================================
class MonteCarloEngine:
    def __init__(self, num_sims=100000):
        self.num_sims = num_sims

    def run_simulation(self, current_price, volatility_daily, drift_daily, horizon_days):
        if horizon_days <= 0: return current_price, current_price
        Z = np.random.normal(0, 1, self.num_sims)
        term1 = (drift_daily - 0.5 * volatility_daily ** 2) * horizon_days
        term2 = volatility_daily * np.sqrt(horizon_days) * Z
        sim_prices = current_price * np.exp(term1 + term2)
        return np.percentile(sim_prices, 1), np.percentile(sim_prices, 99)

    def validate_prediction(self, lstm_target, current_price, vol_annual, drift_annual, horizon_days):
        vol_daily = vol_annual / np.sqrt(252)
        drift_daily = (drift_annual / 252) * 0.5
        low, high = self.run_simulation(current_price, vol_daily, drift_daily, horizon_days)

        if lstm_target > high: return False, f"Hallucination (> {high:.2f})", high
        if lstm_target < low: return False, f"Crash Overshoot (< {low:.2f})", low
        return True, "Valid", lstm_target


# ===============================================================
# PREDICTION
# ===============================================================
def compute_price_targets(df_full, model_lgb, model_lstm, scaler, seq_len, feature_order):
    close = float(df_full["Close"].iloc[-1])

    # ML
    try:
        X_row = df_full[feature_order].iloc[-1].values.astype(np.float32).reshape(1, -1)
        Xs = scaler.transform(X_row)
        lgb_p = model_lgb.predict(Xs)[0]

        if len(df_full) >= seq_len:
            seq = df_full[feature_order].tail(seq_len).values.astype(np.float32).reshape(1, seq_len, -1)
            lstm_p = model_lstm.predict(seq, verbose=0)[0]
        else:
            lstm_p = np.zeros_like(lgb_p)

        raw_pred = (lgb_p + lstm_p) / 2.0
    except Exception as e:
        print("WARN: Prediction failed:", e)
        raw_pred = np.zeros(len(HORIZONS))

    # ATR (Smart Find)
    atr_col = None
    for c in ["ATR_14", "ATRr_14", "ATR"]:
        if c in df_full.columns:
            atr_col = c
            break
    atr = float(df_full[atr_col].iloc[-1]) if atr_col else close * 0.01

    # Monte Carlo Prep
    vol_factor = atr / close
    vol_annual = vol_factor * np.sqrt(252)
    try:
        drifts = np.log(df_full["Close"] / df_full["Close"].shift(1))
        drift_annual = drifts.tail(252).mean() * 252
        if np.isnan(drift_annual): drift_annual = 0.0
    except:
        drift_annual = 0.0

    mc_engine = MonteCarloEngine()
    mc_notes = []

    # Hybrid + Clamp + MC
    atr_exp = close + (atr * np.array([ATR_MULT[h] for h in HORIZONS]))
    ml_prices = close * (1.0 + raw_pred)
    hybrid = []

    for i, h in enumerate(HORIZONS):
        w_ml = WEIGHT_ML_H[h]
        w_atr = WEIGHT_ATR_H[h]
        raw_h = ml_prices[i] * w_ml + atr_exp[i] * w_atr

        # Beta Clamp
        max_move = vol_factor * np.sqrt(h) * 2.0
        clamped = max(close * (1 - max_move), min(close * (1 + max_move), raw_h))

        # MC Check
        valid, msg, safe = mc_engine.validate_prediction(clamped, close, vol_annual, drift_annual, h)
        if not valid:
            clamped = safe
            mc_notes.append(f"{h}d: {msg}")

        hybrid.append(clamped)

    return {"raw": raw_pred, "hybrid": np.array(hybrid), "mc_notes": mc_notes}


def compute_final_confidence(pred_ret, trend, macro, vol):
    ml_conf = np.clip(abs(pred_ret) * 950, 0, 60)
    base = ml_conf * CONF_WEIGHT_ML + trend * CONF_WEIGHT_TREND + macro * CONF_WEIGHT_MACRO
    if vol == "High-Risk":
        base *= VOL_PENALTY_HIGH
    elif vol == "Volatile":
        base *= VOL_PENALTY_VOLATILE
    return int(min(100, base))


def classify_signal(pred_ret, conf, thresh=35):
    if conf < thresh: return "HOLD"
    if pred_ret < SELL_THRESHOLD: return "SELL"
    if pred_ret > BUY_THRESHOLD: return "BUY"
    return "HOLD"


def compute_risk_flags(trend, macro, vol, senti):
    w = []
    if vol == "High-Risk": w.append("⚠ High Volatility")
    if macro < 40: w.append("⚠ Macro Headwinds")
    if trend < 40: w.append("⚠ Weak Trend")
    if abs(senti) < SENTIMENT_NEUTRAL:
        w.append("⚠ Neutral Sentiment")
    elif senti < SENTIMENT_NEG:
        w.append("⚠ Negative Sentiment")
    if not w: w.append("No major risks")
    return w


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
    print("HORIZON | SIGNAL | CONFIDENCE | PRICE TARGET")
    print("----------------------------------------")

    for i, h in enumerate(HORIZONS):
        p_ret = (preds["hybrid"][i] - close) / close
        conf = compute_final_confidence(p_ret, trend, macro, vol)
        sig = classify_signal(p_ret, conf)
        print(f"{str(h).ljust(8)} | {sig.ljust(6)} | {str(conf).ljust(10)} | {preds['hybrid'][i]:.2f}")

    print("----------------------------------------")
    print("RISK FLAGS:")
    for r in risk: print(" -", r)

    if preds["mc_notes"]:
        print("MONTE CARLO ALERTS:")
        for n in preds["mc_notes"]: print(" -", n)
    print("========================================")


# ===============================================================
# MAIN
# ===============================================================
def run_prediction(ticker):
    global DEBUG_MODE
    timer = Timer()

    # Paths
    model_dir = "."
    if os.path.exists(os.path.join("../test", MODEL_JOBLIB)): model_dir = "../test"

    models = load_models(model_dir, timer)
    df_macro = load_macro(model_dir, timer)
    df_senti = load_sentiment(model_dir, timer)
    df_sect = load_sectors(model_dir)
    df_price = load_price(ticker, timer)

    df_feat, trend, macro, vol, df_full = engineer_features(
        df_price, df_macro, df_senti, df_sect, models["features"], timer
    )

    # PREDICT
    timer.mark("Prediction - ScalePrepare")
    preds = compute_price_targets(df_full, models["lgbm"], models["lstm"],
                                  models["scaler"], models["seq_len"], models["features"])
    timer.mark("Prediction")

    # REPORT
    senti = df_full["sentiment_score"].iloc[-1] if "sentiment_score" in df_full.columns else 0
    risk = compute_risk_flags(trend, macro, vol, senti)
    print_report(ticker, df_price["Close"].iloc[-1], preds, trend, macro, vol, risk, timer)


def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEBUG_MODE = args.debug

    print(f"\n[ChronoStox v8.4] Initializing for: {args.ticker.upper()}")
    print("========================================")
    try:
        run_prediction(args.ticker.upper())
    except Exception as e:
        print("FATAL ERROR:", e)
        import traceback;
        traceback.print_exc()


if __name__ == "__main__":
    main()