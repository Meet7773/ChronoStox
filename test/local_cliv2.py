# ===============================================================
# ChronoStox v8 - Hybrid Quant Engine (ATR + ML Model C)
# Full CLI Version - corrected & self-contained
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

HISTORY_DAYS = 400        # fetch extra to stabilize indicators
MACRO_FILE = "macro_features.parquet"
# Use the exact local filename you mentioned earlier
SENTIMENT_FILE = "ticker_sentiment_scores (1).parquet"
SECTOR_FILE = "ticker.csv"
MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"

DEBUG_MODE = False   # toggled by --debug

# Horizons
HORIZONS = [5, 21, 63, 126, 252]

# ATR multipliers for targets
ATR_MULT = {
    5:   1.0,
    21:  1.8,
    63:  3.0,
    126: 4.8,
    252: 7.2
}


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
        # try auto-find
        try:
            candidate = auto_find_sentiment_file()
            print(f"Using detected sentiment file: {candidate}")
            path = candidate
        except Exception:
            print(f"FATAL: Sentiment file '{path}' not found.")
            sys.exit(1)

    df_s = safe_read_parquet(path)

    # Normalise date
    if "Date" not in df_s.columns:
        # maybe index-based
        if isinstance(df_s.index, pd.DatetimeIndex):
            df_s = df_s.reset_index().rename(columns={"index": "Date"})
        else:
            print("FATAL: Sentiment parquet missing 'Date' column.")
            print("Columns found:", df_s.columns)
            sys.exit(1)

    df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce")
    df_s["Date"] = df_s["Date"].dt.tz_localize(None)
    df_s = df_s.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Standardize ticker col name
    ticker_col = None
    for c in df_s.columns:
        if c.lower() in ["ticker", "tickeryf", "ticker_yf", "symbol", "stock"]:
            ticker_col = c
            break
    if ticker_col is None:
        # try heuristics
        for c in df_s.columns:
            if "ticker" in c.lower() or "symbol" in c.lower():
                ticker_col = c
                break

    if ticker_col is None:
        print("FATAL: Sentiment file missing ticker column.")
        print("Columns available:", df_s.columns)
        sys.exit(1)

    df_s = df_s.rename(columns={ticker_col: "Ticker_YF"})

    # Standardize sentiment_score name
    if "sentiment_score" not in df_s.columns:
        found = None
        for c in df_s.columns:
            if "sentiment" in c.lower():
                found = c
                break
        if found:
            df_s = df_s.rename(columns={found: "sentiment_score"})
    if "sentiment_score" not in df_s.columns:
        print("FATAL: Sentiment file missing sentiment_score column.")
        print("Columns available:", df_s.columns)
        sys.exit(1)

    timer.mark("Sentiment Loading")
    if DEBUG_MODE:
        print("\n===== DEBUG: SENTIMENT FEATURES =====")
        print(df_s.head(5))
        print(df_s.tail(5))
        print("Sentiment columns:", df_s.columns.tolist())

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
        df = yf.Ticker(ticker).history(period=f"{HISTORY_DAYS}d", interval="1d")
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
    try:
        # Normalize ticker column in sentiment
        df_senti_local = df_senti.rename(columns={"Ticker_YF": "Ticker_YF"})
        ticker = df["Ticker_YF"].iloc[0]
        ssub = df_senti_local[df_senti_local["Ticker_YF"] == ticker].sort_values("Date")
        if ssub.empty:
            # fallback: use market-wide daily sentiment if present (Date-indexed)
            if "sentiment_score" in df_senti_local.columns and "Ticker_YF" not in df_senti_local.columns:
                ssub = df_senti_local[["Date", "sentiment_score"]].sort_values("Date")
            else:
                # create empty
                ssub = pd.DataFrame(columns=["Date", "sentiment_score"])

        df = pd.merge_asof(df, ssub[["Date", "sentiment_score"]], on="Date", direction="backward")
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
                score += 25
            else:
                score += 5

        # MACD hist
        if "MACDh_12_26_9" in latest and latest["MACDh_12_26_9"] > 0:
            score += 25

        # RSI
        if "RSI_14" in latest:
            rsi = latest["RSI_14"]
            if 50 < rsi < 70:
                score += 25
            elif rsi >= 70:
                score += 10
            else:
                score += 5

        # close_to_ema200
        if "close_to_ema200" in latest:
            ratio = latest["close_to_ema200"]
            if ratio > 1.03:
                score += 25
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
    try:
        latest = df.iloc[-1]
        atrp = None
        if "ATRr_14" in df.columns:
            atrp = latest.get("ATRr_14", None)
            if atrp is not None:
                atrp = atrp * 100
        if atrp is None:
            if "ATR_14" in latest:
                atrp = (latest["ATR_14"] / latest["Close"]) * 100
        if atrp is None:
            return "Normal"

        if atrp < 1:
            return "Calm"
        elif atrp < 2:
            return "Normal"
        elif atrp < 4:
            return "Volatile"
        else:
            return "High-Risk"
    except:
        return "Normal"


# ===============================================================
# PRICE TARGET ENGINE (ATR + ML Hybrid)
# ===============================================================
def compute_price_targets(df_full, model_lgb, model_lstm, scaler, seq_len, feature_order):
    """
    df_full: full merged df (with Close & ATR & feature cols)
    feature_order: list of expected feature column names in the same order as scaler/model
    """
    close = float(df_full["Close"].iloc[-1])

    # ML raw preds
    try:
        X_row = df_full[feature_order].iloc[-1].values.astype(np.float32).reshape(1, -1)
        Xs = scaler.transform(X_row)

        # LGBM
        raw_lgb = model_lgb.predict(Xs)[0]

        # LSTM
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

    # ATR bands (use ATR_14 if available)
    try:
        atr = float(df_full["ATR_14"].iloc[-1]) if "ATR_14" in df_full.columns else (close * 0.01)
    except:
        atr = close * 0.01

    # multiplier vector
    atr_mult = np.array([ATR_MULT[h] for h in HORIZONS])
    atr_expansion = close + (atr * atr_mult)

    # hybrid: 70% ML (relative returns) + 30% ATR band (absolute)
    # convert raw_pred (returns) to price: close * (1 + raw_pred)
    ml_prices = close * (1.0 + raw_pred)
    hybrid = ml_prices * 0.70 + atr_expansion * 0.30

    return {
        "raw": raw_pred,
        "atr": atr_expansion,
        "hybrid": hybrid
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

    # sanity
    if df_features.shape[1] != len(feature_order):
        # if mismatch, attempt to reorder columns
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
        print("ATR bands:", preds["atr"])
        print("Hybrid:", preds["hybrid"])

    return preds


# ===============================================================
#   SIGNAL ENGINE / CONFIDENCE / RISK / REPORT (unchanged)
# ===============================================================
def classify_signal(pred_ret):
    if pred_ret < -0.015:
        return "SELL"
    elif pred_ret > 0.020:
        return "BUY"
    else:
        return "HOLD"


def compute_confidence(pred_ret, trend_score, macro_score, vol_regime):
    ml_conf = min(100, abs(pred_ret) * 1600)
    base = (ml_conf * 0.4) + (trend_score * 0.3) + (macro_score * 0.2)
    if vol_regime == "High-Risk":
        base *= 0.55
    elif vol_regime == "Volatile":
        base *= 0.75
    return int(min(100, base))


def compute_risk_flags(trend_score, macro_score, vol_regime, sentiment, df):
    warnings = []
    if vol_regime == "High-Risk":
        warnings.append("⚠ Market volatility extremely high")
    if macro_score < 40:
        warnings.append("⚠ Macro headwinds detected")
    if trend_score < 40:
        warnings.append("⚠ Weak price trend")
    if "sentiment_score" in df.columns:
        if abs(df["sentiment_score"].iloc[-1]) < 0.02:
            warnings.append("⚠ Neutral sentiment (low conviction)")
        elif df["sentiment_score"].iloc[-1] < -0.15:
            warnings.append("⚠ Strong negative sentiment")
    if len(warnings) == 0:
        warnings.append("No major risk flags")
    return warnings


def print_report(ticker, close_price, preds, trend_score, macro_score, vol_regime, warnings, timer):
    print("========================================")
    print(f"ChronoStox v8 Quant Signal Report")
    print(f"Ticker       : {ticker}")
    print(f"Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Last Close   : {close_price:.2f}")
    print("----------------------------------------")
    print(f"Trend Score  : {trend_score}/100")
    print(f"Macro Score  : {macro_score}/100")
    print(f"Vol Regime   : {vol_regime}")
    print("----------------------------------------")
    print(f"Model Latency: {timer.delta('Prediction'):.4f}s")
    print("----------------------------------------")
    print("HORIZON | SIGNAL | CONFIDENCE | PRICE TARGET")
    print("----------------------------------------")

    for i, h in enumerate(HORIZONS):
        pred = float(preds["raw"][i])
        signal = classify_signal(pred)
        conf = compute_confidence(pred, trend_score, macro_score, vol_regime)
        target = float(preds["hybrid"][i])
        print(f"{str(h).ljust(8)} | {signal.ljust(6)} | {str(conf).ljust(10)} | {target:.2f}")

    print("----------------------------------------")
    print("RISK FLAGS:")
    for w in warnings:
        print(" -", w)
    print("========================================")


# ===============================================================
# MASTER PREDICTION PIPELINE
# ===============================================================
def run_prediction(ticker):
    global DEBUG_MODE
    timer = Timer()

    # 1) Load models
    models = load_models(".", timer)

    # 2) Load macro
    df_macro = load_macro(".", timer)

    # 3) Load sentiment
    df_senti = load_sentiment(".", timer)

    # 3.5) Load sectors
    df_sectors = load_sectors()

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
    parser = argparse.ArgumentParser(description="ChronoStox v8 CLI")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., RELIANCE.NS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    DEBUG_MODE = args.debug

    print(f"\n[ChronoStox v8] Initializing for: {args.ticker.upper()}")
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
