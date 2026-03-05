# ===============================================================
# ChronoStox v10 - Hybrid Quant Engine (Regime-Aware)
# INTEGRATION:
#    - HMM "BEAR" regime -> Overrides BUY to HOLD.
#    - HMM "BULL" regime -> Allows aggressive signals.
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
import json

# TF IMPORT
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    tf.get_logger().setLevel("ERROR")
except Exception as e:
    print(f"FATAL: TensorFlow load error: {e}", file=sys.stderr)
    sys.exit(1)

# IMPORT HMM MODULE
try:
    import hmm_regime_detector as hmm_module
except ImportError:
    print("FATAL: Could not import 'hmm_regime_detector_v3.py'. Make sure it is in the same directory.",
          file=sys.stderr)
    sys.exit(1)

warnings.filterwarnings("ignore")

# ===============================================================
# CONFIG
# ===============================================================
MODEL_VERSION = "v10.0 (Regime-Aware Engine)"
HISTORY_DAYS = 400
MACRO_FILE = "macro_features.parquet"
SENTIMENT_FILE = "sentiment_clean.parquet"
SECTOR_FILE = "ticker.csv"
MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"

DEBUG_MODE = False
GLOBAL_END_DATE = None
HORIZONS = [5, 21, 63, 126, 252]
ATR_MULT = {5: 1.0, 21: 1.8, 63: 3.0, 126: 4.8, 252: 7.2}
WEIGHT_ML_H = {5: 0.62, 21: 0.55, 63: 0.50, 126: 0.42, 252: 0.35}
WEIGHT_ATR_H = {h: 1 - WEIGHT_ML_H[h] for h in HORIZONS}

# ===============================================================
#  STRATEGY KNOBS (v10)
# ===============================================================
STRATEGY_CONFIG = {
    # Base Thresholds (Modified by Regime)
    "BUY_THRESHOLDS": {5: 2.0, 21: 3.0, 63: 3.5, 126: 5.0, 252: 8.0},
    "SELL_THRESHOLDS": {5: -1.5, 21: -2.5, 63: -3.0, 126: -4.0, 252: -5.0},
    "CONFIDENCE_MAX_BUY": 15.0,
    "CONFIDENCE_MAX_SELL": -10.0,

    # HMM REGIME OVERRIDES
    # If HMM detects 'BEAR' or 'Crash', force HOLD?
    "HMM_BEAR_LOCKOUT": True,
    # If HMM detects 'BULL', lower buy thresholds by this multiplier?
    "HMM_BULL_AGGRESSION": 0.8  # e.g., 2.0% becomes 1.6%
}


# (Standard Data Loaders & Feature Engineering - Same as v9.5)
# ... (Timer Class) ...
class Timer:
    def __init__(self): self._t0, self._last, self.times = time.time(), time.time(), {}

    def mark(self, name):
        now = time.time();
        elapsed = now - self._last;
        self.times[name] = elapsed
        if DEBUG_MODE: print(f"[TIMER] {name} finished in {elapsed:.4f}s", file=sys.stderr)
        self._last = now

    def delta(self, name): return float(self.times.get(name, 0.0))

    def total(self): return time.time() - self._t0


# ... (safe_read_parquet) ...
def safe_read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"FATAL: Failed to read parquet {path}: {e}", file=sys.stderr); sys.exit(1)


# ... (load_models) ...
def load_models(model_dir, timer):
    try:
        joblib_path, keras_path = os.path.join(model_dir, MODEL_JOBLIB), os.path.join(model_dir, MODEL_KERAS)
        if not os.path.exists(joblib_path) or not os.path.exists(keras_path):
            print(f"FATAL: Model files not found.\nChecked for: {joblib_path}\nChecked for: {keras_path}",
                  file=sys.stderr);
            sys.exit(1)
        bundle, lstm_model = joblib.load(joblib_path), load_model(keras_path, compile=False)
    except Exception as e:
        print(f"FATAL: Error loading model files: {e}", file=sys.stderr); sys.exit(1)
    models = {
        "scaler": bundle["scaler"], "lgbm": bundle["model_lgbm"],
        "features": bundle["features"], "horizons": bundle["horizons"],
        "seq_len": bundle["lstm_sequence_length"], "lstm": lstm_model
    }
    if models["horizons"] != HORIZONS: print("FATAL: Horizon mismatch.", file=sys.stderr); sys.exit(1)
    timer.mark("Model Loading")
    return models


# ... (load_macro) ...
def load_macro(data_dir, timer):
    path = os.path.join(data_dir, MACRO_FILE)
    if not os.path.exists(path): print(f"FATAL: Macro file '{path}' not found.", file=sys.stderr); sys.exit(1)
    df_macro = safe_read_parquet(path)
    if "Date" not in df_macro.columns:
        if isinstance(df_macro.index, pd.DatetimeIndex):
            df_macro = df_macro.reset_index().rename(columns={"index": "Date"})
        else:
            print("FATAL: Macro file missing a datetime index or 'Date' column.", file=sys.stderr); sys.exit(1)
    df_macro["Date"] = pd.to_datetime(df_macro["Date"], errors="coerce").dt.tz_localize(None)
    df_macro = df_macro.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    timer.mark("Macro Loading")
    return df_macro


# ... (load_sentiment) ...
def load_sentiment(data_dir, timer):
    path = os.path.join(data_dir, SENTIMENT_FILE)
    if not os.path.exists(path): print(f"FATAL: Sentiment file '{path}' not found.", file=sys.stderr); sys.exit(1)
    df_s = safe_read_parquet(path)
    if "Date" not in df_s.columns:
        if isinstance(df_s.index, pd.DatetimeIndex):
            df_s = df_s.reset_index().rename(columns={"index": "Date"})
        else:
            print("FATAL: Sentiment parquet missing 'Date' column.", file=sys.stderr); sys.exit(1)
    df_s["Date"] = pd.to_datetime(df_s["Date"], errors="coerce").dt.tz_localize(None)
    df_s = df_s.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    ticker_col = next((c for c in df_s.columns if c.lower() in ["ticker", "tickeryf", "ticker_yf", "symbol"]), None)
    if ticker_col is None: print("FATAL: Sentiment file missing ticker column.", file=sys.stderr); sys.exit(1)
    df_s = df_s.rename(columns={ticker_col: "Ticker_YF"})
    df_s["Ticker_YF"] = df_s["Ticker_YF"].astype(str).str.upper().str.strip()
    if "sentiment_score" not in df_s.columns:
        sentiment_col = next((c for c in df_s.columns if "sentiment" in c.lower()), None)
        if sentiment_col:
            df_s = df_s.rename(columns={sentiment_col: "sentiment_score"})
        else:
            print("FATAL: Sentiment file missing sentiment_score column.", file=sys.stderr); sys.exit(1)
    df_s["sentiment_score"] = pd.to_numeric(df_s["sentiment_score"], errors="coerce").fillna(0.0)
    timer.mark("Sentiment Loading")
    return df_s


# ... (load_sectors) ...
def load_sectors(data_dir=None):
    path = SECTOR_FILE if data_dir is None else os.path.join(data_dir, SECTOR_FILE)
    if not os.path.exists(path): print(f"FATAL: ticker.csv not found at {path}", file=sys.stderr); sys.exit(1)
    try:
        df = pd.read_csv(path, low_memory=False);
        cols = df.columns.tolist()
        ticker_col, sector_col = None, None
        for c in cols:
            if str(c).lower() in ["ticker", "ticker_yf", "symbol"]: ticker_col = c
            if "sector" in str(c).lower(): sector_col = c
        if not ticker_col or not sector_col:
            df = pd.read_csv(path, header=None, low_memory=False);
            cols = df.columns.tolist()
            ticker_col, sector_col = cols[0], (cols[2] if len(cols) > 2 else cols[-1])
        df = df.rename(columns={ticker_col: "Ticker_YF", sector_col: "Sector"})
        df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip().str.upper()
        return df[["Ticker_YF", "Sector"]].drop_duplicates().set_index("Ticker_YF")
    except Exception as e:
        print(f"FATAL: Could not parse {path}. Error: {e}", file=sys.stderr); sys.exit(1)


# ... (load_price) ...
def load_price(ticker, timer):
    try:
        yf_obj = yf.Ticker(ticker)
        if GLOBAL_END_DATE:
            end_dt = pd.to_datetime(GLOBAL_END_DATE)
            start_dt = end_dt - pd.Timedelta(days=HISTORY_DAYS)
            df = yf_obj.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), interval="1d")
        else:
            df = yf_obj.history(period=f"{HISTORY_DAYS}d", interval="1d")
    except Exception as e:
        print(f"FATAL: Failed to fetch price data for {ticker}: {e}", file=sys.stderr); sys.exit(1)
    if df.empty: print(f"FATAL: No price data returned for {ticker}.", file=sys.stderr); sys.exit(1)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Ticker_YF"] = ticker
    timer.mark("Price Fetching")
    return df


# ... (engineer_features) ...
def engineer_features(df_price, df_macro, df_senti, df_sector_map, expected_cols, timer):
    ticker = df_price["Ticker_YF"].iloc[0].upper().strip()
    df = df_price.copy()
    try:
        df.ta.adx(length=14, append=True);
        df.ta.atr(length=14, append=True);
        df.ta.ema(length=50, append=True);
        df.ta.ema(length=200, append=True)
        bb = df.ta.bbands(length=5, append=False);
        df["BBB_5_2.0"] = bb.iloc[:, 3];
        df["BBP_5_2.0"] = bb.iloc[:, 4]
        mac = df.ta.macd(append=False);
        df["MACDh_12_26_9"] = mac.iloc[:, 1];
        df["MACDs_12_26_9"] = mac.iloc[:, 2]
        df.ta.rsi(append=True)
        ema200_col = next((c for c in df.columns if "EMA" in c and "200" in str(c)), None)
        df["close_to_ema200"] = df["Close"] / (df[ema200_col] + 1e-9) if ema200_col else np.nan
    except Exception as e:
        print("FATAL: TA computation failed:", e, file=sys.stderr); sys.exit(1)
    try:
        df = df.sort_values("Date");
        df_macro_sorted = df_macro.sort_values("Date")
        df = pd.merge_asof(df, df_macro_sorted, on="Date", direction="backward")
    except Exception as e:
        print("FATAL: Macro merge failed:", e, file=sys.stderr); sys.exit(1)
    try:
        ssub = df_senti[df_senti["Ticker_YF"] == ticker]
        if ssub.empty:
            df["sentiment_score"] = 0.0
        else:
            ssub_sorted = ssub.sort_values("Date")[["Date", "sentiment_score"]]
            df = pd.merge_asof(df, ssub_sorted, on="Date", direction="backward")
            df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    except Exception as e:
        print("FATAL: Sentiment merge failed:", e, file=sys.stderr); sys.exit(1)
    try:
        sector = str(df_sector_map.loc[ticker, "Sector"]) if ticker in df_sector_map.index else "nan"
        df["Sector"] = sector
        sector_cols = [c for c in expected_cols if c.startswith("Sector_")]
        for s_col in sector_cols: df[s_col] = 1.0 if sector == s_col.replace("Sector_", "") else 0.0
    except Exception as e:
        print("FATAL: Sector merge failed:", e, file=sys.stderr); sys.exit(1)
    try:
        interaction_cols = [c for c in expected_cols if c.startswith("sentiment_x_Sector_")]
        for i_col in interaction_cols:
            s_col = i_col.replace("sentiment_x_", "");
            df[i_col] = df[s_col] * df["sentiment_score"] if s_col in df.columns else 0.0
    except Exception as e:
        print("FATAL: Interaction creation failed:", e, file=sys.stderr); sys.exit(1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    missing_cols = set(expected_cols) - set(df.columns)
    for col in missing_cols: df[col] = 0.0
    try:
        df_final_features = df[expected_cols].copy()
    except Exception as e:
        print("FATAL: Feature mismatch. Missing columns:", e, file=sys.stderr); sys.exit(1)
    timer.mark("Feature Engineering")
    return df_final_features, df


# ===============================================================
#   CORE V10 PREDICTION LOGIC
# ===============================================================
def get_specialist_predictions(models, df_features, timer):
    seq_len, scaler, model_lgb, model_lstm = models["seq_len"], models["scaler"], models["lgbm"], models["lstm"]
    try:
        X_row = df_features.iloc[-1].values.astype(np.float32).reshape(1, -1)
        Xs = scaler.transform(X_row)
        raw_lgb = model_lgb.predict(Xs)[0]

        if len(df_features) >= seq_len:
            seq_df = df_features.tail(seq_len).values.astype(np.float32)
            seq_s = scaler.transform(seq_df)
            lstm_in = seq_s.reshape(1, seq_len, -1)
            raw_lstm = model_lstm.predict(lstm_in, verbose=0)[0]
        else:
            raw_lstm = np.zeros_like(raw_lgb)

        timer.mark("Specialist Prediction")
        return raw_lgb, raw_lstm
    except Exception as e:
        print(f"FATAL: Specialist prediction failed. {e}", file=sys.stderr);
        sys.exit(1)


def compute_hybrid_targets(raw_lgb, raw_lstm, close_price, timer):
    raw_pred_reg = (raw_lgb + raw_lstm) / 2.0
    ml_prices = close_price * (1.0 + raw_pred_reg)
    atr = (close_price * 0.01)
    atr_mult = np.array([ATR_MULT[h] for h in HORIZONS])
    atr_expansion = close_price + (atr * atr_mult)

    hybrid_targets = []
    for i, h in enumerate(HORIZONS):
        target = (ml_prices[i] * WEIGHT_ML_H[h]) + (atr_expansion[i] * WEIGHT_ATR_H[h])
        hybrid_targets.append(target)
    timer.mark("Price Target Gen")
    return hybrid_targets


# ===============================================================
#   !!! V10 SIGNAL LOGIC (Regime-Aware) !!!
# ===============================================================
def compute_signals(hybrid_targets, close_price, regime_label, timer):
    """
    Uses v9.5 deterministic thresholds, but ADJUSTS them based on HMM Regime.
    """
    signals = []

    # 1. Get Regime Multipliers
    buy_mult = 1.0
    sell_mult = 1.0
    force_hold = False

    if "BEAR" in regime_label or "Crash" in regime_label:
        if STRATEGY_CONFIG["HMM_BEAR_LOCKOUT"]:
            force_hold = True  # No Buying allowed
    elif "BULL" in regime_label:
        buy_mult = STRATEGY_CONFIG["HMM_BULL_AGGRESSION"]  # Lower threshold = easier to buy

    # Unpack base knobs
    buy_thresholds = STRATEGY_CONFIG["BUY_THRESHOLDS"]
    sell_thresholds = STRATEGY_CONFIG["SELL_THRESHOLDS"]
    conf_max_buy = STRATEGY_CONFIG["CONFIDENCE_MAX_BUY"] / 100.0
    conf_max_sell = STRATEGY_CONFIG["CONFIDENCE_MAX_SELL"] / 100.0

    for i, h in enumerate(HORIZONS):
        target = hybrid_targets[i]
        try:
            # 2. Apply Regime Modifier to Thresholds
            buy_thresh_h = (buy_thresholds[h] * buy_mult) / 100.0
            sell_thresh_h = (sell_thresholds[h] * sell_mult) / 100.0

            expected_return = (target - close_price) / close_price

            # 3. Determine SIGNAL
            signal = "HOLD"

            if force_hold:
                signal = "HOLD"  # Regime Lockout
            elif expected_return > buy_thresh_h:
                signal = "BUY"
            elif expected_return < sell_thresh_h:
                signal = "SELL"

            # 4. Determine CONFIDENCE
            confidence = 0.0
            if signal == "BUY":
                confidence = np.interp(expected_return, [buy_thresh_h, conf_max_buy], [50.0, 100.0])
            elif signal == "SELL":
                confidence = np.interp(expected_return, [sell_thresh_h, conf_max_sell], [50.0, 100.0])
            else:
                if expected_return > 0:
                    confidence = 100 - np.interp(expected_return, [0, buy_thresh_h], [0, 100])
                else:
                    confidence = 100 - np.interp(expected_return, [0, sell_thresh_h], [0, 100])

            confidence = np.clip(confidence, 0, 100)

            signals.append({
                "signal": signal,
                "confidence": confidence,
                "expected_return_pct": expected_return * 100
            })

        except Exception:
            signals.append({"signal": "FAIL", "confidence": 0, "expected_return_pct": 0})

    timer.mark("Signal Gen")
    return signals


# ===============================================================
#   REPORTING
# ===============================================================
def print_report(ticker, close_price, hybrid_targets, signals, regime_label, timer, as_json=False):
    report = {
        "metadata": {
            "ticker": ticker, "model_version": MODEL_VERSION,
            "generated_at": datetime.now().isoformat(),
            "last_close": f"{close_price:.2f}",
            "regime": regime_label
        },
        "signals": []
    }

    for i, h in enumerate(HORIZONS):
        sig = signals[i]
        report["signals"].append({
            "horizon_days": h, "signal": sig['signal'],
            "confidence_pct": round(sig['confidence'], 2),
            "price_target": round(hybrid_targets[i], 2),
            "expected_return_pct": round(sig['expected_return_pct'], 2)
        })

    if as_json:
        print(json.dumps(report));
        return

    print("========================================")
    print(f"ChronoStox {MODEL_VERSION} Report")
    print(f"Ticker       : {ticker}")
    print(f"Generated    : {report['metadata']['generated_at']}")
    print(f"Last Close   : {report['metadata']['last_close']}")
    print("----------------------------------------")
    print(f"Market Regime: {regime_label}  <-- (HMM Detected)")
    print("----------------------------------------")

    header = "HORIZON | SIGNAL | CONF   | EXP. RETURN | TARGET"
    print(header);
    print("-" * len(header))

    for sig in report['signals']:
        h, p = sig['horizon_days'], sig['expected_return_pct']
        signal_str = sig['signal'].ljust(6)
        conf_str = f"{sig['confidence_pct']:.1f}%".ljust(6)
        return_str = f"{p: >+6.2f}%".ljust(11)
        target_str = f"{sig['price_target']:.2f}"
        print(f"{str(h).ljust(8)}| {signal_str}| {conf_str} | {return_str} | {target_str}")
    print("========================================")


# ===============================================================
# MASTER PIPELINE
# ===============================================================
def run_prediction(ticker, as_json=False):
    global DEBUG_MODE
    timer = Timer()

    # 1. Load Data & Models
    model_dir = "." if os.path.exists(MODEL_JOBLIB) else "../test"
    models = load_models(model_dir, timer)
    df_macro = load_macro(model_dir, timer)
    df_senti = load_sentiment(model_dir, timer)
    df_sectors = load_sectors(model_dir)
    df_raw = load_price(ticker, timer)
    close_price = float(df_raw["Close"].iloc[-1])

    # 2. Feature Engineering
    df_feat, df_full_merged = engineer_features(
        df_raw, df_macro, df_senti, df_sectors, models["features"], timer
    )

    # 3. HMM Regime Detection (The New Brain)
    # We perform this on-the-fly using the raw price history
    try:
        hmm_df = hmm_module.fetch_data(ticker)  # Re-fetch 2y for HMM
        hmm_model, hmm_states = hmm_module.fit_hmm(hmm_df)
        regime_map = hmm_module.interpret_states(hmm_model, hmm_states)
        current_state = hmm_states["state"].iloc[-1]
        regime_label = regime_map[current_state]
        timer.mark("HMM Regime Detect")
    except Exception as e:
        regime_label = "Unknown (HMM Failed)"
        if DEBUG_MODE: print(f"HMM Error: {e}", file=sys.stderr)

    # 4. Prediction
    raw_lgb, raw_lstm = get_specialist_predictions(models, df_feat, timer)
    hybrid_targets = compute_hybrid_targets(raw_lgb, raw_lstm, close_price, timer)

    # 5. Signals (Regime-Aware)
    signals = compute_signals(hybrid_targets, close_price, regime_label, timer)

    # 6. Output
    print_report(ticker, close_price, hybrid_targets, signals, regime_label, timer, as_json=as_json)


def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser(description=f"ChronoStox {MODEL_VERSION} CLI")
    parser.add_argument("ticker", type=str, help="Ticker symbol")
    parser.add_argument("-j", "--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEBUG_MODE = args.debug

    try:
        run_prediction(args.ticker.upper(), as_json=args.json)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()