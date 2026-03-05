# ===============================================================
# ChronoStox v9.5 — Out-of-Sample Backtester (Combined+ Edition)
#
# SIGNAL STRATEGIES (all tracked):
#   A) HYBRID: bidirectional ATR + ML price targets
#   B) META-LEARNER: XGBoost classifier on raw LGBM+LSTM outputs
#   C) COMBINED: trade only when BOTH hybrid and meta agree
#   D) REGIME-AWARE: HMM modulates signal filtering
#   E) COMBINED+ (default): combined + confidence floor + multi-horizon
#      + per-ticker quality filter
#
# Guardrails: volatility clamp (prevents hallucination)
#
# USAGE:
#   cd test
#   python backtest_oos.py
#   python backtest_oos.py --tickers RELIANCE.NS TCS.NS INFY.NS
#   python backtest_oos.py --start 2025-11-15 --end 2026-02-15
#   python backtest_oos.py --weekly     (default, simulate every Friday)
#   python backtest_oos.py --daily      (simulate every trading day — slow)
# ===============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import time
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"       # Fix Windows HMM/loky crash
os.environ["JOBLIB_VERBOSITY"] = "0"
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tf.get_logger().setLevel("ERROR")
except Exception as e:
    print("FATAL: TensorFlow load error:", e)
    sys.exit(1)

import pandas_ta as ta

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠ hmmlearn not installed — HMM regime detection disabled. pip install hmmlearn")

warnings.filterwarnings("ignore")

# ===============================================================
# CONFIG
# ===============================================================

MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"
MACRO_FILE = "macro_features.parquet"
SENTIMENT_FILE = "sentiment_clean.parquet"
SECTOR_FILE = "ticker.csv"

HORIZONS = [5, 21, 63, 126, 252]
ATR_MULT = {5: 1.0, 21: 1.8, 63: 3.0, 126: 4.8, 252: 7.2}

WEIGHT_ML_H = {5: 0.62, 21: 0.55, 63: 0.50, 126: 0.42, 252: 0.35}
WEIGHT_ATR_H = {h: 1 - WEIGHT_ML_H[h] for h in HORIZONS}

# Default tickers — comprehensive cross-section of Indian market
# Covers: Large-cap, Mid-cap, PSU, across all sectors
DEFAULT_TICKERS = [
    # === Banking & Financial ===
    "HDFCBANK.NS",     # Large-cap Private Bank
    "ICICIBANK.NS",    # Large-cap Private Bank
    "SBIN.NS",         # PSU Bank
    "KOTAKBANK.NS",    # Large-cap Private Bank
    "BAJFINANCE.NS",   # NBFC
    "BANKBARODA.NS",   # PSU Mid-cap Bank
    "PNB.NS",          # PSU Bank
    "HDFCLIFE.NS",     # Insurance
    # === Technology ===
    "TCS.NS",          # Large-cap IT
    "INFY.NS",         # Large-cap IT
    "WIPRO.NS",        # Large-cap IT
    "HCLTECH.NS",      # Large-cap IT
    "TECHM.NS",        # Mid-cap IT
    "LTIM.NS",         # Mid-cap IT
    # === Energy & Oil ===
    "RELIANCE.NS",     # Conglomerate/Energy
    "ONGC.NS",         # PSU Oil
    "BPCL.NS",         # PSU Oil
    "COALINDIA.NS",    # PSU Mining
    "NTPC.NS",         # PSU Power
    "POWERGRID.NS",    # PSU Utilities
    # === FMCG & Consumer ===
    "HINDUNILVR.NS",   # Large-cap FMCG
    "ITC.NS",          # Large-cap FMCG
    "NESTLEIND.NS",    # FMCG
    "TITAN.NS",        # Consumer Discretionary
    "DABUR.NS",        # Mid-cap FMCG
    "MARICO.NS",       # Mid-cap FMCG
    # === Pharma & Healthcare ===
    "SUNPHARMA.NS",    # Large-cap Pharma
    "DRREDDY.NS",      # Large-cap Pharma
    "CIPLA.NS",        # Large-cap Pharma
    "APOLLOHOSP.NS",   # Healthcare
    # === Auto & Manufacturing ===
    "MARUTI.NS",       # Large-cap Auto
    "M&M.NS",          # Large-cap Auto
    "TATAMOTORS.NS",   # Large-cap Auto
    "BAJAJ-AUTO.NS",   # Two-wheeler
    "EICHERMOT.NS",    # Mid-cap Auto
    # === Industrials & Infra ===
    "LT.NS",           # Large-cap Infra
    "ADANIENT.NS",     # Conglomerate
    "ADANIPORTS.NS",   # Ports/Infra
    "ULTRACEMCO.NS",   # Cement
    "GRASIM.NS",       # Diversified
    # === Telecom & Media ===
    "BHARTIARTL.NS",   # Large-cap Telecom
    # === Metals & Mining ===
    "TATASTEEL.NS",    # Steel
    "HINDALCO.NS",     # Aluminium
    "JSWSTEEL.NS",     # Steel
    # === Real Estate ===
    "DLF.NS",          # Real Estate
    # === Defensive/Others ===
    "ASIANPAINT.NS",   # Paints
    "BRITANNIA.NS",    # FMCG/Defensive
]

# ===============================================================
# DATA LOADING (same as engine)
# ===============================================================

def safe_read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"FATAL: Failed to read parquet {path}: {e}")
        sys.exit(1)


def load_models_once(model_dir):
    joblib_path = os.path.join(model_dir, MODEL_JOBLIB)
    keras_path = os.path.join(model_dir, MODEL_KERAS)

    bundle = joblib.load(joblib_path)
    lstm_model = load_model(keras_path, compile=False)

    return {
        "scaler": bundle["scaler"],
        "lgbm": bundle["model_lgbm"],
        "meta_models": bundle.get("meta_models", {}),
        "features": bundle["features"],
        "horizons": bundle["horizons"],
        "seq_len": bundle["lstm_sequence_length"],
        "lstm": lstm_model
    }


def load_macro_once(data_dir):
    path = os.path.join(data_dir, MACRO_FILE)
    df = safe_read_parquet(path)
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Date"})
        else:
            print("FATAL: Macro file missing Date column.")
            sys.exit(1)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def load_sentiment_once(data_dir):
    path = os.path.join(data_dir, SENTIMENT_FILE)
    df = safe_read_parquet(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        print("FATAL: Sentiment missing Date"); sys.exit(1)

    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Standardise columns
    for c in df.columns:
        if c.lower() in ["ticker", "tickeryf", "ticker_yf", "symbol"]:
            df = df.rename(columns={c: "Ticker_YF"}); break
    if "Ticker_YF" not in df.columns:
        for c in df.columns:
            if "ticker" in c.lower():
                df = df.rename(columns={c: "Ticker_YF"}); break

    if "sentiment_score" not in df.columns:
        for c in df.columns:
            if "sentiment" in c.lower():
                df = df.rename(columns={c: "sentiment_score"}); break

    return df


def load_sectors_once(data_dir):
    path = os.path.join(data_dir, SECTOR_FILE)
    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        df = pd.read_csv(path, header=None, low_memory=False)

    cols = df.columns.tolist()
    ticker_col, sector_col = None, None
    for c in cols:
        if str(c).lower() in ["ticker", "ticker_yf", "symbol"]: ticker_col = c
        if "sector" in str(c).lower(): sector_col = c
    if not ticker_col: ticker_col = cols[0]
    if not sector_col: sector_col = cols[2] if len(cols) > 2 else cols[-1]

    df = df.rename(columns={ticker_col: "Ticker_YF", sector_col: "Sector"})
    df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip().str.upper()
    return df[["Ticker_YF", "Sector"]].drop_duplicates()


def fetch_price_history(ticker, start_date="2024-01-01"):
    """Fetch full price history for a ticker via yfinance."""
    try:
        df = yf.Ticker(ticker).history(start=start_date, interval="1d")
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df["Ticker_YF"] = ticker
        return df
    except Exception as e:
        print(f"  ⚠ Failed to fetch {ticker}: {e}")
        return None


# ===============================================================
# FEATURE ENGINEERING (adapted from engine)
# ===============================================================

def engineer_features_for_slice(df_price, df_macro, df_senti, df_sector, expected_cols):
    """
    Run feature engineering on a price slice.
    Returns (df_final, df_full) or (None, None) on failure.
    """
    try:
        df = df_price.copy()

        # TA indicators
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)

        bb = df.ta.bbands(length=5, append=False)
        if bb is not None and not bb.empty:
            df["BBB_5_2.0"] = bb.iloc[:, 3] if "BBB_5_2.0" not in bb.columns else bb["BBB_5_2.0"]
            df["BBP_5_2.0"] = bb.iloc[:, 4] if "BBP_5_2.0" not in bb.columns else bb["BBP_5_2.0"]

        mac = df.ta.macd(append=False)
        if mac is not None and not mac.empty and mac.shape[1] >= 3:
            df["MACDh_12_26_9"] = mac.iloc[:, 1]
            df["MACDs_12_26_9"] = mac.iloc[:, 2]

        df.ta.rsi(append=True)

        ema200 = [c for c in df.columns if "EMA" in c and "200" in str(c)]
        df["close_to_ema200"] = df["Close"] / (df[ema200[0]] + 1e-9) if ema200 else np.nan

        # Macro merge
        df = df.sort_values("Date").reset_index(drop=True)
        df = pd.merge_asof(df, df_macro.sort_values("Date"), on="Date", direction="backward")

        # Sentiment merge
        ticker = df["Ticker_YF"].iloc[0].upper().strip()
        df_sl = df_senti.copy()
        df_sl["Ticker_YF"] = df_sl["Ticker_YF"].astype(str).str.upper().str.strip()
        ssub = df_sl[df_sl["Ticker_YF"] == ticker]
        if ssub.empty:
            ssub = pd.DataFrame({"Date": df["Date"], "sentiment_score": np.zeros(len(df))})
        ssub = ssub.sort_values("Date")[["Date", "sentiment_score"]]
        df = pd.merge_asof(df, ssub, on="Date", direction="backward", allow_exact_matches=True)

        # Sector
        sec_row = df_sector[df_sector["Ticker_YF"] == ticker]
        sector = sec_row["Sector"].iloc[0] if not sec_row.empty else "nan"
        sector = str(sector).strip()
        df["Sector"] = sector

        sectors = ["Communication Services", "Consumer Cyclical", "Consumer Defensive",
                    "Energy", "Financial Services", "Healthcare", "Industrials",
                    "Real Estate", "Technology", "Utilities", "nan"]
        for s in sectors:
            df[f"Sector_{s}"] = 1.0 if sector == s else 0.0

        # Interactions
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0
        for c in df.columns:
            if str(c).startswith("Sector_"):
                df[f"sentiment_x_{c}"] = df[c].astype(float) * df["sentiment_score"].astype(float)

        # Cleanup
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in set(expected_cols) - set(df.columns):
            df[col] = 0.0

        df_final = df[expected_cols].copy()
        return df_final, df

    except Exception as e:
        return None, None


# ===============================================================
# PREDICTION (FIXED — bidirectional ATR, meta-learner driven)
# ===============================================================

def predict_at_date(df_full, models, dynamic_weights=None):
    """
    Run prediction using the full merged dataframe.
    Returns dict with raw predictions, bidirectional hybrid targets,
    and meta-learner signals.
    """
    close = float(df_full["Close"].iloc[-1])
    feature_order = models["features"]
    scaler = models["scaler"]
    model_lgb = models["lgbm"]
    model_lstm = models["lstm"]
    seq_len = models["seq_len"]

    try:
        X_row = df_full[feature_order].iloc[-1].values.astype(np.float32).reshape(1, -1)
        Xs = scaler.transform(X_row)
        raw_lgb = model_lgb.predict(Xs)[0]

        if len(df_full) >= seq_len:
            seq = df_full[feature_order].tail(seq_len).values.astype(np.float32).reshape(1, seq_len, -1)
            raw_lstm = model_lstm.predict(seq, verbose=0)[0]
        else:
            raw_lstm = np.zeros_like(raw_lgb)

        if dynamic_weights is not None:
            w_lgb = dynamic_weights.get("lgbm", 0.5)
            w_lstm = dynamic_weights.get("lstm", 0.5)
        else:
            w_lgb = 0.5
            w_lstm = 0.5
            
        raw_avg = (raw_lgb * w_lgb) + (raw_lstm * w_lstm)
    except Exception:
        raw_lgb = np.zeros(len(HORIZONS))
        raw_lstm = np.zeros(len(HORIZONS))
        raw_avg = np.zeros(len(HORIZONS))

    # ATR
    atr_col = None
    for c in ["ATR_14", "ATRr_14", "ATR"]:
        if c in df_full.columns:
            atr_col = c; break
    atr = float(df_full[atr_col].iloc[-1]) if atr_col else close * 0.01

    vol_factor = atr / close

    # --- FIX: BIDIRECTIONAL ATR expansion ---
    # ATR component now goes UP when ML predicts positive, DOWN when negative
    # This removes the permanent bullish bias from the old version
    ml_prices = close * (1.0 + raw_avg)

    hybrid = []
    for i, h in enumerate(HORIZONS):
        atr_offset = atr * ATR_MULT[h]
        ml_direction = np.sign(raw_avg[i])  # +1 if bullish, -1 if bearish, 0 if flat

        # ATR expansion follows ML direction
        if ml_direction >= 0:
            atr_price = close + atr_offset
        else:
            atr_price = close - atr_offset

        raw_h = ml_prices[i] * WEIGHT_ML_H[h] + atr_price * WEIGHT_ATR_H[h]

        # GUARDRAIL: volatility-based clamp (prevents hallucination)
        max_move = vol_factor * np.sqrt(h) * 2.0
        clamped = max(close * (1 - max_move), min(close * (1 + max_move), raw_h))
        hybrid.append(clamped)

    # Meta-learner signals
    meta_signals = {}
    meta_models = models.get("meta_models", {})
    for i, h in enumerate(HORIZONS):
        if h in meta_models:
            try:
                X_meta = np.array([[raw_lgb[i], raw_lstm[i]]], dtype=np.float32)
                proba = meta_models[h].predict_proba(X_meta)[0]
                pred_class = int(meta_models[h].predict(X_meta)[0])
                meta_signals[h] = {
                    "class": pred_class,     # 0=Sell, 1=Hold, 2=Buy
                    "proba": proba.tolist(),
                    "signal": ["SELL", "HOLD", "BUY"][pred_class],
                    "confidence": float(max(proba) - sorted(proba)[-2])  # gap between top two
                }
            except:
                meta_signals[h] = {"class": 1, "proba": [0, 1, 0], "signal": "HOLD", "confidence": 0.0}

    return {
        "close": close,
        "raw_lgb": raw_lgb,
        "raw_lstm": raw_lstm,
        "raw_avg": raw_avg,
        "hybrid": np.array(hybrid),
        "predicted_returns": {h: (hybrid[i] - close) / close for i, h in enumerate(HORIZONS)},
        "meta_signals": meta_signals
    }


# ===============================================================
# METRICS COMPUTATION (tracks both hybrid and meta-learner)
# ===============================================================

def compute_metrics(results_df):
    """Compute comprehensive backtest metrics from results."""
    metrics = {}

    for h in HORIZONS:
        pred_col = f"pred_ret_{h}d"
        actual_col = f"actual_ret_{h}d"

        if pred_col not in results_df.columns or actual_col not in results_df.columns:
            continue

        df = results_df.dropna(subset=[pred_col, actual_col])
        if len(df) == 0:
            continue

        pred = df[pred_col].values
        actual = df[actual_col].values

        # Directional accuracy (hybrid-based)
        pred_dir = np.sign(pred)
        actual_dir = np.sign(actual)
        dir_acc = np.mean(pred_dir == actual_dir) * 100

        # MAE and RMSE (as %)
        mae = np.mean(np.abs(pred - actual)) * 100
        rmse = np.sqrt(np.mean((pred - actual) ** 2)) * 100

        # Correlation
        corr = np.corrcoef(pred, actual)[0, 1] if len(pred) > 2 else 0.0

        # Mean predicted vs mean actual
        mean_pred = np.mean(pred) * 100
        mean_actual = np.mean(actual) * 100

        # --- Hybrid-based signals (pred_ret > 0.5% = BUY, < -0.5% = SELL) ---
        buy_mask = pred > 0.005
        sell_mask = pred < -0.005
        hold_mask = ~buy_mask & ~sell_mask

        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        hold_count = int(hold_mask.sum())

        buy_actual = actual[buy_mask].mean() * 100 if buy_mask.any() else None
        sell_actual = actual[sell_mask].mean() * 100 if sell_mask.any() else None

        buy_win_rate = (actual[buy_mask] > 0).mean() * 100 if buy_mask.any() else None
        sell_win_rate = (actual[sell_mask] < 0).mean() * 100 if sell_mask.any() else None

        # --- Meta-learner signal analysis ---
        meta_col = f"meta_signal_{h}d"
        meta_acc = None
        meta_buy_count = 0
        meta_sell_count = 0
        meta_hold_count = 0
        meta_buy_win_rate = None
        meta_sell_win_rate = None
        meta_buy_avg_ret = None
        meta_sell_avg_ret = None

        if meta_col in results_df.columns:
            dm = df.dropna(subset=[meta_col])
            if len(dm) > 0:
                meta_correct = 0
                for _, row in dm.iterrows():
                    sig = row[meta_col]
                    act = row[actual_col]
                    if sig == "BUY" and act > 0:
                        meta_correct += 1
                    elif sig == "SELL" and act < 0:
                        meta_correct += 1
                    elif sig == "HOLD" and abs(act) < 0.05:
                        meta_correct += 1
                meta_acc = meta_correct / len(dm) * 100

                # Meta signal counts & win rates
                meta_buy_mask = dm[meta_col] == "BUY"
                meta_sell_mask = dm[meta_col] == "SELL"
                meta_hold_mask = dm[meta_col] == "HOLD"

                meta_buy_count = int(meta_buy_mask.sum())
                meta_sell_count = int(meta_sell_mask.sum())
                meta_hold_count = int(meta_hold_mask.sum())

                meta_buy_actuals = dm.loc[meta_buy_mask, actual_col].values
                meta_sell_actuals = dm.loc[meta_sell_mask, actual_col].values

                if len(meta_buy_actuals) > 0:
                    meta_buy_win_rate = (meta_buy_actuals > 0).mean() * 100
                    meta_buy_avg_ret = meta_buy_actuals.mean() * 100
                if len(meta_sell_actuals) > 0:
                    meta_sell_win_rate = (meta_sell_actuals < 0).mean() * 100
                    meta_sell_avg_ret = meta_sell_actuals.mean() * 100

        metrics[h] = {
            "n_samples": len(df),
            "directional_accuracy": round(dir_acc, 1),
            "mae_pct": round(mae, 2),
            "rmse_pct": round(rmse, 2),
            "correlation": round(corr, 3),
            "mean_predicted_pct": round(mean_pred, 2),
            "mean_actual_pct": round(mean_actual, 2),
            # Hybrid-based signals
            "hybrid_buy_count": buy_count,
            "hybrid_sell_count": sell_count,
            "hybrid_hold_count": hold_count,
            "hybrid_buy_avg_ret": round(buy_actual, 2) if buy_actual is not None else "N/A",
            "hybrid_buy_win_rate": round(buy_win_rate, 1) if buy_win_rate is not None else "N/A",
            "hybrid_sell_avg_ret": round(sell_actual, 2) if sell_actual is not None else "N/A",
            "hybrid_sell_win_rate": round(sell_win_rate, 1) if sell_win_rate is not None else "N/A",
            # Meta-learner signals
            "meta_accuracy": round(meta_acc, 1) if meta_acc is not None else "N/A",
            "meta_buy_count": meta_buy_count,
            "meta_sell_count": meta_sell_count,
            "meta_hold_count": meta_hold_count,
            "meta_buy_win_rate": round(meta_buy_win_rate, 1) if meta_buy_win_rate is not None else "N/A",
            "meta_sell_win_rate": round(meta_sell_win_rate, 1) if meta_sell_win_rate is not None else "N/A",
            "meta_buy_avg_ret": round(meta_buy_avg_ret, 2) if meta_buy_avg_ret is not None else "N/A",
            "meta_sell_avg_ret": round(meta_sell_avg_ret, 2) if meta_sell_avg_ret is not None else "N/A",
        }

    return metrics


# ===============================================================
# HMM REGIME DETECTION
# ===============================================================

def detect_regime_for_slice(df_price_slice):
    """
    Fit a 3-state GaussianHMM on the price slice and return the current regime.
    Returns: str — one of 'BULL', 'BEAR', 'NEUTRAL'
    Uses adaptive interpretation (same logic as hmm_regime_detector.py).
    """
    if not HMM_AVAILABLE:
        return "NEUTRAL"

    try:
        df = df_price_slice.copy()
        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
        df["range_vol"] = ((df["High"] - df["Low"]) / df["Close"]) * 100
        df = df.dropna(subset=["log_ret", "range_vol"])

        if len(df) < 60:  # need enough data
            return "NEUTRAL"

        X = df[["log_ret", "range_vol"]].values

        model = GaussianHMM(n_components=3, covariance_type="full",
                            n_iter=500, random_state=420)
        model.fit(X)
        states = model.predict(X)
        df["state"] = states

        # Adaptive interpretation
        global_avg_ret = df["log_ret"].mean()
        global_avg_vol = df["range_vol"].mean()

        current_state = states[-1]

        mask = df["state"] == current_state
        state_ret = df.loc[mask, "log_ret"].mean()
        state_vol = df.loc[mask, "range_vol"].mean()

        # Classify
        if state_ret > global_avg_ret and state_vol < global_avg_vol:
            return "BULL"      # Stable uptrend
        elif state_ret > 0 and state_vol > (global_avg_vol * 1.2):
            return "BULL"      # Volatile uptrend — still bullish
        elif state_ret < -0.1:
            return "BEAR"      # Crash
        elif state_ret < 0 and state_vol > global_avg_vol:
            return "BEAR"      # Panic
        else:
            return "NEUTRAL"

    except Exception:
        return "NEUTRAL"


def compute_portfolio_metrics(results_df):
    """
    THREE portfolio strategies computed:
      1) META-ONLY:    trade on meta-learner signal alone
      2) COMBINED:     trade only when hybrid + meta agree (confidence sizing)
      3) HYBRID-ONLY:  trade on hybrid price direction alone
    All use 21d horizon. Returns dict with all three.
    """
    strategies = {}

    # --- Strategy 1: META-ONLY (flat 20%) ---
    strategies["meta_only"] = _run_portfolio_sim(
        results_df, signal_mode="meta", label="META-ONLY"
    )

    # --- Strategy 2: COMBINED (hybrid + meta agree, confidence sizing) ---
    strategies["combined"] = _run_portfolio_sim(
        results_df, signal_mode="combined", label="COMBINED"
    )

    # --- Strategy 3: HYBRID-ONLY (flat 20%) ---
    strategies["hybrid_only"] = _run_portfolio_sim(
        results_df, signal_mode="hybrid", label="HYBRID-ONLY"
    )

    # --- Strategy 4: REGIME-AWARE (HMM modulated combined) ---
    strategies["regime_aware"] = _run_portfolio_sim(
        results_df, signal_mode="regime", label="REGIME-AWARE"
    )

    # --- Strategy 5: COMBINED+ (multi-horizon + confidence floor + ticker quality) ---
    strategies["combined_plus"] = _run_portfolio_sim(
        results_df, signal_mode="combined_plus", label="COMBINED+"
    )

    return strategies


def _run_portfolio_sim(results_df, signal_mode="combined", label=""):
    """Run a single portfolio simulation with the given signal mode."""
    portfolio = {"equity": [100.0], "trades": 0, "wins": 0, "losses": 0,
                 "buy_trades": 0, "sell_trades": 0, "buy_wins": 0, "sell_wins": 0,
                 "skipped_disagreement": 0, "regime_overrides": 0,
                 "skipped_confidence": 0, "skipped_horizon": 0, "skipped_ticker": 0}

    actual_col = "actual_ret_21d"
    meta_col = "meta_signal_21d"
    pred_col = "pred_ret_21d"
    conf_col = "meta_confidence_21d"
    regime_col = "hmm_regime"

    df = results_df.dropna(subset=[actual_col]).copy()
    if meta_col not in df.columns or pred_col not in df.columns:
        return _empty_portfolio()

    df = df.dropna(subset=[meta_col, pred_col])
    df = df.sort_values("sim_date")

    # Per-ticker win rate tracking (for combined_plus)
    ticker_track = {}

    for _, row in df.iterrows():
        meta_signal = row[meta_col]
        hybrid_ret = row[pred_col]
        actual = row[actual_col]
        confidence = row.get(conf_col, 0.05)
        if pd.isna(confidence):
            confidence = 0.05

        # Determine hybrid direction
        if hybrid_ret > 0.005:
            hybrid_signal = "BUY"
        elif hybrid_ret < -0.005:
            hybrid_signal = "SELL"
        else:
            hybrid_signal = "HOLD"

        # Determine trade signal based on mode
        if signal_mode == "regime":
            # REGIME-AWARE: use HMM regime to modulate signal filtering
            regime = row.get(regime_col, "NEUTRAL") if regime_col in df.columns else "NEUTRAL"
            if pd.isna(regime):
                regime = "NEUTRAL"

            if regime == "BULL":
                # In BULL: accept BUY if EITHER meta or hybrid says BUY
                # Only accept SELL if BOTH agree (high conviction short)
                if meta_signal == "BUY" or hybrid_signal == "BUY":
                    trade_signal = "BUY"
                    portfolio["regime_overrides"] += 1
                elif meta_signal == "SELL" and hybrid_signal == "SELL":
                    trade_signal = "SELL"
                else:
                    continue
            elif regime == "BEAR":
                # In BEAR: accept SELL if EITHER meta or hybrid says SELL
                # Only accept BUY if BOTH agree (high conviction long)
                if meta_signal == "SELL" or hybrid_signal == "SELL":
                    trade_signal = "SELL"
                    portfolio["regime_overrides"] += 1
                elif meta_signal == "BUY" and hybrid_signal == "BUY":
                    trade_signal = "BUY"
                else:
                    continue
            else:
                # NEUTRAL: strict combined (both must agree)
                if meta_signal == "HOLD" or hybrid_signal == "HOLD":
                    continue
                if meta_signal != hybrid_signal:
                    portfolio["skipped_disagreement"] += 1
                    continue
                trade_signal = meta_signal

            position_size = min(0.30, max(0.10, 0.10 + confidence * 2.0))

        elif signal_mode == "combined_plus":
            # COMBINED+: combined + multi-horizon + confidence floor + ticker quality
            # Step 1: Both hybrid & meta must agree on 21d (same as combined)
            if meta_signal == "HOLD" or hybrid_signal == "HOLD":
                continue
            if meta_signal != hybrid_signal:
                portfolio["skipped_disagreement"] += 1
                continue
            trade_signal = meta_signal

            # Step 2: Confidence floor — meta must be decisive
            if confidence < 0.08:
                portfolio["skipped_confidence"] += 1
                continue

            # Step 3: Multi-horizon confirmation — 5d meta must agree with 21d direction
            meta_5d_col = "meta_signal_5d"
            if meta_5d_col in df.columns:
                meta_5d = row.get(meta_5d_col, "HOLD")
                if pd.isna(meta_5d):
                    meta_5d = "HOLD"
                # 5d meta should either agree or be HOLD (not contradict)
                if trade_signal == "BUY" and meta_5d == "SELL":
                    portfolio["skipped_horizon"] += 1
                    continue
                if trade_signal == "SELL" and meta_5d == "BUY":
                    portfolio["skipped_horizon"] += 1
                    continue

            # Step 4: Per-ticker quality filter
            tkr = row.get("ticker", "")
            if tkr in ticker_track:
                t = ticker_track[tkr]
                if t["total"] >= 3 and t["wins"] / t["total"] < 0.35:
                    # This ticker has been consistently wrong → skip
                    portfolio["skipped_ticker"] += 1
                    continue

            position_size = min(0.30, max(0.10, 0.10 + confidence * 2.0))

        elif signal_mode == "combined":
            # COMBINED: both must agree, skip on disagreement
            if meta_signal == "HOLD" or hybrid_signal == "HOLD":
                continue
            if meta_signal != hybrid_signal:
                portfolio["skipped_disagreement"] += 1
                continue
            trade_signal = meta_signal  # they agree
            position_size = min(0.30, max(0.10, 0.10 + confidence * 2.0))
        elif signal_mode == "meta":
            if meta_signal == "HOLD":
                continue
            trade_signal = meta_signal
            position_size = 0.20
        elif signal_mode == "hybrid":
            if hybrid_signal == "HOLD":
                continue
            trade_signal = hybrid_signal
            position_size = 0.20
        else:
            continue

        portfolio["trades"] += 1
        tkr = row.get("ticker", "")

        if trade_signal == "BUY":
            pnl = actual * position_size
            portfolio["buy_trades"] += 1
            if actual > 0:
                portfolio["buy_wins"] += 1
        elif trade_signal == "SELL":
            pnl = -actual * position_size
            portfolio["sell_trades"] += 1
            if actual < 0:
                portfolio["sell_wins"] += 1
        else:
            continue

        # Slippage (0.15%) + commission (0.09%)
        pnl -= 0.0024

        new_eq = portfolio["equity"][-1] * (1 + pnl)
        portfolio["equity"].append(new_eq)

        if pnl > 0:
            portfolio["wins"] += 1
            if tkr:
                ticker_track.setdefault(tkr, {"wins": 0, "total": 0})
                ticker_track[tkr]["wins"] += 1
                ticker_track[tkr]["total"] += 1
        else:
            portfolio["losses"] += 1
            if tkr:
                ticker_track.setdefault(tkr, {"wins": 0, "total": 0})
                ticker_track[tkr]["total"] += 1

    eq = np.array(portfolio["equity"])
    returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])

    total_ret = (eq[-1] / eq[0] - 1) * 100
    max_dd = np.min(eq / np.maximum.accumulate(eq) - 1) * 100 if len(eq) > 1 else 0
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(52) if len(returns) > 1 else 0
    win_rate = portfolio["wins"] / max(1, portfolio["trades"]) * 100

    buy_wr = portfolio["buy_wins"] / max(1, portfolio["buy_trades"]) * 100
    sell_wr = portfolio["sell_wins"] / max(1, portfolio["sell_trades"]) * 100

    return {
        "label": label,
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": portfolio["trades"],
        "buy_trades": portfolio["buy_trades"],
        "sell_trades": portfolio["sell_trades"],
        "win_rate_pct": round(win_rate, 1),
        "buy_win_rate_pct": round(buy_wr, 1),
        "sell_win_rate_pct": round(sell_wr, 1),
        "final_equity": round(eq[-1], 2),
        "skipped_disagreement": portfolio["skipped_disagreement"],
        "regime_overrides": portfolio.get("regime_overrides", 0),
        "skipped_confidence": portfolio.get("skipped_confidence", 0),
        "skipped_horizon": portfolio.get("skipped_horizon", 0),
        "skipped_ticker": portfolio.get("skipped_ticker", 0),
    }


def _empty_portfolio():
    return {
        "label": "", "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
        "total_trades": 0, "buy_trades": 0, "sell_trades": 0,
        "win_rate_pct": 0.0, "buy_win_rate_pct": 0.0, "sell_win_rate_pct": 0.0,
        "final_equity": 100.0, "skipped_disagreement": 0, "regime_overrides": 0,
        "skipped_confidence": 0, "skipped_horizon": 0, "skipped_ticker": 0,
    }


# ===============================================================
# PRINT REPORT
# ===============================================================

def print_report(metrics, portfolio_metrics, results_df, n_tickers, n_sim_dates, oos_start, oos_end):
    print("\n" + "=" * 80)
    print("  ChronoStox v9.5 — OUT-OF-SAMPLE BACKTEST REPORT (Combined+ Edition)")
    print("=" * 80)
    print(f"  Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  OOS Period    : {oos_start} → {oos_end}")
    print(f"  Tickers Tested: {n_tickers}")
    print(f"  Simulation Pts: {n_sim_dates}")
    print("-" * 80)

    # === HYBRID RETURNS TABLE ===
    print(f"\n  HYBRID PRICE TARGET ANALYSIS (bidirectional ATR):")
    print(f"  {'HORIZON':<8} | {'N':<5} | {'DIR ACC':<8} | {'MAE%':<7} | {'CORR':<6} | {'Pred→Act':<18} | {'BUY':<5} | {'SELL':<5} | {'HOLD':<5}")
    print("  " + "-" * 88)

    for h in HORIZONS:
        if h not in metrics:
            print(f"  {h}d       | —")
            continue
        m = metrics[h]
        pred_act = f"{m['mean_predicted_pct']:+.1f}→{m['mean_actual_pct']:+.1f}%"
        print(
            f"  {str(h) + 'd':<8} | "
            f"{m['n_samples']:<5} | "
            f"{m['directional_accuracy']:>5.1f}%  | "
            f"{m['mae_pct']:>5.2f}% | "
            f"{m['correlation']:>5.3f} | "
            f"{pred_act:<18} | "
            f"{m['hybrid_buy_count']:<5} | "
            f"{m['hybrid_sell_count']:<5} | "
            f"{m['hybrid_hold_count']:<5}"
        )

    # === META-LEARNER TABLE ===
    print(f"\n  META-LEARNER SIGNAL ANALYSIS:")
    print(f"  {'HORIZON':<8} | {'META ACC':<9} | {'BUY':<5} | {'BUY WR':<7} | {'BUY Avg':<8} | {'SELL':<5} | {'SELL WR':<8} | {'SELL Avg':<9} | {'HOLD':<5}")
    print("  " + "-" * 92)

    for h in HORIZONS:
        if h not in metrics:
            print(f"  {h}d       | —")
            continue
        m = metrics[h]
        ma = m['meta_accuracy']
        ma_str = f"{ma}%" if ma != "N/A" else "N/A"
        bwr = m['meta_buy_win_rate']
        bwr_str = f"{bwr}%" if bwr != "N/A" else "N/A"
        swr = m['meta_sell_win_rate']
        swr_str = f"{swr}%" if swr != "N/A" else "N/A"
        bar = m['meta_buy_avg_ret']
        bar_str = f"{bar:+.2f}%" if bar != "N/A" else "N/A"
        sar = m['meta_sell_avg_ret']
        sar_str = f"{sar:+.2f}%" if sar != "N/A" else "N/A"

        print(
            f"  {str(h) + 'd':<8} | "
            f"{ma_str:<9} | "
            f"{m['meta_buy_count']:<5} | "
            f"{bwr_str:<7} | "
            f"{bar_str:<8} | "
            f"{m['meta_sell_count']:<5} | "
            f"{swr_str:<8} | "
            f"{sar_str:<9} | "
            f"{m['meta_hold_count']:<5}"
        )

    # Per-horizon detailed
    print("\n" + "-" * 80)
    print("  DETAILED BREAKDOWN:")
    print("-" * 80)
    for h in HORIZONS:
        if h not in metrics:
            continue
        m = metrics[h]
        print(f"\n  {h}-DAY HORIZON:")
        print(f"    Samples           : {m['n_samples']}")
        print(f"    Hybrid Dir. Acc   : {m['directional_accuracy']}%")
        print(f"    MAE               : {m['mae_pct']}%")
        print(f"    Pred vs Actual    : {m['mean_predicted_pct']}% predicted → {m['mean_actual_pct']}% actual")
        print(f"    --- Hybrid Signals ---")
        print(f"    BUY  : {m['hybrid_buy_count']} (win rate: {m['hybrid_buy_win_rate']}%, avg ret: {m['hybrid_buy_avg_ret']}%)")
        print(f"    SELL : {m['hybrid_sell_count']} (win rate: {m['hybrid_sell_win_rate']}%, avg ret: {m['hybrid_sell_avg_ret']}%)")
        print(f"    HOLD : {m['hybrid_hold_count']}")
        print(f"    --- Meta-Learner ---")
        if m['meta_accuracy'] != 'N/A':
            print(f"    Meta Accuracy     : {m['meta_accuracy']}%")
        print(f"    Meta BUY  : {m['meta_buy_count']} (win rate: {m['meta_buy_win_rate']}%, avg ret: {m['meta_buy_avg_ret']}%)")
        print(f"    Meta SELL : {m['meta_sell_count']} (win rate: {m['meta_sell_win_rate']}%, avg ret: {m['meta_sell_avg_ret']}%)")
        print(f"    Meta HOLD : {m['meta_hold_count']}")

    # Portfolio comparison
    print("\n" + "-" * 80)
    print("  PORTFOLIO STRATEGY COMPARISON (21d horizon):")
    print("-" * 80)
    print(f"  {'Strategy':<18} | {'Return':>8} | {'MaxDD':>8} | {'Sharpe':>7} | {'Trades':>7} | {'Win%':>6} | {'Equity':>9}")
    print("  " + "-" * 80)

    # Order: regime first (recommended), then combined, meta, hybrid
    for key in ["adaptive", "combined_plus", "combined", "regime_aware", "meta_only", "hybrid_only"]:
        pm = portfolio_metrics.get(key, _empty_portfolio())
        label = {
            "adaptive": "\u2b50 ADAPTIVE",
            "combined_plus": "   COMBINED+",
            "combined": "   COMBINED",
            "regime_aware": "   REGIME-AWARE",
            "meta_only": "   META-ONLY",
            "hybrid_only": "   HYBRID-ONLY"
        }.get(key, key)
        trades_str = f"{pm['total_trades']} ({pm['buy_trades']}B/{pm['sell_trades']}S)"
        print(
            f"  {label:<18} | "
            f"{pm['total_return_pct']:>+7.1f}% | "
            f"{pm['max_drawdown_pct']:>+7.1f}% | "
            f"{pm['sharpe_ratio']:>+6.2f} | "
            f"{trades_str:>7} | "
            f"{pm['win_rate_pct']:>5.1f}% | "
            f"\u20b9{pm['final_equity']:>7.2f}"
        )

    # Detail on COMBINED+
    cp = portfolio_metrics.get("combined_plus", _empty_portfolio())
    print(f"\n  \u2b50 COMBINED+ STRATEGY DETAILS:")
    print(f"    Base             : Combined (hybrid + meta 21d must agree)")
    print(f"    + Confidence     : meta gap \u2265 0.08 required")
    print(f"    + Multi-horizon  : 5d meta must not contradict 21d direction")
    print(f"    + Ticker quality : skip ticker if win rate < 35% after 3+ trades")
    print(f"    Position Sizing  : 10-30% (confidence-based)")
    print(f"    Skipped (disagree)  : {cp.get('skipped_disagreement', 0)}")
    print(f"    Skipped (low conf)  : {cp.get('skipped_confidence', 0)}")
    print(f"    Skipped (horizon)   : {cp.get('skipped_horizon', 0)}")
    print(f"    Skipped (bad ticker): {cp.get('skipped_ticker', 0)}")
    print(f"    Buy Win Rate     : {cp['buy_win_rate_pct']}%")
    print(f"    Sell Win Rate    : {cp['sell_win_rate_pct']}%")

    # Per-ticker performance (21d horizon)
    if "actual_ret_21d" in results_df.columns and "pred_ret_21d" in results_df.columns:
        print("\n" + "-" * 80)
        print("  PER-TICKER 21d PERFORMANCE:")
        print("-" * 80)
        print(f"  {'Ticker':<18} | {'N':>3} | {'DirAcc':>6} | {'MetaAcc':>7} | {'Pred':>6} | {'Actual':>7} | {'HMM Mode':>9}")
        print("  " + "-" * 72)

        df21 = results_df.dropna(subset=["actual_ret_21d", "pred_ret_21d"])
        for tkr in sorted(df21["ticker"].unique()):
            tdf = df21[df21["ticker"] == tkr]
            n = len(tdf)
            if n == 0:
                continue
            pred = tdf["pred_ret_21d"].values
            actual = tdf["actual_ret_21d"].values
            dir_acc = np.mean(np.sign(pred) == np.sign(actual)) * 100
            avg_pred = np.mean(pred) * 100
            avg_actual = np.mean(actual) * 100

            # Meta accuracy for this ticker
            meta_acc_str = "N/A"
            if "meta_signal_21d" in tdf.columns:
                mc = 0
                mdf = tdf.dropna(subset=["meta_signal_21d"])
                for _, r in mdf.iterrows():
                    s, a = r["meta_signal_21d"], r["actual_ret_21d"]
                    if (s == "BUY" and a > 0) or (s == "SELL" and a < 0) or (s == "HOLD" and abs(a) < 0.05):
                        mc += 1
                if len(mdf) > 0:
                    meta_acc_str = f"{mc / len(mdf) * 100:.0f}%"

            # Dominant HMM regime
            hmm_str = "N/A"
            if "hmm_regime" in tdf.columns:
                regime_counts_tkr = tdf["hmm_regime"].value_counts()
                if len(regime_counts_tkr) > 0:
                    hmm_str = regime_counts_tkr.index[0]

            short_tkr = tkr.replace(".NS", "")
            print(
                f"  {short_tkr:<18} | "
                f"{n:>3} | "
                f"{dir_acc:>5.0f}% | "
                f"{meta_acc_str:>7} | "
                f"{avg_pred:>+5.1f}% | "
                f"{avg_actual:>+6.1f}% | "
                f"{hmm_str:>9}"
            )

    print("=" * 80)


# ===============================================================
# MAIN BACKTEST LOOP
# ===============================================================

def run_backtest(tickers, oos_start, oos_end, frequency="weekly"):
    t0 = time.time()

    # --- Step 1: Load everything once ---
    print("[1/6] Loading models...")
    model_dir = "."
    if not os.path.exists(os.path.join(".", MODEL_JOBLIB)):
        if os.path.exists(os.path.join("../test", MODEL_JOBLIB)):
            model_dir = "../test"
        else:
            print("FATAL: Cannot find model files.")
            sys.exit(1)

    models = load_models_once(model_dir)
    df_macro = load_macro_once(model_dir)
    df_senti = load_sentiment_once(model_dir)
    df_sector = load_sectors_once(model_dir)
    print(f"  ✅ Models loaded ({len(models['features'])} features, seq_len={models['seq_len']})")
    print(f"  ✅ Meta-learners: {list(models['meta_models'].keys()) if models['meta_models'] else 'NONE'}")

    # --- Step 2: Fetch price history for all tickers ---
    print(f"\n[2/6] Fetching price history for {len(tickers)} tickers...")
    price_cache = {}
    failed_tickers = []

    # Fetch from 2 years before OOS start to have enough history
    fetch_start = (pd.to_datetime(oos_start) - pd.Timedelta(days=800)).strftime("%Y-%m-%d")

    for i, tkr in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {tkr}...", end=" ")
        df_px = fetch_price_history(tkr, start_date=fetch_start)
        if df_px is not None and len(df_px) > 100:
            price_cache[tkr] = df_px
            print(f"✅ {len(df_px)} rows")
        else:
            failed_tickers.append(tkr)
            print("❌ skipped")

    if not price_cache:
        print("FATAL: No ticker data fetched.")
        sys.exit(1)

    print(f"  → {len(price_cache)} tickers ready, {len(failed_tickers)} failed")

    # --- Step 3: Generate simulation dates ---
    print(f"\n[3/6] Generating simulation dates ({frequency})...")
    start_dt = pd.to_datetime(oos_start)
    end_dt = pd.to_datetime(oos_end)

    if frequency == "daily":
        # Get actual trading days from any ticker's data
        sample_tkr = list(price_cache.keys())[0]
        trading_days = price_cache[sample_tkr]["Date"]
        sim_dates = trading_days[(trading_days >= start_dt) & (trading_days <= end_dt)].tolist()
    else:
        # Weekly (every Friday-ish)
        all_dates = pd.date_range(start_dt, end_dt, freq="W-FRI")
        sim_dates = all_dates.tolist()

    # Leave buffer at end for forward returns to materialize
    today = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    max_verify_date = {
        5: today - pd.Timedelta(days=8),    # 5 trading days + buffer
        21: today - pd.Timedelta(days=30),   # 21 trading days + buffer
        63: today - pd.Timedelta(days=90),
        126: today - pd.Timedelta(days=180),
        252: today - pd.Timedelta(days=365),
    }

    print(f"  → {len(sim_dates)} simulation dates")

    # --- Step 4: Run predictions ---
    print(f"\n[4/6] Running predictions + HMM regime detection...")
    all_results = []
    total_preds = len(sim_dates) * len(price_cache)
    completed = 0
    regime_counts = defaultdict(int)
    
    # Track short-term proxy errors for Adaptive Averaging
    proxy_errors = {}
    proxy_errors_dates = {}

    for sim_date in sim_dates:
        for tkr, df_full_px in price_cache.items():
            completed += 1

            # Slice price data up to sim_date (simulate standing on that date)
            df_slice = df_full_px[df_full_px["Date"] <= sim_date].copy()

            if len(df_slice) < 100:
                continue

            # Feature engineering
            df_feat, df_merged = engineer_features_for_slice(
                df_slice, df_macro, df_senti, df_sector, models["features"]
            )

            if df_feat is None:
                continue

            # HMM regime detection on the price slice
            regime = detect_regime_for_slice(df_slice)
            regime_counts[regime] += 1

            # Dynamic Weights (Proxy Error on 5d horizon)
            # Find the index of the 5d horizon (usually 0)
            h5_idx = HORIZONS.index(5) if 5 in HORIZONS else 0
            
            # Simple 10-period rolling proxy engine
            dyn_weights = {"lgbm": 0.5, "lstm": 0.5}
            if tkr in proxy_errors and len(proxy_errors[tkr]) >= 3:
                lgb_errs = [err[0] for err in proxy_errors[tkr][-10:]]
                lstm_errs = [err[1] for err in proxy_errors[tkr][-10:]]
                
                sum_lgb_err = sum(lgb_errs)
                sum_lstm_err = sum(lstm_errs)
                total_err = sum_lgb_err + sum_lstm_err
                
                if total_err > 0:
                    # Inverse weighting: lower error gets higher weight!
                    # Bound to prevent 0 or 1 extremes
                    raw_w_lgb = sum_lstm_err / total_err
                    raw_w_lstm = sum_lgb_err / total_err
                    dyn_weights["lgbm"] = max(0.1, min(0.9, raw_w_lgb))
                    dyn_weights["lstm"] = max(0.1, min(0.9, raw_w_lstm))

            # Predict
            try:
                pred = predict_at_date(df_merged, models, dynamic_weights=dyn_weights)
            except Exception:
                continue
                
            # Log the proxy error for the *previous* simulated dates if we have actuals
            # Look back 5 days to score the models
            eval_date = sim_date - pd.Timedelta(days=7) # approximate 5 trading days
            past_slice = df_slice[df_slice["Date"] <= eval_date]
            if not past_slice.empty:
                # Get the actual return from eval_date to sim_date
                past_close = past_slice["Close"].iloc[-1]
                current_close = df_slice["Close"].iloc[-1]
                actual_ret_5d = (current_close - past_close) / past_close
                
                # Fetch what the models predicted 5 days ago (if recorded)
                # To save compute, we just extract from all_results if it exists
                past_res = [r for r in all_results if r["ticker"] == tkr and r["sim_date"] == past_slice["Date"].iloc[-1]]
                if past_res:
                    p_res = past_res[0]
                    lgb_past_pred = p_res.get(f"raw_ml_ret_5d_lgbm", p_res.get("raw_ml_ret_5d", 0.0))
                    lstm_past_pred = p_res.get(f"raw_ml_ret_5d_lstm", p_res.get("raw_ml_ret_5d", 0.0))
                    
                    # Store proxy errors
                    err_lgb = abs(actual_ret_5d - lgb_past_pred)
                    err_lstm = abs(actual_ret_5d - lstm_past_pred)
                    
                    if tkr not in proxy_errors:
                        proxy_errors[tkr] = []
                    # Avoid appending duplicates for the same eval_date
                    if not proxy_errors[tkr] or past_slice["Date"].iloc[-1] != proxy_errors_dates.get(tkr):
                        proxy_errors[tkr].append((err_lgb, err_lstm))
                        proxy_errors_dates[tkr] = past_slice["Date"].iloc[-1]

            # Record
            result = {
                "ticker": tkr,
                "sim_date": sim_date,
                "close_at_sim": pred["close"],
                "hmm_regime": regime,
            }


            # Raw ML predictions (for transparency and proxy scoring)
            for i, h in enumerate(HORIZONS):
                result[f"raw_ml_ret_{h}d"] = pred["raw_avg"][i]
                result[f"raw_ml_ret_{h}d_lgbm"] = pred["raw_lgb"][i]
                result[f"raw_ml_ret_{h}d_lstm"] = pred["raw_lstm"][i]
                
            # Log dynamic weights for analysis
            result["weight_lgbm"] = dyn_weights["lgbm"]
            result["weight_lstm"] = dyn_weights["lstm"]

            # Predicted returns + targets from hybrid
            for i, h in enumerate(HORIZONS):
                result[f"pred_ret_{h}d"] = pred["predicted_returns"][h]
                result[f"pred_target_{h}d"] = pred["hybrid"][i]

                # Meta-learner signal
                if h in pred["meta_signals"]:
                    result[f"meta_signal_{h}d"] = pred["meta_signals"][h]["signal"]
                    result[f"meta_proba_buy_{h}d"] = pred["meta_signals"][h]["proba"][2]
                    result[f"meta_proba_sell_{h}d"] = pred["meta_signals"][h]["proba"][0]
                    result[f"meta_confidence_{h}d"] = pred["meta_signals"][h]["confidence"]

                # Actual returns (if enough future data exists)
                if sim_date <= max_verify_date.get(h, pd.Timestamp("2000-01-01")):
                    future_dates = df_full_px[df_full_px["Date"] > sim_date].head(h)
                    if len(future_dates) >= max(1, h * 0.7):  # allow some flexibility
                        actual_close = future_dates["Close"].iloc[-1]
                        result[f"actual_ret_{h}d"] = (actual_close - pred["close"]) / pred["close"]
                        result[f"actual_close_{h}d"] = actual_close

            all_results.append(result)

            if completed % 20 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / completed) * (total_preds - completed)
                print(f"  [{completed}/{total_preds}] {tkr} @ {sim_date.strftime('%Y-%m-%d')} — ETA: {int(eta)}s")

    results_df = pd.DataFrame(all_results)
    print(f"  ✅ {len(results_df)} predictions recorded")
    if regime_counts:
        print(f"  📊 Regimes detected: " + ", ".join(f"{k}: {v}" for k, v in sorted(regime_counts.items())))

    # --- Step 5: Compute metrics ---
    print(f"\n[5/6] Computing metrics...")
    metrics = compute_metrics(results_df)
    portfolio_metrics = compute_portfolio_metrics(results_df)

    # Print report
    print_report(
        metrics, portfolio_metrics, results_df,
        len(price_cache), len(sim_dates),
        oos_start, oos_end
    )

    # Save CSVs
    results_df.to_csv("backtest_results.csv", index=False)
    print(f"\n💾 Results saved → backtest_results.csv")

    # Save metrics
    metrics_rows = []
    for h, m in metrics.items():
        m["horizon"] = h
        metrics_rows.append(m)
    pd.DataFrame(metrics_rows).to_csv("backtest_metrics.csv", index=False)
    print(f"💾 Metrics saved → backtest_metrics.csv")

    elapsed = time.time() - t0
    print(f"\n⏱ Total runtime: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    return results_df, metrics, portfolio_metrics


# ===============================================================
# CLI
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="ChronoStox v9.5 OOS Backtester (Combined+ Edition)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to test (default: 20 Nifty stocks)")
    parser.add_argument("--start", default="2025-11-17",
                        help="OOS start date (default: 2025-11-17)")
    parser.add_argument("--end", default=None,
                        help="OOS end date (default: today)")
    parser.add_argument("--daily", action="store_true",
                        help="Simulate every trading day (slow)")
    parser.add_argument("--weekly", action="store_true", default=True,
                        help="Simulate every Friday (default)")

    args = parser.parse_args()

    tickers = args.tickers if args.tickers else DEFAULT_TICKERS
    oos_start = args.start
    oos_end = args.end if args.end else datetime.now().strftime("%Y-%m-%d")
    freq = "daily" if args.daily else "weekly"

    print(f"\n{'=' * 60}")
    print(f"  ChronoStox v9.4 — OOS Backtester (Quant Overlay Edition)")
    print(f"{'=' * 60}")
    print(f"  OOS Window : {oos_start} → {oos_end}")
    print(f"  Tickers    : {len(tickers)}")
    print(f"  Frequency  : {freq}")
    print(f"{'=' * 60}")

    run_backtest(tickers, oos_start, oos_end, frequency=freq)


if __name__ == "__main__":
    main()
