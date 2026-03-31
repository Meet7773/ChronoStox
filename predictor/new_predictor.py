# predictor/predictor.py
import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# external libs used in main repo
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import load_model



# Model filenames (match your repo)
MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"

# Same horizons + blending rules as your backtester
HORIZONS = [5, 21, 63, 126, 252]
ATR_MULT = {5: 1.0, 21: 1.8, 63: 3.0, 126: 4.8, 252: 7.2}
WEIGHT_ML_H = {5: 0.62, 21: 0.55, 63: 0.50, 126: 0.42, 252: 0.35}
WEIGHT_ATR_H = {h: 1 - WEIGHT_ML_H.get(h, 0.5) for h in HORIZONS}

# -----------------------
# Helpers
# -----------------------
def safe_read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        # caller will handle None
        return None

def _normalize_dates(df, date_col="Date"):
    df = df.copy()
    if date_col not in df.columns:
        df = df.reset_index().rename(columns={"index": date_col})
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    return df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

# -----------------------
# Loading data & models
# -----------------------
def load_models_once(model_dir):
    """
    Returns a dict with keys:
      - scaler
      - lgbm
      - meta_models (dict by horizon)
      - features (list)
      - horizons (list)
      - lstm_sequence_length (int)
      - lstm (keras model)
    """
    bundle_path = os.path.join(model_dir, MODEL_JOBLIB)
    keras_path = os.path.join(model_dir, MODEL_KERAS)

    bundle = joblib.load(bundle_path)
    lstm_model = load_model(keras_path, compile=False)

    out = {
        "scaler": bundle["scaler"],
        "lgbm": bundle["model_lgbm"],
        "meta_models": bundle.get("meta_models", {}),
        "features": bundle["features"],
        "horizons": bundle.get("horizons", HORIZONS),
        "seq_len": bundle.get("lstm_sequence_length", 60),
        "lstm": lstm_model
    }
    return out

def load_macro_once(data_dir):
    df = safe_read_parquet(os.path.join(data_dir, "macro_features.parquet"))
    if df is None:
        return pd.DataFrame()
    return _normalize_dates(df, "Date")

def load_sentiment_once(data_dir):
    df = safe_read_parquet(os.path.join(data_dir, "sentiment_clean.parquet"))
    if df is None:
        return pd.DataFrame()
    df = _normalize_dates(df, "Date")
    # unify column names
    for c in df.columns:
        if str(c).lower() in ["ticker", "tickeryf", "ticker_yf", "symbol"]:
            df = df.rename(columns={c: "Ticker_YF"})
        if "sentiment" in str(c).lower():
            df = df.rename(columns={c: "sentiment_score"})
    if "Ticker_YF" in df.columns:
        df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.upper().str.strip()
    return df

def load_sectors_once(data_dir):
    path = os.path.join(data_dir, "ticker.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Ticker_YF", "Sector"])
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, header=None, low_memory=False)
    cols = list(df.columns)
    ticker_col = next((c for c in cols if str(c).lower() in ["ticker", "ticker_yf", "symbol"]), cols[0])
    sector_col = next((c for c in cols if "sector" in str(c).lower()), cols[-1])
    df = df.rename(columns={ticker_col: "Ticker_YF", sector_col: "Sector"})
    df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip().str.upper()
    return df[["Ticker_YF", "Sector"]].drop_duplicates()

# -----------------------
# Price history helper
# -----------------------
def fetch_price_history(ticker, start_date="2018-01-01"):
    """
    Returns a pandas DataFrame with Date, Open, High, Low, Close, Volume, Ticker_YF
    """
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, interval="1d")
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df = _normalize_dates(df, "Date")
        df["Ticker_YF"] = ticker
        return df
    except Exception:
        return None

# -----------------------
# Feature engineering for a ticker slice
# -----------------------
def engineer_features_for_slice(df_price, df_macro, df_senti, df_sector, expected_features):
    """
    Given raw price df (as fetched by fetch_price_history), returns:
      - df_feat : dataframe with technical indicators and final merged features (subset used for modelling)
      - df_merged: full frame (with Date, Close, ATR_final, sentiment_score, Sector columns)
    """
    try:
        df = df_price.copy().sort_values("Date").reset_index(drop=True)
        if len(df) < 30:
            return None, None

        # Basic TA features (keep consistent names used by models)
        # ADX / ATR / EMA / RSI / MACD / BBANDS
        try:
            df.ta.adx(length=14, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)

            bb = df.ta.bbands(length=5, append=False)
            if bb is not None and not bb.empty:
                # pandas_ta returns columns default names, try to keep backward-compatible keys
                if "BBB_5_2.0" in bb.columns:
                    df["BBB_5_2.0"] = bb["BBB_5_2.0"]
                    df["BBP_5_2.0"] = bb["BBP_5_2.0"]
                else:
                    # fallback: use third and fourth columns (typical layout)
                    df["BBB_5_2.0"] = bb.iloc[:, 3]
                    df["BBP_5_2.0"] = bb.iloc[:, 4]
            mac = df.ta.macd(append=False)
            if mac is not None and not mac.empty and mac.shape[1] >= 3:
                df["MACDh_12_26_9"] = mac.iloc[:, 1]
                df["MACDs_12_26_9"] = mac.iloc[:, 2]
            df.ta.rsi(append=True)
        except Exception:
            # if pandas_ta fails for any reason, continue with whatever columns exist
            pass

        # close_to_ema200
        ema200 = [c for c in df.columns if "EMA" in c and "200" in str(c)]
        df["close_to_ema200"] = df["Close"] / (df[ema200[0]] + 1e-9) if ema200 else np.nan

        # Merge macro (backward fill by date)
        if df_macro is None or df_macro.empty:
            df_macro = pd.DataFrame({"Date": df["Date"], "macro_dummy": 0.0})
        df = pd.merge_asof(df.sort_values("Date"), df_macro.sort_values("Date"), on="Date", direction="backward")

        # Merge sentiment for ticker
        ticker = str(df["Ticker_YF"].iloc[0]).upper().strip()
        df_s = pd.DataFrame({"Date": df["Date"], "sentiment_score": np.zeros(len(df))})
        if df_senti is not None and not df_senti.empty and "Ticker_YF" in df_senti.columns:
            ssub = df_senti[df_senti["Ticker_YF"] == ticker]
            if not ssub.empty:
                ssub = ssub.sort_values("Date")[["Date", "sentiment_score"]]
                df = pd.merge_asof(df.sort_values("Date"), ssub, on="Date", direction="backward", allow_exact_matches=True)
            else:
                df = pd.merge_asof(df.sort_values("Date"), df_s.sort_values("Date"), on="Date", direction="backward", allow_exact_matches=True)
        else:
            df = pd.merge_asof(df.sort_values("Date"), df_s.sort_values("Date"), on="Date", direction="backward", allow_exact_matches=True)

        # Sector one-hot (keep same list as backtester)
        sec_row = pd.DataFrame()
        try:
            sec_row = df_sector[df_sector["Ticker_YF"] == ticker]
        except Exception:
            pass
        sector = sec_row["Sector"].iloc[0] if not sec_row.empty else "nan"
        sector = str(sector).strip()
        df["Sector"] = sector
        sector_list = ["Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy",
                       "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities", "nan"]
        for s in sector_list:
            df[f"Sector_{s}"] = 1.0 if sector == s else 0.0

        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0

        # sentiment × sector interactions
        for c in list(df.columns):
            if str(c).startswith("Sector_"):
                df[f"sentiment_x_{c}"] = df[c].astype(float) * df["sentiment_score"].astype(float)

        # Replace infinities & NaNs consistently
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Ensure expected features are present
        for col in set(expected_features) - set(df.columns):
            df[col] = 0.0

        # ATR final for quick volatility
        atr_col = next((c for c in ["ATR_14", "ATRr_14", "ATR"] if c in df.columns), None)
        df["ATR_final"] = df[atr_col] if atr_col else df["Close"] * 0.01

        # Return two frames:
        # df_feat: only expected features (in order), and Date/Close for caller convenience
        # df_merged: full merged frame (keeps ATR_final, Date, Close, sentiment, etc.)
        df_feat = df[["Date", "Close"] + list(expected_features)].copy()
        df_merged = df.copy()
        return df_feat, df_merged
    except Exception:
        return None, None

# -----------------------
# Prediction wrapper
# -----------------------
def predict_at_date(df_merged, models):
    """
    df_merged: full dataframe returned by engineer_features_for_slice, with last row the date-of-inference
    models: output of load_models_once

    Returns:
      {
         "close": float,
         "predicted_returns": { horizon: pct_return (float) },
         "predicted_targets": { horizon: target_price (float) },
         "meta_signals": { horizon: {"signal": "BUY/HOLD/SELL", "confidence": float} },
         "raw": { ... (optional raw arrays) }
      }
    """
    out = {
        "close": None,
        "predicted_returns": {},
        "predicted_targets": {},
        "meta_signals": {},
        "raw": {}
    }

    try:
        features = models["features"]
        scaler = models["scaler"]
        lgbm = models["lgbm"]
        lstm = models["lstm"]
        seq_len = models["seq_len"]
        meta_models = models.get("meta_models", {})
        horizons = models.get("horizons", HORIZONS)

        if df_merged is None or len(df_merged) < 2:
            return out

        latest_idx = len(df_merged) - 1
        close = float(df_merged["Close"].iloc[latest_idx])
        out["close"] = close

        # assemble X_row and seq
        X_row = df_merged[features].iloc[latest_idx].values.astype(np.float32)
        if latest_idx + 1 >= seq_len:
            seq = df_merged[features].iloc[latest_idx - seq_len + 1: latest_idx + 1].values.astype(np.float32)
        else:
            # pad with zeros at top to maintain seq_len shape
            pad = np.zeros((seq_len - (latest_idx + 1), len(features)), dtype=np.float32)
            seq = np.vstack([pad, df_merged[features].iloc[0:latest_idx + 1].values.astype(np.float32)])

        # scale & predict LGBM
        Xs = scaler.transform(X_row.reshape(1, -1))
        raw_lgb = lgbm.predict(Xs).ravel()

        # predict LSTM (use model call to avoid predict overhead)
        seq_in = np.expand_dims(seq, axis=0)  # (1, seq_len, n_features)
        raw_lstm = lstm(seq_in, training=False).numpy().ravel()

        # Combine raw ML outputs via equal weighting baseline
        # If models produce vector per horizon, ensure shapes match
        # raw_lgb and raw_lstm should be arrays length == len(horizons)
        raw_lgb = np.array(raw_lgb, dtype=float).ravel()
        raw_lstm = np.array(raw_lstm, dtype=float).ravel()
        if raw_lgb.shape != raw_lstm.shape:
            # fallback: broadcast shorter to longer or trim to len(horizons)
            minlen = min(len(raw_lgb), len(raw_lstm), len(horizons))
            raw_lgb = raw_lgb[:minlen]
            raw_lstm = raw_lstm[:minlen]

        # naive equal weight; keep meta for advanced dynamic weighting outside this helper
        raw_avg = (raw_lgb + raw_lstm) / 2.0

        # compute hybrid targets for each horizon
        for i, h in enumerate(horizons):
            if i >= len(raw_avg):
                break
            # predicted ML price = close * (1 + ml_ret)
            ml_prices = close * (1.0 + float(raw_avg[i]))
            atr = float(df_merged["ATR_final"].iloc[latest_idx])
            atr_offset = atr * ATR_MULT.get(h, 1.0)
            atr_price = (close + atr_offset) if (raw_avg[i] >= 0) else (close - atr_offset)
            raw_h = ml_prices * WEIGHT_ML_H.get(h, 0.5) + atr_price * WEIGHT_ATR_H.get(h, 0.5)

            # volatility clamp (prevent absurd targets)
            vol_factor = max(atr / max(close, 1e-9), 0.0001)
            max_move = vol_factor * np.sqrt(h) * 2.0
            clamped = max(close * (1 - max_move), min(close * (1 + max_move), raw_h))

            pred_ret = (clamped - close) / close
            out["predicted_returns"][h] = float(pred_ret)
            out["predicted_targets"][h] = float(clamped)

            # meta learner signals
            if h in meta_models:
                try:
                    # meta expects features like [raw_lgb, raw_lstm] for the horizon
                    X_meta = np.array([[raw_lgb[i], raw_lstm[i]]], dtype=np.float32)
                    meta = meta_models[h]
                    proba = meta.predict_proba(X_meta)[0]
                    pred_class = int(meta.predict(X_meta)[0])
                    # class mapping: assume 0=SELL,1=HOLD,2=BUY
                    mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
                    signal = mapping.get(pred_class, "HOLD")
                    confidence = float(max(proba) - sorted(proba)[-2]) if len(proba) > 1 else float(max(proba))
                    out["meta_signals"][h] = {"signal": signal, "confidence": confidence}
                except Exception:
                    out["meta_signals"][h] = {"signal": "HOLD", "confidence": 0.0}

        # return raw arrays for debugging/audit
        out["raw"]["raw_lgb"] = raw_lgb.tolist()
        out["raw"]["raw_lstm"] = raw_lstm.tolist()
        out["raw"]["raw_avg"] = raw_avg.tolist()
        out["raw"]["close"] = close

        return out
    except Exception:
        return out

# If module is executed directly, run a quick smoke test if possible
if __name__ == "__main__":
    print("predictor module loaded. Run the live engine or import functions:")
    print("  load_models_once, load_macro_once, load_sentiment_once, load_sectors_once,")
    print("  fetch_price_history, engineer_features_for_slice, predict_at_date")




