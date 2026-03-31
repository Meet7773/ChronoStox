# backtest_oos_fast_with_full_report.py
# Fast tensor-batched backtester with detailed reporting & trade ledger
# Optimized for local execution.

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
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
    print("⚠ hmmlearn not installed — HMM regime detection disabled.")

warnings.filterwarnings("ignore")

# ------------------ CONFIG ------------------
MODEL_JOBLIB = "sector_model_v7_UNIVERSAL_20251113_063532.joblib"
MODEL_KERAS = "final_lstm_20251113_124137.keras"
MACRO_FILE = "macro_features.parquet"
SENTIMENT_FILE = "sentiment_clean.parquet"
SECTOR_FILE = "ticker.csv"

HORIZONS = [5, 21, 63, 126, 252]
ATR_MULT = {5: 1.0, 21: 1.8, 63: 3.0, 126: 4.8, 252: 7.2}
WEIGHT_ML_H = {5: 0.62, 21: 0.55, 63: 0.50, 126: 0.42, 252: 0.35}
WEIGHT_ATR_H = {h: 1 - WEIGHT_ML_H[h] for h in HORIZONS}

SLIPPAGE_BPS = 10
BROKERAGE_BPS = 3
STT_BPS = 10
EXCHANGE_FEES_BPS = 0.35
TOTAL_BUY_FEE_PCT = (SLIPPAGE_BPS + BROKERAGE_BPS + EXCHANGE_FEES_BPS) / 10000.0
TOTAL_SELL_FEE_PCT = (SLIPPAGE_BPS + BROKERAGE_BPS + STT_BPS + EXCHANGE_FEES_BPS) / 10000.0

DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS"]


# ------------------ UTIL ------------------

def get_dynamic_tickers():
    try:
        df_nse = pd.read_csv('https://archives.nseindia.com/content/indices/ind_nifty200list.csv')
        tickers = [str(sym).strip() + ".NS" for sym in df_nse["Symbol"].dropna().tolist()]
        return tickers
    except Exception as e:
        print(f"⚠️ Failed to fetch live NSE 200 list: {e}\n⚠️ Falling back to DEFAULT_TICKERS.")
        return DEFAULT_TICKERS


# ------------------ DATA LOADERS ------------------

def safe_read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"FATAL: Failed to read parquet {path}: {e}")
        sys.exit(1)


def load_models_once(model_dir):
    bundle = joblib.load(os.path.join(model_dir, MODEL_JOBLIB))
    lstm_model = load_model(os.path.join(model_dir, MODEL_KERAS), compile=False)
    return {
        "scaler": bundle["scaler"], "lgbm": bundle["model_lgbm"],
        "meta_models": bundle.get("meta_models", {}), "features": bundle["features"],
        "horizons": bundle["horizons"], "seq_len": bundle["lstm_sequence_length"],
        "lstm": lstm_model
    }


def load_macro_once(data_dir):
    df = safe_read_parquet(os.path.join(data_dir, MACRO_FILE))
    if "Date" not in df.columns: df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def load_sentiment_once(data_dir):
    df = safe_read_parquet(os.path.join(data_dir, SENTIMENT_FILE))
    if "Date" not in df.columns: df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in df.columns:
        if c.lower() in ["ticker", "tickeryf", "ticker_yf", "symbol"]: df = df.rename(columns={c: "Ticker_YF"})
        if "sentiment" in c.lower(): df = df.rename(columns={c: "sentiment_score"})
    return df


def load_sectors_once(data_dir):
    try:
        df = pd.read_csv(os.path.join(data_dir, SECTOR_FILE), low_memory=False)
    except:
        df = pd.read_csv(os.path.join(data_dir, SECTOR_FILE), header=None, low_memory=False)
    cols = df.columns.tolist()
    ticker_col = next((c for c in cols if str(c).lower() in ["ticker", "ticker_yf", "symbol"]), cols[0])
    sector_col = next((c for c in cols if "sector" in str(c).lower()), cols[-1])
    df = df.rename(columns={ticker_col: "Ticker_YF", sector_col: "Sector"})
    df["Ticker_YF"] = df["Ticker_YF"].astype(str).str.strip().str.upper()
    return df[["Ticker_YF", "Sector"]].drop_duplicates()


def fetch_price_history(ticker, start_date="2018-01-01"):
    try:
        df = yf.Ticker(ticker).history(start=start_date, interval="1d")
        if df.empty: return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(
            drop=True)
        df["Ticker_YF"] = ticker
        return df
    except Exception:
        return None


# ------------------ FEATURE PRECOMPUTE ------------------

def precompute_features(df_price, df_macro, df_senti, df_sector, expected_cols):
    try:
        df = df_price.copy()
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

        df = df.sort_values("Date").reset_index(drop=True)
        df = pd.merge_asof(df, df_macro.sort_values("Date"), on="Date", direction="backward")

        ticker = df["Ticker_YF"].iloc[0].upper().strip()
        df_sl = df_senti.copy()
        df_sl["Ticker_YF"] = df_sl["Ticker_YF"].astype(str).str.upper().str.strip()
        ssub = df_sl[df_sl["Ticker_YF"] == ticker]
        if ssub.empty: ssub = pd.DataFrame({"Date": df["Date"], "sentiment_score": np.zeros(len(df))})
        ssub = ssub.sort_values("Date")[['Date', 'sentiment_score']]
        df = pd.merge_asof(df, ssub, on="Date", direction="backward", allow_exact_matches=True)

        sec_row = df_sector[df_sector["Ticker_YF"] == ticker]
        sector = sec_row["Sector"].iloc[0] if not sec_row.empty else "nan"
        sector = str(sector).strip()
        df["Sector"] = sector
        for s in ["Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy", "Financial Services",
                  "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities", "nan"]:
            df[f"Sector_{s}"] = 1.0 if sector == s else 0.0

        if "sentiment_score" not in df.columns: df["sentiment_score"] = 0.0
        for c in df.columns:
            if str(c).startswith("Sector_"):
                df[f"sentiment_x_{c}"] = df[c].astype(float) * df["sentiment_score"].astype(float)

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in set(expected_cols) - set(df.columns): df[col] = 0.0

        atr_col = next((c for c in ["ATR_14", "ATRr_14", "ATR"] if c in df.columns), None)
        df["ATR_final"] = df[atr_col] if atr_col else df["Close"] * 0.01

        return df
    except Exception as e:
        return None


# ------------------ HMM ------------------

def detect_regime_for_slice(df_price_slice):
    if not HMM_AVAILABLE or df_price_slice is None or len(df_price_slice) < 60: return "NEUTRAL"
    try:
        df = df_price_slice.copy()
        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
        df["range_vol"] = ((df["High"] - df["Low"]) / df["Close"]) * 100
        df = df.dropna(subset=["log_ret", "range_vol"])
        if len(df) < 60: return "NEUTRAL"

        X = df[["log_ret", "range_vol"]].values
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=420)
        model.fit(X)
        states = model.predict(X)
        df["state"] = states

        global_avg_ret, global_avg_vol = df["log_ret"].mean(), df["range_vol"].mean()
        current_state = states[-1]
        mask = df["state"] == current_state
        state_ret, state_vol = df.loc[mask, "log_ret"].mean(), df.loc[mask, "range_vol"].mean()

        if state_ret > global_avg_ret and state_vol < global_avg_vol:
            return "BULL"
        elif state_ret > 0 and state_vol > (global_avg_vol * 1.2):
            return "BULL"
        elif state_ret < -0.1:
            return "BEAR"
        elif state_ret < 0 and state_vol > global_avg_vol:
            return "BEAR"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


# ------------------ METRICS COMPUTATION ------------------

def compute_metrics(results_df):
    """
    Build per-horizon metrics including comprehensive win rates and average returns.
    """
    metrics = {}
    if results_df is None or results_df.empty:
        return metrics

    for h in HORIZONS:
        pred_col = f"pred_ret_{h}d"
        actual_col = f"actual_ret_{h}d"
        meta_col = f"meta_signal_{h}d"

        if pred_col not in results_df.columns: continue
        dfh = results_df.dropna(subset=[pred_col])
        if len(dfh) == 0: continue

        # Require actual to compute precision stats
        if actual_col in dfh.columns:
            df_valid = dfh.dropna(subset=[actual_col])
        else:
            df_valid = dfh.copy()

        n_samples = len(df_valid)
        if n_samples == 0: continue

        preds = df_valid[pred_col].astype(float)
        actuals = df_valid[actual_col].astype(float) if actual_col in df_valid.columns else pd.Series(
            [np.nan] * len(df_valid))

        sign_eq = (np.sign(preds) == np.sign(actuals)) & (np.sign(actuals) != 0)
        directional_accuracy = float(sign_eq.sum() / max(1, len(preds))) * 100.0
        mae_pct = float((preds - actuals).abs().mean() * 100.0)
        corr = float(preds.corr(actuals)) if len(preds) > 1 and actuals.notna().sum() > 1 else 0.0

        mean_pred = float(preds.mean() * 100.0)
        mean_actual = float(actuals.mean() * 100.0)

        # Hybrid Signal Deep Dive
        buy_mask = preds > 0.005
        sell_mask = preds < -0.005

        hybrid_buy_count = int(buy_mask.sum())
        hybrid_sell_count = int(sell_mask.sum())
        hybrid_hold_count = int(((preds <= 0.005) & (preds >= -0.005)).sum())

        buy_actuals = actuals[buy_mask].dropna()
        sell_actuals = actuals[sell_mask].dropna()

        hb_win = int((buy_actuals > 0).sum())
        hs_win = int((sell_actuals < 0).sum())

        hb_avg_ret = float(buy_actuals.mean() * 100.0) if not buy_actuals.empty else 0.0
        hs_avg_ret = float(sell_actuals.mean() * 100.0) if not sell_actuals.empty else 0.0

        hb_win_rate = (hb_win / max(1, hybrid_buy_count)) * 100.0
        hs_win_rate = (hs_win / max(1, hybrid_sell_count)) * 100.0

        # Meta Signal Deep Dive
        meta_accuracy = "N/A"
        meta_buy_count = meta_sell_count = meta_hold_count = 0
        meta_buy_win = meta_sell_win = 0
        meta_buy_avg_ret = meta_sell_avg_ret = None

        if meta_col in df_valid.columns:
            metas = df_valid[meta_col].fillna("HOLD").astype(str)
            meta_buy_count = int((metas == "BUY").sum())
            meta_sell_count = int((metas == "SELL").sum())
            meta_hold_count = int((metas == "HOLD").sum())

            def meta_matches(row):
                m = row.get(meta_col, "HOLD")
                a = row.get(actual_col, np.nan)
                if pd.isna(a): return False
                if m == "BUY" and a > 0: return True
                if m == "SELL" and a < 0: return True
                if m == "HOLD" and abs(a) <= 0.005: return True
                return False

            matched = df_valid.apply(meta_matches, axis=1)
            meta_accuracy = f"{round(float(matched.sum() / max(1, len(df_valid)) * 100.0), 1)}%"

            if meta_buy_count > 0 and actual_col in df_valid.columns:
                mb_actuals = df_valid.loc[df_valid[meta_col] == "BUY", actual_col].dropna().astype(float)
                meta_buy_win = int((mb_actuals > 0).sum())
                meta_buy_avg_ret = float(mb_actuals.mean() * 100.0) if not mb_actuals.empty else 0.0
            if meta_sell_count > 0 and actual_col in df_valid.columns:
                ms_actuals = df_valid.loc[df_valid[meta_col] == "SELL", actual_col].dropna().astype(float)
                meta_sell_win = int((ms_actuals < 0).sum())
                meta_sell_avg_ret = float(ms_actuals.mean() * 100.0) if not ms_actuals.empty else 0.0

        metrics[h] = {
            "n_samples": int(n_samples),
            "directional_accuracy": round(float(directional_accuracy), 1),
            "mae_pct": round(float(mae_pct), 2),
            "correlation": round(float(corr) if not pd.isna(corr) else 0.0, 3),
            "mean_predicted_pct": round(float(mean_pred), 1),
            "mean_actual_pct": round(float(mean_actual), 1),

            "hybrid_buy_count": hybrid_buy_count,
            "hybrid_buy_win_rate": round(hb_win_rate, 1),
            "hybrid_buy_avg_ret": round(hb_avg_ret, 2),
            "hybrid_sell_count": hybrid_sell_count,
            "hybrid_sell_win_rate": round(hs_win_rate, 1),
            "hybrid_sell_avg_ret": round(hs_avg_ret, 2),
            "hybrid_hold_count": hybrid_hold_count,

            "meta_accuracy": meta_accuracy,
            "meta_buy_count": int(meta_buy_count),
            "meta_buy_win_rate": f"{round(meta_buy_win / max(1, meta_buy_count) * 100.0, 1)}%" if meta_buy_count > 0 else "N/A",
            "meta_buy_avg_ret": (round(meta_buy_avg_ret, 2) if meta_buy_avg_ret is not None else "N/A"),
            "meta_sell_count": int(meta_sell_count),
            "meta_sell_win_rate": f"{round(meta_sell_win / max(1, meta_sell_count) * 100.0, 1)}%" if meta_sell_count > 0 else "N/A",
            "meta_sell_avg_ret": (round(meta_sell_avg_ret, 2) if meta_sell_avg_ret is not None else "N/A"),
            "meta_hold_count": int(meta_hold_count)
        }
    return metrics


# ------------------ PORTFOLIO SIM & LEDGER ------------------

def compute_portfolio_metrics(results_df, trade_horizon=5):
    def _run_portfolio_sim(results_df, signal_mode="combined", label=""):
        portfolio = {"equity": [100.0], "trades": 0, "wins": 0, "losses": 0, "buy_trades": 0, "sell_trades": 0,
                     "buy_wins": 0, "sell_wins": 0, "skipped_disagreement": 0, "regime_overrides": 0,
                     "skipped_confidence": 0, "skipped_horizon": 0, "skipped_ticker": 0}

        actual_col = f"actual_ret_{trade_horizon}d"
        meta_col = f"meta_signal_{trade_horizon}d"
        pred_col = f"pred_ret_{trade_horizon}d"
        conf_col = f"meta_confidence_{trade_horizon}d"
        regime_col = "hmm_regime"
        alt_horizon = 21 if trade_horizon == 5 else 5

        trade_log = []  # The Ledger

        if actual_col not in results_df.columns:
            return {"label": label, "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
                    "total_trades": 0, "buy_trades": 0, "sell_trades": 0, "win_rate_pct": 0.0, "buy_win_rate_pct": 0.0,
                    "sell_win_rate_pct": 0.0, "final_equity": 100.0, "trade_log": []}

        df = results_df.dropna(subset=[actual_col]).copy()
        if meta_col not in df.columns or pred_col not in df.columns:
            return {"label": label, "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
                    "total_trades": 0, "buy_trades": 0, "sell_trades": 0, "win_rate_pct": 0.0, "buy_win_rate_pct": 0.0,
                    "sell_win_rate_pct": 0.0, "final_equity": 100.0, "trade_log": []}

        df = df.dropna(subset=[meta_col, pred_col]).sort_values("sim_date")

        ticker_track, current_positions, current_equity = {}, {}, 100.0
        prev_macro_regime = "NEUTRAL"
        grouped = df.groupby("sim_date")

        for sim_date, group in grouped:
            target_weights, ticker_alloc_info = {}, {}

            current_macro = group[regime_col].mode()[0] if regime_col in df.columns and not group[
                regime_col].empty else "NEUTRAL"
            if pd.isna(current_macro): current_macro = "NEUTRAL"
            if current_macro != prev_macro_regime:
                ticker_track.clear()
                prev_macro_regime = current_macro

            for _, row in group.iterrows():
                tkr = row.get("ticker", "")
                meta_signal, hybrid_ret, actual = row[meta_col], row[pred_col], row[actual_col]
                confidence = row.get(conf_col, 0.05)
                if pd.isna(confidence): confidence = 0.05

                hybrid_signal = "BUY" if hybrid_ret > 0.005 else "SELL" if hybrid_ret < -0.005 else "HOLD"
                trade_signal, raw_weight, is_override = "HOLD", 0.0, False
                regime = row.get(regime_col, "NEUTRAL") if regime_col in df.columns else "NEUTRAL"
                if pd.isna(regime): regime = "NEUTRAL"

                if signal_mode == "regime":
                    if regime == "BULL":
                        if meta_signal == "BUY" or hybrid_signal == "BUY":
                            trade_signal = "BUY"
                        elif meta_signal == "SELL" and hybrid_signal == "SELL":
                            trade_signal = "SELL"
                    elif regime == "BEAR":
                        if meta_signal == "SELL" or hybrid_signal == "SELL":
                            trade_signal = "SELL"
                        elif meta_signal == "BUY" and hybrid_signal == "BUY":
                            trade_signal = "BUY"
                    else:
                        if meta_signal == hybrid_signal and meta_signal != "HOLD": trade_signal = meta_signal
                    raw_weight = min(0.30, max(0.10, 0.10 + confidence * 2.0))

                elif signal_mode == "combined_plus":
                    if regime == "BULL":
                        if meta_signal == "BUY" or hybrid_signal == "BUY":
                            trade_signal = "BUY"
                            if meta_signal != "BUY": is_override = True
                        elif meta_signal == "SELL" and hybrid_signal == "SELL" and confidence >= 0.05:
                            trade_signal = "SELL"
                    elif regime == "BEAR":
                        if (meta_signal == "SELL" or hybrid_signal == "SELL") and confidence >= 0.05:
                            trade_signal = "SELL"
                            if meta_signal != "SELL": is_override = True
                        elif meta_signal == "BUY" and hybrid_signal == "BUY":
                            trade_signal = "BUY"
                    else:
                        if meta_signal == hybrid_signal and meta_signal != "HOLD":
                            if not (meta_signal == "SELL" and confidence < 0.05): trade_signal = meta_signal
                        elif meta_signal != hybrid_signal and meta_signal != "HOLD" and hybrid_signal != "HOLD":
                            portfolio["skipped_disagreement"] += 1

                    if trade_signal != "HOLD" and not is_override:
                        if confidence < 0.08:
                            portfolio["skipped_confidence"] += 1
                            continue
                        meta_alt = row.get(f"meta_signal_{alt_horizon}d", "HOLD")
                        if pd.isna(meta_alt): meta_alt = "HOLD"
                        if (trade_signal == "BUY" and meta_alt == "SELL") or (
                                trade_signal == "SELL" and meta_alt == "BUY"):
                            portfolio["skipped_horizon"] += 1
                            continue

                    if trade_signal != "HOLD" and tkr in ticker_track:
                        t = ticker_track[tkr]
                        if t["total"] >= 3 and t["wins"] / t["total"] < 0.35:
                            portfolio["skipped_ticker"] += 1
                            continue

                    eff_conf = max(confidence, 0.15) if is_override else confidence
                    raw_weight = 0.10 + eff_conf * 2.0

                elif signal_mode == "combined":
                    if meta_signal == hybrid_signal and meta_signal != "HOLD":
                        trade_signal = meta_signal
                        raw_weight = 0.10 + confidence * 2.0
                    elif meta_signal != "HOLD" and hybrid_signal != "HOLD":
                        portfolio["skipped_disagreement"] += 1
                elif signal_mode == "meta":
                    if meta_signal != "HOLD": trade_signal, raw_weight = meta_signal, 0.20
                elif signal_mode == "hybrid":
                    if hybrid_signal != "HOLD": trade_signal, raw_weight = hybrid_signal, 0.20

                if trade_signal != "HOLD":
                    risk_adj_weight = min(raw_weight / max(abs(hybrid_ret), 0.01), 5.0)
                    target_weights[tkr] = -risk_adj_weight if trade_signal == "SELL" else risk_adj_weight
                    ticker_alloc_info[tkr] = {"signal": trade_signal, "actual_fwd_ret": actual,
                                              "close_px": row.get("close_at_sim", 0.0)}

            total_abs_weight = sum(abs(w) for w in target_weights.values())
            if total_abs_weight > 0.0:
                scale_factor = 1.0 / max(1.0, total_abs_weight)
                for tkr in target_weights: target_weights[tkr] *= scale_factor

            date_pnl = 0.0
            all_tickers = set(current_positions.keys()).union(set(target_weights.keys()))

            for tkr in all_tickers:
                old_w, new_w = current_positions.get(tkr, 0.0), target_weights.get(tkr, 0.0)
                weight_diff = new_w - old_w
                fee = weight_diff * TOTAL_BUY_FEE_PCT if weight_diff > 0 else abs(
                    weight_diff) * TOTAL_SELL_FEE_PCT if weight_diff < 0 else 0.0
                date_pnl -= fee

                if new_w != 0.0:
                    info = ticker_alloc_info[tkr]
                    trade_sig, actual_ret, close_px = info["signal"], info["actual_fwd_ret"], info["close_px"]
                    trade_pnl = abs(new_w) * actual_ret if new_w > 0 else abs(new_w) * -actual_ret
                    date_pnl += trade_pnl

                    # --- LEDGER ENTRY ---
                    trade_log.append({
                        "Date": sim_date.strftime("%Y-%m-%d"),
                        "Ticker": tkr,
                        "Action": trade_sig,
                        "Price": round(close_px, 2),
                        "Alloc_Pct": round(abs(new_w) * 100, 2),
                        "Fwd_Ret_Pct": round(actual_ret * 100, 2),
                        "PnL_Impact": round(trade_pnl * 100, 4)
                    })

                    portfolio["trades"] += 1
                    if trade_sig == "BUY":
                        portfolio["buy_trades"] += 1
                        if actual_ret > 0: portfolio["buy_wins"] += 1
                    elif trade_sig == "SELL":
                        portfolio["sell_trades"] += 1
                        if actual_ret < 0: portfolio["sell_wins"] += 1

                    ticker_track.setdefault(tkr, {"wins": 0, "total": 0})
                    ticker_track[tkr]["total"] += 1
                    if trade_pnl > 0:
                        portfolio["wins"] += 1
                        ticker_track[tkr]["wins"] += 1
                    else:
                        portfolio["losses"] += 1

            current_positions = target_weights
            current_equity *= (1.0 + date_pnl)
            portfolio["equity"].append(current_equity)

        eq = np.array(portfolio["equity"])
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
        total_ret = (eq[-1] / eq[0] - 1) * 100
        max_dd = np.min(eq / np.maximum.accumulate(eq) - 1) * 100 if len(eq) > 1 else 0
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(52) if len(returns) > 1 else 0

        return {
            "label": label, "total_return_pct": round(total_ret, 2), "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": portfolio["trades"], "buy_trades": portfolio["buy_trades"],
            "sell_trades": portfolio["sell_trades"],
            "win_rate_pct": round(portfolio["wins"] / max(1, portfolio["trades"]) * 100, 1),
            "buy_win_rate_pct": round(portfolio["buy_wins"] / max(1, portfolio["buy_trades"]) * 100, 1) if portfolio[
                                                                                                               "buy_trades"] > 0 else 0.0,
            "sell_win_rate_pct": round(portfolio["sell_wins"] / max(1, portfolio["sell_trades"]) * 100, 1) if portfolio[
                                                                                                                  "sell_trades"] > 0 else 0.0,
            "final_equity": round(eq[-1], 2), "skipped_disagreement": portfolio["skipped_disagreement"],
            "regime_overrides": portfolio.get("regime_overrides", 0),
            "skipped_confidence": portfolio.get("skipped_confidence", 0),
            "skipped_horizon": portfolio.get("skipped_horizon", 0),
            "skipped_ticker": portfolio.get("skipped_ticker", 0),
            "trade_log": trade_log
        }

    return {
        "meta_only": _run_portfolio_sim(results_df, "meta", "META-ONLY"),
        "combined": _run_portfolio_sim(results_df, "combined", "COMBINED"),
        "hybrid_only": _run_portfolio_sim(results_df, "hybrid", "HYBRID-ONLY"),
        "regime_aware": _run_portfolio_sim(results_df, "regime", "REGIME-AWARE"),
        "combined_plus": _run_portfolio_sim(results_df, "combined_plus", "COMBINED+")
    }


def compute_per_ticker_stats(results_df, trade_horizon=5):
    """Generates the per-ticker performance summary for the specified horizon."""
    actual_col = f"actual_ret_{trade_horizon}d"
    pred_col = f"pred_ret_{trade_horizon}d"
    meta_col = f"meta_signal_{trade_horizon}d"

    if actual_col not in results_df.columns: return pd.DataFrame()
    df_valid = results_df.dropna(subset=[pred_col, actual_col]).copy()
    if df_valid.empty: return pd.DataFrame()

    def get_dir_acc(g):
        sign_eq = np.sign(g[pred_col]) == np.sign(g[actual_col])
        return round((sign_eq.sum() / len(g)) * 100)

    def get_meta_acc(g):
        if meta_col not in g.columns: return "N/A"

        def match(row):
            m, a = row[meta_col], row[actual_col]
            if m == "BUY" and a > 0: return True
            if m == "SELL" and a < 0: return True
            if m == "HOLD" and abs(a) <= 0.005: return True
            return False

        return round((g.apply(match, axis=1).sum() / len(g)) * 100)

    stats = []
    for tkr, group in df_valid.groupby("ticker"):
        stats.append({
            "Ticker": tkr,
            "N": len(group),
            "DirAcc": get_dir_acc(group),
            "MetaAcc": get_meta_acc(group),
            "Pred": round(group[pred_col].mean() * 100, 1),
            "Actual": round(group[actual_col].mean() * 100, 1),
            "HMM Mode": group["hmm_regime"].mode()[0] if "hmm_regime" in group.columns else "NEUTRAL"
        })
    return pd.DataFrame(stats).sort_values("Ticker").reset_index(drop=True)


# ------------------ REPORT ------------------

def print_report(metrics, portfolio_metrics, results_df, n_tickers, n_sim_dates, oos_start, oos_end, trade_horizon=5):
    print("\n" + "=" * 80)
    print("  ChronoStox v9.5 — OUT-OF-SAMPLE BACKTEST REPORT (Combined+ Edition)")
    print("=" * 80)
    print(f"  Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  OOS Period    : {oos_start} → {oos_end}")
    print(f"  Tickers Tested: {n_tickers}")
    print(f"  Simulation Pts: {n_sim_dates}")
    print("-" * 80)

    print(f"\n  HYBRID PRICE TARGET ANALYSIS (bidirectional ATR):")
    print(
        f"  {'HORIZON':<8} | {'N':<5} | {'DIR ACC':<8} | {'MAE%':<7} | {'CORR':<6} | {'Pred→Act':<18} | {'BUY':<5} | {'SELL':<5} | {'HOLD':<5}")
    print("  " + "-" * 88)

    for h in HORIZONS:
        if h not in metrics: continue
        m = metrics[h]
        if m['n_samples'] == 0:
            print(f"  {str(h) + 'd':<8} | —")
            continue
        pred_act = f"{m['mean_predicted_pct']:+.1f}→{m['mean_actual_pct']:+.1f}%"
        print(
            f"  {str(h) + 'd':<8} | {m['n_samples']:<5} | {m['directional_accuracy']:>5.1f}%  | {m['mae_pct']:>5.2f}% | {m['correlation']:>5.3f} | {pred_act:<18} | {m['hybrid_buy_count']:<5} | {m['hybrid_sell_count']:<5} | {m['hybrid_hold_count']:<5}")

    print(f"\n  META-LEARNER SIGNAL ANALYSIS:")
    print(
        f"  {'HORIZON':<8} | {'META ACC':<9} | {'BUY':<5} | {'BUY WR':<7} | {'BUY Avg':<8} | {'SELL':<5} | {'SELL WR':<8} | {'SELL Avg':<9} | {'HOLD':<5}")
    print("  " + "-" * 92)

    for h in HORIZONS:
        if h not in metrics: continue
        m = metrics[h]
        if m['n_samples'] == 0:
            print(f"  {str(h) + 'd':<8} | —")
            continue

        mb_avg = m.get('meta_buy_avg_ret', 'N/A')
        ms_avg = m.get('meta_sell_avg_ret', 'N/A')
        mb_str = f"{mb_avg:>+6.2f}%" if isinstance(mb_avg, float) else f"{mb_avg:>7}"
        ms_str = f"{ms_avg:>+6.2f}%" if isinstance(ms_avg, float) else f"{ms_avg:>8}"

        print(
            f"  {str(h) + 'd':<8} | {m['meta_accuracy']:<9} | {m.get('meta_buy_count', 0):<5} | {m.get('meta_buy_win_rate', 'N/A'):<7} | {mb_str:<8} | {m.get('meta_sell_count', 0):<5} | {m.get('meta_sell_win_rate', 'N/A'):<8} | {ms_str:<9} | {m.get('meta_hold_count', 0):<5}")

    print("\n" + "-" * 80)
    print("  DETAILED BREAKDOWN:")
    print("-" * 80)
    for h in HORIZONS:
        if h not in metrics or metrics[h]['n_samples'] == 0: continue
        m = metrics[h]

        hb_avg = m.get('hybrid_buy_avg_ret', 0.0)
        hs_avg = m.get('hybrid_sell_avg_ret', 0.0)
        mb_avg = m.get('meta_buy_avg_ret', 'N/A')
        ms_avg = m.get('meta_sell_avg_ret', 'N/A')
        mb_str = f"{mb_avg:+.2f}%" if isinstance(mb_avg, float) else str(mb_avg)
        ms_str = f"{ms_avg:+.2f}%" if isinstance(ms_avg, float) else str(ms_avg)

        print(f"\n  {h}-DAY HORIZON:")
        print(f"    Samples           : {m['n_samples']}")
        print(f"    Hybrid Dir. Acc   : {m['directional_accuracy']}%")
        print(f"    MAE               : {m['mae_pct']}%")
        print(f"    Pred vs Actual    : {m['mean_predicted_pct']}% predicted → {m['mean_actual_pct']}% actual")
        print(f"    --- Hybrid Signals ---")
        print(f"    BUY  : {m['hybrid_buy_count']} (win rate: {m['hybrid_buy_win_rate']}%, avg ret: {hb_avg:+.2f}%)")
        print(f"    SELL : {m['hybrid_sell_count']} (win rate: {m['hybrid_sell_win_rate']}%, avg ret: {hs_avg:+.2f}%)")
        print(f"    HOLD : {m['hybrid_hold_count']}")
        print(f"    --- Meta-Learner ---")
        print(f"    Meta Accuracy     : {m['meta_accuracy']}")
        print(f"    Meta BUY  : {m['meta_buy_count']} (win rate: {m['meta_buy_win_rate']}, avg ret: {mb_str})")
        print(f"    Meta SELL : {m['meta_sell_count']} (win rate: {m['meta_sell_win_rate']}, avg ret: {ms_str})")
        print(f"    Meta HOLD : {m['meta_hold_count']}")

    print("\n" + "-" * 80)
    print(f"  PORTFOLIO STRATEGY COMPARISON ({trade_horizon}d horizon):")
    print("-" * 80)
    print(
        f"  {'Strategy':<18} | {'Return':>8} | {'MaxDD':>8} | {'Sharpe':>7} | {'Trades':>15} | {'Win%':>6} | {'Equity':>9}")
    print("  " + "-" * 80)

    for key in ["combined_plus", "combined", "regime_aware", "meta_only", "hybrid_only"]:
        pm = portfolio_metrics.get(key, {"final_equity": 100.0, "total_trades": 0, "buy_trades": 0, "sell_trades": 0,
                                         "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
                                         "win_rate_pct": 0.0})
        label = {"combined_plus": "⭐ COMBINED+", "combined": "   COMBINED", "regime_aware": "   REGIME-AWARE",
                 "meta_only": "   META-ONLY", "hybrid_only": "   HYBRID-ONLY"}.get(key, key)
        trades_str = f"{pm['total_trades']} ({pm.get('buy_trades', 0)}B/{pm.get('sell_trades', 0)}S)"
        print(
            f"  {label:<18} | {pm['total_return_pct']:>+7.1f}% | {pm['max_drawdown_pct']:>+7.1f}% | {pm['sharpe_ratio']:>+6.2f} | {trades_str:>15} | {pm['win_rate_pct']:>5.1f}% | \u20b9{pm['final_equity']:>7.2f}")

    cp = portfolio_metrics.get("combined_plus", {})
    if cp:
        alt_horizon = 21 if trade_horizon == 5 else 5
        print("\n  ⭐ COMBINED+ STRATEGY DETAILS:")
        print(f"    Base             : Combined (hybrid + meta {trade_horizon}d must agree)")
        print("    + Confidence     : meta gap ≥ 0.08 required")
        print(f"    + Multi-horizon  : {alt_horizon}d meta must not contradict {trade_horizon}d direction")
        print("    + Ticker quality : skip ticker if win rate < 35% after 3+ trades")
        print("    Position Sizing  : 10-30% (confidence-based)")
        print(f"    Skipped (disagree)  : {cp.get('skipped_disagreement', 0)}")
        print(f"    Skipped (low conf)  : {cp.get('skipped_confidence', 0)}")
        print(f"    Skipped (horizon)   : {cp.get('skipped_horizon', 0)}")
        print(f"    Skipped (bad ticker): {cp.get('skipped_ticker', 0)}")
        print(f"    Buy Win Rate     : {cp.get('buy_win_rate_pct', 0.0)}%")
        print(f"    Sell Win Rate    : {cp.get('sell_win_rate_pct', 0.0)}%")

    print("\n" + "-" * 80)
    print(f"  PER-TICKER {trade_horizon}d PERFORMANCE:")
    print("-" * 80)
    df_tkr = compute_per_ticker_stats(results_df, trade_horizon)
    if not df_tkr.empty:
        print(
            f"  {'Ticker':<18} | {'N':>3} | {'DirAcc':>6} | {'MetaAcc':>7} | {'Pred':>6} | {'Actual':>7} | {'HMM Mode':>10}")
        print("  " + "-" * 70)
        # Print top 30 to terminal to avoid console spam
        for _, row in df_tkr.head(30).iterrows():
            print(
                f"  {row['Ticker']:<18} | {row['N']:>3} | {row['DirAcc']:>5}% | {str(row['MetaAcc']) + '%':>7} | {row['Pred']:>5.1f}% | {row['Actual']:>6.1f}% | {row['HMM Mode']:>10}")
        if len(df_tkr) > 30:
            print(f"  ... (+ {len(df_tkr) - 30} more tickers. Full list saved to 'per_ticker_stats.csv')")
        df_tkr.to_csv("per_ticker_stats.csv", index=False)

    print("\n" + "-" * 80)
    print("  LEDGER (COMBINED+) - RECENT TRADES:")
    print("-" * 80)
    trade_log = cp.get("trade_log", [])
    if trade_log:
        df_ledger = pd.DataFrame(trade_log)
        print(
            f"  {'Date':<10} | {'Ticker':<12} | {'Action':<6} | {'Price':>8} | {'Alloc%':>7} | {'FwdRet%':>8} | {'PnL Impact':>10}")
        print("  " + "-" * 76)
        # Print last 15 trades
        for _, row in df_ledger.tail(15).iterrows():
            print(
                f"  {row['Date']:<10} | {row['Ticker']:<12} | {row['Action']:<6} | {row['Price']:>8.2f} | {row['Alloc_Pct']:>6.2f}% | {row['Fwd_Ret_Pct']:>+7.2f}% | {row['PnL_Impact']:>+10.4f}")
        df_ledger.to_csv("combat_ledger.csv", index=False)
        print(f"  \n  -> Full {len(df_ledger)}-trade ledger successfully dumped to 'combat_ledger.csv'")
    else:
        print("  No trades executed by COMBINED+ strategy in this period.")

    print("=" * 80)


# ------------------ BACKTEST ENGINE (FAST) ------------------

def run_backtest(tickers, oos_start, oos_end, frequency="weekly"):
    t0 = time.time()
    print("[1/5] Loading models & global data...")
    model_dir = "." if os.path.exists(os.path.join(".", MODEL_JOBLIB)) else "../test"
    models = load_models_once(model_dir)
    df_macro = load_macro_once(model_dir)
    df_senti = load_sentiment_once(model_dir)
    df_sector = load_sectors_once(model_dir)

    scaler = models["scaler"]
    lgbm_model = models["lgbm"]
    lstm_model = models["lstm"]
    meta_models = models.get("meta_models", {})
    seq_len = models["seq_len"]

    print(f"\n[2/5] Fetching price history for {len(tickers)} tickers & Macro Proxy (^NSEI)...")
    fetch_start = (pd.to_datetime(oos_start) - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
    price_cache, precomputed_cache = {}, {}

    nifty_df = fetch_price_history("^NSEI", start_date=fetch_start)

    for i, tkr in enumerate(tickers):
        sys.stdout.write(f"\r  [{i + 1}/{len(tickers)}] Fetching {tkr:<15}")
        sys.stdout.flush()
        df_px = fetch_price_history(tkr, start_date=fetch_start)
        if df_px is not None and len(df_px) > 100: price_cache[tkr] = df_px

    print(f"\n\n[3/5] PRE-COMPUTING Features to C-Memory (NumPy Arrays)...")
    for i, (tkr, df_px) in enumerate(price_cache.items()):
        sys.stdout.write(f"\r  [{i + 1}/{len(price_cache)}] Processing {tkr:<15}")
        sys.stdout.flush()
        df_feat = precompute_features(df_px, df_macro, df_senti, df_sector, models["features"])
        if df_feat is not None:
            precomputed_cache[tkr] = {
                "dates": df_feat["Date"].values.astype('datetime64[D]'),
                "features_np": df_feat[models["features"]].values.astype(np.float32),
                "close_np": df_feat["Close"].values.astype(np.float32),
                "atr_np": df_feat["ATR_final"].values.astype(np.float32),
                "df": df_feat
            }

    start_dt, end_dt = pd.to_datetime(oos_start), pd.to_datetime(oos_end)
    if frequency == "daily" and len(price_cache) > 0:
        sample_tkr = list(price_cache.keys())[0]
        trading_days = price_cache[sample_tkr]["Date"]
        sim_dates = trading_days[(trading_days >= start_dt) & (trading_days <= end_dt)].tolist()
    else:
        sim_dates = pd.date_range(start_dt, end_dt, freq="W-FRI").tolist()

    max_verify_date = {h: pd.to_datetime(datetime.now().strftime("%Y-%m-%d")) - pd.Timedelta(days=h * 1.5) for h in
                       HORIZONS}

    print(f"\n\n[4/5] TENSOR BATCH INFERENCE ({len(sim_dates)} matrices)...")
    all_results = []
    results_dict = {}
    proxy_errors = {}

    total_preds = len(sim_dates) * len(precomputed_cache)
    completed = 0

    for sim_date in sim_dates:
        sim_date_np = np.datetime64(sim_date.date())
        macro_regime = "NEUTRAL"
        if nifty_df is not None:
            nifty_slice = nifty_df[nifty_df["Date"] <= sim_date]
            macro_regime = detect_regime_for_slice(nifty_slice)

        batch_inputs = []

        for tkr, cache in precomputed_cache.items():
            idx = np.searchsorted(cache["dates"], sim_date_np, side='right') - 1
            if idx < 100: continue

            eval_date_np = np.datetime64((sim_date - pd.Timedelta(days=7)).date())
            idx_past = np.searchsorted(cache["dates"], eval_date_np, side='right') - 1

            dyn_weights = {"lgbm": 0.5, "lstm": 0.5}
            if idx_past >= 100:
                past_date_key = cache["dates"][idx_past]
                past_res = results_dict.get((tkr, past_date_key))

                if past_res:
                    actual_ret_5d = (cache["close_np"][idx] - cache["close_np"][idx_past]) / cache["close_np"][idx_past]
                    err_lgb = abs(actual_ret_5d - past_res.get("raw_ml_ret_5d_lgbm", 0.0))
                    err_lstm = abs(actual_ret_5d - past_res.get("raw_ml_ret_5d_lstm", 0.0))

                    proxy_errors.setdefault(tkr, []).append((err_lgb, err_lstm))
                    if len(proxy_errors[tkr]) >= 3:
                        lgb_errs = [e[0] for e in proxy_errors[tkr][-10:]]
                        lstm_errs = [e[1] for e in proxy_errors[tkr][-10:]]
                        total_err = sum(lgb_errs) + sum(lstm_errs)
                        if total_err > 0:
                            dyn_weights["lgbm"] = max(0.1, min(0.9, sum(lstm_errs) / total_err))
                            dyn_weights["lstm"] = max(0.1, min(0.9, sum(lgb_errs) / total_err))

            X_row = cache["features_np"][idx]
            seq = cache["features_np"][idx - seq_len + 1: idx + 1] if (idx + 1) >= seq_len else np.zeros(
                (seq_len, X_row.shape[0]), dtype=np.float32)

            batch_inputs.append({
                "tkr": tkr, "idx": idx, "X_row": X_row, "seq": seq,
                "close": cache["close_np"][idx], "atr": cache["atr_np"][idx],
                "dyn_weights": dyn_weights
            })
            completed += 1

        if not batch_inputs: continue

        X_mat = np.array([b["X_row"] for b in batch_inputs])
        seq_mat = np.array([b["seq"] for b in batch_inputs])

        Xs_mat = scaler.transform(X_mat)
        raw_lgb_mat = lgbm_model.predict(Xs_mat)

        raw_lstm_mat = lstm_model(seq_mat, training=False).numpy()

        for b_idx, b in enumerate(batch_inputs):
            tkr = b["tkr"]
            close = b["close"]
            atr = b["atr"]
            vol_factor = atr / close

            raw_lgb = raw_lgb_mat[b_idx]
            raw_lstm = raw_lstm_mat[b_idx]
            raw_avg = (raw_lgb * b["dyn_weights"]["lgbm"]) + (raw_lstm * b["dyn_weights"]["lstm"])

            ml_prices = close * (1.0 + raw_avg)
            hybrid = []
            for i, h in enumerate(HORIZONS):
                atr_offset = atr * ATR_MULT[h]
                atr_price = close + atr_offset if np.sign(raw_avg[i]) >= 0 else close - atr_offset
                raw_h = ml_prices[i] * WEIGHT_ML_H[h] + atr_price * WEIGHT_ATR_H[h]

                max_move = vol_factor * np.sqrt(h) * 2.0
                clamped = max(close * (1 - max_move), min(close * (1 + max_move), raw_h))
                hybrid.append(clamped)

            meta_signals = {}
            for i, h in enumerate(HORIZONS):
                if h in meta_models:
                    try:
                        X_meta = np.array([[raw_lgb[i], raw_lstm[i]]], dtype=np.float32)
                        proba = meta_models[h].predict_proba(X_meta)[0]
                        pred_class = int(meta_models[h].predict(X_meta)[0])
                        meta_signals[h] = {
                            "signal": ["SELL", "HOLD", "BUY"][pred_class],
                            "confidence": float(max(proba) - sorted(proba)[-2])
                        }
                    except:
                        meta_signals[h] = {"signal": "HOLD", "confidence": 0.0}

            result = {
                "ticker": tkr, "sim_date": sim_date, "close_at_sim": close,
                "hmm_regime": macro_regime, "weight_lgbm": b["dyn_weights"]["lgbm"],
                "weight_lstm": b["dyn_weights"]["lstm"]
            }

            for i, h in enumerate(HORIZONS):
                result[f"raw_ml_ret_{h}d"] = raw_avg[i]
                result[f"raw_ml_ret_{h}d_lgbm"] = raw_lgb[i]
                result[f"raw_ml_ret_{h}d_lstm"] = raw_lstm[i]
                result[f"pred_ret_{h}d"] = (hybrid[i] - close) / close
                result[f"pred_target_{h}d"] = hybrid[i]

                if h in meta_signals:
                    result[f"meta_signal_{h}d"] = meta_signals[h]["signal"]
                    result[f"meta_confidence_{h}d"] = meta_signals[h]["confidence"]

                if sim_date <= max_verify_date.get(h, pd.Timestamp("2000-01-01")):
                    df_full = precomputed_cache[tkr]["df"]
                    future_dates = df_full[df_full["Date"] > sim_date].head(h)
                    if len(future_dates) >= max(1, int(h * 0.7)):
                        result[f"actual_ret_{h}d"] = (future_dates["Close"].iloc[-1] - close) / close

            all_results.append(result)
            results_dict[(tkr, np.datetime64(sim_date.date()))] = result

        if completed % 1000 < 200:
            sys.stdout.write(f"\r  -> Tensor Batch Computed {completed}/{total_preds} predictions...")
            sys.stdout.flush()

    results_df = pd.DataFrame(all_results)
    print(f"\n  ✅ {len(results_df)} predictions recorded")

    print("\n[5/5] Finalizing Metrics & Generating Files...")

    metrics = compute_metrics(results_df)

    # --- DYNAMIC HORIZON TOGGLE ---
    primary_horizon = 5  # Change this to 21, 63, etc., to change the ledger's tracking logic

    portfolio_metrics = compute_portfolio_metrics(results_df, trade_horizon=primary_horizon)

    print_report(metrics, portfolio_metrics, results_df, len(precomputed_cache), len(sim_dates), oos_start, oos_end,
                 trade_horizon=primary_horizon)

    results_df.to_csv("backtest_results.csv", index=False)

    # Safely write metrics without breaking if N/A is present
    safe_metrics = []
    for h, m in metrics.items():
        if m['n_samples'] > 0:
            safe_metrics.append({**m, "horizon": h})
    if safe_metrics:
        pd.DataFrame(safe_metrics).to_csv("backtest_metrics.csv", index=False)

    print(f"\n⏱ Total runtime: {int((time.time() - t0) // 60)}m {int((time.time() - t0) % 60)}s")


# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2025-11-17")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--daily", action="store_true")
    args = parser.parse_args()
    run_backtest(args.tickers if args.tickers else get_dynamic_tickers(), args.start, args.end,
                 "daily" if args.daily else "weekly")


if __name__ == "__main__":
    main()


# Live wrapper logic intact (engineer_features_for_slice & predict_at_date)
def engineer_features_for_slice(df_price, df_macro, df_senti, df_sector, expected_features):
    try:
        df_full = precompute_features(df_price, df_macro, df_senti, df_sector, expected_features)
        if df_full is None: return None, None
        for c in expected_features:
            if c not in df_full.columns: df_full[c] = 0.0
        cols = ["Date", "Close"] + [c for c in expected_features if c in df_full.columns]
        return df_full[cols].copy(), df_full
    except Exception as e:
        print(f"⚠️ engineer_features_for_slice failed: {e}")
        return None, None


def predict_at_date(df_merged, models):
    out = {"close": None, "predicted_returns": {}, "predicted_targets": {}, "meta_signals": {}, "raw": {}}
    try:
        if df_merged is None or len(df_merged) < 1: return out
        features = models.get("features", [])
        scaler = models.get("scaler")
        lgbm_model = models.get("lgbm")
        lstm_model = models.get("lstm")
        meta_models = models.get("meta_models", {})
        horizons = models.get("horizons", HORIZONS)
        seq_len = models.get("seq_len", models.get("lstm_sequence_length", 60))

        latest_idx = len(df_merged) - 1
        close = float(df_merged["Close"].iloc[latest_idx])
        out["close"] = close

        X_row = df_merged[features].iloc[latest_idx].values.astype(np.float32)
        if latest_idx + 1 >= seq_len:
            seq = df_merged[features].iloc[latest_idx - seq_len + 1: latest_idx + 1].values.astype(np.float32)
        else:
            pad = np.zeros((seq_len - (latest_idx + 1), len(features)), dtype=np.float32)
            seq = np.vstack([pad, df_merged[features].iloc[0:latest_idx + 1].values.astype(np.float32)])

        Xs = scaler.transform(X_row.reshape(1, -1))
        raw_lgb = np.array(lgbm_model.predict(Xs)).ravel()
        seq_in = np.expand_dims(seq, axis=0)
        raw_lstm = lstm_model(seq_in, training=False).numpy().ravel()

        raw_lgb = np.array(raw_lgb, dtype=float).ravel()
        raw_lstm = np.array(raw_lstm, dtype=float).ravel()
        minlen = min(len(raw_lgb), len(raw_lstm), len(horizons))
        raw_lgb, raw_lstm = raw_lgb[:minlen], raw_lstm[:minlen]
        raw_avg = (raw_lgb + raw_lstm) / 2.0

        for i, h in enumerate(horizons[:minlen]):
            ml_price = close * (1.0 + float(raw_avg[i]))
            atr = float(df_merged.get("ATR_final", df_merged.get("ATR_14", df_merged["Close"] * 0.01)).iloc[latest_idx])
            atr_offset = atr * ATR_MULT.get(h, 1.0)
            atr_price = (close + atr_offset) if (raw_avg[i] >= 0) else (close - atr_offset)
            raw_h = ml_price * WEIGHT_ML_H.get(h, 0.5) + atr_price * WEIGHT_ATR_H.get(h, 0.5)

            vol_factor = max(atr / max(close, 1e-9), 0.0001)
            max_move = vol_factor * np.sqrt(h) * 2.0
            clamped = max(close * (1 - max_move), min(close * (1 + max_move), raw_h))

            pred_ret = (clamped - close) / close
            out["predicted_returns"][h] = float(pred_ret)
            out["predicted_targets"][h] = float(clamped)

            if h in meta_models:
                try:
                    X_meta = np.array([[raw_lgb[i], raw_lstm[i]]], dtype=np.float32)
                    meta = meta_models[h]
                    proba = meta.predict_proba(X_meta)[0]
                    pred_class = int(meta.predict(X_meta)[0])
                    mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
                    out["meta_signals"][h] = {"signal": mapping.get(pred_class, "HOLD"),
                                              "confidence": float(max(proba) - sorted(proba)[-2]) if len(
                                                  proba) > 1 else float(max(proba))}
                except:
                    out["meta_signals"][h] = {"signal": "HOLD", "confidence": 0.0}

        out["raw"]["raw_lgb"], out["raw"]["raw_lstm"], out["raw"]["raw_avg"], out["raw"][
            "close"] = raw_lgb.tolist(), raw_lstm.tolist(), raw_avg.tolist(), close
        return out
    except Exception as e:
        print(f"⚠️ predict_at_date failed: {e}")
        return out