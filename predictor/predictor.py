import os
import sys
import json
import joblib
import warnings
import argparse
import logging
import atexit
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress annoying TF/Keras and Loky warnings for a clean terminal
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import pandas_ta as ta
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_FILE = os.path.join(PROJECT_ROOT, "predictor", "portfolio.json")
LOG_FILE = os.path.join(PROJECT_ROOT, "predictor", "engine.log")
LOCK_FILE = os.path.join(PROJECT_ROOT, "predictor", ".engine.lock")
MODEL_DIR = os.path.join(PROJECT_ROOT, "test")

STARTING_CAPITAL = 100000.0
MIN_TRADE_VALUE = 5000.0
MAX_POSITIONS = 25

SLIPPAGE_BPS = 10
BROKERAGE_BPS = 3
STT_BPS = 10
EXCHANGE_FEES_BPS = 0.35

HORIZONS = [5, 21, 63, 126, 252]
ATR_MULT = {5: 1.0, 21: 1.8, 63: 3.0, 126: 4.8, 252: 7.2}
WEIGHT_ML_H = {5: 0.62, 21: 0.55, 63: 0.50, 126: 0.42, 252: 0.35}
WEIGHT_ATR_H = {h: 1 - WEIGHT_ML_H[h] for h in HORIZONS}

# ---------------------------------------------------------
# SETUP LOGGING & SMART LOCKS
# ---------------------------------------------------------
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def check_and_set_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            print(f"❌ ERROR: Engine is already running (PID: {old_pid})")
            sys.exit(1)
        except (ValueError, OSError):
            os.remove(LOCK_FILE)
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.remove(LOCK_FILE) if os.path.exists(LOCK_FILE) else None)


check_and_set_lock()


# ---------------------------------------------------------
# PORTFOLIO & EXECUTION
# ---------------------------------------------------------
def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {"cash": STARTING_CAPITAL, "positions": {}, "history": [],
                "last_deposit_month": datetime.now().strftime("%Y-%m")}
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


def save_portfolio(pf):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(pf, f, indent=4)


def execute_trade(pf, ticker, action, shares, price, date):
    value = shares * price
    fees = value * ((SLIPPAGE_BPS + BROKERAGE_BPS + EXCHANGE_FEES_BPS) / 10000)
    pnl_pct = 0.0

    if action == "SELL": fees += value * (STT_BPS / 10000)

    if action == "BUY":
        if pf["cash"] < (value + fees): return False
        pf["cash"] -= (value + fees)
        if ticker not in pf["positions"]:
            pf["positions"][ticker] = {"shares": shares, "avg_price": price, "buy_date": date}
        else:
            old = pf["positions"][ticker]
            new_shares = old["shares"] + shares
            pf["positions"][ticker]["avg_price"] = ((old["avg_price"] * old["shares"]) + (price * shares)) / new_shares
            pf["positions"][ticker]["shares"] = new_shares
    elif action == "SELL":
        if ticker not in pf["positions"]: return False
        pos = pf["positions"][ticker]
        if pos.get("buy_date") == date: return False  # Prevent day-trading loops

        buy_value = pos["avg_price"] * shares
        pnl_pct = (((value - fees) - buy_value) / buy_value) * 100 if buy_value > 0 else 0.0

        pf["cash"] += (value - fees)
        pos["shares"] -= shares
        if pos["shares"] <= 0: del pf["positions"][ticker]

    pf["history"].append(
        {"date": date, "ticker": ticker, "action": action, "shares": shares, "price": price, "value": value,
         "fees": fees, "pnl_pct": round(pnl_pct, 2) if action == "SELL" else 0.0})
    save_portfolio(pf)
    return True


# ---------------------------------------------------------
# REGIME DETECTION (INDEX TUNED)
# ---------------------------------------------------------
def detect_regime(df):
    if df is None or len(df) < 200: return "NEUTRAL"
    df = df.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
    df["range_vol"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df = df.dropna()
    if len(df) < 200: return "NEUTRAL"

    X = df[["log_ret", "range_vol"]].values
    try:
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=42)
        model.fit(X)
        states = model.predict(X)
        df["state"] = states

        cur = states[-1]
        state = df[df["state"] == cur]
        state_ret, state_vol = state["log_ret"].mean(), state["range_vol"].mean()
        global_ret, global_vol = df["log_ret"].mean(), df["range_vol"].mean()

        # --- TUNED FOR LOW-VOLATILITY INDICES (Nifty 50) ---
        if state_ret > global_ret and state_vol <= (global_vol * 1.1):
            return "BULL"
        elif state_ret > 0 and state_vol > (global_vol * 1.1):
            return "BULL"
        elif state_ret < -0.05:
            return "BEAR"  # Sensitive crash trigger
        elif state_ret < 0 and state_vol > global_vol:
            return "BEAR"
        return "NEUTRAL"
    except Exception as e:
        logging.warning(f"HMM Regime detection failed: {e}")
        return "NEUTRAL"


# ---------------------------------------------------------
# STANDALONE DATA PIPELINE & ML ENGINE
# ---------------------------------------------------------
def engineer_features_standalone(df_price, df_macro, df_senti, df_sector, expected_cols):
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
        ssub = ssub.sort_values("Date")[["Date", "sentiment_score"]]
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
            if str(c).startswith("Sector_"): df[f"sentiment_x_{c}"] = df[c].astype(float) * df[
                "sentiment_score"].astype(float)

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in set(expected_cols) - set(df.columns): df[col] = 0.0

        atr_col = next((c for c in ["ATR_14", "ATRr_14", "ATR"] if c in df.columns), None)
        df["ATR_final"] = df[atr_col] if atr_col else df["Close"] * 0.01

        return df
    except Exception:
        return None


def predict_live(df, models, horizon=21):
    close = float(df["Close"].iloc[-1])
    atr = float(df["ATR_final"].iloc[-1])
    feature_order, scaler = models["features"], models["scaler"]
    model_lgb, model_lstm, seq_len = models["lgbm"], models["lstm"], models["seq_len"]

    X_row = df[feature_order].iloc[-1].values.astype(np.float32).reshape(1, -1)
    Xs = scaler.transform(X_row)
    raw_lgb = model_lgb.predict(Xs)[0]

    if len(df) >= seq_len:
        seq = df[feature_order].tail(seq_len).values.astype(np.float32).reshape(1, seq_len, -1)
        raw_lstm = model_lstm(seq, training=False).numpy()[0]
    else:
        raw_lstm = np.zeros_like(raw_lgb)

    raw_avg = (raw_lgb * 0.5) + (raw_lstm * 0.5)
    vol_factor = atr / close
    ml_prices = close * (1.0 + raw_avg)

    hybrid_targets = {}
    for i, h in enumerate(HORIZONS):
        atr_offset = atr * ATR_MULT[h]
        atr_price = close + atr_offset if np.sign(raw_avg[i]) >= 0 else close - atr_offset
        raw_h = ml_prices[i] * WEIGHT_ML_H[h] + atr_price * WEIGHT_ATR_H[h]
        max_move = vol_factor * np.sqrt(h) * 2.0
        clamped = max(close * (1 - max_move), min(close * (1 + max_move), raw_h))
        hybrid_targets[h] = clamped

    meta_signal, meta_conf = "HOLD", 0.0
    meta_models = models.get("meta_models", {})
    if horizon in meta_models:
        h_idx = HORIZONS.index(horizon)
        try:
            X_meta = np.array([[raw_lgb[h_idx], raw_lstm[h_idx]]], dtype=np.float32)
            proba = meta_models[horizon].predict_proba(X_meta)[0]
            pred_class = int(meta_models[horizon].predict(X_meta)[0])
            meta_signal = ["SELL", "HOLD", "BUY"][pred_class]
            meta_conf = float(max(proba) - sorted(proba)[-2])
        except:
            pass

    target = hybrid_targets.get(horizon, close)
    hybrid_ret = (target - close) / close
    hybrid_sig = "BUY" if hybrid_ret > 0.005 else "SELL" if hybrid_ret < -0.005 else "HOLD"

    return {"meta": meta_signal, "hybrid": hybrid_sig, "confidence": meta_conf, "volatility": max(vol_factor, 0.01),
            "price": close}


# ---------------------------------------------------------
# SIGNAL FILTERING & ALLOCATION
# ---------------------------------------------------------
def filter_signals(signals, regime, pf):
    filtered = []
    for s in signals:
        meta, hybrid, conf, ticker = s["meta"], s["hybrid"], s.get("confidence", 0.0), s["ticker"]

        if ticker in pf["positions"]:
            avg_px = pf["positions"][ticker]["avg_price"]
            if (((s["price"] - avg_px) / avg_px) * 100) <= -15.0:
                s["signal"], s["confidence"] = "SELL", 0.99
                filtered.append(s)
                continue

        if regime == "BULL":
            if meta == "BUY" or hybrid == "BUY":
                s["signal"] = "BUY"; filtered.append(s)
            elif meta == "SELL" and hybrid == "SELL" and conf >= 0.05:
                s["signal"] = "SELL"; filtered.append(s)
        elif regime == "BEAR":
            if (meta == "SELL" or hybrid == "SELL") and conf >= 0.05:
                s["signal"] = "SELL"; filtered.append(s)
            elif meta == "BUY" and hybrid == "BUY":
                s["signal"] = "BUY"; filtered.append(s)
        else:
            if meta == hybrid and meta != "HOLD":
                if meta == "SELL" and conf < 0.05: continue
                s["signal"] = meta
                filtered.append(s)

    return filtered


def allocate(signals, pf):
    buys = [s for s in signals if s["signal"] == "BUY"]
    if not buys: return []

    cash, deployable = pf["cash"], max(0.0, pf["cash"] - max(20000.0, pf["cash"] * 0.15))
    candidates = []
    for s in buys:
        if s["confidence"] < 0.08: continue
        raw = min(0.3, max(0.1, 0.1 + s["confidence"] * 2.0))
        candidates.append((s, min(raw / max(s["volatility"], 0.01), 5.0)))

    if not candidates: return []
    candidates.sort(key=lambda x: x[1], reverse=True)

    total_w = sum(w for _, w in candidates)
    weights = [w / total_w for _, w in candidates]

    trades, planned_positions = [], len(pf["positions"])
    for (s, _), w in zip(candidates, weights):
        is_new = s["ticker"] not in pf["positions"]
        if is_new and planned_positions >= MAX_POSITIONS: continue
        alloc = w * deployable
        if alloc < MIN_TRADE_VALUE: continue

        shares = int(alloc // s["price"])
        if shares > 0 and (shares * s["price"]) >= MIN_TRADE_VALUE:
            trades.append({"ticker": s["ticker"], "shares": shares, "price": s["price"], "action": "BUY",
                           "confidence": s["confidence"]})
            if is_new: planned_positions += 1
    return trades


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ChronoStox Live Execution Engine (Standalone)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to test")
    args = parser.parse_args()

    pf = load_portfolio()
    today, cur_month = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m")

    if not args.dry_run and pf["last_deposit_month"] != cur_month:
        pf["cash"] += 100000.0
        pf["last_deposit_month"] = cur_month
        save_portfolio(pf)
        logging.info(f"Monthly deposit of ₹1,00,000.00 triggered for {cur_month}.")

    print(f"\n{'=' * 50}\n ⚙️ ChronoStox Live Engine (Standalone)\n" + (
        " 🛡️ DRY RUN MODE ENABLED\n" if args.dry_run else "") + f"{'=' * 50}")
    print(f"Cash Available : ₹{pf['cash']:,.2f}\nOpen Positions : {len(pf['positions'])}")
    for tkr, pos in pf["positions"].items(): print(f"  - {tkr}: {pos['shares']} shares @ ₹{pos['avg_price']:.2f}")
    print("-" * 50)

    # 1. Load Models (using loaders from backtest_oos strictly to read disk)
    sys.path.append(os.path.join(PROJECT_ROOT, "test"))
    import backtest_oos as b_oos
    print("\n[+] Initializing Models & Data...")
    df_macro, df_senti, df_sector = b_oos.load_macro_once(MODEL_DIR), b_oos.load_sentiment_once(
        MODEL_DIR), b_oos.load_sectors_once(MODEL_DIR)
    models = b_oos.load_models_once(MODEL_DIR)

    if args.tickers:
        target_tickers = [t.upper() if t.endswith('.NS') else t.upper() + '.NS' for t in args.tickers]
    else:
        target_tickers = list(dict.fromkeys(list(pf["positions"].keys()) + b_oos.get_dynamic_tickers()))[:200]

    # 2. Macro Regime (Index Tuned)
    lookback_start = (datetime.now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    nifty_df = b_oos.fetch_price_history("^NSEI", lookback_start)
    regime = detect_regime(nifty_df) if nifty_df is not None else "NEUTRAL"
    color = "🟢" if regime == "BULL" else "🔴" if regime == "BEAR" else "⚪"
    print(f"[*] Current Macro Regime: {color} {regime}\n[*] Scanning {len(target_tickers)} stocks...")

    # 3. Predict Sequential
    raw_signals = []
    H_DESIRED = 21
    horizon = H_DESIRED if H_DESIRED in models.get("horizons", []) else (models.get("horizons", [None])[0] or H_DESIRED)

    for idx, tkr in enumerate(target_tickers):
        sys.stdout.write(f"\r  -> Processing {idx + 1}/{len(target_tickers)}: {tkr:<15}")
        sys.stdout.flush()

        df_px = b_oos.fetch_price_history(tkr, lookback_start)
        if df_px is None or len(df_px) < 100: continue

        df_merged = engineer_features_standalone(df_px, df_macro, df_senti, df_sector, models["features"])
        if df_merged is None: continue

        try:
            res = predict_live(df_merged, models, horizon)
            res["ticker"] = tkr
            raw_signals.append(res)
        except Exception as e:
            logging.exception(f"Prediction failed for {tkr}: {e}")

    print("\n[+] Inference Complete.")

    # 4. Filter & Execute
    signals = filter_signals(raw_signals, regime, pf)
    print(f"\n[DEBUG] Raw Signals Generated : {len(raw_signals)}\n[DEBUG] Passed Regime Filter  : {len(signals)}")

    planned, sell_count, skipped_same_day = [], 0, 0
    for s in signals:
        if s["signal"] == "SELL" and s["ticker"] in pf["positions"]:
            pos = pf["positions"][s["ticker"]]
            if pos.get("buy_date") == today: skipped_same_day += 1; continue
            planned.append({"ticker": s["ticker"], "shares": pos["shares"], "price": s["price"], "action": "SELL",
                            "confidence": s["confidence"]})
            sell_count += 1

    print(f"[DEBUG] Planned Sells         : {sell_count} (Skipped Same-Day: {skipped_same_day})")

    buys = allocate(signals, pf)
    print(f"[DEBUG] Planned Buys          : {len(buys)}")
    planned.extend(buys)

    if not planned:
        print("\n⚠️ No actionable trades met the execution threshold today. Holding positions.")
        logging.info(f"Routine scan complete. Regime: {regime}. No actionable trades.")
        return

    print("\n" + "=" * 60 + "\n  PLANNED EXECUTIONS MANIFEST\n" + "=" * 60)
    for pt in planned:
        val = pt['shares'] * pt['price']
        color_prefix = "🟢" if pt['action'] == "BUY" else "🔴"
        pnl_str = ""
        if pt['action'] == "SELL":
            avg_px = pf["positions"][pt['ticker']]["avg_price"]
            expected_pnl = ((pt['price'] - avg_px) / avg_px) * 100
            pnl_str = f" | ⚠️ CIRCUIT BREAKER: {expected_pnl:+.1f}%" if expected_pnl <= -15.0 else f" | Est. PnL: {expected_pnl:+.1f}%"
        print(
            f"{color_prefix} [{pt['action']}] {pt['shares']:>3} x {pt['ticker']:<15} @ ₹{pt['price']:<8.2f} = ₹{val:<9.2f} (Conf: {pt.get('confidence', 0.0):.2f}){pnl_str}")
    print("=" * 60)

    if args.dry_run: return print("\n🛡️  DRY RUN COMPLETED - Exiting without execution.")

    if input("\nExecute? (y/n): ").lower() == "y":
        exec_count = 0
        logging.info(f"--- Trade Execution Initiated: {today} | Regime: {regime} ---")
        for t in planned:
            if execute_trade(pf, t["ticker"], t["action"], t["shares"], t["price"], today):
                exec_count += 1
                logging.info(
                    f"EXECUTED: {t['action']} {t['shares']} shares of {t['ticker']} @ {t['price']} (Conf: {t['confidence']:.2f})")
        print(f"\n✅ {exec_count} trades executed and logged to portfolio.json and engine.log")
        logging.info(f"--- Execution Complete. Total trades: {exec_count} ---")
    else:
        print("\nAborted. No trades executed.")


if __name__ == "__main__":
    main()