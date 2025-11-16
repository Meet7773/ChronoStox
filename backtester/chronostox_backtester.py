# ============================================================
# ChronoStox v8 — FULL BACKTESTER (Part 1/3)
# Multi-Asset | LGBM + LSTM Hybrid | TA + Macro + Sentiment
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

pd.set_option("mode.chained_assignment", None)


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config(path="config_backtester.json"):
    if not os.path.exists(path):
        raise FileNotFoundError("config_backtester.json not found!")

    with open(path, "r") as f:
        cfg = json.load(f)

    return cfg


# ============================================================
# LOAD MODELS (Joblib + Keras)
# ============================================================

def load_models(cfg):
    joblib_path = cfg["MODEL_JOBLIB"]
    keras_path  = cfg["MODEL_KERAS"]

    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f"Missing: {joblib_path}")

    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"Missing: {keras_path}")

    bundle = joblib.load(joblib_path)
    lstm  = load_model(keras_path, compile=False)

    return {
        "scaler": bundle["scaler"],
        "lgbm": bundle["model_lgbm"],
        "features": bundle["features"],
        "seq_len": bundle["lstm_sequence_length"],
        "lstm": lstm
    }


# ============================================================
# LOAD INPUT FILES
# ============================================================

def load_price_data(cfg):
    path = cfg["PRICE_FILE"]
    ftype = cfg.get("PRICE_FILE_TYPE", "csv").lower()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Price file not found: {path}")

    # ---- Load file ----
    if ftype == "csv":
        df = pd.read_csv(path)
    elif ftype == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("PRICE_FILE_TYPE must be 'csv' or 'parquet'.")

    # ---- If Date is in INDEX, convert to column ----
    if "Date" not in df.columns:
        # Try using index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Date"})
        else:
            raise ValueError("Price data missing 'Date' column AND index is not DatetimeIndex.")

    # ---- Normalize ----
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)

    if "Ticker" not in df.columns:
        raise ValueError("Price parquet must contain a 'Ticker' column.")

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    df = df.dropna(subset=["Date"])

    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)




def load_parquet(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_parquet(path)
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Date"])

    return df.sort_values("Date").reset_index(drop=True)


def load_sector_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Try to find reasonable tickers and sector columns
    tcol = None
    scol = None

    for c in df.columns:
        if c.lower() in ["ticker", "tickeryf", "symbol"]:
            tcol = c
        if "sector" in c.lower():
            scol = c

    if tcol is None:
        tcol = df.columns[0]
    if scol is None:
        scol = df.columns[-1]

    df = df.rename(columns={tcol: "Ticker_YF", scol: "Sector"})
    df["Ticker_YF"] = (
        df["Ticker_YF"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df[["Ticker_YF", "Sector"]].drop_duplicates()


# ============================================================
# MERGE FEATURES (TA + MACRO + SENTIMENT + SECTOR)
# ============================================================

# ---- TA INDICATORS (using pandas_ta) ----
import pandas_ta as ta

def compute_ta(df):
    df = df.copy()

    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)

    # Bollinger Bands
    bb = df.ta.bbands(length=5, append=False)
    if bb is not None and not bb.empty:
        df["BBB_5_2.0"] = bb.iloc[:, 3]
        df["BBP_5_2.0"] = bb.iloc[:, 4]

    # MACD
    mac = df.ta.macd(append=False)
    if mac is not None and not mac.empty:
        df["MACDh_12_26_9"] = mac.iloc[:, 1]
        df["MACDs_12_26_9"] = mac.iloc[:, 2]

    df.ta.rsi(append=True)

    # close/EMA200 ratio
    ema200_cols = [c for c in df.columns if "EMA_200" in c]
    if ema200_cols:
        em = ema200_cols[0]
        df["close_to_ema200"] = df["Close"] / (df[em] + 1e-8)
    else:
        df["close_to_ema200"] = 0.0

    return df


# ---- MACRO ASOF MERGE ----
def merge_macro(df, df_macro):
    return pd.merge_asof(
        df.sort_values("Date"),
        df_macro.sort_values("Date"),
        on="Date",
        direction="backward"
    )


# ---- SENTIMENT ASOF MERGE ----
def merge_sentiment(df, df_senti):
    tkr = df["Ticker"].iloc[0]
    ssub = df_senti[df_senti["Ticker_YF"] == tkr]

    if ssub.empty:
        # Neutral sentiment fallback
        ssub = pd.DataFrame({"Date": df["Date"], "sentiment_score": 0.0})

    return pd.merge_asof(
        df.sort_values("Date"),
        ssub.sort_values("Date")[["Date", "sentiment_score"]],
        on="Date",
        direction="backward"
    )


# ---- SECTOR / INTERACTIONS ----
SECTOR_CLASSES = [
    "Communication Services", "Consumer Cyclical", "Consumer Defensive",
    "Energy", "Financial Services", "Healthcare", "Industrials",
    "Real Estate", "Technology", "Utilities", "nan"
]

def apply_sector_features(df, df_sector):
    tkr = df["Ticker"].iloc[0]
    sec = df_sector[df_sector["Ticker_YF"] == tkr]

    sector = "nan"
    if not sec.empty:
        sector = str(sec["Sector"].iloc[0]).strip()

    df["Sector"] = sector

    # one-hot encode
    for s in SECTOR_CLASSES:
        df[f"Sector_{s}"] = 1.0 if s == sector else 0.0

    # interactions
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0

    for s in SECTOR_CLASSES:
        cname = f"Sector_{s}"
        df[f"sentiment_x_{cname}"] = df[cname] * df["sentiment_score"]

    return df


# ============================================================
# FINAL FEATURE PIPELINE PER TICKER
# ============================================================

def build_features_for_ticker(df_px, df_macro, df_senti, df_sector, expected_cols):
    # 1) TA
    df = compute_ta(df_px)

    # 2) macro
    df = merge_macro(df, df_macro)

    # 3) sentiment
    df = merge_sentiment(df, df_senti)

    # 4) sector + interactions
    df = apply_sector_features(df, df_sector)

    # 5) cleanup
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 6) ensure expected columns
    missing = set(expected_cols) - set(df.columns)
    for m in missing:
        df[m] = 0.0

    df = df[expected_cols]

    return df

# ============================================================
# HYBRID PREDICTION (LGBM + LSTM + ATR)
# ============================================================

HORIZONS = [5, 21, 63, 126, 252]
ATR_MULT = {
    5:   1.0,
    21:  1.8,
    63:  3.0,
    126: 4.8,
    252: 7.2
}

def compute_price_targets(df_raw, df_feat_full, model_lgb, model_lstm, scaler, seq_len, feature_order):
    """
    df_raw: full df with Close + ATR etc.
    df_feat_full: full merged features with same index as df_raw
    """
    close = float(df_raw["Close"].iloc[-1])

    # ML raw preds ------------------------------------------------------
    try:
        X = df_feat_full[feature_order].iloc[-1].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        raw_lgb = model_lgb.predict(X_scaled)[0]
    except Exception as e:
        print("LGBM failed:", e)
        raw_lgb = np.zeros(len(HORIZONS))

    try:
        if len(df_feat_full) >= seq_len:
            seq = df_feat_full[feature_order].tail(seq_len).values
            seq = seq.reshape(1, seq_len, -1)
            raw_lstm = model_lstm.predict(seq, verbose=0)[0]
        else:
            raw_lstm = np.zeros(len(HORIZONS))
    except Exception as e:
        print("LSTM failed:", e)
        raw_lstm = np.zeros(len(HORIZONS))

    raw_pred = (raw_lgb + raw_lstm) / 2.0

    # ATR expansion -----------------------------------------------------
    try:
        atr = float(df_raw["ATR_14"].iloc[-1])
    except KeyError:
        atr = close * 0.01

    atr_targets = close + atr * np.array([ATR_MULT[h] for h in HORIZONS])

    # Hybrid final ------------------------------------------------------
    ml_pp = close * (1 + raw_pred)
    hybrid = 0.70 * ml_pp + 0.30 * atr_targets

    return hybrid, raw_pred


def classify_signal(pred_ret):
    """Convert ML return to BUY/SELL/HOLD"""
    if pred_ret < -0.015:
        return "SELL"
    elif pred_ret > 0.020:
        return "BUY"
    else:
        return "HOLD"


# ============================================================
# PORTFOLIO ENGINE
# ============================================================

class Portfolio:
    def __init__(self, initial_cash, cfg):
        self.cash = initial_cash
        self.cfg = cfg
        self.positions = {}  # ticker → {qty, entry_price, entry_date}
        self.equity_history = []
        self.trade_log = []

    def value(self, prices_today):
        """Compute total equity."""
        val = self.cash
        for tkr, pos in self.positions.items():
            if tkr in prices_today:
                val += pos["qty"] * prices_today[tkr]
        return val

    def record_equity(self, date, prices_today):
        self.equity_history.append({
            "Date": date,
            "Equity": self.value(prices_today)
        })

    def can_enter_new(self):
        # Do not exceed max positions
        if len(self.positions) >= self.cfg["MAX_POSITIONS"]:
            return False
        # Ensure enough cash
        if self.cash < self.cfg["MIN_CASH_RATIO"] * (self.cash + sum(
                p["qty"] * p["entry_price"] for p in self.positions.values())):
            return False
        return True

    def enter(self, ticker, price, date, side):
        """
        side ∈ {"BUY", "SELL"}.
        For SELL → short sell: qty negative.
        """
        pos_size = self.cfg["POSITION_SIZE"]
        capital = self.cash * pos_size

        if capital <= 0:
            return

        qty = capital / price
        qty = round(qty, 4)

        if side == "SELL":
            qty = -qty  # short

        cost = qty * price
        fee = abs(cost) * self.cfg["COMMISSION"]

        self.cash -= cost
        self.cash -= fee

        self.positions[ticker] = {
            "qty": qty,
            "entry_price": price,
            "entry_date": date
        }

        self.trade_log.append({
            "date": date,
            "ticker": ticker,
            "action": "ENTER_" + side,
            "entry_price": price,
            "exit_price": None,
            "qty": qty,
            "pnl_abs": 0.0,
            "pnl_pct": 0.0,
            "hold_days": 0
        })

    def exit(self, ticker, price, date, reason):
        """Close position."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        qty = pos["qty"]

        cost = qty * price * -1   # reverse trade
        fee = abs(qty * price) * self.cfg["COMMISSION"]

        self.cash += cost
        self.cash -= fee

        entry_price = pos["entry_price"]
        pnl_abs = qty * (price - entry_price) * -1
        pnl_pct = pnl_abs / (abs(entry_price * qty) + 1e-8)

        hold_days = (date - pos["entry_date"]).days

        self.trade_log.append({
            "date": date,
            "ticker": ticker,
            "action": "EXIT_" + reason,
            "entry_price": entry_price,
            "exit_price": price,
            "qty": qty,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "hold_days": hold_days
        })

        del self.positions[ticker]

    def evaluate_stops(self, ticker, price, date):
        """Check SL/TP for one ticker."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        qty = pos["qty"]
        entry = pos["entry_price"]

        ret = (price - entry) / entry

        if qty < 0:  # short
            ret = -ret

        if ret <= self.cfg["STOPLOSS"]:
            self.exit(ticker, price, date, "STOPLOSS")
            return

        if ret >= self.cfg["TAKEPROFIT"]:
            self.exit(ticker, price, date, "TAKEPROFIT")
            return


# ============================================================
# BACKTEST PER-DAY SIGNAL HANDLING
# ============================================================

def process_daily_signals(date, todays_rows, feat_groups, models, portfolio, cfg):
    """todays_rows: dict[ticker] = close_price"""

    prices_today = {t: todays_rows[t]["Close"] for t in todays_rows}

    # (1) First update equity history
    portfolio.record_equity(date, prices_today)

    # (2) Evaluate stops
    for tkr in list(portfolio.positions.keys()):
        if tkr in prices_today:
            portfolio.evaluate_stops(tkr, prices_today[tkr], date)

    # (3) Generate signals for each ticker
    for tkr, row in todays_rows.items():

        # Build a rolling feature window for THIS ticker
        df_feat = feat_groups[tkr]["feat"]
        df_raw = feat_groups[tkr]["raw"]

        idx = row["__idx__"]
        if idx < models["seq_len"] + 20:
            # not enough history
            continue

        df_raw_slice  = df_raw.iloc[:idx+1].copy()
        df_feat_slice = df_feat.iloc[:idx+1].copy()

        # Hybrid prediction
        hybrid, raw = compute_price_targets(
            df_raw_slice,
            df_feat_slice,
            models["lgbm"],
            models["lstm"],
            models["scaler"],
            models["seq_len"],
            models["features"]
        )

        pred_ret = raw[1]  # 21-day horizon (medium-term)

        signal = classify_signal(pred_ret)

        # (4) Manage positions accordingly
        if signal == "BUY":
            if tkr not in portfolio.positions and portfolio.can_enter_new():
                portfolio.enter(tkr, row["Close"], date, "BUY")

        elif signal == "SELL":
            if tkr in portfolio.positions and portfolio.positions[tkr]["qty"] > 0:
                # long → exit
                portfolio.exit(tkr, row["Close"], date, "SIGNAL_SELL")
            else:
                # open short if allowed
                if tkr not in portfolio.positions and portfolio.can_enter_new():
                    portfolio.enter(tkr, row["Close"], date, "SELL")

        # HOLD → do nothing

    return

# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(cfg_path="config_backtester.json", initial_cash=1_000_000):
    # --------------------------------------------------------
    # LOAD CONFIG + MODELS + DATA
    # --------------------------------------------------------
    cfg = load_config(cfg_path)
    models = load_models(cfg)

    print("Loading price CSV...")
    df_px = load_price_data(cfg)

    print("Loading macro...")
    df_macro = load_parquet(cfg["MACRO_FILE"])

    print("Loading sentiment...")
    df_senti = load_parquet(cfg["SENTIMENT_FILE"])
    if "Ticker" in df_senti.columns:
        df_senti = df_senti.rename(columns={"Ticker": "Ticker_YF"})
    df_senti["Ticker_YF"] = (
        df_senti["Ticker_YF"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    print("Loading sectors...")
    df_sector = load_sector_csv(cfg["TICKER_FILE"])


    # --------------------------------------------------------
    # AUTO-DETECT DATE RANGE
    # --------------------------------------------------------
    if cfg["AUTO_DETECT_DATE_RANGE"]:
        min_dates = [
            df_px["Date"].min(),
            df_macro["Date"].min(),
            df_senti["Date"].min()
        ]
        max_dates = [
            df_px["Date"].max(),
            df_macro["Date"].max(),
            df_senti["Date"].max()
        ]
        start = max(min_dates)
        end   = min(max_dates)
    else:
        start = pd.to_datetime(cfg["START_DATE"])
        end   = pd.to_datetime(cfg["END_DATE"])

    print(f"Backtest window: {start.date()} → {end.date()}")


    # --------------------------------------------------------
    # PREPARE TICKER GROUPS
    # --------------------------------------------------------
    tickers = df_px["Ticker"].unique().tolist()
    print(f"Tickers found: {len(tickers)}")

    feat_groups = {}  # ticker → {"raw", "feat"}

    for tkr in tickers:
        px = df_px[df_px["Ticker"] == tkr].copy()
        px = px.sort_values("Date").reset_index(drop=True)

        # raw copy for ATR calculations
        df_raw = compute_ta(px.copy())

        # build full feature matrix
        df_feat = build_features_for_ticker(
            df_raw.copy(),
            df_macro,
            df_senti,
            df_sector,
            models["features"]
        )

        df_raw["__idx__"] = range(len(df_raw))
        df_feat["__idx__"] = range(len(df_feat))

        feat_groups[tkr] = {
            "raw": df_raw,
            "feat": df_feat
        }


    # --------------------------------------------------------
    # MAIN DAILY LOOP
    # --------------------------------------------------------
    dates = pd.date_range(start, end, freq="D")
    portfolio = Portfolio(initial_cash, cfg)

    for d in dates:
        todays_rows = {}

        # collect rows for each ticker ON this date
        for tkr in tickers:
            df_raw = feat_groups[tkr]["raw"]
            sel = df_raw[df_raw["Date"] == d]
            if not sel.empty:
                todays_rows[tkr] = sel.iloc[0]

        if len(todays_rows) == 0:
            continue

        process_daily_signals(
            d,
            todays_rows,
            feat_groups,
            models,
            portfolio,
            cfg
        )

    # --------------------------------------------------------
    # FINAL EQUITY RECORD
    # --------------------------------------------------------
    if len(dates) > 0:
        last = dates[-1]
        last_prices = {}
        for tkr in tickers:
            df_raw = feat_groups[tkr]["raw"]
            sel = df_raw[df_raw["Date"] == last]
            if not sel.empty:
                last_prices[tkr] = sel.iloc[0]["Close"]
        portfolio.record_equity(last, last_prices)


    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    eq = pd.DataFrame(portfolio.equity_history)
    eq.to_csv("equity_curve.csv", index=False)

    tr = pd.DataFrame(portfolio.trade_log)
    tr.to_csv("trades.csv", index=False)

    print("Backtest complete.")
    print(f"Final Equity: {eq['Equity'].iloc[-1]:,.2f}")
    print("Saved: equity_curve.csv, trades.csv")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_backtest()
