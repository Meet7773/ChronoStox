# ===============================================================
# ChronoStox v10.1 Module - Adaptive HMM Regime Detector
# Fixes Windows multiprocessing crash & makes logic adaptive.
# ===============================================================

import os

# FORCE SINGLE THREADING TO FIX WINDOWS CRASH
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_VERBOSITY"] = "0"

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import argparse
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")


def fetch_data(ticker, period="2y"):
    print(f"[+] Fetching {period} data for {ticker}...", file=sys.stderr)
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty: raise ValueError("No data found.")
    except Exception as e:
        print(f"FATAL: Yahoo Finance error: {e}")
        sys.exit(1)

    df = df.reset_index()

    # Calculate features & scale them for HMM convergence
    # Log Return (Direction)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
    # Normalized Range (Volatility)
    df["range_vol"] = ((df["High"] - df["Low"]) / df["Close"]) * 100

    df = df.dropna()
    return df


def fit_hmm(df, n_components=3):
    X_train = df[["log_ret", "range_vol"]].values

    print(f"[+] Training HMM with {n_components} hidden states...", file=sys.stderr)

    # n_iter=10000 for convergence
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=10000, random_state=420)
    model.fit(X_train)

    df["state"] = model.predict(X_train)
    return model, df


def interpret_states(model, df):
    """
    New ADAPTIVE interpretation logic.
    Compares state metrics to the *stock's global average*.
    """
    # Global Averages for this specific stock
    global_avg_ret = df["log_ret"].mean()
    global_avg_vol = df["range_vol"].mean()

    state_stats = []
    for i in range(model.n_components):
        mask = (df["state"] == i)
        avg_ret = df.loc[mask, "log_ret"].mean()
        avg_vol = df.loc[mask, "range_vol"].mean()
        count = mask.sum()

        state_stats.append({
            "state_id": i,
            "avg_daily_return": avg_ret,
            "avg_volatility": avg_vol,
            "days_active": count
        })

    print("\n[+] Regime Analysis (Adaptive):")
    print(f"{'ID':<3} | {'AVG RET':<9} | {'AVG VOL':<9} | {'INTERPRETATION'}")
    print("-" * 55)

    regime_map = {}

    for s in state_stats:
        # Logic: Compare to GLOBAL average
        ret = s["avg_daily_return"]
        vol = s["avg_volatility"]

        label = "Neutral"

        # If returns are better than average AND vol is lower than average
        if ret > global_avg_ret and vol < global_avg_vol:
            label = "BULL (Stable)"

        # If returns are significantly negative
        elif ret < -0.1:
            label = "BEAR (Crash)"

        # If returns are high but vol is ALSO high (The IDEA.NS case)
        elif ret > 0 and vol > (global_avg_vol * 1.2):
            label = "BULL (Volatile)"

        # If returns are low/negative and vol is high
        elif ret < 0 and vol > global_avg_vol:
            label = "BEAR (Panic)"

        regime_map[s["state_id"]] = label

        r_str = f"{ret:+.2f}%"
        v_str = f"{vol:.2f}%"
        print(f"{s['state_id']:<3} | {r_str:<9} | {v_str:<9} | {label}")

    return regime_map


def main():
    parser = argparse.ArgumentParser(description="HMM Market Regime Detector v10.1")
    parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., RELIANCE.NS)")
    args = parser.parse_args()

    try:
        df = fetch_data(args.ticker)
        model, df_states = fit_hmm(df, n_components=3)
        regime_map = interpret_states(model, df_states)

        current_state = df_states["state"].iloc[-1]
        current_label = regime_map[current_state]

        print(f"\n[RESULT] Current Regime for {args.ticker}: ** {current_label} **")

        # Determine Signal Implication
        if "BEAR" in current_label:
            print("-> ACTION: Force HOLD / Sell Rallies")
        elif "BULL (Stable)" in current_label:
            print("-> ACTION: Allow BUY Signals")
        else:
            print("-> ACTION: Caution / Reduce Size")

    except Exception as e:
        print(f"Error: {e}")
        # import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()