import os
import sys
import pandas as pd
from datetime import datetime
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_VERBOSITY"] = "0"
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

# Import core engine functions from the backtester to reuse data pipelines
from backtest_oos import (
    DEFAULT_TICKERS, load_models_once, load_macro_once,
    load_sentiment_once, load_sectors_once, fetch_price_history,
    engineer_features_for_slice, predict_at_date
)

def main():
    print("=" * 70)
    print("  ChronoStox — Long-Term Predictor (6-Month & 12-Month Targets)")
    print("=" * 70)
    
    # Load all models and data (from current directory '.')
    print("\n[1/3] Loading models and datasets...")
    models = load_models_once(".")
    df_macro = load_macro_once(".")
    df_senti = load_sentiment_once(".")
    df_sector = load_sectors_once(".")
    
    print("\n[2/3] Fetching latest valid tickers from NSE Indices...")
    try:
        # Tries to get the Nifty Total Market (top 750 liquid stocks)
        url = "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
        df_tickers = pd.read_csv(url)
        
        # The column in NSE CSV is usually 'Symbol'
        if 'Symbol' in df_tickers.columns:
            nse_tickers = [str(sym).strip() + ".NS" for sym in df_tickers['Symbol'].dropna()]
            target_tickers = nse_tickers[:750]  # Start with the whole Nifty 750
            print(f"  \u2705 Dynamically loaded {len(target_tickers)} live tickers from NSE Total Market Index.")
        else:
            raise ValueError("CSV format unexpected. Missing 'Symbol' column.")
            
    except Exception as e:
        print(f"  \u26a0\ufe0f Could not fetch live NSE list ({e}). Falling back to defaults.")
        target_tickers = DEFAULT_TICKERS
    
    results = []
    
    print(f"\n[3/3] Fetching latest market data and predicting for {len(target_tickers)} tickers...")
    for idx, tkr in enumerate(target_tickers, 1):
        print(f"  [{idx}/{len(target_tickers)}] {tkr}...", end="")
        sys.stdout.flush()
        
        # Fetch up to today
        df_price = fetch_price_history(tkr, start_date="2024-01-01")
        if df_price is None or len(df_price) < 60:
            print(" \u274c No data")
            continue
            
        # Engineer features using the latest slice
        df_final, df_full = engineer_features_for_slice(df_price, df_macro, df_senti, df_sector, models["features"])
        if df_final is None or df_full is None:
            print(" \u274c Feature error")
            continue
            
        # Predict
        try:
            pred = predict_at_date(df_full, models)
            
            # Predict returns for 6m (126d) and 12m (252d)
            ret_126 = pred["predicted_returns"].get(126, 0.0)
            ret_252 = pred["predicted_returns"].get(252, 0.0)
            
            current_price = pred["close"]
            
            # Compute targets based on current price and predicted returns
            tgt_126 = current_price * (1.0 + ret_126)
            tgt_252 = current_price * (1.0 + ret_252)
            
            results.append({
                "Ticker": tkr.replace(".NS", ""),
                "Current Price": round(current_price, 2),
                "6M Target": round(tgt_126, 2),
                "6M Ret": ret_126,
                "12M Target": round(tgt_252, 2),
                "12M Ret": ret_252
            })
            print(" \u2705")
        except Exception as e:
            print(f" \u274c Error: {e}")
            
    # Sort by 1-year expected return, descending
    results.sort(key=lambda x: x["12M Ret"], reverse=True)
    
    print("\n[3/3] Generating Report...")
    print("\n=" * 80)
    print(f"  CHRONOSTOX LONG-TERM TARGETS (As of {datetime.now().strftime('%Y-%m-%d')})")
    print("=" * 80)
    print(f"  {'Ticker':<12} | {'Current Price':>13} | {'6M Target':>10} | {'6M Ret':>8} | {'12M Target':>10} | {'12M Ret':>8}")
    print("-" * 80)
    
    for r in results:
        ret_6m_str = f"{r['6M Ret'] * 100:+.1f}%"
        ret_12m_str = f"{r['12M Ret'] * 100:+.1f}%"
        print(f"  {r['Ticker']:<12} | \u20b9{r['Current Price']:>12.2f} | \u20b9{r['6M Target']:>9.2f} | {ret_6m_str:>8} | \u20b9{r['12M Target']:>9.2f} | {ret_12m_str:>8}")

    print("=" * 80)
    
    # Save to CSV in root\predictions folder
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pred_dir = os.path.join(root_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"long_term_targets_{timestamp}.csv"
        filepath = os.path.join(pred_dir, filename)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(filepath, index=False)
        print(f"\n\U0001f4be Saved predictions to: {filepath}")
    except Exception as e:
        print(f"\n\u274c Failed to save CSV: {e}")

    print("\n✅ Done. You can compare these 12-Month targets against financial analyst consensus.\n")

if __name__ == "__main__":
    main()
