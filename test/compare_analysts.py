import os
import sys
import glob
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

def get_latest_predictions_file():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_dir = os.path.join(root_dir, "predictions")
    
    if not os.path.exists(pred_dir):
        print(f"\u274c No predictions directory found at {pred_dir}")
        return None
        
    # Get all prediction CSVs and find the latest
    csv_files = glob.glob(os.path.join(pred_dir, "long_term_targets_*.csv"))
    if not csv_files:
        print("\u274c No prediction CSV files found. Please run predict_long_term.py first.")
        return None
        
    latest_file = max(csv_files, key=os.path.getctime)
    return latest_file

def fetch_analyst_data(ticker_symbol):
    try:
        # Suppress yfinance output
        tkr = yf.Ticker(ticker_symbol)
        info = tkr.info
        
        return {
            "Analyst Mean": info.get("targetMeanPrice", None),
            "Analyst High": info.get("targetHighPrice", None),
            "Analyst Low": info.get("targetLowPrice", None),
            "Recommendation": info.get("recommendationKey", "N/A").upper().replace("_", " "),
            "Rec Score": info.get("recommendationMean", None) # 1 (Strong Buy) to 5 (Sell)
        }
    except Exception:
        return {
            "Analyst Mean": None, "Analyst High": None, 
            "Analyst Low": None, "Recommendation": "ERROR", "Rec Score": None
        }

def main():
    print("=" * 90)
    print("  ChronoStox — AI vs. Wall St Analyst Consensus Comparison")
    print("=" * 90)
    
    latest_file = get_latest_predictions_file()
    if not latest_file:
        sys.exit(1)
        
    print(f"\n\U0001f4c2 Loading latest ML predictions: {os.path.basename(latest_file)}")
    try:
        df_ml = pd.read_csv(latest_file)
    except Exception as e:
        print(f"\u274c Error reading CSV: {e}")
        sys.exit(1)
        
    if df_ml.empty:
        print("\u274c Predictions file is empty.")
        sys.exit(1)
        
    print(f"\n\U0001f50d Fetching Analyst Consensus for {len(df_ml)} tickers from Yahoo Finance...")
    
    comparison_results = []
    
    for idx, row in df_ml.iterrows():
        tkr_clean = str(row['Ticker']).strip()
        tkr_yf = f"{tkr_clean}.NS"
        
        print(f"  [{idx+1}/{len(df_ml)}] {tkr_clean}...", end="")
        sys.stdout.flush()
        
        analyst_data = fetch_analyst_data(tkr_yf)
        
        ml_target = float(row.get('12M Target', 0))
        current_price = float(row.get('Current Price', 0))
        mean_analyst_tgt = analyst_data['Analyst Mean']
        
        # Calculate AI vs Analyst divergence
        divergence = 0.0
        divergence_str = "N/A"
        if mean_analyst_tgt is not None and mean_analyst_tgt > 0:
            divergence = ((ml_target - mean_analyst_tgt) / mean_analyst_tgt) * 100
            divergence_str = f"{divergence:+.1f}%"
            print(" \u2705")
        else:
            print(" \u26a0\ufe0f No analyst data")
            
        comparison_results.append({
            "Ticker": tkr_clean,
            "Current Price": current_price,
            "ML 12M Target": ml_target,
            "Analyst Mean": mean_analyst_tgt if mean_analyst_tgt else 0.0,
            "Analyst High": analyst_data['Analyst High'] if analyst_data['Analyst High'] else 0.0,
            "AI vs Analyst \u0394": divergence_str,
            "AI Divergence %": divergence,  # Hidden float for sorting
            "Recommendation": analyst_data['Recommendation'],
            "Rec Score": analyst_data['Rec Score'] if analyst_data['Rec Score'] else 9.9
        })

    # Convert to DataFrame
    df_cmp = pd.DataFrame(comparison_results)
    
    # Filter out ones with no analyst data for the clean report view
    df_report = df_cmp[df_cmp["Analyst Mean"] > 0].copy()
    
    # Sort by absolute divergence (biggest disagreements first)
    df_report['Abs_Divergence'] = df_report['AI Divergence %'].abs()
    df_report = df_report.sort_values(by="Abs_Divergence", ascending=False)
    
    print("\n\n" + "=" * 105)
    print(f"  AI vs. ANALYST CONSENSUS REPORT (Sorted by Biggest Disagreements)")
    print("=" * 105)
    print(f"  {'Ticker':<12} | {'Current Pr':>10} | {'ML Target':>10} | {'Street Mean':>11} | {'AI vs Street /u0394':>16} | {'Rating':>15}")
    print("-" * 105)
    
    for _, r in df_report.iterrows():
        # Highlighting logic for terminal
        divergence_val = r['AI Divergence %']
        ticker_str = r['Ticker']
        
        # Determine if AI is Bullish or Bearish relative to the street
        stance = ""
        if divergence_val > 10:
            stance = "(AI Very Bullish)"
        elif divergence_val < -10:
            stance = "(AI Very Bearish)"
            
        rec_str = str(r['Recommendation'])[:15]

        print(
            f"  {ticker_str:<12} | "
            f"₹{r['Current Price']:>9.2f} | "
            f"₹{r['ML 12M Target']:>9.2f} | "
            f"₹{r['Analyst Mean']:>10.2f} | "
            f"{r['AI vs Analyst Δ']:>16} | "
            f"{rec_str:>15} {stance}"
        )
    print("=" * 105)
    
    # Save the comparison to CSV
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_dir = os.path.join(root_dir, "predictions")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmp_filename = f"analyst_comparison_{timestamp}.csv"
    cmp_filepath = os.path.join(pred_dir, cmp_filename)
    
    # Drop the hidden absolute sort column before saving
    if 'Abs_Divergence' in df_cmp.columns:
        df_cmp = df_cmp.drop(columns=['Abs_Divergence'])
        
    df_cmp.to_csv(cmp_filepath, index=False)
    print(f"\n\U0001f4be Saved detailed comparison to: {cmp_filepath}\n")

if __name__ == "__main__":
    main()
