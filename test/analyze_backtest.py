import pandas as pd
import numpy as np
import sys, io

# Force UTF-8 so unicode chars don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CSV_PATH = r"../predictions/backtest_results_03032026.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    df["sim_date"] = pd.to_datetime(df["sim_date"])
    
    print("=" * 80)
    print("  CHRONOSTOX BACKTEST DEEP ANALYSIS")
    print("=" * 80)
    print(f"  Total predictions: {len(df)}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Date range: {df['sim_date'].min().date()} → {df['sim_date'].max().date()}")
    
    # =============================================
    # 1. DYNAMIC WEIGHT ANALYSIS
    # =============================================
    print("\n" + "=" * 80)
    print("  1. ADAPTIVE WEIGHT BEHAVIOR")
    print("=" * 80)
    
    if "weight_lgbm" in df.columns and "weight_lstm" in df.columns:
        w_lgb = df["weight_lgbm"]
        w_lstm = df["weight_lstm"]
        
        print(f"  LightGBM weight  — Mean: {w_lgb.mean():.4f}  Min: {w_lgb.min():.4f}  Max: {w_lgb.max():.4f}  Std: {w_lgb.std():.4f}")
        print(f"  LSTM weight      — Mean: {w_lstm.mean():.4f}  Min: {w_lstm.min():.4f}  Max: {w_lstm.max():.4f}  Std: {w_lstm.std():.4f}")
        
        # Check if weights actually moved
        unique_weights = df[["weight_lgbm", "weight_lstm"]].drop_duplicates()
        print(f"  Unique weight combinations: {len(unique_weights)}")
        
        # Weight evolution over time
        wt = df.groupby("sim_date")[["weight_lgbm", "weight_lstm"]].mean()
        print("\n  Weight evolution over time:")
        print(f"  {'Date':<12} | {'LGB Weight':>10} | {'LSTM Weight':>11}")
        print("  " + "-" * 40)
        for dt, row in wt.iterrows():
            print(f"  {dt.strftime('%Y-%m-%d'):<12} | {row['weight_lgbm']:>10.4f} | {row['weight_lstm']:>11.4f}")
    else:
        print("  ⚠ No weight columns found.")
    
    # =============================================
    # 2. VERIFIABLE HORIZON ACCURACY
    # =============================================
    print("\n" + "=" * 80)
    print("  2. MODEL ACCURACY BY HORIZON (Verifiable)")
    print("=" * 80)
    
    for h in [5, 21, 63]:
        pred_col = f"pred_ret_{h}d"
        actual_col = f"actual_ret_{h}d"
        
        if pred_col not in df.columns or actual_col not in df.columns:
            continue
        
        sub = df.dropna(subset=[pred_col, actual_col])
        if len(sub) == 0:
            continue
        
        pred = sub[pred_col].values
        actual = sub[actual_col].values
        
        dir_acc = np.mean(np.sign(pred) == np.sign(actual)) * 100
        mae = np.mean(np.abs(pred - actual)) * 100
        rmse = np.sqrt(np.mean((pred - actual) ** 2)) * 100
        corr = np.corrcoef(pred, actual)[0, 1] if len(pred) > 2 else 0.0
        mean_pred = np.mean(pred) * 100
        mean_actual = np.mean(actual) * 100
        bias = mean_pred - mean_actual
        
        print(f"\n  {h}-DAY HORIZON ({len(sub)} samples):")
        print(f"    Direction Accuracy : {dir_acc:.1f}%")
        print(f"    MAE                : {mae:.2f}%")
        print(f"    RMSE               : {rmse:.2f}%")
        print(f"    Correlation        : {corr:.3f}")
        print(f"    Mean Predicted     : {mean_pred:+.2f}%")
        print(f"    Mean Actual        : {mean_actual:+.2f}%")
        print(f"    Bias (Pred-Actual) : {bias:+.2f}%  {'⚠ BULLISH BIAS' if bias > 1.0 else '✅ LOW BIAS' if abs(bias) < 1.0 else '⚠ BEARISH BIAS'}")
    
    # =============================================
    # 3. RAW LGBM vs LSTM ACCURACY (HEAD TO HEAD)
    # =============================================
    print("\n" + "=" * 80)
    print("  3. LIGHTGBM vs LSTM HEAD-TO-HEAD (Raw Predictions)")
    print("=" * 80)
    
    for h in [5, 21, 63]:
        lgb_col = f"raw_ml_ret_{h}d_lgbm"
        lstm_col = f"raw_ml_ret_{h}d_lstm"
        actual_col = f"actual_ret_{h}d"
        
        if lgb_col not in df.columns or lstm_col not in df.columns or actual_col not in df.columns:
            continue
        
        sub = df.dropna(subset=[lgb_col, lstm_col, actual_col])
        if len(sub) == 0:
            continue
        
        lgb_pred = sub[lgb_col].values
        lstm_pred = sub[lstm_col].values
        actual = sub[actual_col].values
        
        lgb_dir = np.mean(np.sign(lgb_pred) == np.sign(actual)) * 100
        lstm_dir = np.mean(np.sign(lstm_pred) == np.sign(actual)) * 100
        
        lgb_mae = np.mean(np.abs(lgb_pred - actual)) * 100
        lstm_mae = np.mean(np.abs(lstm_pred - actual)) * 100
        
        lgb_corr = np.corrcoef(lgb_pred, actual)[0, 1] if len(lgb_pred) > 2 else 0
        lstm_corr = np.corrcoef(lstm_pred, actual)[0, 1] if len(lstm_pred) > 2 else 0
        
        # Who wins more often?
        lgb_closer = np.sum(np.abs(lgb_pred - actual) < np.abs(lstm_pred - actual))
        lstm_closer = np.sum(np.abs(lstm_pred - actual) < np.abs(lgb_pred - actual))
        ties = len(sub) - lgb_closer - lstm_closer
        
        winner = "LightGBM" if lgb_closer > lstm_closer else "LSTM"
        
        print(f"\n  {h}-DAY HORIZON ({len(sub)} samples):")
        print(f"    {'Metric':<20} | {'LightGBM':>10} | {'LSTM':>10} | {'Winner':>10}")
        print(f"    {'-'*60}")
        print(f"    {'Dir. Accuracy':<20} | {lgb_dir:>9.1f}% | {lstm_dir:>9.1f}% | {'LGB' if lgb_dir > lstm_dir else 'LSTM':>10}")
        print(f"    {'MAE':<20} | {lgb_mae:>9.2f}% | {lstm_mae:>9.2f}% | {'LGB' if lgb_mae < lstm_mae else 'LSTM':>10}")
        print(f"    {'Correlation':<20} | {lgb_corr:>10.3f} | {lstm_corr:>10.3f} | {'LGB' if lgb_corr > lstm_corr else 'LSTM':>10}")
        print(f"    {'Closer to Actual':<20} | {lgb_closer:>10} | {lstm_closer:>10} | {winner:>10}")
    
    # =============================================
    # 4. HMM REGIME DISTRIBUTION
    # =============================================
    print("\n" + "=" * 80)
    print("  4. MARKET REGIME DISTRIBUTION")
    print("=" * 80)
    
    if "hmm_regime" in df.columns:
        regime_counts = df["hmm_regime"].value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"    {regime:<10}: {count:>4} ({pct:.1f}%)")
        
        # Accuracy by regime
        for h in [21]:
            actual_col = f"actual_ret_{h}d"
            pred_col = f"pred_ret_{h}d"
            if actual_col in df.columns and pred_col in df.columns:
                print(f"\n  21-DAY Direction Accuracy by Regime:")
                for regime in df["hmm_regime"].unique():
                    sub = df[(df["hmm_regime"] == regime)].dropna(subset=[pred_col, actual_col])
                    if len(sub) > 5:
                        pred = sub[pred_col].values
                        actual = sub[actual_col].values
                        acc = np.mean(np.sign(pred) == np.sign(actual)) * 100
                        print(f"    {regime:<10}: {acc:.1f}% ({len(sub)} samples)")
    
    # =============================================
    # 5. TOP/BOTTOM PERFORMERS
    # =============================================
    print("\n" + "=" * 80)
    print("  5. TOP/BOTTOM TICKERS (21d Directional Accuracy)")
    print("=" * 80)
    
    if "actual_ret_21d" in df.columns and "pred_ret_21d" in df.columns:
        ticker_stats = []
        for tkr in df["ticker"].unique():
            sub = df[df["ticker"] == tkr].dropna(subset=["pred_ret_21d", "actual_ret_21d"])
            if len(sub) < 3:
                continue
            pred = sub["pred_ret_21d"].values
            actual = sub["actual_ret_21d"].values
            acc = np.mean(np.sign(pred) == np.sign(actual)) * 100
            avg_actual = np.mean(actual) * 100
            ticker_stats.append({"Ticker": tkr.replace(".NS",""), "DirAcc": acc, "N": len(sub), "AvgRet": avg_actual})
        
        ts = pd.DataFrame(ticker_stats).sort_values("DirAcc", ascending=False)
        
        print("\n  🏆 TOP 10:")
        print(f"    {'Ticker':<15} | {'Dir.Acc':>7} | {'N':>3} | {'Avg Ret':>8}")
        print(f"    {'-'*40}")
        for _, r in ts.head(10).iterrows():
            print(f"    {r['Ticker']:<15} | {r['DirAcc']:>6.0f}% | {r['N']:>3} | {r['AvgRet']:>+7.1f}%")
        
        print("\n  💀 BOTTOM 10:")
        print(f"    {'Ticker':<15} | {'Dir.Acc':>7} | {'N':>3} | {'Avg Ret':>8}")
        print(f"    {'-'*40}")
        for _, r in ts.tail(10).iterrows():
            print(f"    {r['Ticker']:<15} | {r['DirAcc']:>6.0f}% | {r['N']:>3} | {r['AvgRet']:>+7.1f}%")
    
    # =============================================
    # 6. META-LEARNER SIGNAL QUALITY
    # =============================================
    print("\n" + "=" * 80)
    print("  6. META-LEARNER SIGNAL QUALITY")
    print("=" * 80)
    
    for h in [5, 21]:
        meta_col = f"meta_signal_{h}d"
        actual_col = f"actual_ret_{h}d"
        conf_col = f"meta_confidence_{h}d"
        
        if meta_col not in df.columns or actual_col not in df.columns:
            continue
        
        sub = df.dropna(subset=[meta_col, actual_col])
        
        for sig in ["BUY", "SELL", "HOLD"]:
            ss = sub[sub[meta_col] == sig]
            if len(ss) == 0:
                continue
            actual = ss[actual_col].values
            avg_ret = np.mean(actual) * 100
            win = np.mean(actual > 0) * 100 if sig == "BUY" else (np.mean(actual < 0) * 100 if sig == "SELL" else 0)
            avg_conf = ss[conf_col].mean() if conf_col in ss.columns else 0
            print(f"    {h}d {sig:<4}: {len(ss):>4} signals | AvgRet: {avg_ret:>+6.2f}% | WinRate: {win:>5.1f}% | AvgConf: {avg_conf:.3f}")
    
    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
