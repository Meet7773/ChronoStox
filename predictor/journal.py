import os
import json

PORTFOLIO_FILE = "portfolio_legacy.json"


def main():
    if not os.path.exists(PORTFOLIO_FILE):
        print("❌ No portfolio.json found.")
        return

    with open(PORTFOLIO_FILE, "r") as f:
        pf = json.load(f)

    history = pf.get("history", [])
    if not history:
        print("⚠️ No trade history found.")
        return

    print(f"\n{'=' * 85}")
    print(f" 📖 CHRONOSTOX TRADE JOURNAL (Realized PnL)")
    print(f"{'=' * 85}")
    print(f" {'DATE':<12} | {'TICKER':<15} | {'ACT':<4} | {'QTY':<4} | {'PRICE':<8} | {'FEES':<6} | {'PnL %':<8}")
    print("-" * 85)

    total_realized_pnl = 0.0
    win_count = 0
    loss_count = 0

    for t in history:
        date = t.get("date", "Unknown")
        tkr = t.get("ticker", "Unknown")
        act = t.get("action", "UNK")
        qty = t.get("shares", 0)
        px = t.get("price", 0.0)
        fees = t.get("fees", 0.0)
        pnl_pct = t.get("pnl_pct", 0.0)

        # Color coding
        if act == "BUY":
            color_prefix = "\033[94m"  # Blue
            pnl_str = "-"
        else:
            if pnl_pct > 0:
                color_prefix = "\033[92m"  # Green
                pnl_str = f"+{pnl_pct:.2f}%"
                win_count += 1
            elif pnl_pct < 0:
                color_prefix = "\033[91m"  # Red
                pnl_str = f"{pnl_pct:.2f}%"
                loss_count += 1
            else:
                color_prefix = "\033[93m"  # Yellow
                pnl_str = "0.00%"

        reset = "\033[0m"

        print(
            f" {date:<12} | {tkr:<15} | {color_prefix}{act:<4}{reset} | {qty:<4} | ₹{px:<7.2f} | ₹{fees:<5.2f} | {color_prefix}{pnl_str:<8}{reset}")

    print("-" * 85)

    total_trades = win_count + loss_count
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

    print(f" Total Closed Trades : {total_trades}")
    print(f" Win Rate            : {win_rate:.1f}% ({win_count}W / {loss_count}L)")
    print(f" Current Cash        : ₹{pf.get('cash', 0.0):,.2f}")
    print(f"{'=' * 85}\n")


if __name__ == "__main__":
    main()