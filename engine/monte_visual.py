import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import argparse

# Cyberpunk Style Settings
plt.style.use('dark_background')
COLORS = {
    'bg': '#0d1117',
    'grid': '#21262d',
    'path': '#238636',  # Dim Matrix Green
    'mean': '#58a6ff',  # Neon Blue
    'upper': '#3fb950',  # Bright Green
    'lower': '#f85149',  # Bright Red
    'target': '#d29922'  # Gold
}


def fetch_data(ticker):
    print(f"[*] Fetching data for {ticker}...")
    df = yf.download(ticker, period="400d", interval="1d", progress=False)
    if df.empty:
        raise ValueError("No data found.")

    # FIX: Handle yfinance MultiIndex columns (Close, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Try to extract 'Close' safely
        try:
            data = df.xs('Close', axis=1, level=0)
            # If multiple columns remain (weird case), take the first
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            return data
        except KeyError:
            pass

    # Standard case or fallback
    if 'Close' in df.columns:
        data = df['Close']
        # Ensure it's a Series, not a 1-col DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        return data

    # Last ditch effort: grab the first column
    return df.iloc[:, 0]


def monte_carlo_paths(last_price, daily_vol, daily_drift, days, simulations):
    # Generate Z-matrix (Random Shocks)
    # Shape: (days, simulations)
    Z = np.random.normal(0, 1, (days, simulations))

    # Initialize price matrix
    price_paths = np.zeros((days, simulations))
    price_paths[0] = last_price

    # Vectorized GBM Logic
    drift_term = (daily_drift - 0.5 * daily_vol ** 2)
    shock_term = daily_vol * Z

    # Daily Multipliers
    daily_returns = np.exp(drift_term + shock_term)

    # Cumulative Product to get price path
    price_paths = np.vstack([np.full(simulations, last_price), daily_returns]).cumprod(axis=0)

    return price_paths


def plot_simulation(ticker, price_paths, days):
    plt.figure(figsize=(14, 8), facecolor=COLORS['bg'])
    ax = plt.gca()
    ax.set_facecolor(COLORS['bg'])

    # 1. Plot the "Chaos" (First 100 paths only)
    plt.plot(price_paths[:, :100], color=COLORS['path'], alpha=0.05, linewidth=1)

    # 2. Calculate Statistical Cones
    p05 = np.percentile(price_paths, 5, axis=1)
    p50 = np.percentile(price_paths, 50, axis=1)
    p95 = np.percentile(price_paths, 95, axis=1)
    p99 = np.percentile(price_paths, 99, axis=1)
    p01 = np.percentile(price_paths, 1, axis=1)

    x_axis = np.arange(len(p50))

    # 3. Plot the Cones
    plt.fill_between(x_axis, p01, p99, color=COLORS['grid'], alpha=0.3, label='99% Confidence (Extreme)')
    plt.fill_between(x_axis, p05, p95, color=COLORS['mean'], alpha=0.2, label='90% Confidence (Likely)')
    plt.plot(x_axis, p50, color=COLORS['mean'], linewidth=2, linestyle='--', label='Median Path')

    # 4. Plot the Boundaries
    plt.plot(x_axis, p99, color=COLORS['upper'], linewidth=1.5, linestyle='-', label='Max Ceiling (99th)')
    plt.plot(x_axis, p01, color=COLORS['lower'], linewidth=1.5, linestyle='-', label='Max Floor (1st)')

    # Decoration
    plt.title(f"MONTE CARLO SIMULATION: {ticker} [{days} Days]", color='white', fontsize=16, pad=20)
    plt.xlabel("Days into Future", color='gray')
    plt.ylabel("Price", color='gray')
    plt.grid(color=COLORS['grid'], linestyle=':', linewidth=0.5)
    plt.tick_params(colors='gray')

    leg = plt.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
    for text in leg.get_texts():
        text.set_color('gray')

    # Stats Box
    last_p50 = p50[-1]
    last_p99 = p99[-1]
    last_p01 = p01[-1]

    stats_text = (
        f"Start Price: {price_paths[0, 0]:.2f}\n"
        f"Median Target: {last_p50:.2f}\n"
        f"Max Ceiling (99%): {last_p99:.2f}\n"
        f"Max Floor (1%): {last_p01:.2f}\n"
        f"Simulations: {price_paths.shape[1]}"
    )
    plt.text(0.02, 0.95, stats_text, transform=ax.transAxes,
             color='#00ff41', fontfamily='monospace',
             bbox=dict(facecolor='black', alpha=0.7, edgecolor='#00ff41'))

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str, help="Ticker (e.g. IDEA.NS)")
    parser.add_argument("--days", type=int, default=252, help="Forecast horizon")
    parser.add_argument("--sims", type=int, default=5000, help="Number of simulations")
    args = parser.parse_args()

    # 1. Get Data
    prices = fetch_data(args.ticker)

    # Ensure we have a Series of floats
    prices = prices.astype(float)
    current_price = float(prices.iloc[-1])

    # 2. Calculate Physics (Drift & Vol)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # FIX: Force scalar conversion for printing
    vol_daily = float(log_returns.std())
    vol_annual = vol_daily * np.sqrt(252)

    # Conservative Drift (50% of historical)
    drift_daily = float(log_returns.mean()) * 0.5

    print(f"[*] Current Price: {current_price:.2f}")
    print(f"[*] Daily Volatility: {vol_daily:.4f} ({vol_annual * 100:.2f}% Annual)")
    print(f"[*] Daily Drift Bias: {drift_daily:.6f}")

    # 3. Run Engine
    print(f"[*] Running {args.sims} simulations for {args.days} days...")
    paths = monte_carlo_paths(current_price, vol_daily, drift_daily, args.days, args.sims)

    # 4. Render
    plot_simulation(args.ticker, paths, args.days)


if __name__ == "__main__":
    main()