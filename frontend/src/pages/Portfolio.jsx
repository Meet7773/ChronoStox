import React, { useCallback, useEffect, useState } from "react";
import axios from "axios";
import TickerSearch from "../components/TickerSearch.jsx";

const API_URL = "http://127.0.0.1:8000";
const USER_ID = "college_project_user";

function formatINR(value) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(value ?? 0);
}

function TradeWidget({ onTradeSuccess }) {
  const [ticker, setTicker] = useState("");
  const [quantity, setQuantity] = useState("1");
  const [price, setPrice] = useState("");
  const [status, setStatus] = useState({ message: "", tone: "info", loading: false });

  const reset = () => {
    setTicker("");
    setQuantity("1");
    setPrice("");
  };

  const submitTrade = async (action) => {
    if (status.loading) return;
    const trimmedTicker = ticker.trim().toUpperCase();
    const qty = Number.parseInt(quantity, 10);
    const px = Number.parseFloat(price);

    if (!trimmedTicker) {
      setStatus({ message: "Enter a ticker symbol.", tone: "danger", loading: false });
      return;
    }
    if (!Number.isFinite(qty) || qty <= 0) {
      setStatus({ message: "Quantity must be a positive integer.", tone: "danger", loading: false });
      return;
    }
    if (!Number.isFinite(px) || px <= 0) {
      setStatus({ message: "Price must be a positive number.", tone: "danger", loading: false });
      return;
    }

    setStatus({ message: `Submitting ${action.toLowerCase()} order…`, tone: "info", loading: true });
    try {
      const response = await axios.post(`${API_URL}/trade`, {
        userId: USER_ID,
        ticker: trimmedTicker,
        quantity: qty,
        price: px,
        action,
      });
      const next = response.data.newPortfolio ?? null;
      setStatus({ message: response.data.message ?? "Trade executed.", tone: "success", loading: false });
      reset();
      onTradeSuccess(next);
    } catch (error) {
      const detail = error.response?.data?.detail ?? "Trade failed. Check API connectivity.";
      setStatus({ message: detail, tone: "danger", loading: false });
    }
  };

  return (
    <div className="space-y-4 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
      <div>
        <h3 className="text-lg font-semibold text-[var(--color-text)]">Execute Order</h3>
        <p className="text-sm text-[var(--color-text-muted)]">Send simulated trades to the ChronoStox portfolio.</p>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="sm:col-span-1">
          <TickerSearch
            label="Ticker"
            value={ticker}
            onInputChange={(value) => setTicker(value.toUpperCase())}
            onSelect={(item) => setTicker(item.ticker)}
            helper="Pulls from ticker catalog"
          />
        </div>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-[var(--color-text-muted)]">Quantity</span>
          <input
            type="number"
            min={1}
            value={quantity}
            onChange={(event) => setQuantity(event.target.value)}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 focus:border-[var(--color-accent)] focus:outline-none"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-[var(--color-text-muted)]">Price</span>
          <input
            type="number"
            min={0.01}
            step={0.01}
            value={price}
            onChange={(event) => setPrice(event.target.value)}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 focus:border-[var(--color-accent)] focus:outline-none"
          />
        </label>
      </div>
      <div className="flex flex-col gap-2 sm:flex-row">
        <button
          type="button"
          onClick={() => submitTrade("BUY")}
          disabled={status.loading}
          className="flex-1 rounded-lg bg-[var(--color-accent)] px-4 py-2 font-semibold text-white transition hover:bg-[var(--color-accent)]/90 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {status.loading ? "Processing…" : "Buy"}
        </button>
        <button
          type="button"
          onClick={() => submitTrade("SELL")}
          disabled={status.loading}
          className="flex-1 rounded-lg border border-[var(--color-border)] px-4 py-2 font-semibold text-[var(--color-text)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {status.loading ? "Processing…" : "Sell"}
        </button>
      </div>
      {status.message && (
        <div
          className={`rounded-lg border px-3 py-2 text-sm ${
            status.tone === "success"
              ? "border-green-400 text-green-600"
              : status.tone === "danger"
              ? "border-red-400 text-red-500"
              : "border-[var(--color-border)] text-[var(--color-text-muted)]"
          }`}
        >
          {status.message}
        </div>
      )}
    </div>
  );
}

function Portfolio() {
  const [portfolio, setPortfolio] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPortfolio = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/portfolio/${USER_ID}`);
      setPortfolio(response.data);
    } catch (err) {
      const detail = err.response?.data?.detail ?? `Network error: ${err.message}`;
      setError(detail);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPortfolio();
  }, [fetchPortfolio]);

  const handleTradeSuccess = (next) => {
    if (next) {
      setPortfolio(next);
    } else {
      fetchPortfolio();
    }
  };

  if (isLoading && !portfolio) {
    return (
      <div className="flex h-48 items-center justify-center rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 text-sm text-[var(--color-text-muted)]">
        Loading portfolio…
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4 rounded-2xl border border-red-200 bg-red-50 p-6 text-sm text-red-600">
        <div>
          <h3 className="text-lg font-semibold text-red-700">Unable to load portfolio</h3>
          <p>{error}</p>
        </div>
        <button
          type="button"
          onClick={fetchPortfolio}
          className="rounded-lg bg-red-600 px-4 py-2 font-semibold text-white transition hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!portfolio) {
    return null;
  }

  const totalHoldingsValue = portfolio.holdings.reduce((sum, holding) => sum + holding.quantity * holding.avgPrice, 0);
  const cash = portfolio.virtualCash ?? 0;
  const totalEquity = cash + totalHoldingsValue;

  return (
    <section className="space-y-8">
      <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-3xl font-semibold text-[var(--color-text)]">Portfolio Overview</h2>
          <p className="text-[var(--color-text-muted)]">User: {portfolio.userId}</p>
        </div>
      </header>

      <div className="grid gap-5 md:grid-cols-3">
        <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5 shadow-sm">
          <p className="text-xs uppercase text-[var(--color-text-muted)]">Total Equity</p>
          <p className="mt-2 text-2xl font-semibold text-[var(--color-text)]">{formatINR(totalEquity)}</p>
        </div>
        <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5 shadow-sm">
          <p className="text-xs uppercase text-[var(--color-text-muted)]">Cash</p>
          <p className="mt-2 text-2xl font-semibold text-[var(--color-text)]">{formatINR(cash)}</p>
        </div>
        <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5 shadow-sm">
          <p className="text-xs uppercase text-[var(--color-text-muted)]">Invested Value</p>
          <p className="mt-2 text-2xl font-semibold text-[var(--color-text)]">{formatINR(totalHoldingsValue)}</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-12">
        <div className="lg:col-span-4">
          <TradeWidget onTradeSuccess={handleTradeSuccess} />
        </div>
        <div className="card lg:col-span-8">
          <div className="p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-[var(--color-text)]">Positions</h3>
              <span className="text-xs text-[var(--color-text-muted)]">{portfolio.holdings.length} holdings</span>
            </div>
            <div className="mt-4 overflow-x-auto rounded-xl border border-[var(--color-border)]">
              <table className="min-w-full divide-y divide-[var(--color-border)]">
                <thead className="bg-[var(--color-surface-muted)]/40 text-xs uppercase tracking-wide text-[var(--color-text-muted)]">
                  <tr>
                    <th className="px-4 py-3 text-left">Ticker</th>
                    <th className="px-4 py-3 text-right">Quantity</th>
                    <th className="px-4 py-3 text-right">Avg Price</th>
                    <th className="px-4 py-3 text-right">Book Value</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-surface-elevated)] text-sm">
                  {portfolio.holdings.length ? (
                    portfolio.holdings.map((holding) => {
                      const bookValue = holding.quantity * holding.avgPrice;
                      return (
                        <tr key={holding.ticker} className="hover:bg-[var(--color-accent-soft)]/60">
                          <td className="px-4 py-3 font-semibold text-[var(--color-text)]">{holding.ticker}</td>
                          <td className="px-4 py-3 text-right text-[var(--color-text)]">{holding.quantity}</td>
                          <td className="px-4 py-3 text-right text-[var(--color-text)]">{formatINR(holding.avgPrice)}</td>
                          <td className="px-4 py-3 text-right text-[var(--color-text)]">{formatINR(bookValue)}</td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={4} className="px-4 py-6 text-center text-sm text-[var(--color-text-muted)]">
                        No holdings yet. Execute a trade to build your portfolio.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Portfolio;
