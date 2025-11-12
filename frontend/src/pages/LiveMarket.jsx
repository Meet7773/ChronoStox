import React, { useState } from "react";
import axios from "axios";
import TickerSearch from "../components/TickerSearch.jsx";

const API_URL = "http://127.0.0.1:8000";

function formatNumber(value, { style = "decimal" } = {}) {
  if (value === null || value === undefined) return "—";
  if (style === "currency") {
    return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(value);
  }
  return new Intl.NumberFormat("en-IN").format(value);
}

function LiveMarket() {
  const [ticker, setTicker] = useState("");
  const [stockData, setStockData] = useState(null);
  const [status, setStatus] = useState({ message: "Search for a NSE ticker (e.g. RELIANCE.NS)", tone: "muted", loading: false });

  const handleSearch = async (event) => {
    event.preventDefault();
    const trimmed = ticker.trim().toUpperCase();
    if (!trimmed) {
      setStatus({ message: "Enter a ticker symbol to continue.", tone: "danger", loading: false });
      setStockData(null);
      return;
    }

    setStatus({ message: `Fetching ${trimmed}…`, tone: "info", loading: true });
    setStockData(null);
    try {
      const response = await axios.get(`${API_URL}/stock/${trimmed}`);
      setStockData(response.data);
      setStatus({ message: `${response.data.ticker} data refreshed.`, tone: "success", loading: false });
    } catch (error) {
      const detail = error.response?.data?.detail ?? "Failed to load quote.";
      setStatus({ message: detail, tone: "danger", loading: false });
      setStockData(null);
    }
  };

  return (
    <section className="space-y-6">
      <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-3xl font-semibold text-[var(--color-text)]">Live Market Lookup</h2>
          <p className="text-[var(--color-text-muted)]">Pull real-time snapshots directly from the FastAPI backend.</p>
        </div>
      </header>

      <form onSubmit={handleSearch} className="flex flex-col gap-3 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-sm md:flex-row md:items-end">
        <div className="flex-1">
          <TickerSearch
            label="Ticker"
            value={ticker}
            onInputChange={setTicker}
            onSelect={(item) => setTicker(item.ticker)}
            helper="Start typing to pick from NSE/BSE coverage"
          />
        </div>
        <button
          type="submit"
          disabled={status.loading}
          className="rounded-lg bg-[var(--color-accent)] px-6 py-2 font-semibold text-white transition hover:bg-[var(--color-accent)]/90 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {status.loading ? "Searching…" : "Fetch"}
        </button>
      </form>

      <div
        className={`rounded-2xl border px-4 py-3 text-sm ${
          status.tone === "success"
            ? "border-green-200 bg-green-50 text-green-700"
            : status.tone === "danger"
            ? "border-red-200 bg-red-50 text-red-600"
            : status.tone === "info"
            ? "border-blue-200 bg-blue-50 text-blue-600"
            : "border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-text-muted)]"
        }`}
      >
        {status.message}
      </div>

      {stockData && (
        <div className="card">
          <div className="p-6">
            <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
              <div>
                <p className="text-sm uppercase text-[var(--color-text-muted)]">{stockData.ticker}</p>
                <h3 className="text-2xl font-semibold text-[var(--color-text)]">{stockData.companyName}</h3>
              </div>
              <div className="text-right">
                <p className="text-sm text-[var(--color-text-muted)]">Last Price</p>
                <p className="text-3xl font-semibold text-[var(--color-text)]">{formatNumber(stockData.currentPrice, { style: "currency" })}</p>
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                <p className="text-xs uppercase text-[var(--color-text-muted)]">Previous Close</p>
                <p className="mt-1 text-lg font-semibold text-[var(--color-text)]">{formatNumber(stockData.previousClose, { style: "currency" })}</p>
              </div>
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                <p className="text-xs uppercase text-[var(--color-text-muted)]">Day High / Low</p>
                <p className="mt-1 text-lg font-semibold text-[var(--color-text)]">
                  {formatNumber(stockData.dayHigh, { style: "currency" })}
                  <span className="text-[var(--color-text-muted)]"> / </span>
                  {formatNumber(stockData.dayLow, { style: "currency" })}
                </p>
              </div>
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                <p className="text-xs uppercase text-[var(--color-text-muted)]">Market Cap</p>
                <p className="mt-1 text-lg font-semibold text-[var(--color-text)]">
                  {stockData.marketCap ? formatNumber(stockData.marketCap, { style: "currency" }) : "—"}
                </p>
              </div>
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                <p className="text-xs uppercase text-[var(--color-text-muted)]">Volume</p>
                <p className="mt-1 text-lg font-semibold text-[var(--color-text)]">
                  {stockData.volume ? formatNumber(stockData.volume) : "—"}
                </p>
              </div>
            </div>

            {stockData.longSummary && (
              <div className="mt-6 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                <p className="text-xs uppercase text-[var(--color-text-muted)]">Business Summary</p>
                <p className="mt-2 text-sm leading-relaxed text-[var(--color-text)]">{stockData.longSummary}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

export default LiveMarket;
