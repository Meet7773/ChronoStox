import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import Sparkline from "../components/Sparkline.jsx";
import TickerSearch from "../components/TickerSearch.jsx";

const API_URL = "http://127.0.0.1:8000";

const SCENARIOS = [
  {
    id: "gfc",
    name: "2008 Financial Crisis",
    ticker: "ICICIBANK.NS",
    start: "2007-10-01",
    end: "2009-04-01",
    description: "Trade through the credit crunch and global deleveraging shock.",
  },
  {
    id: "covid",
    name: "COVID-19 Crash",
    ticker: "RELIANCE.NS",
    start: "2020-01-01",
    end: "2020-06-01",
    description: "Navigate the volatility during the early pandemic months.",
  },
  {
    id: "dotcom",
    name: "Dot-Com Aftermath",
    ticker: "INFY.NS",
    start: "1999-01-01",
    end: "2001-12-31",
    description: "Experience the rollercoaster of early IT giants.",
  },
  {
    id: "taper",
    name: "Taper Tantrum",
    ticker: "HDFCBANK.NS",
    start: "2013-01-01",
    end: "2013-12-31",
    description: "See how emerging markets reacted to the first FED taper scare.",
  },
];

const PERIOD_OPTIONS = [
  { label: "1 Month", value: "1mo" },
  { label: "3 Months", value: "3mo" },
  { label: "6 Months", value: "6mo" },
  { label: "1 Year", value: "1y" },
  { label: "2 Years", value: "2y" },
  { label: "5 Years", value: "5y" },
];

function formatCurrency(value) {
  if (value === null || value === undefined) return "₹0";
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(value);
}

function formatPercent(value) {
  if (!Number.isFinite(value)) return "0%";
  const prefix = value >= 0 ? "+" : "-";
  return `${prefix}${Math.abs(value).toFixed(2)}%`;
}

function TradeSimulator() {
  const [mode, setMode] = useState("scenario");
  const [scenario, setScenario] = useState(SCENARIOS[0]);
  const [customTicker, setCustomTicker] = useState({ ticker: "RELIANCE.NS", name: "Reliance Industries Ltd." });
  const [period, setPeriod] = useState("6mo");
  const [history, setHistory] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [quantity, setQuantity] = useState("10");
  const [status, setStatus] = useState({ message: "", tone: "muted" });

  const activeTicker = mode === "scenario" ? scenario.ticker : (customTicker?.ticker || "");
  const activeLabel = mode === "scenario" ? `${scenario.ticker} • ${scenario.name}` : (customTicker?.ticker || "Select a ticker");

  const fetchHistory = useCallback(async () => {
    if (!activeTicker) {
      setHistory([]);
      setCursor(0);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const params = mode === "scenario"
        ? { start: scenario.start, end: scenario.end }
        : { period };

      const response = await axios.get(`${API_URL}/history/${activeTicker}`, { params });
      const data = response.data
        .map((point) => ({ ...point, date: point.date }))
        .sort((a, b) => new Date(a.date) - new Date(b.date));

      setHistory(data);
      setCursor(Math.max(data.length - 1, 0));
    } catch (err) {
      setHistory([]);
      setCursor(0);
      setError(err.response?.data?.detail ?? "Failed to load historical prices.");
    } finally {
      setIsLoading(false);
    }
  }, [activeTicker, mode, period, scenario]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const activePoint = history[cursor];

  const metrics = useMemo(() => {
    if (!history.length) return null;
    const first = history[0];
    const last = activePoint ?? history[history.length - 1];
    const changeAbs = last.close - first.close;
    const changePct = first.close ? (changeAbs / first.close) * 100 : 0;
    return {
      startDate: new Date(first.date).toLocaleDateString(),
      endDate: new Date(last.date).toLocaleDateString(),
      startClose: first.close,
      currentClose: last.close,
      changeAbs,
      changePct,
    };
  }, [history, activePoint]);

  const handleTrade = async (action) => {
    if (!activePoint || !activeTicker) return;
    const qty = Number.parseInt(quantity, 10);
    if (!Number.isFinite(qty) || qty <= 0) {
      setStatus({ message: "Quantity must be a positive integer.", tone: "danger" });
      return;
    }

    setStatus({ message: `Submitting ${action} order…`, tone: "info" });
    try {
      const response = await axios.post(`${API_URL}/trade`, {
        userId: "college_project_user",
        ticker: activeTicker,
        quantity: qty,
        price: activePoint.close,
        action,
      });
      setStatus({ message: response.data.message ?? "Trade executed.", tone: "success" });
    } catch (err) {
      setStatus({
        message: err.response?.data?.detail ?? "Trade failed. Check API connectivity.",
        tone: "danger",
      });
    }
  };

  const handleScenarioSelect = (item) => {
    setScenario(item);
    setMode("scenario");
  };

  return (
    <section className="space-y-10">
      <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-3xl font-semibold text-[var(--color-text)]">Trade Simulator</h2>
          <p className="text-[var(--color-text-muted)]">
            Rewind historic regimes or replay custom periods with real prices, then execute simulated orders instantly.
          </p>
        </div>
        <div className="inline-flex rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] p-1">
          <button
            type="button"
            onClick={() => setMode("scenario")}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
              mode === "scenario"
                ? "bg-[var(--color-accent)] text-white"
                : "text-[var(--color-text-muted)] hover:bg-[var(--color-accent-soft)] hover:text-[var(--color-text)]"
            }`}
          >
            Scenario Replay
          </button>
          <button
            type="button"
            onClick={() => setMode("custom")}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
              mode === "custom"
                ? "bg-[var(--color-accent)] text-white"
                : "text-[var(--color-text-muted)] hover:bg-[var(--color-accent-soft)] hover:text-[var(--color-text)]"
            }`}
          >
            Custom Timeline
          </button>
        </div>
      </header>

      {mode === "scenario" ? (
        <div className="grid gap-4 lg:grid-cols-4">
          {SCENARIOS.map((item) => {
            const selected = item.id === scenario.id;
            return (
              <button
                type="button"
                key={item.id}
                onClick={() => handleScenarioSelect(item)}
                className={`flex h-full flex-col justify-between rounded-2xl border px-4 py-4 text-left transition ${
                  selected
                    ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
                    : "border-[var(--color-border)] bg-[var(--color-surface-elevated)] hover:border-[var(--color-accent)]"
                }`}
              >
                <div className="space-y-2">
                  <p className="text-xs uppercase text-[var(--color-text-muted)]">{item.start} → {item.end}</p>
                  <h3 className="text-lg font-semibold text-[var(--color-text)]">{item.name}</h3>
                  <p className="text-sm text-[var(--color-text-muted)]">{item.description}</p>
                </div>
                <span className="text-xs font-semibold text-[var(--color-text-muted)]">Ticker: {item.ticker}</span>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="card p-6">
          <div className="grid gap-4 md:grid-cols-3">
            <TickerSearch
              label="Ticker"
              value={customTicker?.ticker ?? ""}
              onSelect={(item) => setCustomTicker(item)}
              helper="Start typing to search NSE/BSE coverage"
            />
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-[var(--color-text-muted)]">Period</span>
              <select
                value={period}
                onChange={(event) => setPeriod(event.target.value)}
                className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 focus:border-[var(--color-accent)] focus:outline-none"
              >
                {PERIOD_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="flex flex-col justify-end gap-2">
              <button
                type="button"
                onClick={fetchHistory}
                className="rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[var(--color-accent)]/90"
              >
                Refresh Series
              </button>
              <p className="text-xs text-[var(--color-text-muted)]">
                Uses the `/history/{ticker}` endpoint with the selected timeframe.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[3fr,2fr]">
        <div className="card overflow-hidden">
          <div className="border-b border-[var(--color-border)] bg-[var(--color-surface-muted)]/40 px-6 py-4">
            <p className="text-sm font-semibold text-[var(--color-text)]">{activeLabel}</p>
            {metrics && (
              <p className="text-xs text-[var(--color-text-muted)]">{metrics.startDate} → {metrics.endDate}</p>
            )}
          </div>
          <div className="space-y-5 p-6">
            {isLoading ? (
              <div className="flex h-48 items-center justify-center text-sm text-[var(--color-text-muted)]">
                Loading price history…
              </div>
            ) : error ? (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                {error}
              </div>
            ) : history.length ? (
              <>
                <Sparkline data={history} accessor={(point) => point.close} />
                <input
                  type="range"
                  min={0}
                  max={Math.max(history.length - 1, 0)}
                  value={cursor}
                  onChange={(event) => setCursor(Number(event.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
                {activePoint && (
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
                      <p className="text-xs text-[var(--color-text-muted)]">Date</p>
                      <p className="text-sm font-semibold text-[var(--color-text)]">
                        {new Date(activePoint.date).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
                      <p className="text-xs text-[var(--color-text-muted)]">Close</p>
                      <p className="text-sm font-semibold text-[var(--color-text)]">{formatCurrency(activePoint.close)}</p>
                    </div>
                    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
                      <p className="text-xs text-[var(--color-text-muted)]">High / Low</p>
                      <p className="text-sm font-semibold text-[var(--color-text)]">
                        {formatCurrency(activePoint.high)}
                        <span className="text-[var(--color-text-muted)]"> / </span>
                        {formatCurrency(activePoint.low)}
                      </p>
                    </div>
                    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
                      <p className="text-xs text-[var(--color-text-muted)]">Volume</p>
                      <p className="text-sm font-semibold text-[var(--color-text)]">
                        {activePoint.volume ? activePoint.volume.toLocaleString("en-IN") : "—"}
                      </p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex h-48 items-center justify-center text-sm text-[var(--color-text-muted)]">
                Select a scenario or ticker to load data.
              </div>
            )}
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-semibold text-[var(--color-text)]">Execute Trade</h3>
          <p className="text-sm text-[var(--color-text-muted)]">
            Orders post to `/trade` and refresh the Mongo-backed portfolio for user <code className="font-mono">college_project_user</code>.
          </p>
          <div className="mt-5 grid gap-3">
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
            <div className="flex flex-col gap-2 sm:flex-row">
              <button
                type="button"
                onClick={() => handleTrade("BUY")}
                disabled={!activePoint || isLoading}
                className="flex-1 rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[var(--color-accent)]/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Buy @ {activePoint ? formatCurrency(activePoint.close) : "—"}
              </button>
              <button
                type="button"
                onClick={() => handleTrade("SELL")}
                disabled={!activePoint || isLoading}
                className="flex-1 rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm font-semibold text-[var(--color-text)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-60"
              >
                Sell @ {activePoint ? formatCurrency(activePoint.close) : "—"}
              </button>
            </div>
            {status.message && (
              <div
                className={`rounded-lg border px-3 py-2 text-sm ${
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
            )}
          </div>

          {metrics && (
            <div className="mt-6 grid gap-4">
              <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                <p className="text-xs text-[var(--color-text-muted)]">Return Over Window</p>
                <p className="text-xl font-semibold text-[var(--color-text)]">{formatPercent(metrics.changePct)}</p>
                <p className="text-sm text-[var(--color-text-muted)]">{formatCurrency(metrics.changeAbs)}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default TradeSimulator;
