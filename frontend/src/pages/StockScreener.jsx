import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";
const PAGE_SIZE = 25;

function createCsv(rows) {
  const header = ["Ticker", "Name", "Exchange", "Category", "Country"];
  const body = rows
    .map((row) => [row.ticker, row.name ?? "", row.exchange ?? "", row.category ?? "", row.country ?? ""].join(","))
    .join("\n");
  return [header.join(","), body].join("\n");
}

function downloadCsv(rows) {
  const csv = createCsv(rows);
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "chronostox_tickers.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function StockScreener() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [exchange, setExchange] = useState("ALL");
  const [country, setCountry] = useState("ALL");
  const [page, setPage] = useState(1);

  useEffect(() => {
    let isActive = true;
    async function bootstrap() {
      setIsLoading(true);
      try {
        const response = await axios.get(`${API_URL}/tickers`, { params: { limit: 250 } });
        if (!isActive) return;
        setResults(response.data);
      } catch (err) {
        if (!isActive) return;
        setError(err.response?.data?.detail ?? "Unable to load tickers.");
      } finally {
        if (isActive) setIsLoading(false);
      }
    }
    bootstrap();
    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    const handle = setTimeout(async () => {
      setIsLoading(true);
      setError(null);
      try {
        const params = { limit: 250 };
        if (query.trim().length >= 2) {
          params.query = query.trim();
        }
        const response = await axios.get(`${API_URL}/tickers`, { params });
        setResults(response.data);
        setPage(1);
      } catch (err) {
        setError(err.response?.data?.detail ?? "Unable to search tickers.");
      } finally {
        setIsLoading(false);
      }
    }, 220);

    return () => clearTimeout(handle);
  }, [query]);

  const exchanges = useMemo(() => {
    const set = new Set(results.map((item) => item.exchange).filter(Boolean));
    return ["ALL", ...Array.from(set).sort()];
  }, [results]);

  const countries = useMemo(() => {
    const set = new Set(results.map((item) => item.country).filter(Boolean));
    return ["ALL", ...Array.from(set).sort()];
  }, [results]);

  const filtered = useMemo(() => {
    return results.filter((item) => {
      const exchangeMatch = exchange === "ALL" || item.exchange === exchange;
      const countryMatch = country === "ALL" || item.country === country;
      return exchangeMatch && countryMatch;
    });
  }, [results, exchange, country]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <section className="space-y-8">
      <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-3xl font-semibold text-[var(--color-text)]">Stock Screener</h2>
          <p className="text-[var(--color-text-muted)]">
            Explore NSE/BSE coverage sourced from <code className="font-mono">data/ticker.csv</code> with instant search and filtering.
          </p>
        </div>
        <button
          type="button"
          onClick={() => downloadCsv(filtered)}
          className="rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[var(--color-accent)]/90"
        >
          Download CSV
        </button>
      </header>

      <div className="card p-6">
        <div className="grid gap-4 md:grid-cols-4">
          <label className="md:col-span-2">
            <span className="text-sm text-[var(--color-text-muted)]">Search</span>
            <input
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Type at least 2 characters"
              className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 focus:border-[var(--color-accent)] focus:outline-none"
            />
          </label>
          <label>
            <span className="text-sm text-[var(--color-text-muted)]">Exchange</span>
            <select
              value={exchange}
              onChange={(event) => {
                setExchange(event.target.value);
                setPage(1);
              }}
              className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 text-sm focus:border-[var(--color-accent)] focus:outline-none"
            >
              {exchanges.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="text-sm text-[var(--color-text-muted)]">Country</span>
            <select
              value={country}
              onChange={(event) => {
                setCountry(event.target.value);
                setPage(1);
              }}
              className="mt-1 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 text-sm focus:border-[var(--color-accent)] focus:outline-none"
            >
              {countries.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
        {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      </div>

      <div className="card overflow-hidden">
        <div className="border-b border-[var(--color-border)] bg-[var(--color-surface-muted)]/40 px-6 py-4">
          <div className="flex items-center justify-between text-sm text-[var(--color-text-muted)]">
            <span>
              Showing <strong className="text-[var(--color-text)]">{filtered.length}</strong> matches
            </span>
            {isLoading && <span>Updating…</span>}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-[var(--color-border)]">
            <thead className="bg-[var(--color-surface)] text-xs uppercase text-[var(--color-text-muted)]">
              <tr>
                <th className="px-6 py-3 text-left">Ticker</th>
                <th className="px-6 py-3 text-left">Name</th>
                <th className="px-6 py-3 text-left">Exchange</th>
                <th className="px-6 py-3 text-left">Category</th>
                <th className="px-6 py-3 text-left">Country</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-surface-elevated)] text-sm">
              {paged.map((row) => (
                <tr key={`${row.ticker}-${row.exchange}`} className="hover:bg-[var(--color-accent-soft)]/60">
                  <td className="px-6 py-3 font-semibold text-[var(--color-text)]">{row.ticker}</td>
                  <td className="px-6 py-3 text-[var(--color-text)]">{row.name}</td>
                  <td className="px-6 py-3 text-[var(--color-text-muted)]">{row.exchange || "—"}</td>
                  <td className="px-6 py-3 text-[var(--color-text-muted)]">{row.category || "—"}</td>
                  <td className="px-6 py-3 text-[var(--color-text-muted)]">{row.country || "—"}</td>
                </tr>
              ))}
              {!paged.length && !isLoading && (
                <tr>
                  <td colSpan={5} className="px-6 py-8 text-center text-sm text-[var(--color-text-muted)]">
                    No results match the current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between border-t border-[var(--color-border)] bg-[var(--color-surface)] px-6 py-3 text-sm text-[var(--color-text-muted)]">
          <span>
            Page {page} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setPage((prev) => Math.max(1, prev - 1))}
              disabled={page === 1}
              className="rounded-lg border border-[var(--color-border)] px-3 py-1 transition hover:border-[var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              Prev
            </button>
            <button
              type="button"
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={page === totalPages}
              className="rounded-lg border border-[var(--color-border)] px-3 py-1 transition hover:border-[var(--color-accent)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

export default StockScreener;
