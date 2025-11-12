import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

function debounce(fn, delay) {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), delay);
  };
}

const fetchTickers = debounce(async (query, limit, setState) => {
  try {
    const params = {};
    if (query) params.query = query;
    if (limit) params.limit = limit;
    const response = await axios.get(`${API_URL}/tickers`, { params });
    setState({ status: "success", items: response.data });
  } catch (error) {
    setState({ status: "error", items: [], message: error.response?.data?.detail ?? "Search failed." });
  }
}, 180);

function TickerSearch({
  label = "Ticker",
  placeholder = "Search NSE or BSE symbols",
  limit = 15,
  onSelect,
  value = "",
  helper,
  onInputChange,
}) {
  const [input, setInput] = useState(value);
  const [state, setState] = useState({ status: "idle", items: [], message: "" });
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!input || input.length < 2) {
      setState({ status: "idle", items: [], message: "" });
      return;
    }
    setState((prev) => ({ ...prev, status: "loading" }));
    fetchTickers(input, limit, setState);
  }, [input, limit]);

  useEffect(() => {
    setInput(value);
  }, [value]);

  const suggestions = useMemo(() => state.items || [], [state.items]);

  return (
    <div className="relative">
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-[var(--color-text-muted)]">{label}</span>
        <input
          type="text"
          value={input}
          onChange={(event) => {
            setInput(event.target.value);
            setOpen(true);
            onInputChange?.(event.target.value);
          }}
          onFocus={() => setOpen(Boolean(input))}
          placeholder={placeholder}
          className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-3 py-2 focus:border-[var(--color-accent)] focus:outline-none"
        />
      </label>
      {helper && <p className="mt-1 text-xs text-[var(--color-text-muted)]">{helper}</p>}
      {open && input.length >= 2 && (
        <div className="absolute z-20 mt-2 max-h-64 w-full overflow-y-auto rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-elevated)] shadow-xl">
          {state.status === "loading" && (
            <p className="px-3 py-2 text-xs text-[var(--color-text-muted)]">Searching…</p>
          )}
          {state.status === "error" && (
            <p className="px-3 py-2 text-xs text-red-500">{state.message}</p>
          )}
          {state.status !== "error" && suggestions.length === 0 && input.length >= 2 && state.status !== "loading" && (
            <p className="px-3 py-2 text-xs text-[var(--color-text-muted)]">No matches.</p>
          )}
          {suggestions.map((item) => (
            <button
              type="button"
              key={`${item.ticker}-${item.exchange}`}
              onClick={() => {
                setInput(item.ticker);
                setOpen(false);
                onInputChange?.(item.ticker);
                onSelect?.(item);
              }}
              className="flex w-full flex-col items-start gap-0.5 border-b border-[var(--color-border)] px-3 py-2 text-left text-sm transition hover:bg-[var(--color-accent-soft)]"
            >
              <span className="font-semibold text-[var(--color-text)]">{item.ticker}</span>
              <span className="text-xs text-[var(--color-text-muted)]">
                {item.name}
                {item.exchange ? ` · ${item.exchange}` : ""}
                {item.country ? ` · ${item.country}` : ""}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default TickerSearch;
