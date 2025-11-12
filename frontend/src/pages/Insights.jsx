import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import Sparkline from "../components/Sparkline.jsx";

const API_URL = "http://127.0.0.1:8000";

const PLAYBOOK = [
  {
    title: "Sector Rotation",
    description: "Track relative strength across IT, banking, and industrials to time rebalancing.",
  },
  {
    title: "Event Risk",
    description: "Mark regulatory announcements and earnings to capture volatility regime shifts.",
  },
  {
    title: "Macro Watch",
    description: "Keep an eye on bond yields and crude oil to contextualise equity moves.",
  },
];

function formatChange(value) {
  if (!Number.isFinite(value)) return "0.00%";
  const prefix = value >= 0 ? "+" : "";
  return `${prefix}${value.toFixed(2)}%`;
}

function Insights() {
  const [indices, setIndices] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    async function loadIndices() {
      setIsLoading(true);
      setError(null);
      try {
        const response = await axios.get(`${API_URL}/indices`);
        if (!isMounted) return;
        setIndices(response.data);
      } catch (err) {
        if (!isMounted) return;
        setError(err.response?.data?.detail ?? "Unable to retrieve index data.");
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    loadIndices();
    return () => {
      isMounted = false;
    };
  }, []);

  const groupedIndices = useMemo(() => {
    return indices.reduce((acc, index) => {
      const region = index.region || "Global";
      if (!acc[region]) {
        acc[region] = [];
      }
      acc[region].push(index);
      return acc;
    }, {});
  }, [indices]);

  return (
    <section className="space-y-8">
      <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-3xl font-semibold text-[var(--color-text)]">Insights Dashboard</h2>
          <p className="text-[var(--color-text-muted)]">
            Daily pulse of the market with index heatmaps, strategy notes, and actionable themes.
          </p>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-12">
        <div className="card lg:col-span-8">
          <div className="p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold text-[var(--color-text)]">Key Indices</h3>
              {isLoading && <span className="text-xs text-[var(--color-text-muted)]">Updating…</span>}
            </div>
            {error ? (
              <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                {error}
              </div>
            ) : (
              <div className="mt-6 space-y-6">
                {Object.entries(groupedIndices).map(([region, items]) => (
                  <article key={region} className="rounded-3xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-sm">
                    <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-4">
                      <h4 className="text-sm font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
                        {region}
                      </h4>
                      <span className="text-xs text-[var(--color-text-muted)]">{items.length} indices</span>
                    </div>
                    <div className="mt-4 grid gap-4 md:grid-cols-2">
                      {items.map((index) => (
                        <div
                          key={index.ticker}
                          className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface-elevated)] p-4 transition hover:border-[var(--color-accent)]"
                        >
                          <div className="flex items-start justify-between">
                            <div>
                              <p className="text-xs uppercase text-[var(--color-text-muted)]">{index.ticker}</p>
                              <h5 className="text-lg font-semibold text-[var(--color-text)]">{index.name}</h5>
                            </div>
                            <div
                              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                                index.changePct >= 0
                                  ? "bg-green-100 text-green-600"
                                  : "bg-red-100 text-red-600"
                              }`}
                            >
                              {formatChange(index.changePct)}
                            </div>
                          </div>
                          <p className="mt-2 text-sm text-[var(--color-text-muted)]">
                            Last close{" "}
                            <span className="font-medium text-[var(--color-text)]">{index.lastClose.toFixed(2)}</span>
                          </p>
                          <div className="mt-4">
                            <Sparkline data={index.history} accessor={(point) => point.close} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </article>
                ))}
                {!indices.length && !isLoading && (
                  <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 text-center text-sm text-[var(--color-text-muted)]">
                    No index data available. Try refreshing once the API is online.
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <aside className="card lg:col-span-4">
          <div className="space-y-6 p-6">
            <div>
              <h3 className="text-lg font-semibold text-[var(--color-text)]">Strategy Playbook</h3>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                Reference notes to guide risk positioning and watchlist reviews.
              </p>
            </div>
            <ul className="space-y-4">
              {PLAYBOOK.map((item) => (
                <li key={item.title} className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
                  <p className="text-sm font-semibold text-[var(--color-text)]">{item.title}</p>
                  <p className="mt-1 text-sm text-[var(--color-text-muted)]">{item.description}</p>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </div>

      <div className="card">
        <div className="p-6">
          <h3 className="text-xl font-semibold text-[var(--color-text)]">Daily Briefing</h3>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
              <p className="text-xs uppercase text-[var(--color-text-muted)]">Flows</p>
              <p className="mt-2 text-lg font-semibold text-[var(--color-text)]">FII net buy ₹1,240 Cr</p>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">Broad-based buying across banking and oil & gas.</p>
            </div>
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
              <p className="text-xs uppercase text-[var(--color-text-muted)]">Earnings</p>
              <p className="mt-2 text-lg font-semibold text-[var(--color-text)]">ACC, LT, and Kotak report today</p>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">Watch guidance on margin normalisation for cement.</p>
            </div>
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
              <p className="text-xs uppercase text-[var(--color-text-muted)]">Global Cues</p>
              <p className="mt-2 text-lg font-semibold text-[var(--color-text)]">US futures modestly higher</p>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">Dollar index eases while Brent holds at $82/bbl.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Insights;

