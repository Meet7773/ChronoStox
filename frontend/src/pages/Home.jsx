import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import Sparkline from "../components/Sparkline.jsx";

const API_URL = "http://127.0.0.1:8000";

const FEATURE_CARDS = [
  {
    title: "Connected Portfolio",
    body: "Place simulated orders, manage positions, and track P&L backed by a persistent FastAPI + MongoDB stack.",
    cta: { label: "Go to Portfolio", to: "/portfolio" },
  },
  {
    title: "Scenario Lab",
    body: "Replay historic regimes or build your own timeline, then stress-test strategies with realistic execution.",
    cta: { label: "Open Trade Simulator", to: "/trade-simulator" },
  },
  {
    title: "Market Radar",
    body: "Scan live quotes, screen equities via curated fundamentals, and monitor global macro signals in one view.",
    cta: { label: "Explore Live Tools", to: "/market" },
  },
];

function formatChange(changePct) {
  if (changePct === null || changePct === undefined) return "";
  const prefix = changePct >= 0 ? "+" : "-";
  const value = Math.abs(changePct).toFixed(2);
  return `${prefix}${value}%`;
}

function Home() {
  const [indices, setIndices] = useState([]);
  const [status, setStatus] = useState({ message: "Fetching global indices…", tone: "muted", loading: true });

  useEffect(() => {
    let active = true;
    async function loadIndices() {
      setStatus({ message: "Fetching global indices…", tone: "muted", loading: true });
      try {
        const response = await axios.get(`${API_URL}/indices`);
        if (!active) return;
        setIndices(response.data);
        setStatus({ message: "", tone: "muted", loading: false });
      } catch (error) {
        if (!active) return;
        const detail = error.response?.data?.detail ?? "Unable to load index data.";
        setStatus({ message: detail, tone: "danger", loading: false });
      }
    }
    loadIndices();
    return () => {
      active = false;
    };
  }, []);

  const indicesByRegion = useMemo(() => {
    return indices.reduce((acc, item) => {
      const region = item.region || "Global";
      if (!acc[region]) {
        acc[region] = [];
      }
      acc[region].push(item);
      return acc;
    }, {});
  }, [indices]);

  return (
    <div className="space-y-12">
      <section className="relative overflow-hidden rounded-3xl border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-8 py-10 shadow-2xl">
        <div className="absolute inset-0 -z-10 bg-gradient-to-br from-[var(--color-accent-soft)] via-transparent to-transparent" aria-hidden />
        <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
          <div className="space-y-6">
            <span className="inline-flex items-center rounded-full bg-[var(--color-accent-soft)] px-4 py-1 text-sm font-medium text-[var(--color-accent)]">
              ChronoStox 2.0
            </span>
            <h1 className="text-4xl font-semibold leading-tight text-[var(--color-text)] sm:text-5xl">
              Master markets with a unified virtual trading workspace.
            </h1>
            <p className="text-lg text-[var(--color-text-muted)]">
              Pull live data, rehearse strategies from past crises, and manage a simulated portfolio built on FastAPI, React, and MongoDB.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                to="/portfolio"
                className="rounded-xl bg-[var(--color-accent)] px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-500/30 transition hover:bg-[var(--color-accent)]/90"
              >
                Launch Portfolio
              </Link>
              <Link
                to="/trade-simulator"
                className="rounded-xl border border-[var(--color-border)] px-5 py-3 text-sm font-semibold text-[var(--color-text)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
              >
                Try Trade Simulator
              </Link>
            </div>
            <div className="flex flex-col gap-2 text-sm text-[var(--color-text-muted)] sm:flex-row sm:items-center">
              <span>Live market data · Persistent portfolios · Global coverage</span>
            </div>
          </div>
          <div className="h-full rounded-3xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-inner">
            <h2 className="text-sm font-semibold uppercase tracking-[0.25em] text-[var(--color-text-muted)]">Quick Pulse</h2>
            {status.message && (
              <div
                className={`mt-3 rounded-lg border px-3 py-2 text-xs ${
                  status.tone === "danger"
                    ? "border-red-200 bg-red-50 text-red-600"
                    : "border-[var(--color-border)] bg-[var(--color-surface-elevated)] text-[var(--color-text-muted)]"
                }`}
              >
                {status.message}
              </div>
            )}
            <div className="mt-5 grid gap-4">
              {indices.slice(0, 4).map((index) => (
                <div key={index.ticker} className="flex items-center justify-between rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-4 py-3">
                  <div>
                    <p className="text-xs uppercase text-[var(--color-text-muted)]">{index.region}</p>
                    <p className="text-base font-semibold text-[var(--color-text)]">{index.name}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-semibold text-[var(--color-text)]">{index.lastClose.toFixed(2)}</p>
                    <p
                      className={`text-xs font-medium ${
                        index.changePct >= 0 ? "text-[var(--color-success)]" : "text-red-500"
                      }`}
                    >
                      {formatChange(index.changePct)}
                    </p>
                  </div>
                </div>
              ))}
              {!indices.length && !status.loading && (
                <p className="text-sm text-[var(--color-text-muted)]">No live data available. Start the API and refresh.</p>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="space-y-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-[var(--color-text)]">Global Indices Dashboard</h2>
            <p className="text-sm text-[var(--color-text-muted)]">Live snapshots across India, the US, Europe, and Asia-Pacific.</p>
          </div>
          <Link
            to="/insights"
            className="rounded-lg border border-[var(--color-border)] px-4 py-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
          >
            Open Insights
          </Link>
        </div>
        <div className="grid gap-6 lg:grid-cols-2">
          {Object.entries(indicesByRegion).map(([region, items]) => (
            <article key={region} className="card overflow-hidden">
              <div className="border-b border-[var(--color-border)] bg-[var(--color-surface-muted)]/40 px-6 py-4">
                <p className="text-sm font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">{region}</p>
              </div>
              <div className="space-y-4 p-6">
                {items.map((index) => (
                  <div key={index.ticker} className="grid gap-4 md:grid-cols-[2fr,1fr] md:items-center">
                    <div>
                      <p className="text-sm font-medium text-[var(--color-text)]">{index.name}</p>
                      <p className="text-xs uppercase text-[var(--color-text-muted)]">{index.ticker}</p>
                      <div className="mt-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm">
                        <span className="font-semibold text-[var(--color-text)]">{index.lastClose.toFixed(2)}</span>
                        <span
                          className={`ml-2 font-medium ${
                            index.changePct >= 0 ? "text-[var(--color-success)]" : "text-red-500"
                          }`}
                        >
                          {formatChange(index.changePct)}
                        </span>
                      </div>
                    </div>
                    <Sparkline data={index.history} accessor={(point) => point.close} />
                  </div>
                ))}
              </div>
            </article>
          ))}
          {!indices.length && status.loading && (
            <div className="card p-6 text-sm text-[var(--color-text-muted)]">Loading market overview…</div>
          )}
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-3">
        {FEATURE_CARDS.map((feature) => (
          <article key={feature.title} className="card flex flex-col justify-between p-6">
            <div className="space-y-3">
              <h3 className="text-xl font-semibold text-[var(--color-text)]">{feature.title}</h3>
              <p className="text-sm text-[var(--color-text-muted)]">{feature.body}</p>
            </div>
            <Link
              to={feature.cta.to}
              className="mt-6 inline-flex items-center text-sm font-semibold text-[var(--color-accent)] hover:text-[var(--color-accent)]/80"
            >
              {feature.cta.label}
              <span className="ml-2">→</span>
            </Link>
          </article>
        ))}
      </section>
    </div>
  );
}

export default Home;
