import React, { useMemo } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { useTheme } from "./context/ThemeContext.jsx";

const navigation = [
  { label: "Home", to: "/" },
  { label: "Portfolio", to: "/portfolio" },
  { label: "Live Market", to: "/market" },
  { label: "Trade Simulator", to: "/trade-simulator" },
  { label: "Stock Screener", to: "/screener" },
  { label: "Insights", to: "/insights" },
];

function App() {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  const activePath = useMemo(() => {
    if (location.pathname === "/") return "/";
    const [, firstSegment] = location.pathname.split("/");
    return firstSegment ? `/${firstSegment}` : "/";
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-[var(--color-surface)] text-[var(--color-text)] transition-colors">
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 pb-12 pt-6 md:px-8">
        <header className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface-elevated)] px-5 py-4 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.25em] text-[var(--color-text-muted)]">ChronoStox</p>
              <h1 className="mt-1 text-2xl font-semibold text-[var(--color-text)]">Trading Platform</h1>
            </div>
            <div className="flex flex-col gap-3 md:flex-row md:items-center">
              <nav className="flex flex-wrap gap-2 rounded-full bg-[var(--color-surface)] p-1">
                {navigation.map((item) => {
                  const isActive = activePath === item.to;
                  return (
                    <Link
                      key={item.to}
                      to={item.to}
                      className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                        isActive
                          ? "bg-[var(--color-accent)] text-white shadow-md"
                          : "text-[var(--color-text-muted)] hover:bg-[var(--color-accent-soft)] hover:text-[var(--color-text)]"
                      }`}
                    >
                      {item.label}
                    </Link>
                  );
                })}
              </nav>
              <button
                type="button"
                onClick={toggleTheme}
                className="flex h-10 w-10 items-center justify-center rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] text-xl transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]"
                aria-label="Toggle color theme"
              >
                {theme === "dark" ? "🌙" : "🌞"}
              </button>
            </div>
          </div>
        </header>

        <main className="mt-8 flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default App;
