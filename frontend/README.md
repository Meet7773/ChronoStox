# ChronoStox Frontend

React + Vite single-page interface for the ChronoStox trading simulator. It interacts with the FastAPI backend (`http://127.0.0.1:8000`) to retrieve portfolio data, execute trades, simulate historic scenarios, and visualise market insights.

## Available Scripts

```bash
npm run dev       # start Vite dev server (default http://127.0.0.1:5173)
npm run build     # production build
npm run preview   # preview the production build locally
npm run lint      # run ESLint checks
```

## Styling

Tailwind CSS plus a custom light/dark design system (defined in `tailwind.config.js` and `src/index.css`) provide the visual foundation. Toggle the theme from the header or adjust the design tokens to suit your brand.

## App Structure

- `src/main.jsx` – React entry point with router configuration.
- `src/App.jsx` – Layout shell, navigation, and theme toggle.
- `src/context/ThemeContext.jsx` – Light/dark state management.
- `src/components/Sparkline.jsx` – Reusable mini chart for trends.
- `src/pages/Portfolio.jsx` – Portfolio dashboard and trading ticket.
- `src/pages/LiveMarket.jsx` – Live ticker lookup backed by `/stock/{ticker}`.
- `src/pages/TradeSimulator.jsx` – Scenario-based replay with trade execution.
- `src/pages/StockScreener.jsx` – Quick filters over sample large-cap coverage.
- `src/pages/Insights.jsx` – Index overviews and strategy notes from `/indices`.

Add further pages by extending the router inside `src/main.jsx`.
