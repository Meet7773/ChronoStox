# ChronoStox Web Platform

ChronoStox is a virtual trading platform that simulates end-to-end equity trading. The web platform replaces the earlier Streamlit prototype with a production-style three-tier stack: React + Tailwind for the UI, FastAPI for business logic, and MongoDB Atlas for persistent storage.

This repository contains everything required to spin up the web tier locally, run the REST API, and connect to a MongoDB Atlas cluster. The machine learning layer (LSTM predictors) can plug into the FastAPI service when ready via additional endpoints.

## Architecture

- **Frontend** – `frontend/` – Vite + React + Tailwind CSS SPA.
- **Backend** – `api/` – FastAPI service with MongoDB atlas integration.
- **Database** – MongoDB Atlas (shared via connection string in environment variables).

```
┌────────────┐      HTTPS       ┌────────────┐       MongoDB Atlas
│  React SPA │  ─────────────▶  │  FastAPI   │  ──▶  (users, trades)
└────────────┘     REST         └────────────┘
```

## Prerequisites

- Node.js ≥ 20.x with npm
- Python ≥ 3.11
- MongoDB Atlas cluster (free tier works perfectly)

## Backend Setup (FastAPI)

```bash
cd api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `api/.env` (copy `api/env.example`) and fill in your Atlas credentials:

```env
MONGO_USER=your_mongo_username
MONGO_PASS=your_mongo_password
MONGO_HOST=your-cluster.mongodb.net
```

Run the API:

```bash
uvicorn api.main:app --reload
```

The service listens on `http://127.0.0.1:8000`. Swagger UI is available at `/docs`.

## Frontend Setup (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Vite will start the development server at `http://127.0.0.1:5173`. The SPA automatically talks to the FastAPI service running on port 8000.

## Key Features

- **Portfolio Dashboard** – Shows virtual cash, holdings, and provides a trade ticket to execute BUY/SELL orders.
- **Trade Execution Loop** – Validates orders, updates MongoDB, and logs every trade in an immutable ledger.
- **Live Market Lookup** – Proxies `yfinance` to fetch real-time quotes and company data for any ticker.
- **Terminal UI** – Custom “hacker terminal” aesthetic built with Tailwind utilities.

## API Overview

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Health check. |
| `/portfolio/{userId}` | GET | Fetches or creates the user’s portfolio. |
| `/trade` | POST | Executes BUY or SELL orders and logs trades. |
| `/stock/{ticker}` | GET | Returns live market data for the requested ticker. |

Sample trade payload:

```json
{
  "userId": "college_project_user",
  "ticker": "RELIANCE.NS",
  "quantity": 10,
  "price": 2800,
  "action": "BUY"
}
```

## Development Tips

- Run `npm run lint` in `frontend/` and use FastAPI’s built-in validation to keep responses clean.
- MongoDB automatically seeds a portfolio with ₹100,000 virtual cash for new users.
- The frontend expects camelCase fields (e.g., `virtualCash`, `avgPrice`) which the backend now emits via response aliases.
- For production builds, configure CORS origins and environment variables to match your hosting setup.

## Future Enhancements

- User authentication and role-based access.
- Integration of ML/LSTM prediction endpoints.
- Historical charting and performance analytics.
- Notifications for executed trades and market events.

ChronoStox provides a solid base for experimentation with algorithmic trading strategies, portfolio analytics, and full-stack deployment pipelines. Happy hacking!

