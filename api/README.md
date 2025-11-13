# ChronoStox Backend API

FastAPI backend with SQLite database for ChronoStox trading platform.

## Setup

1. Install dependencies:
```bash
cd api
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --reload --port 8000
```

The database file (`chronostox.db`) will be created automatically in the `api/` directory on first run.

## API Endpoints

- `GET /` - Health check
- `GET /portfolio/{user_id}` - Get or create user portfolio
- `POST /trade` - Execute a trade (BUY/SELL)
- `GET /stock/{ticker}` - Get current stock data
- `GET /history/{ticker}` - Get historical stock data (supports `period`, `start`, `end` params)
- `GET /indices` - Get major market indices data
- `GET /tickers` - Get ticker list from local CSV

## Database Schema

- **users**: User portfolios with virtual cash
- **holdings**: User stock holdings (ticker, quantity, avg_price)
- **trades**: Trade history log

## Trade Request Format

```json
{
  "userId": "college_project_user",
  "ticker": "RELIANCE.NS",
  "quantity": 10,
  "action": "BUY",
  "price": 2800.0  // Optional, will fetch if not provided
}
```

