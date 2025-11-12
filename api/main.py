import csv
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Import database module
from . import database

# Set up logging
log = logging.getLogger("main")
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Connects to MongoDB on startup, disconnects on shutdown.
    """
    log.info("API startup...")
    database.connect_to_mongo()
    if database.db is None:  # Check if connection was successful
        log.warning("Database connection failed. API is running in a degraded state.")
    else:
        log.info("API started and connected to database.")
    yield  # API is now running
    log.info("API shutdown...")
    database.close_mongo_connection()
    log.info("API shut down and disconnected from database.")


# Create the FastAPI app instance with the lifespan handler
app = FastAPI(
    title="ChronoStox API",
    description="Backend for the ChronoStox virtual trading platform.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This allows your React frontend (on http://localhost:5173)
# to make requests to this API (on http://localhost:8000).
origins = [
    "http://localhost:5173",  # React default dev port
    "http://localhost:3000",  # Common React port
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- Pydantic Models (Data Validation) ---

class Holding(BaseModel):
    ticker: str
    quantity: int
    avg_price: float = Field(..., alias="avgPrice")
    model_config = ConfigDict(populate_by_name=True)


class Portfolio(BaseModel):
    id: str = Field(..., alias="_id")
    user_id: str = Field(..., alias="userId")
    virtual_cash: float = Field(..., alias="virtualCash")
    holdings: List[Holding]
    model_config = ConfigDict(populate_by_name=True)


class TradeRequest(BaseModel):
    user_id: str = Field("college_project_user", alias="userId")  # Default user
    ticker: str
    quantity: int
    price: float
    action: str  # "BUY" or "SELL"
    model_config = ConfigDict(populate_by_name=True)


class TradeResponse(BaseModel):
    status: str
    message: str
    new_portfolio: Optional[Portfolio] = Field(None, alias="newPortfolio")
    model_config = ConfigDict(populate_by_name=True)


class StockData(BaseModel):
    ticker: str
    company_name: str = Field(..., alias="companyName")
    current_price: float = Field(..., alias="currentPrice")
    day_high: float = Field(..., alias="dayHigh")
    day_low: float = Field(..., alias="dayLow")
    previous_close: float = Field(..., alias="previousClose")
    market_cap: Optional[int] = Field(None, alias="marketCap")
    volume: Optional[int]
    long_summary: Optional[str] = Field(None, alias="longSummary")
    model_config = ConfigDict(populate_by_name=True)


class IndexHistoryPoint(BaseModel):
    date: datetime
    close: float
    model_config = ConfigDict(populate_by_name=True)


class IndexSummary(BaseModel):
    name: str
    ticker: str
    last_close: float = Field(..., alias="lastClose")
    change: float
    change_pct: float = Field(..., alias="changePct")
    history: List[IndexHistoryPoint]
    region: Optional[str] = None
    model_config = ConfigDict(populate_by_name=True)


class PriceHistoryPoint(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    model_config = ConfigDict(populate_by_name=True)


class TickerSummary(BaseModel):
    ticker: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    category: Optional[str] = Field(None, alias="category")
    country: Optional[str] = None
    model_config = ConfigDict(populate_by_name=True)


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for basic API health check."""
    return {"status": "ChronoStox API is running."}


@app.get("/portfolio/{user_id}", response_model=Portfolio, response_model_by_alias=True)
def get_user_portfolio(user_id: str):
    """
    Retrieves a user's portfolio or creates a new one.
    """
    log.info(f"Fetching portfolio for user: {user_id}")
    portfolio = database.get_portfolio(user_id)

    if portfolio is None:
        # This checks if the db connection failed
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Could not connect to the database."
        )

    # This checks if the user was found (which it always should be, as we create one)
    if not portfolio:
        raise HTTPException(status_code=404, detail="User portfolio not found.")

    return Portfolio.model_validate(portfolio)


@app.post("/trade", response_model=TradeResponse, response_model_by_alias=True)
def post_trade(trade: TradeRequest):
    """
    Executes a trade (BUY or SELL) for a user.
    """
    log.info(f"Processing trade for {trade.user_id}: {trade.action} {trade.quantity} {trade.ticker}")
    result = database.execute_trade(
        user_id=trade.user_id,
        ticker=trade.ticker,
        quantity=trade.quantity,
        price=trade.price,
        action=trade.action
    )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    if result.get("new_portfolio"):
        result["new_portfolio"] = Portfolio.model_validate(result["new_portfolio"])

    return TradeResponse.model_validate(result)


@app.get("/stock/{ticker}", response_model=StockData, response_model_by_alias=True)
def get_stock_data(ticker: str):
    """
    Gets live stock data from yfinance.
    """
    log.info(f"Fetching yfinance data for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get('regularMarketPrice') is None:
            # Try with ".NS" suffix for Indian stocks if not already present
            if not ticker.endswith(".NS"):
                log.info(f"No data for {ticker}, retrying with {ticker}.NS")
                stock = yf.Ticker(f"{ticker}.NS")
                info = stock.info
                ticker = f"{ticker}.NS"  # Update ticker for response

            if not info or info.get('regularMarketPrice') is None:
                raise HTTPException(status_code=404, detail=f"Stock data not found for ticker: {ticker}")

        # Map yfinance data to our StockData model
        data = StockData(
            ticker=ticker,
            companyName=info.get('longName', 'N/A'),
            currentPrice=info.get('regularMarketPrice', 0.0),
            dayHigh=info.get('dayHigh', 0.0),
            dayLow=info.get('dayLow', 0.0),
            previousClose=info.get('previousClose', 0.0),
            marketCap=info.get('marketCap'),
            volume=info.get('regularMarketVolume'),
            longSummary=info.get('longBusinessSummary')
        )
        return data

    except Exception as e:
        log.error(f"Error fetching yfinance data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {e}")


INDEX_TICKERS: List[Dict[str, str]] = [
    {"name": "NIFTY 50", "symbol": "^NSEI", "region": "India"},
    {"name": "SENSEX", "symbol": "^BSESN", "region": "India"},
    {"name": "NIFTY Bank", "symbol": "^NSEBANK", "region": "India"},
    {"name": "NIFTY IT", "symbol": "^CNXIT", "region": "India"},
    {"name": "S&P 500", "symbol": "^GSPC", "region": "United States"},
    {"name": "NASDAQ 100", "symbol": "^NDX", "region": "United States"},
    {"name": "Dow Jones", "symbol": "^DJI", "region": "United States"},
    {"name": "FTSE 100", "symbol": "^FTSE", "region": "United Kingdom"},
    {"name": "DAX", "symbol": "^GDAXI", "region": "Europe"},
    {"name": "CAC 40", "symbol": "^FCHI", "region": "Europe"},
    {"name": "Nikkei 225", "symbol": "^N225", "region": "Japan"},
    {"name": "TOPIX", "symbol": "^TOPX", "region": "Japan"},
    {"name": "Hang Seng", "symbol": "^HSI", "region": "Hong Kong"},
    {"name": "ASX 200", "symbol": "^AXJO", "region": "Australia"},
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKER_CSV_PATH = DATA_DIR / "ticker.csv"


def _load_ticker_catalog() -> List[Dict[str, Optional[str]]]:
    catalog: List[Dict[str, Optional[str]]] = []
    if not TICKER_CSV_PATH.exists():
        log.warning("Ticker CSV not found at %s", TICKER_CSV_PATH)
        return catalog

    try:
        with TICKER_CSV_PATH.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ticker = (row.get("Ticker") or "").strip()
                if not ticker:
                    continue
                catalog.append(
                    {
                        "ticker": ticker,
                        "name": (row.get("Name") or "").strip() or None,
                        "exchange": (row.get("Exchange") or "").strip() or None,
                        "category": (row.get("Category Name") or "").strip() or None,
                        "country": (row.get("Country") or "").strip() or None,
                    }
                )
    except Exception as exc:
        log.error("Failed to load ticker catalog: %s", exc)
        return []

    log.info("Loaded %s tickers from %s", len(catalog), TICKER_CSV_PATH)
    return catalog


TICKER_CATALOG: List[Dict[str, Optional[str]]] = _load_ticker_catalog()


@app.get("/indices", response_model=List[IndexSummary], response_model_by_alias=True)
def get_indices():
    summaries: List[IndexSummary] = []

    for entry in INDEX_TICKERS:
        name = entry["name"]
        ticker = entry["symbol"]
        region = entry.get("region")
        log.info(f"Fetching index data for {name} ({ticker})")
        try:
            history = yf.Ticker(ticker).history(period="1mo", interval="1d")
        except Exception as exc:
            log.error(f"Failed to fetch index {ticker}: {exc}")
            continue

        if history.empty or len(history) < 2:
            log.warning(f"Insufficient data for index {ticker}")
            continue

        if getattr(history.index, "tz", None) is not None:
            history.index = history.index.tz_localize(None)

        closes = history["Close"].dropna()
        if closes.empty or len(closes) < 2:
            continue

        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        change = last_close - prev_close
        change_pct = (change / prev_close * 100.0) if prev_close else 0.0

        history_points = [
            IndexHistoryPoint(date=index.to_pydatetime(), close=float(value))
            for index, value in closes.tail(30).items()
        ]

        summaries.append(
            IndexSummary(
                name=name,
                ticker=ticker,
                lastClose=last_close,
                change=change,
                changePct=change_pct,
                history=history_points,
                region=region,
            )
        )

    if not summaries:
        raise HTTPException(status_code=502, detail="Failed to fetch index data.")

    return summaries


@app.get(
    "/history/{ticker}",
    response_model=List[PriceHistoryPoint],
    response_model_by_alias=True,
)
def get_price_history(
    ticker: str,
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    period: str = Query("1mo"),
):
    """
    Provides OHLCV history for a ticker. Use date range or period (e.g. 1mo, 3mo).
    """
    log.info(f"Fetching history for {ticker} (start={start}, end={end}, period={period})")

    try:
        yf_ticker = yf.Ticker(ticker)
        if start is not None or end is not None:
            history = yf_ticker.history(
                start=start.isoformat() if start else None,
                end=end.isoformat() if end else None,
                interval="1d",
            )
        else:
            history = yf_ticker.history(period=period, interval="1d")
    except Exception as exc:
        log.error(f"Failed to fetch history for {ticker}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch price history.")

    if history.empty:
        raise HTTPException(status_code=404, detail="No history available for ticker.")

    if getattr(history.index, "tz", None) is not None:
        history.index = history.index.tz_localize(None)

    records: List[PriceHistoryPoint] = []
    for index, row in history.iterrows():
        try:
            volume_value: Optional[float] = None
            volume_raw = row.get("Volume")
            if volume_raw is not None and volume_raw == volume_raw:
                volume_value = float(volume_raw)

            records.append(
                PriceHistoryPoint(
                    date=index.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=volume_value,
                )
            )
        except Exception as exc:
            log.debug(f"Skipping malformed row for {ticker}: {exc}")

    if not records:
        raise HTTPException(status_code=404, detail="No valid history points found.")

    return records


@app.get(
    "/tickers",
    response_model=List[TickerSummary],
    response_model_by_alias=True,
)
def search_tickers(
    query: Optional[str] = Query(None, min_length=1, description="Filter by ticker, name, or sector."),
    limit: int = Query(25, ge=1, le=200),
):
    """
    Returns tickers sourced from data/ticker.csv. Optional fuzzy search with query tokens.
    """
    if not TICKER_CATALOG:
        log.warning("Ticker catalog unavailable.")
        return []

    catalog = TICKER_CATALOG

    if query:
        tokens = [token for token in query.lower().split() if token]
        if tokens:
            def matches(entry: Dict[str, Optional[str]]) -> bool:
                haystack = " ".join(
                    filter(
                        None,
                        [
                            entry.get("ticker", ""),
                            entry.get("name", ""),
                            entry.get("category", ""),
                            entry.get("exchange", ""),
                            entry.get("country", ""),
                        ],
                    )
                ).lower()
                return all(token in haystack for token in tokens)

            catalog = [entry for entry in catalog if matches(entry)]

    limited = catalog[:limit]
    return [TickerSummary.model_validate(entry) for entry in limited]