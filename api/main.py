import logging
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Import database functions
from database import get_portfolio, execute_trade, connect_to_mongo, close_mongo_connection, db

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
    connect_to_mongo()
    if db is None:  # Check if connection was successful
        log.warning("Database connection failed. API is running in a degraded state.")
    else:
        log.info("API started and connected to database.")
    yield  # API is now running
    log.info("API shutdown...")
    close_mongo_connection()
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


class Portfolio(BaseModel):
    id: str = Field(..., alias="_id")
    user_id: str = Field(..., alias="userId")
    virtual_cash: float = Field(..., alias="virtualCash")
    holdings: List[Holding]


class TradeRequest(BaseModel):
    user_id: str = Field("college_project_user", alias="userId")  # Default user
    ticker: str
    quantity: int
    price: float
    action: str  # "BUY" or "SELL"


class TradeResponse(BaseModel):
    status: str
    message: str
    new_portfolio: Optional[Portfolio] = Field(None, alias="newPortfolio")


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


# --- API Endpoints ---

@app.get("/")
def read_root():
    """Root endpoint for basic API health check."""
    return {"status": "ChronoStox API is running."}


@app.get("/portfolio/{user_id}", response_model=Portfolio, response_model_by_alias=False)
def get_user_portfolio(user_id: str):
    """
    Retrieves a user's portfolio or creates a new one.
    """
    log.info(f"Fetching portfolio for user: {user_id}")
    portfolio = get_portfolio(user_id)

    if portfolio is None:
        # This checks if the db connection failed
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Could not connect to the database."
        )

    # This checks if the user was found (which it always should be, as we create one)
    if not portfolio:
        raise HTTPException(status_code=404, detail="User portfolio not found.")

    return portfolio


@app.post("/trade", response_model=TradeResponse, response_model_by_alias=False)
def post_trade(trade: TradeRequest):
    """
    Executes a trade (BUY or SELL) for a user.
    """
    log.info(f"Processing trade for {trade.user_id}: {trade.action} {trade.quantity} {trade.ticker}")
    result = execute_trade(
        user_id=trade.user_id,
        ticker=trade.ticker,
        quantity=trade.quantity,
        price=trade.price,
        action=trade.action
    )

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@app.get("/stock/{ticker}", response_model=StockData, response_model_by_alias=False)
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
                raise HTTPException(status_code=4404, detail=f"Stock data not found for ticker: {ticker}")

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