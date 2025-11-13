"""
FastAPI backend for ChronoStox.
Provides REST API endpoints for portfolio management, stock data, and trading.
"""
import os
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext

from database import (
    init_db,
    get_portfolio,
    execute_trade,
    create_user,
    authenticate_user,
    get_simulation_portfolio,
    execute_simulation_trade,
    reset_simulation,
    get_simulation_trades,
)

# Setup logging
log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')

# Initialize FastAPI app
app = FastAPI(title="ChronoStox API", version="2.0.0")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    log.info("ChronoStox API started")


# --- Pydantic Models ---
class TradeRequest(BaseModel):
    userId: str
    ticker: str
    quantity: int
    action: str  # "BUY" or "SELL"
    price: Optional[float] = None  # Optional, will fetch if not provided


class AuthRequest(BaseModel):
    userId: str
    password: str


class SimulationTradeRequest(BaseModel):
    userId: str
    ticker: str
    quantity: int
    price: float
    action: str
    scenario: Optional[str] = None


# --- Helper functions ---
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return pwd_context.verify(password, stored_hash)
    except Exception:
        return False


# --- API Endpoints ---

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "ChronoStox API is running"}


@app.post("/auth/signup")
def signup(request: AuthRequest):
    """Registers a new user."""
    if len(request.userId.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if len(request.password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password must be 72 characters or fewer."
        )

    password_hash = hash_password(request.password)
    result = create_user(request.userId.strip(), password_hash)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    portfolio = get_portfolio(request.userId.strip())
    simulation = get_simulation_portfolio(request.userId.strip())
    return {
        "status": "success",
        "message": "Signup successful.",
        "portfolio": portfolio,
        "simulation": simulation,
    }


@app.post("/auth/login")
def login(request: AuthRequest):
    """Authenticates a user and returns portfolio information."""
    stored_hash = authenticate_user(request.userId)
    if not stored_hash or not verify_password(request.password, stored_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    portfolio = get_portfolio(request.userId)
    if not portfolio:
        raise HTTPException(status_code=404, detail="User portfolio not found.")

    simulation = get_simulation_portfolio(request.userId)
    return {
        "status": "success",
        "message": "Login successful.",
        "portfolio": portfolio,
        "simulation": simulation,
    }


@app.get("/portfolio/{user_id}")
def get_user_portfolio(user_id: str):
    """
    Fetches an existing user's portfolio.
    """
    try:
        portfolio = get_portfolio(user_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="User not found.")
        return portfolio
    except Exception as e:
        log.error(f"Error fetching portfolio for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trade")
def execute_trade_endpoint(trade: TradeRequest):
    """
    Executes a buy or sell trade.
    If price is not provided, fetches current market price from yfinance.
    """
    try:
        # Ensure user exists
        if not get_portfolio(trade.userId):
            raise HTTPException(status_code=404, detail="User not found.")

        # If price not provided, fetch current price
        if trade.price is None:
            try:
                ticker_obj = yf.Ticker(trade.ticker)
                info = ticker_obj.info
                current_price = info.get("regularMarketPrice") or info.get("currentPrice")
                if not current_price:
                    # Fallback: try to get from history
                    hist = ticker_obj.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist["Close"].iloc[-1])
                    else:
                        raise HTTPException(status_code=400, detail=f"Could not fetch price for {trade.ticker}")
            except Exception as e:
                log.error(f"Error fetching price for {trade.ticker}: {e}")
                raise HTTPException(status_code=400, detail=f"Could not fetch price for {trade.ticker}: {str(e)}")
        else:
            current_price = trade.price
        
        # Execute trade
        result = execute_trade(
            user_id=trade.userId,
            ticker=trade.ticker.upper(),
            quantity=trade.quantity,
            price=current_price,
            action=trade.action
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{ticker}")
def get_stock_data(ticker: str):
    """
    Fetches current stock data from yfinance.
    Returns current price, company info, and basic metrics.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Get current price
        current_price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        
        if not current_price:
            # Fallback: try history
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
            else:
                raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        # Get previous close for change calculation
        previous_close = info.get("previousClose") or current_price
        change = current_price - previous_close
        change_pct = (change / previous_close * 100) if previous_close else 0
        
        return {
            "ticker": ticker.upper(),
            "currentPrice": float(current_price),
            "previousClose": float(previous_close),
            "change": float(change),
            "changePct": float(change_pct),
            "name": info.get("longName") or info.get("shortName") or ticker,
            "marketCap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "volume": info.get("volume"),
            "avgVolume": info.get("averageVolume"),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching stock data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")


@app.get("/history/{ticker}")
def get_stock_history(
    ticker: str,
    period: str = Query(default="1mo", description="Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"),
    start: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
):
    """
    Fetches historical stock data from yfinance.
    Returns OHLCV data as list of dicts.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Fetch history
        if start and end:
            hist = ticker_obj.history(start=start, end=end, auto_adjust=False)
        else:
            hist = ticker_obj.history(period=period, auto_adjust=False)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {ticker}")
        
        # Convert to list of dicts
        hist.reset_index(inplace=True)
        hist["time"] = hist["Date"].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
        
        result = []
        for _, row in hist.iterrows():
            result.append({
                "time": row["time"],
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
            })
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching history for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@app.get("/simulation/portfolio/{user_id}")
def get_simulation(user_id: str):
    """Returns the simulation portfolio for a user."""
    try:
        # Ensure real user exists before simulation state
        if not get_portfolio(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        return get_simulation_portfolio(user_id)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching simulation portfolio for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/trade")
def simulation_trade(request: SimulationTradeRequest):
    """Executes a simulation trade."""
    try:
        if request.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive.")
        if request.price <= 0:
            raise HTTPException(status_code=400, detail="Price must be positive.")

        # Ensure user exists
        if not get_portfolio(request.userId):
            raise HTTPException(status_code=404, detail="User not found.")

        result = execute_simulation_trade(
            user_id=request.userId,
            ticker=request.ticker.upper(),
            quantity=request.quantity,
            price=request.price,
            action=request.action,
            scenario=request.scenario,
        )

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error executing simulation trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/reset/{user_id}")
def simulation_reset(user_id: str):
    """Resets a user's simulation data."""
    try:
        if not get_portfolio(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        result = reset_simulation(user_id)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        # Return fresh portfolio
        return get_simulation_portfolio(user_id)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error resetting simulation for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulation/trades/{user_id}")
def simulation_trades(user_id: str):
    """Returns simulation trade history."""
    try:
        if not get_portfolio(user_id):
            raise HTTPException(status_code=404, detail="User not found.")
        return get_simulation_trades(user_id)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching simulation trades for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indices")
def get_indices():
    """
    Fetches data for major market indices.
    Returns list of indices with current values and changes.
    """
    try:
        # Major indices to track
        indices_config = [
            {"ticker": "^GSPC", "name": "S&P 500", "region": "US"},
            {"ticker": "^DJI", "name": "Dow Jones", "region": "US"},
            {"ticker": "^IXIC", "name": "NASDAQ", "region": "US"},
            {"ticker": "^NSEI", "name": "Nifty 50", "region": "India"},
            {"ticker": "^BSESN", "name": "BSE Sensex", "region": "India"},
            {"ticker": "^FTSE", "name": "FTSE 100", "region": "UK"},
            {"ticker": "^GDAXI", "name": "DAX", "region": "Germany"},
            {"ticker": "^FCHI", "name": "CAC 40", "region": "France"},
            {"ticker": "^N225", "name": "Nikkei 225", "region": "Japan"},
        ]
        
        results = []
        for idx_config in indices_config:
            try:
                ticker_obj = yf.Ticker(idx_config["ticker"])
                info = ticker_obj.info
                
                # Get current and previous close
                current = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
                previous = info.get("previousClose") or current
                
                if not current:
                    # Fallback to history
                    hist = ticker_obj.history(period="5d")
                    if not hist.empty:
                        current = float(hist["Close"].iloc[-1])
                        previous = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
                    else:
                        continue
                
                change = current - previous
                change_pct = (change / previous * 100) if previous else 0
                
                # Get history for sparkline (last 30 days)
                hist = ticker_obj.history(period="1mo")
                history_list = []
                if not hist.empty:
                    hist.reset_index(inplace=True)
                    for _, row in hist.iterrows():
                        history_list.append({
                            "time": row["Date"].isoformat() if hasattr(row["Date"], 'isoformat') else str(row["Date"]),
                            "close": float(row["Close"]),
                        })
                
                results.append({
                    "ticker": idx_config["ticker"],
                    "name": idx_config["name"],
                    "region": idx_config["region"],
                    "lastClose": float(current),
                    "previousClose": float(previous),
                    "change": float(change),
                    "changePct": float(change_pct),
                    "history": history_list,
                })
            except Exception as e:
                log.warning(f"Failed to fetch data for {idx_config['ticker']}: {e}")
                continue
        
        return results
    except Exception as e:
        log.error(f"Error fetching indices: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching indices: {str(e)}")


@app.get("/tickers")
def get_tickers():
    """
    Serves ticker data from the local data/ticker.csv file.
    Returns list of tickers with name, category, etc.
    """
    try:
        # Path to ticker CSV (relative to project root)
        csv_path = Path(__file__).resolve().parent.parent / "data" / "ticker.csv"
        
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="ticker.csv file not found")
        
        df = pd.read_csv(csv_path)
        
        # Convert to list of dicts
        results = []
        for _, row in df.iterrows():
            result_dict = {}
            for col in df.columns:
                value = row[col]
                # Convert to native Python types
                if pd.isna(value):
                    result_dict[col] = None
                elif isinstance(value, (int, float)):
                    result_dict[col] = float(value) if isinstance(value, float) else int(value)
                else:
                    result_dict[col] = str(value)
            results.append(result_dict)
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error reading ticker CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading ticker data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
