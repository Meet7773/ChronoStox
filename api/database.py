import os
import ssl
import certifi
import logging
import urllib.parse
import datetime
from pathlib import Path
from typing import Optional
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

# Set up logging
log = logging.getLogger("database")
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')

# Load environment variables from .env file (prioritize local api/.env)
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=False)
load_dotenv(override=False)

# Global client variable
client: Optional[MongoClient] = None
db = None
users_collection = None
trades_collection = None


def connect_to_mongo():
    """
    Establishes a connection to MongoDB Atlas using credentials from .env
    and sets the global client and collection variables.
    """
    global client, db, users_collection, trades_collection

    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASS = os.getenv("MONGO_PASS")
    MONGO_HOST = os.getenv("MONGO_HOST")

    if not all([MONGO_USER, MONGO_PASS, MONGO_HOST]):
        log.fatal("Missing MongoDB credentials in .env file.")
        return

    try:
        # URL-encode the username and password
        safe_user = urllib.parse.quote_plus(MONGO_USER)
        safe_pass = urllib.parse.quote_plus(MONGO_PASS)

        # Build the connection string
        MONGODB_URI = f"mongodb+srv://{safe_user}:{safe_pass}@{MONGO_HOST}/?retryWrites=true&w=majority&appName=Cluster0"

        # Create the client, explicitly using certifi for SSL
        client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsCAFile=certifi.where()
        )

        # Test the connection
        client.admin.command('ping')

        # Set global database and collection objects
        db = client["chronostox_db"]
        users_collection = db["users"]
        trades_collection = db["trades"]

        log.info("MongoDB connection successful.")

    except (ConnectionFailure, OperationFailure) as e:
        log.fatal(f"Error connecting to MongoDB. API will not be able to access database.\nError Details: {e}")
        client = None
    except Exception as e:
        log.fatal(f"An unexpected error occurred during MongoDB connection: {e}")
        client = None


def close_mongo_connection():
    """Closes the MongoDB connection."""
    global client
    if client:
        client.close()
        log.info("MongoDB connection closed.")


def get_portfolio(user_id: str):
    """
    Fetches or creates a user's portfolio.
    """
    # Use 'is not None' for truth-value testing on collections
    if users_collection is None:
        log.error("get_portfolio failed: users_collection is not initialized.")
        return None

    try:
        # Find the user's portfolio
        portfolio = users_collection.find_one({"user_id": user_id})

        if portfolio:
            # Convert MongoDB's _id to a string
            portfolio["_id"] = str(portfolio["_id"])
            return portfolio
        else:
            # User not found, create a new portfolio
            new_portfolio = {
                "user_id": user_id,
                "virtual_cash": 100000.00,
                "holdings": []  # Stores list of {"ticker": "XYZ", "quantity": 10, "avg_price": 150.0}
            }
            result = users_collection.insert_one(new_portfolio)
            new_portfolio["_id"] = str(result.inserted_id)
            log.info(f"Created new portfolio for user: {user_id}")
            return new_portfolio

    except Exception as e:
        log.error(f"Error in get_portfolio for {user_id}: {e}")
        return None


def execute_trade(user_id: str, ticker: str, quantity: int, price: float, action: str):
    """
    Executes a buy or sell trade and updates the user's portfolio.
    """
    if users_collection is None or trades_collection is None:
        log.error("execute_trade failed: collections are not initialized.")
        return {"status": "error", "message": "Database not connected."}

    try:
        # Get the user's current portfolio
        portfolio = users_collection.find_one({"user_id": user_id})
        if not portfolio:
            return {"status": "error", "message": "User not found."}

        virtual_cash = portfolio.get("virtual_cash", 0.0)
        holdings = portfolio.get("holdings", [])

        trade_value = quantity * price
        new_holding = True

        if action.upper() == "BUY":
            # Check for sufficient funds
            if virtual_cash < trade_value:
                return {"status": "error", "message": "Insufficient funds."}

            # Update virtual cash
            virtual_cash -= trade_value

            # Update holdings
            for holding in holdings:
                if holding["ticker"] == ticker:
                    new_quantity = holding["quantity"] + quantity
                    new_avg_price = ((holding["avg_price"] * holding["quantity"]) + trade_value) / new_quantity
                    holding["quantity"] = new_quantity
                    holding["avg_price"] = new_avg_price
                    new_holding = False
                    break

            if new_holding:
                holdings.append({"ticker": ticker, "quantity": quantity, "avg_price": price})

        elif action.upper() == "SELL":
            # Check if user owns the stock
            found_holding = None
            for holding in holdings:
                if holding["ticker"] == ticker:
                    found_holding = holding
                    break

            if not found_holding:
                return {"status": "error", "message": "Stock not found in portfolio."}

            # Check for sufficient quantity
            if found_holding["quantity"] < quantity:
                return {"status": "error", "message": "Insufficient quantity to sell."}

            # Update virtual cash
            virtual_cash += trade_value

            # Update holdings
            found_holding["quantity"] -= quantity

            # Remove holding if quantity is zero
            if found_holding["quantity"] == 0:
                holdings = [h for h in holdings if h["ticker"] != ticker]

        else:
            return {"status": "error", "message": "Invalid action."}

        # --- Database Transaction ---
        # 1. Update the user's portfolio
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"virtual_cash": virtual_cash, "holdings": holdings}}
        )

        # 2. Log the trade
        trade_record = {
            "user_id": user_id,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "action": action.upper(),
            "trade_value": trade_value,
            "timestamp": datetime.datetime.now(datetime.UTC)
        }
        trades_collection.insert_one(trade_record)
        # --- End Transaction ---

        log.info(f"Trade executed for {user_id}: {action} {quantity} {ticker} @ {price}")
        return {"status": "success", "message": f"Trade executed.", "new_portfolio": get_portfolio(user_id)}

    except Exception as e:
        log.error(f"Error in execute_trade for {user_id}: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}