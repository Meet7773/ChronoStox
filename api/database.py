"""
Database module using SQLModel with SQLite.
Defines User and Holding models and database initialization.
"""
import logging
from pathlib import Path
from typing import Optional, List
from sqlmodel import SQLModel, Field, Session, create_engine, select
from datetime import datetime, timezone

log = logging.getLogger("database")
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')

# Database file path
DB_PATH = Path(__file__).resolve().parent / "chronostox.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


class User(SQLModel, table=True):
    """User portfolio model."""
    __tablename__ = "users"
    
    user_id: str = Field(primary_key=True, index=True)
    virtual_cash: float = Field(default=100000.00)
    password_hash: str = Field(default="")
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class Holding(SQLModel, table=True):
    """Holding model for user's stock positions."""
    __tablename__ = "holdings"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.user_id")
    ticker: str = Field(index=True)
    quantity: int = Field(default=0)
    avg_price: float = Field(default=0.0)
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trade(SQLModel, table=True):
    """Trade log model."""
    __tablename__ = "trades"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.user_id")
    ticker: str = Field(index=True)
    quantity: int
    price: float
    action: str  # "BUY" or "SELL"
    trade_value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SimulationState(SQLModel, table=True):
    """Simulation portfolio summary per user."""
    __tablename__ = "simulation_state"

    user_id: str = Field(primary_key=True, index=True)
    virtual_cash: float = Field(default=100000.00)
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class SimulationHolding(SQLModel, table=True):
    """Simulation holdings per user."""
    __tablename__ = "simulation_holdings"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, foreign_key="simulation_state.user_id")
    ticker: str = Field(index=True)
    quantity: int = Field(default=0)
    avg_price: float = Field(default=0.0)
    scenario: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))


class SimulationTrade(SQLModel, table=True):
    """Simulation trade log."""
    __tablename__ = "simulation_trades"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, foreign_key="simulation_state.user_id")
    ticker: str = Field(index=True)
    quantity: int
    price: float
    action: str
    scenario: Optional[str] = Field(default=None)
    trade_value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def init_db():
    """Create all tables."""
    SQLModel.metadata.create_all(engine)
    log.info(f"Database initialized at {DB_PATH}")


def get_portfolio(user_id: str) -> Optional[dict]:
    """
    Fetches a user's portfolio.
    Returns a dict with virtualCash and holdings list, or None if user does not exist.
    """
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            return None

        statement = select(Holding).where(Holding.user_id == user_id).where(Holding.quantity > 0)
        holdings = session.exec(statement).all()

        holdings_list = [
            {
                "ticker": h.ticker,
                "quantity": h.quantity,
                "avgPrice": h.avg_price
            }
            for h in holdings
        ]

        return {
            "userId": user.user_id,
            "virtualCash": user.virtual_cash,
            "holdings": holdings_list
        }


def execute_trade(user_id: str, ticker: str, quantity: int, price: float, action: str) -> dict:
    """
    Executes a buy or sell trade and updates the user's portfolio.
    Returns dict with status and message.
    """
    with Session(engine) as session:
        try:
            # Get user
            user = session.get(User, user_id)
            if not user:
                return {"status": "error", "message": "User not found."}

            trade_value = quantity * price
            action_upper = action.upper()

            if action_upper == "BUY":
                # Check for sufficient funds
                if user.virtual_cash < trade_value:
                    return {"status": "error", "message": "Insufficient funds."}

                # Update virtual cash
                user.virtual_cash -= trade_value
                user.updated_at = datetime.now(timezone.utc)

                # Find or create holding
                statement = select(Holding).where(
                    Holding.user_id == user_id,
                    Holding.ticker == ticker
                )
                holding = session.exec(statement).first()

                if holding:
                    # Update existing holding
                    new_quantity = holding.quantity + quantity
                    new_avg_price = ((holding.avg_price * holding.quantity) + trade_value) / new_quantity
                    holding.quantity = new_quantity
                    holding.avg_price = new_avg_price
                    holding.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new holding
                    holding = Holding(
                        user_id=user_id,
                        ticker=ticker,
                        quantity=quantity,
                        avg_price=price
                    )
                    session.add(holding)

            elif action_upper == "SELL":
                # Find holding
                statement = select(Holding).where(
                    Holding.user_id == user_id,
                    Holding.ticker == ticker
                )
                holding = session.exec(statement).first()

                if not holding:
                    return {"status": "error", "message": "Stock not found in portfolio."}

                # Check for sufficient quantity
                if holding.quantity < quantity:
                    return {"status": "error", "message": "Insufficient quantity to sell."}

                # Update virtual cash
                user.virtual_cash += trade_value
                user.updated_at = datetime.now(timezone.utc)

                # Update holding
                holding.quantity -= quantity
                holding.updated_at = datetime.now(timezone.utc)

                # Remove holding if quantity is zero
                if holding.quantity == 0:
                    session.delete(holding)
            else:
                return {"status": "error", "message": "Invalid action. Must be BUY or SELL."}

            # Log the trade
            trade_record = Trade(
                user_id=user_id,
                ticker=ticker,
                quantity=quantity,
                price=price,
                action=action_upper,
                trade_value=trade_value
            )
            session.add(trade_record)

            # Commit all changes
            session.commit()

            log.info(f"Trade executed for {user_id}: {action_upper} {quantity} {ticker} @ {price}")
            return {
                "status": "success",
                "message": "Trade executed successfully.",
                "newPortfolio": get_portfolio(user_id)
            }

        except Exception as e:
            session.rollback()
            log.error(f"Error in execute_trade for {user_id}: {e}")
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


def create_user(user_id: str, password_hash: str) -> dict:
    """Creates a new user with default balances."""
    with Session(engine) as session:
        try:
            existing = session.get(User, user_id)
            if existing:
                return {"status": "error", "message": "User already exists."}

            user = User(
                user_id=user_id,
                password_hash=password_hash,
                virtual_cash=100000.00
            )
            session.add(user)

            # Initialize simulation state for this user
            sim_state = SimulationState(user_id=user_id, virtual_cash=100000.00)
            session.add(sim_state)

            session.commit()
            log.info(f"Created user: {user_id}")
            return {"status": "success", "message": "User created successfully."}
        except Exception as e:
            session.rollback()
            log.error(f"Error creating user {user_id}: {e}")
            return {"status": "error", "message": f"Could not create user: {str(e)}"}


def authenticate_user(user_id: str) -> Optional[str]:
    """Returns the stored password hash for a user, or None."""
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            return None
        return user.password_hash


def get_simulation_portfolio(user_id: str) -> dict:
    """Fetches or creates a user's simulation portfolio."""
    with Session(engine) as session:
        sim_state = session.get(SimulationState, user_id)
        if not sim_state:
            sim_state = SimulationState(user_id=user_id, virtual_cash=100000.00)
            session.add(sim_state)
            session.commit()
            session.refresh(sim_state)

        statement = select(SimulationHolding).where(SimulationHolding.user_id == user_id).where(SimulationHolding.quantity > 0)
        holdings = session.exec(statement).all()

        holdings_list = [
            {
                "ticker": h.ticker,
                "quantity": h.quantity,
                "avgPrice": h.avg_price,
                "scenario": h.scenario
            }
            for h in holdings
        ]

        return {
            "userId": sim_state.user_id,
            "virtualCash": sim_state.virtual_cash,
            "holdings": holdings_list
        }


def execute_simulation_trade(
    user_id: str,
    ticker: str,
    quantity: int,
    price: float,
    action: str,
    scenario: Optional[str] = None
) -> dict:
    """Executes a simulation trade for the given user."""
    with Session(engine) as session:
        try:
            sim_state = session.get(SimulationState, user_id)
            if not sim_state:
                sim_state = SimulationState(user_id=user_id, virtual_cash=100000.00)
                session.add(sim_state)
                session.commit()
                session.refresh(sim_state)

            trade_value = quantity * price
            action_upper = action.upper()

            statement = select(SimulationHolding).where(
                SimulationHolding.user_id == user_id,
                SimulationHolding.ticker == ticker
            )
            holding = session.exec(statement).first()

            if action_upper == "BUY":
                if sim_state.virtual_cash < trade_value:
                    return {"status": "error", "message": "Insufficient simulation cash."}

                sim_state.virtual_cash -= trade_value
                sim_state.updated_at = datetime.now(timezone.utc)

                if holding:
                    new_quantity = holding.quantity + quantity
                    new_avg_price = ((holding.avg_price * holding.quantity) + trade_value) / new_quantity
                    holding.quantity = new_quantity
                    holding.avg_price = new_avg_price
                    holding.scenario = scenario
                    holding.updated_at = datetime.now(timezone.utc)
                else:
                    holding = SimulationHolding(
                        user_id=user_id,
                        ticker=ticker,
                        quantity=quantity,
                        avg_price=price,
                        scenario=scenario
                    )
                    session.add(holding)

            elif action_upper == "SELL":
                if not holding:
                    return {"status": "error", "message": "Simulation holding not found."}
                if holding.quantity < quantity:
                    return {"status": "error", "message": "Insufficient quantity in simulation holdings."}

                sim_state.virtual_cash += trade_value
                sim_state.updated_at = datetime.now(timezone.utc)

                holding.quantity -= quantity
                holding.updated_at = datetime.now(timezone.utc)

                if holding.quantity == 0:
                    session.delete(holding)
            else:
                return {"status": "error", "message": "Invalid action. Must be BUY or SELL."}

            sim_trade = SimulationTrade(
                user_id=user_id,
                ticker=ticker,
                quantity=quantity,
                price=price,
                action=action_upper,
                scenario=scenario,
                trade_value=trade_value
            )
            session.add(sim_trade)

            session.commit()
            log.info(f"Simulation trade executed for {user_id}: {action_upper} {quantity} {ticker} @ {price}")
            return {
                "status": "success",
                "message": "Simulation trade executed successfully.",
                "portfolio": get_simulation_portfolio(user_id)
            }
        except Exception as e:
            session.rollback()
            log.error(f"Error in execute_simulation_trade for {user_id}: {e}")
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


def reset_simulation(user_id: str) -> dict:
    """Resets simulation state for a user."""
    with Session(engine) as session:
        try:
            sim_state = session.get(SimulationState, user_id)
            if not sim_state:
                sim_state = SimulationState(user_id=user_id, virtual_cash=100000.00)
                session.add(sim_state)
            else:
                sim_state.virtual_cash = 100000.00
                sim_state.updated_at = datetime.now(timezone.utc)

            # Clean up via raw deletes for simplicity
            session.execute(
                SimulationHolding.__table__.delete().where(SimulationHolding.user_id == user_id)
            )
            session.execute(
                SimulationTrade.__table__.delete().where(SimulationTrade.user_id == user_id)
            )
            session.commit()
            return {"status": "success", "message": "Simulation reset successfully."}
        except Exception as e:
            session.rollback()
            log.error(f"Error resetting simulation for {user_id}: {e}")
            return {"status": "error", "message": f"Failed to reset simulation: {str(e)}"}


def get_simulation_trades(user_id: str) -> List[dict]:
    """Returns simulation trades for a user ordered by latest."""
    with Session(engine) as session:
        statement = (
            select(SimulationTrade)
            .where(SimulationTrade.user_id == user_id)
            .order_by(SimulationTrade.timestamp.desc())
        )
        trades = session.exec(statement).all()
        return [
            {
                "id": trade.id,
                "userId": trade.user_id,
                "ticker": trade.ticker,
                "quantity": trade.quantity,
                "price": trade.price,
                "action": trade.action,
                "scenario": trade.scenario,
                "tradeValue": trade.trade_value,
                "timestamp": trade.timestamp.isoformat(),
            }
            for trade in trades
        ]
