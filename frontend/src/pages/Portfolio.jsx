import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_URL = '[http://127.0.0.1:8000](http://127.0.0.1:8000)';
const USER_ID = 'college_project_user'; // Hardcoded user ID

// --- TradeWidget Component ---
function TradeWidget({ onTradeSuccess }) {
  const [ticker, setTicker] = useState('');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [action, setAction] = useState('BUY');
  const [status, setStatus] = useState({ message: '', isError: false, isLoading: false });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus({ message: `[EXECUTING_${action}...]`, isError: false, isLoading: true });

    try {
      const response = await axios.post(`${API_URL}/trade`, {
        userId: USER_ID,
        ticker: ticker.toUpperCase(),
        quantity: parseInt(quantity, 10),
        price: parseFloat(price),
        action: action,
      });

      setStatus({ message: `[${action}_CONFIRMED]`, isError: false, isLoading: false });
      // Clear form
      setTicker('');
      setQuantity('');
      setPrice('');
      // Call the parent's refresh function
      onTradeSuccess(response.data.newPortfolio);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || 'Trade failed. Check API connection.';
      setStatus({ message: `[${action}_REJECTED] ${errorMsg}`, isError: true, isLoading: false });
    }
  };

  return (
    <div className="border-2 border-matrix-green p-4">
      <h3 className="text-xl -mt-1 mb-2">[TRADE_EXECUTION]</h3>
      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="ticker">[TICKER]</label>
            <input
              id="ticker"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="w-full bg-matrix-dark border border-matrix-green p-1 focus:outline-none focus:bg-matrix-green focus:text-matrix-dark"
              required
            />
          </div>
          <div>
            <label htmlFor="quantity">[QUANTITY]</label>
            <input
              id="quantity"
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              className="w-full bg-matrix-dark border border-matrix-green p-1 focus:outline-none focus:bg-matrix-green focus:text-matrix-dark"
              min="1"
              step="1"
              required
            />
          </div>
        </div>
        <div className="mt-4">
          <label htmlFor="price">[PRICE]</label>
          <input
            id="price"
            type="number"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            className="w-full bg-matrix-dark border border-matrix-green p-1 focus:outline-none focus:bg-matrix-green focus:text-matrix-dark"
            min="0.01"
            step="0.01"
            required
          />
        </div>
        <div className="flex space-x-4 mt-4">
          <button
            type="submit"
            onClick={() => setAction('BUY')}
            disabled={status.isLoading}
            className="w-full bg-matrix-green text-matrix-dark font-bold py-2 px-4 hover:bg-opacity-80 disabled:opacity-50"
          >
            {status.isLoading && action === 'BUY' ? '[...]' : '[EXECUTE_BUY]'}
          </button>
          <button
            type="submit"
            onClick={() => setAction('SELL')}
            disabled={status.isLoading}
            className="w-full bg-matrix-dark text-matrix-green border-2 border-matrix-green font-bold py-2 px-4 hover:bg-matrix-green hover:text-matrix-dark disabled:opacity-50"
          >
            {status.isLoading && action === 'SELL' ? '[...]' : '[EXECUTE_SELL]'}
          </button>
        </div>
      </form>
      {status.message && (
        <div className={`mt-2 p-2 border ${status.isError ? 'text-red-500 border-red-500' : 'text-matrix-green border-matrix-green'}`}>
          {status.message}
        </div>
      )}
    </div>
  );
}

// --- Portfolio Page Component ---
function Portfolio() {
  const [portfolio, setPortfolio] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Use useCallback to create a stable function reference
  const fetchPortfolio = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/portfolio/${USER_ID}`);
      setPortfolio(response.data);
    } catch (err) {
      const errorMsg = err.response ? `API Error: ${err.response.data.detail}` : `Network Error: ${err.message}. Is the backend running?`;
      setError(errorMsg);
    }
    setIsLoading(false);
  }, []); // Empty dependency array means this function is created once

  // useEffect for initial data load
  useEffect(() => {
    fetchPortfolio();
  }, [fetchPortfolio]); // fetchPortfolio is stable, so this runs once on mount

  // Callback for the TradeWidget to update portfolio state
  // This avoids a full re-fetch and makes the UI instant
  const handleTradeSuccess = (newPortfolio) => {
    setPortfolio(newPortfolio);
  };

  // --- Render Logic ---
  if (isLoading && !portfolio) {
    return <div>[...LOADING_PORTFOLIO_DATA...]</div>;
  }

  if (error) {
    return (
      <div className="border-2 border-red-500 p-4 text-red-500">
        <h2 className="text-xl">[CONNECTION_FAILURE]</h2>
        <p>{error}</p>
        <button
          onClick={fetchPortfolio}
          className="mt-4 bg-matrix-green text-matrix-dark font-bold py-2 px-4 hover:bg-opacity-80"
        >
          [RETRY_CONNECTION]
        </button>
      </div>
    );
  }

  if (!portfolio) {
    return <div>[...NO_PORTFOLIO_DATA...]</div>;
  }

  const { virtualCash, holdings, userId } = portfolio;
  const formattedCash = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(virtualCash);

  return (
    <div>
      <h2 className="text-2xl mb-2">[USER_ID: {userId}]</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        {/* Left Column: Cash & Trading */}
        <div className="md:col-span-1 flex flex-col gap-4">
          <div className="border-2 border-matrix-green p-4">
            <h3 className="text-xl -mt-1 mb-2">[LIQUID_ASSETS]</h3>
            <p className="text-4xl">{formattedCash}</p>
          </div>
          <TradeWidget onTradeSuccess={handleTradeSuccess} />
        </div>

        {/* Right Column: Holdings */}
        <div className="md:col-span-2 border-2 border-matrix-green p-4">
          <h3 className="text-xl -mt-1 mb-2">[HOLDINGS_OVERVIEW]</h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b-2 border-matrix-green text-left">
                  <th className="p-2">[TICKER]</th>
                  <th className="p-2">[QUANTITY]</th>
                  <th className="p-2">[AVG_PRICE]</th>
                  <th className="p-2">[MARKET_VALUE]</th>
                </tr>
              </thead>
              <tbody>
                {holdings.length > 0 ? (
                  holdings.map((stock) => {
                    const marketValue = stock.quantity * stock.avgPrice; // Note: In a real app, this would use live price
                    return (
                      <tr key={stock.ticker} className="border-b border-matrix-green/50">
                        <td className="p-2">{stock.ticker}</td>
                        <td className="p-2">{stock.quantity}</td>
                        <td className="p-2">{new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(stock.avgPrice)}</td>
                        <td className="p-2">{new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(marketValue)}</td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan="4" className="p-2 text-center">[...No securities held...]</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}

export default Portfolio;