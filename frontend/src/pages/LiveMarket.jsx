import React, { useState } from 'react';
import axios from 'axios';

const API_URL = '[http://127.0.0.1:8000](http://127.0.0.1:8000)';

function LiveMarket() {
  const [ticker, setTicker] = useState('');
  const [stockData, setStockData] = useState(null);
  const [status, setStatus] = useState({ message: 'Enter ticker (e.g., RELIANCE.NS)', isError: false, isLoading: false });

  const handleSearch = async (e) => {
    e.preventDefault();
    setStatus({ message: `[FETCHING_${ticker}...]`, isError: false, isLoading: true });
    setStockData(null); // Clear previous data

    try {
      const response = await axios.get(`${API_URL}/stock/${ticker.toUpperCase()}`);
      setStockData(response.data);
      setStatus({ message: `[DATA_LOADED] ${response.data.ticker}`, isError: false, isLoading: false });
    } catch (error) {
      const errorMsg = error.response?.data?.detail || 'Fetch failed. Check API connection.';
      setStatus({ message: `[FETCH_FAILED] ${errorMsg}`, isError: true, isLoading: false });
      setStockData(null);
    }
  };

  // Helper component for displaying data
  const DataRow = ({ label, value }) => {
    if (value === null || typeof value === 'undefined') return null;
    let displayValue = value;
    if (typeof value === 'number' && !label.toLowerCase().includes('volume')) {
      displayValue = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(value);
    }
    if (typeof value === 'number' && label.toLowerCase().includes('volume')) {
      displayValue = new Intl.NumberFormat('en-IN').format(value);
    }
    return (
      <div className="flex justify-between border-b border-matrix-green/30 py-2">
        <span>[{label}]</span>
        <span>{displayValue}</span>
      </div>
    );
  };

  return (
    <div>
      <h2 className="text-2xl mb-2">[LIVE_MARKET_DATA]</h2>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="flex space-x-2 mb-4">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Enter Ticker (e.g. RELIANCE.NS)"
          className="w-full bg-matrix-dark border border-matrix-green p-2 focus:outline-none focus:bg-matrix-green focus:text-matrix-dark placeholder:text-matrix-green/50"
          required
        />
        <button
          type="submit"
          disabled={status.isLoading}
          className="bg-matrix-green text-matrix-dark font-bold py-2 px-6 hover:bg-opacity-80 disabled:opacity-50"
        >
          {status.isLoading ? '[...]' : '[FETCH]'}
        </button>
      </form>

      {/* Status & Data Display */}
      <div className="border-2 border-matrix-green p-4">
        {/* Status Bar */}
        <div className={`mb-4 p-2 border ${status.isError ? 'text-red-500 border-red-500' : 'text-matrix-green border-matrix-green'}`}>
          {status.message}
        </div>

        {/* Stock Data */}
        {stockData && (
          <div>
            <h3 className="text-2xl mb-2">{stockData.companyName} ({stockData.ticker})</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              <DataRow label="CURRENT_PRICE" value={stockData.currentPrice} />
              <DataRow label="PREV_CLOSE" value={stockData.previousClose} />
              <DataRow label="DAY_HIGH" value={stockData.dayHigh} />
              <DataRow label="DAY_LOW" value={stockData.dayLow} />
              <DataRow label="MARKET_CAP" value={stockData.marketCap} />
              <DataRow label="VOLUME" value={stockData.volume} />
            </div>
            {stockData.longSummary && (
              <div className="mt-4">
                <h4 className="text-xl">[BUSINESS_SUMMARY]</h4>
                <p className="mt-2 text-justify text-matrix-green/80">
                  {stockData.longSummary}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default LiveMarket;