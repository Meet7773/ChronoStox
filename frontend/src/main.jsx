import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'

import App from './App.jsx' // The main App shell/layout
import Home from './pages/Home.jsx'
import Portfolio from './pages/Portfolio.jsx' // The portfolio page
import LiveMarket from './pages/LiveMarket.jsx' // The live market page
import TradeSimulator from './pages/TradeSimulator.jsx'
import StockScreener from './pages/StockScreener.jsx'
import Insights from './pages/Insights.jsx'

import './index.css' // Global Tailwind styles
import { ThemeProvider } from './context/ThemeContext.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<App />}>
          <Route index element={<Home />} />
          <Route path="portfolio" element={<Portfolio />} />
            <Route path="market" element={<LiveMarket />} />
            <Route path="trade-simulator" element={<TradeSimulator />} />
            <Route path="screener" element={<StockScreener />} />
            <Route path="insights" element={<Insights />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  </React.StrictMode>,
)