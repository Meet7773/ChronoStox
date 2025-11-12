import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'

import App from './App.jsx' // The main App shell/layout
import Portfolio from './pages/Portfolio.jsx' // The portfolio page
import LiveMarket from './pages/LiveMarket.jsx' // The live market page

import './index.css' // Global Tailwind styles

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        {/* The <App /> component is now the parent "layout" route.
          All other routes render *inside* its <Outlet />.
        */}
        <Route path="/" element={<App />}>
          {/* The `index` route is the default component to show
            at the parent's path (e.g., "/")
          */}
          <Route index element={<Portfolio />} />
          <Route path="market" element={<LiveMarket />} />
          {/* You can add more routes here, like:
          <Route path="screener" element={<StockScreener />} />
          */}
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)