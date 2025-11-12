import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';

function App() {
  const location = useLocation();

  // Helper to determine if a link is active
  const isActive = (path) => location.pathname === path;

  return (
    <div className="min-h-screen bg-matrix-dark text-matrix-green font-mono p-4 selection:bg-matrix-green selection:text-matrix-dark">

      {/* Header & Navigation */}
      <header className="border-b-2 border-matrix-green pb-2 mb-4">
        <h1 className="text-3xl font-bold">[ChronoStox_Terminal]</h1>
        <nav className="flex space-x-4 text-lg mt-2">

          <Link
            to="/"
            className={`px-2 py-1 ${isActive('/') ? 'bg-matrix-green text-matrix-dark' : 'hover:bg-matrix-green hover:text-matrix-dark'}`}
          >
            [Portfolio]
          </Link>

          <Link
            to="/market"
            className={`px-2 py-1 ${isActive('/market') ? 'bg-matrix-green text-matrix-dark' : 'hover:bg-matrix-green hover:text-matrix-dark'}`}
          >
            [Live_Market]
          </Link>

        </nav>
      </header>

      {/* Main Content Area */}
      <main>
        {/* The <Outlet> is the placeholder from react-router-dom.
          It renders the component that matches the current URL path
          (e.g., <Portfolio /> or <LiveMarket />)
        */}
        <Outlet />
      </main>
    </div>
  );
}

export default App;