import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import Sidenav from './components/Sidenav';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Cases from './pages/Cases';
import { StatsProvider } from './contexts/StatsContext';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars } from '@fortawesome/free-solid-svg-icons';
import './App.css';

const AppLayout: React.FC = () => {
  const [sidenavOpen, setSidenavOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  
  const currentPage = location.pathname.substring(1) || 'dashboard';

  const handleNavigate = (page: string) => {
    setSidenavOpen(false);
    navigate(`/${page}`);
  };

  const toggleSidenav = () => {
    setSidenavOpen(!sidenavOpen);
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidenav 
        isOpen={sidenavOpen} 
        onToggle={toggleSidenav}
        currentPage={currentPage}
        onNavigate={handleNavigate}
      />
      
      <div className="flex-1 flex flex-col overflow-hidden lg:ml-0">
        {/* Mobile header */}
        <div className="lg:hidden bg-white shadow-sm border-b border-gray-200 px-4 py-3">
          <button
            onClick={toggleSidenav}
            className="p-2 rounded-md hover:bg-gray-100"
          >
            <FontAwesomeIcon icon={faBars} className="text-gray-500" />
          </button>
        </div>
        
        {/* Main content */}
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/cases" element={<Cases />} />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
};

function App() {
  return (
    <StatsProvider>
      <Router>
        <AppLayout />
      </Router>
    </StatsProvider>
  );
}

export default App;
