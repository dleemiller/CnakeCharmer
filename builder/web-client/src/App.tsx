import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { JobProvider } from './contexts/JobContext';
import { SocketProvider } from './contexts/SocketContext';
import Dashboard from './pages/Dashboard';

const App: React.FC = () => {
  return (
    <Router>
      <SocketProvider>
        <JobProvider>
          <Dashboard />
        </JobProvider>
      </SocketProvider>
    </Router>
  );
};

export default App;