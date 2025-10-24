import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import DiagnosticAnalysis from './pages/DiagnosticAnalysis';
import BatchProcessing from './pages/BatchProcessing';
import ClinicalGuide from './pages/ClinicalGuide';
import './styles/App.css';

function App() {
  const [modelParams, setModelParams] = useState({
    sensitivity: 0.5,
    threshold: 1.0
  });

  return (
    <Router>
      <div className="app">
        <Navbar />
        <div className="main-container">
          <Sidebar modelParams={modelParams} setModelParams={setModelParams} />
          <main className="content">
            <Routes>
              <Route path="/" element={<DiagnosticAnalysis modelParams={modelParams} />} />
              <Route path="/batch" element={<BatchProcessing modelParams={modelParams} />} />
              <Route path="/guide" element={<ClinicalGuide />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;