import React, { useState } from 'react';
import { analyzeImage } from '../services/aiService';
import ResultsDisplay from '../components/ResultsDisplay';
import '../styles/DiagnosticAnalysis.css';

const DiagnosticAnalysis = ({ modelParams }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleAnalysis = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    try {
      const results = await analyzeImage(selectedFile, modelParams);
      setAnalysis(results);
    } catch (error) {
      console.error('Analysis failed:', error);
      // Handle error display
    }
    setLoading(false);
  };

  return (
    <div className="diagnostic-analysis">
      <h2>Single Scan Analysis</h2>
      
      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="file-input"
        />
        {selectedFile && (
          <button 
            onClick={handleAnalysis}
            disabled={loading}
            className="analyze-button"
          >
            {loading ? 'Analyzing...' : 'Analyze Scan'}
          </button>
        )}
      </div>

      {loading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Analyzing scan with AI ensemble...</p>
        </div>
      )}

      {analysis && <ResultsDisplay results={analysis} />}
    </div>
  );
};

export default DiagnosticAnalysis;