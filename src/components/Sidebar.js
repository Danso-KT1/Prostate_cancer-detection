import React from 'react';
import '../styles/Sidebar.css';

const Sidebar = ({ modelParams, setModelParams }) => {
  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <h3>System Control Center</h3>
        <hr />
        
        <h4>AI Models Status</h4>
        <div className="model-status">
          <div className="status-item success">✓ ResU-Net</div>
          <div className="status-item success">✓ Attention U-Net</div>
          <div className="status-item success">✓ U-Net++</div>
        </div>
        
        <hr />
        
        <h4>Detection Parameters</h4>
        <div className="parameter-control">
          <label>Sensitivity Level</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={modelParams.sensitivity}
            onChange={(e) => setModelParams({
              ...modelParams,
              sensitivity: parseFloat(e.target.value)
            })}
          />
          <span>{modelParams.sensitivity}</span>
        </div>
        
        <div className="parameter-control">
          <label>Classification Threshold</label>
          <input
            type="range"
            min="0"
            max="10"
            step="0.5"
            value={modelParams.threshold}
            onChange={(e) => setModelParams({
              ...modelParams,
              threshold: parseFloat(e.target.value)
            })}
          />
          <span>{modelParams.threshold}%</span>
        </div>
      </div>
      
      <div className="sidebar-info">
        <h4>System Information</h4>
        <div className="info-box">
          <p><strong>AI Models:</strong></p>
          <ul>
            <li>ResU-Net</li>
            <li>Attention U-Net</li>
            <li>U-Net++</li>
          </ul>
          <p><strong>Method:</strong> Ensemble majority voting</p>
          <p><strong>Version:</strong> 2.0</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;