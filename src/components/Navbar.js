import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <h1>🏥 ProstateCare AI</h1>
        <p>Advanced Cancer Detection & Staging System</p>
      </div>
      <div className="navbar-links">
        <Link to="/">Diagnostic Analysis</Link>
        <Link to="/batch">Batch Processing</Link>
        <Link to="/guide">Clinical Guide</Link>
      </div>
    </nav>
  );
};

export default Navbar;