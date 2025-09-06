// src/components/Navbar/index.jsx (Simplified for Feel2Stream)
import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import Cookies from 'js-cookie';
import './index.css';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    Cookies.remove('token'); // Remove JWT token
    navigate('/login', { replace: true });
  };

  const isLoggedIn = Cookies.get('token') !== undefined;

  return (
    <nav className="navbar-container">
      <div className="navbar-logo">
        {/* Always send to dashboard if logged in, otherwise login */}
        {/* <Link to={isLoggedIn ? '/dashboard' : '/login'}>
          <img className="nav-logo-img" src="/NavFeel2Stream.png" width="150" />
        </Link> */}
        <h1>Feel2Stream</h1>
      </div>
      <ul className="navbar-links">
        {/* Only show Logout if logged in */}
        {isLoggedIn ? (
          <>
            <li>
              <Link to="/dashboard">Dashboard</Link>
            </li>
            <li>
              <button onClick={handleLogout} className="logout-button">
                Logout
              </button>
            </li>
          </>
        ) : (
          // Only show auth navigation if user not logged in
          <>
            {location.pathname !== '/login' && (
              <li>
                <Link to="/login">Login</Link>
              </li>
            )}
            {location.pathname !== '/register' && (
              <li>
                <Link to="/register">Register</Link>
              </li>
            )}
          </>
        )}
      </ul>
    </nav>
  );
};

export default Navbar;
