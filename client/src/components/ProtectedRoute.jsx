// src/components/ProtectedRoute.jsx
import React from 'react';
import { Navigate } from 'react-router-dom';
import Cookies from 'js-cookie';
import { jwtDecode } from 'jwt-decode';

const ProtectedRoute = ({ children }) => {
  const token = Cookies.get('token');

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  try {
    const decoded = jwtDecode(token);
    const now = Date.now() / 1000;
    if (decoded.exp && decoded.exp < now) {
      Cookies.remove('token');
      alert('Session expired. Please login again.');
      return <Navigate to="/login" replace />;
    }
  } catch (err) {
    Cookies.remove('token');
    return <Navigate to="/login" replace />;
  }

  return children;
};

export default ProtectedRoute;
