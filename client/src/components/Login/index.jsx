// src/components/Login.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Cookies from 'js-cookie';
import { ButtonLoader } from '../Loader';
import './index.css';

const Login = () => {
  const [formData, setFormData] = useState({ email: '', password: '' });
  const [showSubmitError, setShowSubmitError] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const token = Cookies.get('token');
    if (token) {
      navigate('/dashboard', { replace: true });
    }
  }, [navigate]);

  const { email, password } = formData;

  const onChange = (e) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  const onSubmitSuccess = (jwtToken) => {
    Cookies.set('token', jwtToken, { expires: 30, path: '/' });
    navigate('/dashboard', { replace: true });
  };

  const onSubmitFailure = (errorMsg) => {
    setShowSubmitError(true);
    setErrorMsg(errorMsg);
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const res = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/auth/login`,
        { email, password }
      );
      onSubmitSuccess(res.data.token);
    } catch (err) {
      const errorMessage =
        err.response?.data?.error || err.message || 'Login failed.';
      onSubmitFailure(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="particles">
        {[...Array(20)].map((_, i) => {
          const left = Math.floor(Math.random() * 100);
          const delay = (Math.random() * 10).toFixed(2);
          const size = Math.floor(Math.random() * 4) + 4;
          return (
            <span
              key={i}
              style={{
                left: `${left}%`,
                width: `${size}px`,
                height: `${size}px`,
                animationDelay: `${delay}s`,
              }}
            />
          );
        })}
      </div>

      <div className="login-form">
        <h2 className="typewriter-text">Login</h2>
        <form onSubmit={onSubmit}>
          <div className="form-group">
            <input
              type="email"
              placeholder="Email Address"
              name="email"
              value={email}
              onChange={onChange}
              required
            />
          </div>
          <div className="form-group">
            <input
              type="password"
              placeholder="Password"
              name="password"
              value={password}
              onChange={onChange}
              required
            />
          </div>
          <button type="submit" className="login-button">
            {isLoading ? <ButtonLoader /> : 'Login'}
          </button>
        </form>
        {showSubmitError && <p className="error-message">*{errorMsg}</p>}
      </div>
    </div>
  );
};

export default Login;
