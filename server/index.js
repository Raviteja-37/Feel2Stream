// index.js
require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const morgan = require('morgan');
const cors = require('cors');
const connectDB = require('./config/db');

const authRoutes = require('./routes/authRoutes');
const recRoutes = require('./routes/recommendationRoutes');

const app = express();

// Security & parsing
app.use(helmet());
app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(morgan('dev'));
app.use(cors({ origin: process.env.CORS_ORIGIN || '*', credentials: true }));

// Connect DB
connectDB();

// Routes
app.get('/health', (req, res) =>
  res.json({ ok: true, time: new Date().toISOString() })
);
app.use('/api/auth', authRoutes);
app.use('/api', recRoutes);

// 404 handler
app.use((req, res) => res.status(404).json({ message: 'Not found' }));

// Global error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ message: 'Server error' });
});

// Start server
const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
