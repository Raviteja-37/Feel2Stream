// middleware/authMiddleware.js
const { verifyToken } = require('../utils/tokenUtils');

module.exports = function auth(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ message: 'No token provided' });

  const decoded = verifyToken(token);
  if (!decoded)
    return res.status(401).json({ message: 'Invalid or expired token' });

  req.user = { id: decoded.id };
  next();
};
