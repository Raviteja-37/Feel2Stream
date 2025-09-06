// utils/tokenUtils.js
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  console.error('JWT_SECRET not set in .env');
  process.exit(1);
}

exports.generateToken = (userId) => {
  return jwt.sign({ id: userId }, JWT_SECRET, { expiresIn: '7d' });
};

exports.verifyToken = (token) => {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (err) {
    return null;
  }
};

/**
 * Helper: if an Authorization header is present, returns decoded payload or null
 */
exports.decodeAuthHeader = (authorizationHeader) => {
  if (!authorizationHeader) return null;
  const token = authorizationHeader.startsWith('Bearer ')
    ? authorizationHeader.slice(7)
    : authorizationHeader;
  return exports.verifyToken(token);
};
