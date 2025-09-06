// routes/authRoutes.js
const router = require('express').Router();
const { body } = require('express-validator');
const authMiddleware = require('../middleware/authMiddleware');
const { register, login, me } = require('../controllers/authController');

router.post(
  '/register',
  [
    body('username').trim().notEmpty().withMessage('Name required'),
    body('email').isEmail().withMessage('Valid email required'),
    body('password')
      .isLength({ min: 6 })
      .withMessage('Password must be at least 6 chars'),
  ],
  register
);

router.post(
  '/login',
  [
    body('email').isEmail().withMessage('Valid email required'),
    body('password').notEmpty().withMessage('Password required'),
  ],
  login
);

router.get('/me', authMiddleware, me);

module.exports = router;
