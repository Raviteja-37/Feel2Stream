// routes/recommendationRoutes.js
const router = require('express').Router();
const multer = require('multer');
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});
const auth = require('../middleware/authMiddleware');

const recController = require('../controllers/recommendationController');

// ML endpoints (text, voice, face)
router.post('/emotion/text', recController.predictText);
router.post(
  '/emotion/voice',
  upload.single('audio'), // frontend must send field name "audio"
  recController.predictVoice
);
router.post('/emotion/face', upload.single('image'), recController.predictFace);

// Generate recommendations directly from emotion (public but will be saved if Authorization header provided)
router.post('/recs/generate', recController.generateFromEmotion);

// Protected: get my saved recommendations
router.get('/recs/my', auth, recController.getMyRecommendations);

module.exports = router;
