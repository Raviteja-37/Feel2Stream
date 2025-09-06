// controllers/recommendationController.js
const User = require('../models/User');
const mlService = require('../services/mlService');
const recService = require('../services/recommendationService');
const { decodeAuthHeader } = require('../utils/tokenUtils');

async function maybeSaveRecommendationForUser(authHeader, emotion, items) {
  try {
    const decoded = decodeAuthHeader(authHeader);
    if (!decoded || !decoded.id) return;
    const user = await User.findById(decoded.id);
    if (!user) return;

    // save to beginning; cap to last 20
    user.recommendations.unshift({ emotion, items, createdAt: new Date() });
    if (user.recommendations.length > 20) {
      user.recommendations = user.recommendations.slice(0, 20);
    }
    user.lastEmotion = emotion;
    await user.save();
  } catch (err) {
    console.error('Failed to save recommendation for user', err);
  }
}

exports.predictText = async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ message: 'text required' });

    const mlResp = await mlService.predictText(text);
    const emotion = mlResp.emotion || mlResp.label;
    if (!emotion)
      return res.status(500).json({ message: 'ML did not return emotion' });

    const items = await recService.generateRecommendations({
      emotion,
    });

    await maybeSaveRecommendationForUser(
      req.headers.authorization,
      emotion,
      items
    );

    res.json({
      emotion,
      confidence: mlResp.confidence || mlResp.score || null,
      items,
    });
  } catch (err) {
    console.error('predictText error', err);
    res.status(500).json({ message: 'Server error' });
  }
};

exports.predictVoice = async (req, res) => {
  try {
    if (!req.file?.buffer)
      return res.status(400).json({ message: 'audio file required' });

    const mlResp = await mlService.predictVoice(
      req.file.buffer,
      req.file.originalname
    );
    const emotion = mlResp.emotion || mlResp.label;
    if (!emotion)
      return res.status(500).json({ message: 'ML did not return emotion' });

    const items = await recService.generateRecommendations({
      emotion,
    });

    await maybeSaveRecommendationForUser(
      req.headers.authorization,
      emotion,
      items
    );

    res.json({
      emotion,
      confidence: mlResp.confidence || mlResp.score || null,
      items,
    });
  } catch (err) {
    console.error('predictVoice error', err);
    res.status(500).json({ message: 'Server error' });
  }
};

exports.predictFace = async (req, res) => {
  try {
    if (!req.file?.buffer)
      return res.status(400).json({ message: 'image file required' });

    const mlResp = await mlService.predictFace(
      req.file.buffer,
      req.file.originalname
    );
    const emotion = mlResp.emotion || mlResp.label;
    if (!emotion)
      return res.status(500).json({ message: 'ML did not return emotion' });

    const items = await recService.generateRecommendations({
      emotion,
    });

    await maybeSaveRecommendationForUser(
      req.headers.authorization,
      emotion,
      items
    );

    res.json({
      emotion,
      confidence: mlResp.confidence || mlResp.score || null,
      items,
    });
  } catch (err) {
    console.error('predictFace error', err);
    res.status(500).json({ message: 'Server error' });
  }
};

exports.generateFromEmotion = async (req, res) => {
  try {
    const { emotion, interest, customText, genre } = req.body;
    if (!emotion) return res.status(400).json({ message: 'emotion required' });

    const items = await recService.generateRecommendations({
      emotion,
      interest,
      customText,
      genre,
    });

    await maybeSaveRecommendationForUser(
      req.headers.authorization,
      emotion,
      items
    );

    res.json({ emotion, items });
  } catch (err) {
    console.error('generateFromEmotion error', err);
    res.status(500).json({ message: 'Server error' });
  }
};

exports.getMyRecommendations = async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select(
      'recommendations lastEmotion'
    );
    if (!user) return res.status(404).json({ message: 'User not found' });

    res.json({
      lastEmotion: user.lastEmotion,
      recommendations: user.recommendations,
    });
  } catch (err) {
    console.error('getMyRecommendations error', err);
    res.status(500).json({ message: 'Server error' });
  }
};
