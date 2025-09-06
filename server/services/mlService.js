// services/mlService.js
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');

const ML_BASE = process.env.ML_BASE_URL || 'http://localhost:9090';
const client = axios.create({ baseURL: ML_BASE, timeout: 30_000 });

function guessMimeFromFilename(filename) {
  const ext = (path.extname(filename || '') || '').toLowerCase();
  if (ext === '.wav') return 'audio/wav';
  if (ext === '.mp3') return 'audio/mpeg';
  if (ext === '.m4a') return 'audio/mp4';
  if (ext === '.ogg') return 'audio/ogg';
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
  if (ext === '.png') return 'image/png';
  return 'application/octet-stream';
}

async function predictText(text) {
  try {
    const resp = await client.post('/predict/text', { text });
    return resp.data;
  } catch (err) {
    console.error('ML predictText error:', err.response?.data || err.message);
    throw new Error('ML text prediction failed');
  }
}

async function predictVoice(buffer, filename = 'audio.wav') {
  try {
    const form = new FormData();
    // ML expects field name "audio" for the multipart route
    const mime = guessMimeFromFilename(filename);
    form.append('audio', buffer, { filename, contentType: mime });

    const headers = form.getHeaders();
    const resp = await client.post('/predict/voice', form, { headers });
    return resp.data;
  } catch (err) {
    console.error('ML predictVoice error:', err.response?.data || err.message);
    // fallback: try sending JSON base64 (ML supports this)
    try {
      const b64 = buffer.toString('base64');
      const resp2 = await client.post('/predict/voice', {
        audio_b64: b64,
        filename,
      });
      return resp2.data;
    } catch (err2) {
      console.error(
        'ML predictVoice fallback error:',
        err2.response?.data || err2.message
      );
      throw new Error('ML voice prediction failed');
    }
  }
}

async function predictFace(buffer, filename = 'image.jpg') {
  try {
    const form = new FormData();
    const mime = guessMimeFromFilename(filename);
    // ML expects field name "image"
    form.append('image', buffer, { filename, contentType: mime });

    const headers = form.getHeaders();
    const resp = await client.post('/predict/face', form, { headers });
    return resp.data;
  } catch (err) {
    console.error('ML predictFace error:', err.response?.data || err.message);
    // fallback to base64 JSON
    try {
      const b64 = buffer.toString('base64');
      const resp2 = await client.post('/predict/face', {
        image_b64: b64,
        filename,
      });
      return resp2.data;
    } catch (err2) {
      console.error(
        'ML predictFace fallback error:',
        err2.response?.data || err2.message
      );
      throw new Error('ML face prediction failed');
    }
  }
}

module.exports = {
  predictText,
  predictVoice,
  predictFace,
};
