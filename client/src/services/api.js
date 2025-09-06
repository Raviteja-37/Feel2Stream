import axios from 'axios';
import Cookies from 'js-cookie';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

function authHeaders() {
  const token = Cookies.get('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ---------- TEXT ----------
export async function predictText(text) {
  const resp = await client.post(
    '/api/emotion/text',
    { text },
    { headers: authHeaders() }
  );
  return resp.data;
}

// ---------- VOICE ----------
export async function predictVoiceBlob(blob, filename = 'recording.wav') {
  const fd = new FormData();
  fd.append('audio', blob, filename);
  const resp = await client.post('/api/emotion/voice', fd, {
    headers: { ...authHeaders(), 'Content-Type': 'multipart/form-data' },
  });
  return resp.data;
}

// ---------- FACE ----------
export async function predictFaceBlob(blob, filename = 'snapshot.jpg') {
  const fd = new FormData();
  fd.append('image', blob, filename);
  const resp = await client.post('/api/emotion/face', fd, {
    headers: { ...authHeaders() },
  });
  return resp.data;
}

// ---------- GENERATE FROM EMOTION ----------
export async function generateFromEmotion({
  emotion,
  interest,
  customText,
  genre,
}) {
  const body = { emotion };
  if (interest) body.interest = interest;
  if (customText) body.customText = customText;
  if (genre) body.genre = genre;
  const resp = await client.post('/api/recs/generate', body, {
    headers: authHeaders(),
  });
  return resp.data;
}

// ---------- SAVED RECOMMENDATIONS ----------
export async function getMyRecommendations() {
  const resp = await client.get('/api/recs/my', { headers: authHeaders() });
  return resp.data;
}
