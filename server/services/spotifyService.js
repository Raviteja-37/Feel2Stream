// services/spotifyService.js
const axios = require('axios');
const qs = require('qs');

const CLIENT_ID = process.env.SPOTIFY_CLIENT_ID;
const CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET;
if (!CLIENT_ID || !CLIENT_SECRET)
  console.warn('Spotify client id/secret not set');

let spotifyTokenCache = {
  access_token: null,
  expires_at: 0,
};

async function getSpotifyToken() {
  const now = Date.now();
  if (
    spotifyTokenCache.access_token &&
    spotifyTokenCache.expires_at > now + 5000
  ) {
    return spotifyTokenCache.access_token;
  }
  if (!CLIENT_ID || !CLIENT_SECRET) return null;

  try {
    const tokenUrl = 'https://accounts.spotify.com/api/token';
    const data = qs.stringify({ grant_type: 'client_credentials' });
    const headers = {
      Authorization: `Basic ${Buffer.from(
        `${CLIENT_ID}:${CLIENT_SECRET}`
      ).toString('base64')}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    };
    const resp = await axios.post(tokenUrl, data, { headers });
    const token = resp.data.access_token;
    const expires_in = resp.data.expires_in || 3600;
    spotifyTokenCache = {
      access_token: token,
      expires_at: Date.now() + expires_in * 1000,
    };
    return token;
  } catch (err) {
    console.error('Spotify token error', err.response?.data || err.message);
    return null;
  }
}

async function searchTracks(query, limit = 6) {
  try {
    const token = await getSpotifyToken();
    if (!token) return [];
    const resp = await axios.get('https://api.spotify.com/v1/search', {
      headers: { Authorization: `Bearer ${token}` },
      params: { q: query, type: 'track', limit },
    });

    const tracks = (resp.data.tracks?.items || []).map((t) => ({
      type: 'spotify',
      id: t.id,
      name: t.name,
      artists: t.artists.map((a) => a.name).join(', '),
      preview_url: t.preview_url,
      spotify_url: t.external_urls?.spotify,
      album: t.album?.name,
      album_image:
        t.album?.images?.[1]?.url || t.album?.images?.[0]?.url || null,
    }));
    return tracks;
  } catch (err) {
    console.error('Spotify search error', err.response?.data || err.message);
    return [];
  }
}

module.exports = { searchTracks };
