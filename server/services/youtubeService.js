// services/youtubeService.js
const axios = require('axios');
const YT_KEY = process.env.YT_API_KEY;
if (!YT_KEY)
  console.warn(
    'YT_API_KEY not set in .env — YouTube recommendations will fail'
  );

const YT_BASE = 'https://www.googleapis.com/youtube/v3';

async function searchVideos(query, maxResults = 5) {
  if (!YT_KEY) return [];
  try {
    const params = {
      key: YT_KEY,
      part: 'snippet',
      q: query,
      type: 'video',
      maxResults,
    };
    const resp = await axios.get(`${YT_BASE}/search`, { params });
    const items = (resp.data.items || []).map((it) => {
      const vid = it.id?.videoId || it.id;
      return {
        type: 'youtube',
        id: vid,
        title: it.snippet?.title || null,
        channelTitle: it.snippet?.channelTitle || null,
        thumbnail:
          it.snippet?.thumbnails?.medium?.url ||
          it.snippet?.thumbnails?.default?.url ||
          null,
        url: vid ? `https://www.youtube.com/watch?v=${vid}` : null,
      };
    });
    return items;
  } catch (err) {
    console.error('YouTube search error:', err.response?.data || err.message);
    return [];
  }
}

module.exports = { searchVideos };
