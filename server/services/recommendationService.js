// services/recommendationService.js
const youtubeService = require('./youtubeService');
const spotifyService = require('./spotifyService');

const emotionKeywords = {
  happy: ['happy songs', 'uplifting music', 'energetic playlist'],
  sad: ['sad songs', 'emotional music', 'soft piano'],
  angry: ['calm music', 'chill playlist', 'relaxing beats'],
  fear: ['motivational songs', 'confidence boost', 'inspirational music'],
  surprise: ['trending music', 'new releases', 'exciting hits'],
  neutral: ['lofi beats', 'focus music', 'chill background'],
  disgust: ['calm classical', 'peaceful instrumental'],
};

async function generateRecommendations({
  emotion,
  interest,
  customText,
  genre,
} = {}) {
  // build a search query string based on frontend inputs
  let query;
  if (customText && customText.trim()) {
    query = customText.trim();
  } else if (interest && interest.trim()) {
    query = `${emotion} ${interest}`.trim();
  } else if (genre && genre.trim()) {
    query = `${emotion} ${genre} music`.trim();
  } else {
    const keywords = emotionKeywords[emotion] || ['music playlist'];
    query = keywords[Math.floor(Math.random() * keywords.length)];
  }

  // fetch both youtube and spotify results in parallel
  const [youtubeResults, spotifyResults] = await Promise.all([
    youtubeService.searchVideos(query, 5),
    spotifyService.searchTracks(query, 5),
  ]);

  // return combined array (frontend can filter by type if needed)
  return [...youtubeResults, ...spotifyResults];
}

module.exports = {
  generateRecommendations,
};
