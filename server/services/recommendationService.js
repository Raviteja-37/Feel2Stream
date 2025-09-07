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
  let query;

  if (customText && customText.trim()) {
    // Custom text has highest priority
    query = customText.trim();
  } else {
    // Build query intelligently for YouTube and Spotify
    const youtubeParts = [emotion];
    const spotifyParts = [emotion];

    if (interest) youtubeParts.push(interest); // interest may already have language appended
    if (genre) spotifyParts.push(genre); // genre may already have language appended

    youtubeParts.push('video'); // helps YouTube search
    spotifyParts.push('music'); // helps Spotify search

    // Combine into single string for backend (both services use same query field)
    query = `${youtubeParts.join(' ')} | ${spotifyParts.join(' ')}`;
  }

  // Fetch both YouTube and Spotify in parallel
  const [youtubeResults, spotifyResults] = await Promise.all([
    youtubeService.searchVideos(query, 5),
    spotifyService.searchTracks(query, 5),
  ]);

  return [...youtubeResults, ...spotifyResults];
}

module.exports = {
  generateRecommendations,
};
