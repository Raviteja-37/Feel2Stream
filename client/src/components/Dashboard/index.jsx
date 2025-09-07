// src/components/Dashboard.jsx
import React, { useEffect, useRef, useState } from 'react';
import {
  predictText,
  predictVoiceBlob,
  predictFaceBlob,
  generateFromEmotion,
  getMyRecommendations,
} from '../../services/api.js';
import RecordRTC from 'recordrtc';
import Cookies from 'js-cookie';
import ReactPlayer from 'react-player';
import { ButtonLoader } from '../Loader';
import './index.css';

const INTERESTS = [
  'sports',
  'songs',
  'movies',
  'fun',
  'comedy',
  'debates',
  'other',
];
const GENRES = [
  'indie pop',
  'rock',
  'electronic',
  'dance',
  'classical',
  'jazz',
];
const LANGUAGES = [
  'english',
  'hindi',
  'telugu',
  'spanish',
  'french',
  'german',
  'japanese',
  'korean',
];

export default function Dashboard() {
  const [mode, setMode] = useState('text');
  const [textInput, setTextInput] = useState('');
  const [interest, setInterest] = useState('');
  const [genre, setGenre] = useState('');
  const [language, setLanguage] = useState('');
  const [customText, setCustomText] = useState('');
  const [items, setItems] = useState([]);
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  // voice recording
  const recordRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);

  // face webcam
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceIntervalRef = useRef(null);
  const streamVideoRef = useRef(null);
  const [faceActive, setFaceActive] = useState(false);
  const [lastFaceSentAt, setLastFaceSentAt] = useState(0);

  // saved recommendations
  const [savedRecs, setSavedRecs] = useState([]);
  const token = Cookies.get('token');

  useEffect(() => {
    if (token) fetchSavedRecs().catch(() => {});
    return () => {
      stopFaceCapture();
      stopRecording();
    };
    // eslint-disable-next-line
  }, []);

  async function fetchSavedRecs() {
    try {
      const data = await getMyRecommendations();
      setSavedRecs(data.recommendations || []);
    } catch {
      setSavedRecs([]);
    }
  }

  // Helper: build generate payload (language is appended to customText as fallback)
  function buildGeneratePayload(forEmotion) {
    // Build a fallback customText only if user typed something
    const fallbackCustom = customText?.trim() || undefined;

    // Append language to interest (for YouTube) and genre (for Spotify) in backend
    // So we don't need to send extra params
    const interestWithLang = interest
      ? language
        ? `${interest} ${language}`
        : interest
      : undefined;

    const genreWithLang = genre
      ? language
        ? `${genre} ${language}`
        : genre
      : undefined;

    // Build payload
    const payload = {
      emotion: forEmotion,
      interest: interestWithLang, // YouTube
      genre: genreWithLang, // Spotify
      customText: fallbackCustom, // highest priority
    };

    // Remove undefined keys
    Object.keys(payload).forEach((key) => {
      if (payload[key] === undefined) delete payload[key];
    });

    return payload;
  }

  // ---------- TEXT ----------
  async function handleSubmitText(e) {
    e?.preventDefault();
    if (!textInput?.trim()) return;
    setLoading(true);
    setItems([]); // clear old results immediately
    try {
      const resp = await predictText(textInput.trim());
      setEmotion(resp.emotion || resp.label);
      setConfidence(resp.confidence || resp.score || null);

      // If user selected any filter (including language), ask generator for filtered results.
      const anyFilterSelected = Boolean(
        language || interest || genre || customText
      );
      if (anyFilterSelected) {
        const payload = buildGeneratePayload(resp.emotion || resp.label);
        console.log('generateFromEmotion payload (text):', payload);
        const gen = await generateFromEmotion(payload);
        setItems(gen.items || resp.items || []);
      } else {
        // default behaviour: use resp.items, fallback to generator only if empty
        setItems(resp.items || []);
        if (!resp.items || resp.items.length === 0) {
          const payload = buildGeneratePayload(resp.emotion || resp.label);
          console.log('generateFromEmotion payload (text fallback):', payload);
          const gen = await generateFromEmotion(payload);
          setItems(gen.items || []);
        }
      }
    } catch (err) {
      console.error('predictText failed', err);
      alert('Text prediction failed. Check network / server.');
    } finally {
      setLoading(false);
    }
  }

  // ---------- VOICE ----------
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordRef.current = new RecordRTC(stream, {
        type: 'audio',
        mimeType: 'audio/wav',
        recorderType: RecordRTC.StereoAudioRecorder,
        numberOfAudioChannels: 1,
      });
      recordRef.current.startRecording();
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      alert('Could not start recording. Check microphone permissions.');
    }
  }

  async function stopRecording() {
    if (!recordRef.current) return;
    recordRef.current.stopRecording(async () => {
      const blob = recordRef.current.getBlob();
      await sendVoiceBlob(blob, 'recording.wav');
      recordRef.current = null;
      setIsRecording(false);
    });
  }

  async function sendVoiceBlob(blob, filename = 'recording.wav') {
    setLoading(true);
    setItems([]); // clear old results immediately
    try {
      const resp = await predictVoiceBlob(blob, filename);
      setEmotion(resp.emotion || resp.label);
      setConfidence(resp.confidence || resp.score || null);

      const anyFilterSelected = Boolean(
        language || interest || genre || customText
      );
      if (anyFilterSelected) {
        const payload = buildGeneratePayload(resp.emotion || resp.label);
        console.log('generateFromEmotion payload (voice):', payload);
        const gen = await generateFromEmotion(payload);
        setItems(gen.items || resp.items || []);
      } else {
        setItems(resp.items || []);
        if (!resp.items || resp.items.length === 0) {
          const payload = buildGeneratePayload(resp.emotion || resp.label);
          console.log('generateFromEmotion payload (voice fallback):', payload);
          const gen = await generateFromEmotion(payload);
          setItems(gen.items || []);
        }
      }

      await fetchSavedRecs();
    } catch (err) {
      console.error('predictVoice failed', err);
      alert('Voice prediction failed. Check server logs.');
    } finally {
      setLoading(false);
    }
  }

  // ---------- FACE ----------
  async function startFaceCapture() {
    if (faceActive) return;
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      streamVideoRef.current = s;
      videoRef.current.srcObject = s;
      videoRef.current.play();
      setFaceActive(true);

      faceIntervalRef.current = setInterval(async () => {
        const now = Date.now();
        if (now - lastFaceSentAt < 2500) return;
        if (!canvasRef.current || !videoRef.current) return;
        const v = videoRef.current;
        const c = canvasRef.current;
        c.width = v.videoWidth || 320;
        c.height = v.videoHeight || 240;
        const ctx = c.getContext('2d');
        ctx.drawImage(v, 0, 0, c.width, c.height);
        await new Promise((res) => {
          c.toBlob(
            async (blob) => {
              if (blob) {
                setLastFaceSentAt(Date.now());
                await sendFaceBlob(blob);
              }
              res();
            },
            'image/jpeg',
            0.8
          );
        });
      }, 3000);
    } catch (err) {
      console.error('startFaceCapture error', err);
      alert(
        'Could not access webcam. Grant permission or try another browser.'
      );
    }
  }

  async function sendFaceBlob(blob) {
    setLoading(true);
    setItems([]); // clear old results immediately
    try {
      const resp = await predictFaceBlob(blob, 'snapshot.jpg');
      setEmotion(resp.emotion || resp.label);
      setConfidence(resp.confidence || resp.score || null);

      const anyFilterSelected = Boolean(
        language || interest || genre || customText
      );
      if (anyFilterSelected) {
        const payload = buildGeneratePayload(resp.emotion || resp.label);
        console.log('generateFromEmotion payload (face):', payload);
        const gen = await generateFromEmotion(payload);
        setItems(gen.items || resp.items || []);
      } else {
        setItems(resp.items || []);
        if (!resp.items || resp.items.length === 0) {
          const payload = buildGeneratePayload(resp.emotion || resp.label);
          console.log('generateFromEmotion payload (face fallback):', payload);
          const gen = await generateFromEmotion(payload);
          setItems(gen.items || []);
        }
      }

      await fetchSavedRecs();
    } catch (err) {
      console.error('predictFace failed', err);
    } finally {
      setLoading(false);
    }
  }

  function stopFaceCapture() {
    try {
      if (faceIntervalRef.current) {
        clearInterval(faceIntervalRef.current);
        faceIntervalRef.current = null;
      }
      if (streamVideoRef.current) {
        streamVideoRef.current.getTracks().forEach((t) => t.stop());
        streamVideoRef.current = null;
      }
      setFaceActive(false);
    } catch {}
  }

  // ---------- GENERATE ----------
  async function handleGenerateFromEmotion() {
    if (!emotion) {
      alert(
        'No detected emotion yet. Use text/voice/face to get emotion first.'
      );
      return;
    }
    if (!interest && !genre && !language && !customText) {
      alert('Please select at least one filter before generating.');
      return;
    }
    setLoading(true);
    setItems([]);
    try {
      const payload = buildGeneratePayload(emotion);
      console.log('generateFromEmotion payload (manual):', payload);
      const resp = await generateFromEmotion(payload);
      setItems(resp.items || []);
      await fetchSavedRecs();
    } catch (err) {
      console.error('generateFromEmotion error', err);
      alert('Failed to generate recommendations.');
    } finally {
      setLoading(false);
    }
  }

  // ---------- RENDER ----------
  function renderItem(it, idx) {
    if (it.type === 'youtube' && it.id) {
      const embedUrl = `https://www.youtube.com/embed/${it.id}`;
      return (
        <div key={it.id || idx} className="yt-card">
          <div className="yt-title">
            {it.title?.replace(/&quot;/g, '"') || it.name}
          </div>
          <div
            className="player-wrapper"
            style={{ position: 'relative', paddingTop: '56.25%' }}
          >
            <ReactPlayer
              src={embedUrl}
              controls
              width="100%"
              height="100%"
              style={{ position: 'absolute', top: 0, left: 0 }}
            />
          </div>
          <a href={it.url} target="_blank" rel="noreferrer">
            Open on YouTube
          </a>
        </div>
      );
    }

    if (it.type === 'spotify') {
      const spotifyEmbedUrl = it.id
        ? `https://open.spotify.com/embed/track/${it.id}`
        : it.spotify_url;
      return (
        <div key={it.id || idx} className="spotify-card">
          <div className="track-title">{it.title || it.name}</div>
          <div className="track-artist">{it.artist || it.artists || ''}</div>
          {spotifyEmbedUrl ? (
            <iframe
              title={it.title || it.name}
              src={spotifyEmbedUrl}
              width="100%"
              height="152"
              frameBorder="0"
              allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
              loading="lazy"
            />
          ) : it.preview_url ? (
            <audio controls src={it.preview_url} />
          ) : null}
          <a href={it.spotify_url || it.url} target="_blank" rel="noreferrer">
            Open on Spotify
          </a>
        </div>
      );
    }

    return (
      <pre key={idx} style={{ width: '100%', whiteSpace: 'pre-wrap' }}>
        {JSON.stringify(it, null, 2)}
      </pre>
    );
  }

  return (
    <div className="dashboard-root">
      <h2>Welcome to Feel2Stream Dashboard</h2>

      <div className="controls">
        <label>
          Mode:
          <select
            value={mode}
            onChange={(e) => {
              if (mode === 'face') stopFaceCapture();
              if (mode === 'voice') stopRecording();
              setMode(e.target.value);
              setItems([]);
              setEmotion(null);
              setConfidence(null);
            }}
          >
            <option value="text">Text</option>
            <option value="voice">Voice</option>
            <option value="face">Face</option>
          </select>
        </label>

        <label>
          Interest (YouTube):
          <select
            value={interest}
            onChange={(e) => setInterest(e.target.value)}
          >
            <option value="">Select interest</option>
            {INTERESTS.map((it) => (
              <option key={it} value={it}>
                {it}
              </option>
            ))}
          </select>
        </label>

        <label>
          Spotify Genre:
          <select value={genre} onChange={(e) => setGenre(e.target.value)}>
            <option value="">Select genre</option>
            {GENRES.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </label>

        <label>
          Language:
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="">Select language</option>
            {LANGUAGES.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </label>

        <label>
          Custom (other search text):
          <input
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            placeholder="Type custom search text (optional)"
          />
        </label>

        <button
          onClick={handleGenerateFromEmotion}
          disabled={!emotion || loading}
        >
          {loading ? <ButtonLoader /> : 'Generate (refresh results)'}
        </button>
      </div>

      <div className="mode-area">
        {mode === 'text' && (
          <form onSubmit={handleSubmitText} className="text-area">
            <textarea
              placeholder="Type how you feel..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              rows={4}
            />
            <div>
              <button type="submit" disabled={loading}>
                {loading ? <ButtonLoader /> : 'Analyze Text'}
              </button>
            </div>
          </form>
        )}

        {mode === 'voice' && (
          <div className="voice-area">
            <p>Record audio and we will analyze emotion.</p>
            <div>
              {!isRecording ? (
                <button onClick={startRecording}>Start Recording</button>
              ) : (
                <button onClick={stopRecording}>Stop & Analyze</button>
              )}
            </div>
          </div>
        )}

        {mode === 'face' && (
          <div className="face-area">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ width: 320, height: 240, background: '#000' }}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <div>
              {!faceActive ? (
                <button onClick={startFaceCapture}>Start Webcam</button>
              ) : (
                <button onClick={stopFaceCapture}>Stop Webcam</button>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="status">
        <strong>Detected emotion:</strong> {emotion || '—'}{' '}
        {confidence ? `(${(confidence * 100).toFixed(1)}%)` : ''}
      </div>

      <div className="results">
        <h3>Recommendations</h3>
        {loading && (
          <div style={{ margin: 12 }}>
            <ButtonLoader />
          </div>
        )}
        {items.length === 0 && !loading && <div>No items found yet.</div>}
        <div className="results-grid">
          {items.map((it, idx) => renderItem(it, idx))}
        </div>
      </div>

      <div className="saved-recs">
        <h3>Saved Recommendations</h3>
        {savedRecs.length === 0 && <div>No saved recommendations yet.</div>}
        {savedRecs.map((r, i) => (
          <div key={i} className="saved-rec">
            <div>
              <strong>{r.emotion}</strong> —{' '}
              {new Date(r.createdAt).toLocaleString()}
            </div>
            <div style={{ marginTop: 6 }}>
              {(r.items || []).slice(0, 3).map((it, idx) => (
                <div key={idx} style={{ fontSize: 13 }}>
                  {it.title || it.name || JSON.stringify(it).slice(0, 80)}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
