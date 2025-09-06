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
import ReactPlayer from 'react-player'; // ensure installed: npm install react-player
import './index.css'; // optional, create minimal styles if you want

const INTERESTS = [
  'random',
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

export default function Dashboard() {
  const [mode, setMode] = useState('text'); // 'text' | 'voice' | 'face'
  const [textInput, setTextInput] = useState('');
  const [interest, setInterest] = useState('random');
  const [genre, setGenre] = useState('indie pop');
  const [customText, setCustomText] = useState('');
  const [items, setItems] = useState([]); // results (yt + spotify)
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  // voice recording
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [isRecording, setIsRecording] = useState(false);
  const streamAudioRef = useRef(null);
  const recordRef = useRef(null);

  // face webcam
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceIntervalRef = useRef(null);
  const streamVideoRef = useRef(null);
  const [faceActive, setFaceActive] = useState(false);
  const [lastFaceSentAt, setLastFaceSentAt] = useState(0);

  // saved recommendations (for logged-in users)
  const [savedRecs, setSavedRecs] = useState([]);

  const token = Cookies.get('token');

  useEffect(() => {
    if (token) {
      fetchSavedRecs().catch((e) => {
        // ignore
      });
    }
    // cleanup on unmount
    return () => {
      stopFaceCapture();
      stopRecording();
    };
    // eslint-disable-next-line
  }, []);

  // fetch saved recommendations
  async function fetchSavedRecs() {
    try {
      const data = await getMyRecommendations();
      setSavedRecs(data.recommendations || []);
    } catch (err) {
      console.warn('No saved recs / not authenticated', err?.message || err);
      setSavedRecs([]);
    }
  }

  // ---------- TEXT ----------
  async function handleSubmitText(e) {
    e?.preventDefault();
    if (!textInput?.trim()) return;
    setLoading(true);
    try {
      const resp = await predictText(textInput.trim());
      setEmotion(resp.emotion || resp.label);
      setConfidence(resp.confidence || resp.score || null);
      setItems(resp.items || []);
      // optionally, if no items returned, you can call generate endpoint:
      if (!resp.items || resp.items.length === 0) {
        const gen = await generateFromEmotion({
          emotion: resp.emotion || resp.label,
          interest,
          customText,
          genre,
        });
        setItems(gen.items || []);
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
    }
  }

  async function stopRecording() {
    if (!recordRef.current) return;
    recordRef.current.stopRecording(async () => {
      const blob = recordRef.current.getBlob(); // WAV blob
      await sendVoiceBlob(blob, 'recording.wav');
      recordRef.current = null;
      setIsRecording(false);
    });
  }

  async function sendVoiceBlob(blob, filename = 'recording.wav') {
    setLoading(true);
    try {
      const buffer = await blob.arrayBuffer(); // convert Blob to ArrayBuffer
      const resp = await predictVoiceBlob(blob, filename); // pass Buffer
      setEmotion(resp.emotion || resp.label);
      setConfidence(resp.confidence || resp.score || null);
      setItems(resp.items || []);

      if (!resp.items || resp.items.length === 0) {
        const gen = await generateFromEmotion({
          emotion: resp.emotion || resp.label,
          interest,
          customText,
          genre,
        });
        setItems(gen.items || []);
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

      // capture every 3 seconds
      faceIntervalRef.current = setInterval(async () => {
        // avoid overlapping requests: skip if last sent < 2500ms
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
    try {
      const resp = await predictFaceBlob(blob, 'snapshot.jpg');
      setEmotion(resp.emotion || resp.label);
      setConfidence(resp.confidence || resp.score || null);
      setItems(resp.items || []);
      // fallback to generator if no items returned
      if (!resp.items || resp.items.length === 0) {
        const gen = await generateFromEmotion({
          emotion: resp.emotion || resp.label,
          interest,
          customText,
          genre,
        });
        setItems(gen.items || []);
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
    } catch (e) {
      // ignore
    }
  }

  // ---------- GENERATE FROM EMOTION (manual) ----------
  async function handleGenerateFromEmotion() {
    if (!emotion) {
      alert(
        'No detected emotion yet. Use text/voice/face to get emotion first.'
      );
      return;
    }
    setLoading(true);
    try {
      const resp = await generateFromEmotion({
        emotion,
        interest: interest !== 'random' ? interest : undefined,
        customText: customText || undefined,
        genre: genre || undefined,
      });
      setItems(resp.items || []);
      await fetchSavedRecs();
    } catch (err) {
      console.error('generateFromEmotion error', err);
      alert('Failed to generate recommendations.');
    } finally {
      setLoading(false);
    }
  }

  // ---------- RENDER helpers ----------
  function renderItem(it, idx) {
    // ---------- YouTube ----------
    if (it.type === 'youtube' && it.id) {
      // Check for it.id instead of it.url
      // ✅ FIX: Use the more reliable /embed/ URL format
      const embedUrl = `https://www.youtube.com/embed/${it.id}`;

      return (
        <div key={it.id} className="yt-card">
          <div className="yt-title">
            {it.title?.replace(/&quot;/g, '"') || it.name}
          </div>
          <div
            className="player-wrapper"
            style={{ position: 'relative', paddingTop: '56.25%' }}
          >
            <ReactPlayer
              src={embedUrl} // Use the new embedUrl
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

    // ---------- Spotify ----------
    if (it.type === 'spotify') {
      const spotifyEmbedUrl = it.id
        ? `https://open.spotify.com/embed/track/${it.id}`
        : it.spotify_url;

      return (
        <div key={it.id} className="result-item spotify-card">
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

    // ---------- fallback ----------
    return (
      <pre key={idx} style={{ width: '100%', whiteSpace: 'pre-wrap' }}>
        {JSON.stringify(it, null, 2)}
      </pre>
    );
  }

  // UI for mode controls
  return (
    <div className="dashboard-root">
      <h2>Welcome to Feel2Stream Dashboard</h2>

      <div className="controls">
        <label>
          Mode:
          <select
            value={mode}
            onChange={(e) => {
              // clean up any active streams when switching
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
            <option value="face">Face (webcam)</option>
          </select>
        </label>

        <label>
          Interest (YouTube):
          <select
            value={interest}
            onChange={(e) => setInterest(e.target.value)}
          >
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
            {GENRES.map((g) => (
              <option key={g} value={g}>
                {g}
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
          Generate (refresh results for current emotion)
        </button>
      </div>

      <div className="mode-area">
        {/* TEXT */}
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
                {loading ? 'Analyzing...' : 'Analyze Text'}
              </button>
            </div>
          </form>
        )}

        {/* VOICE */}
        {mode === 'voice' && (
          <div className="voice-area">
            <p>Record audio and we will analyze emotion from your voice.</p>
            <div>
              {!isRecording ? (
                <button onClick={startRecording}>Start Recording</button>
              ) : (
                <button onClick={stopRecording}>Stop & Analyze</button>
              )}
            </div>
            <p style={{ color: 'gray' }}>
              Recorded format: browser default (webm/ogg). If backend expects
              WAV and fails, we will need a WAV converter in JS or server-side
              FFmpeg.
            </p>
          </div>
        )}

        {/* FACE */}
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
                <button onClick={startFaceCapture}>
                  Start Webcam (capture every 3s)
                </button>
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
        <h3>Recommendations / Results</h3>
        {loading && <div>Loading results...</div>}
        {items && items.length === 0 && <div>No items found yet.</div>}
        <div className="results-grid">
          {items?.map((it, idx) => renderItem(it, idx))}
        </div>
      </div>

      <div className="saved-recs">
        <h3>Saved Recommendations (your last saved items)</h3>
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
