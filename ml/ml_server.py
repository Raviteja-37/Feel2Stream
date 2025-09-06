# ml_server.py
"""
FastAPI ML prediction server for Feel2Stream

Endpoints:
  GET  /health
  POST /predict/text   -> JSON { "text": "..." }
  POST /predict/voice  -> JSON { "audio_b64": "...", "filename": "x.wav" } OR multipart form with field "audio"
  POST /predict/face   -> JSON { "image_b64": "...", "filename": "x.jpg" } OR multipart form with field "image"

Environment variables (all optional):
  TEXT_VECTORIZER_PATH  - path to text_vectorizer.joblib
  TEXT_CLF_PATH         - path to text_clf.joblib
  TEXT_LABELS_PATH      - path to text_labels.json
  VOICE_MODEL_PATH      - path to voice Keras model (.keras or .h5)
  FACE_MODEL_PATH       - path to face Keras model (.keras or .h5)
  FACE_USE_MOBILENET_PREPROCESS - "1" or "0" (default 1) apply MobileNetV2 preprocess for face (if model expects RGB)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU to avoid GPU/MPS issues on dev machines
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64
import tempfile
import json
import numpy as np
import traceback
from typing import Optional, Dict

# ML libs
try:
    import joblib
    import librosa
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from PIL import Image
    from io import BytesIO
except Exception as e:
    raise RuntimeError("Missing ML dependencies. Install requirements: fastapi uvicorn joblib librosa tensorflow pillow scikit-learn") from e

app = FastAPI(title="Feel2Stream ML server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Config / model paths
# ---------------------------
BASE_DIR = os.path.dirname(__file__) or "."
def _join_candidates(*cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

TEXT_VECTORIZER_PATH = os.environ.get(
    "TEXT_VECTORIZER_PATH",
    _join_candidates(
        os.path.join(BASE_DIR, "text_emotion", "models", "text_vectorizer.joblib"),
        os.path.join(BASE_DIR, "text_emotion", "text_vectorizer.joblib"),
        os.path.join(BASE_DIR, "text_emotion", "models", "text_vectorizer.pkl"),
    ),
)
TEXT_CLF_PATH = os.environ.get(
    "TEXT_CLF_PATH",
    _join_candidates(
        os.path.join(BASE_DIR, "text_emotion", "models", "text_clf.joblib"),
        os.path.join(BASE_DIR, "text_emotion", "text_clf.joblib"),
    ),
)
TEXT_LABELS_PATH = os.environ.get(
    "TEXT_LABELS_PATH",
    _join_candidates(
        os.path.join(BASE_DIR, "text_emotion", "models", "text_labels.json"),
        os.path.join(BASE_DIR, "text_emotion", "text_labels.json"),
    ),
)

VOICE_MODEL_PATH = os.environ.get(
    "VOICE_MODEL_PATH",
    _join_candidates(
        os.path.join(BASE_DIR, "voice_emotion", "voice_emotion_model.keras"),
        os.path.join(BASE_DIR, "voice_emotion", "voice_emotion_model.h5"),
        os.path.join(BASE_DIR, "voice_emotion", "voice_emotion_model.keras"),
    ),
)

FACE_MODEL_PATH = os.environ.get(
    "FACE_MODEL_PATH",
    _join_candidates(
        os.path.join(BASE_DIR, "face_emotion", "face_emotion_model_final.keras"),
        os.path.join(BASE_DIR, "face_emotion", "face_emotion_model.keras"),
        os.path.join(BASE_DIR, "face_emotion", "face_emotion_model.h5"),
    ),
)

FACE_USE_MOBILENET_PREPROCESS = os.environ.get("FACE_USE_MOBILENET_PREPROCESS", "1") == "1"

# ---------------------------
# Globals to be loaded on startup
# ---------------------------
_text_vectorizer = None
_text_clf = None
_text_labels = None

_voice_model = None
_voice_input_shape = None  # (time_steps, n_mfcc)

_face_model = None
_face_input_shape = None  # (h, w, c)
_face_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# optional import for mobilenet preprocess
_mobilenet_preprocess = None
if FACE_USE_MOBILENET_PREPROCESS:
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _mobilenet_preprocess
    except Exception:
        _mobilenet_preprocess = None

# ---------------------------
# Helpers
# ---------------------------
def load_text_models():
    global _text_vectorizer, _text_clf, _text_labels
    if not TEXT_VECTORIZER_PATH or not os.path.exists(TEXT_VECTORIZER_PATH):
        raise FileNotFoundError(f"text vectorizer not found. looked at: {TEXT_VECTORIZER_PATH}")
    if not TEXT_CLF_PATH or not os.path.exists(TEXT_CLF_PATH):
        raise FileNotFoundError(f"text classifier not found. looked at: {TEXT_CLF_PATH}")
    if not TEXT_LABELS_PATH or not os.path.exists(TEXT_LABELS_PATH):
        raise FileNotFoundError(f"text labels json not found. looked at: {TEXT_LABELS_PATH}")

    _text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
    _text_clf = joblib.load(TEXT_CLF_PATH)
    with open(TEXT_LABELS_PATH, "r") as f:
        _text_labels = json.load(f)
    print("✅ Loaded text vectorizer & classifier. labels:", len(_text_labels))

def load_voice_model():
    global _voice_model, _voice_input_shape
    if not VOICE_MODEL_PATH or not os.path.exists(VOICE_MODEL_PATH):
        raise FileNotFoundError(f"voice model not found. looked at: {VOICE_MODEL_PATH}")
    _voice_model = load_model(VOICE_MODEL_PATH, compile=False)
    # attempt to infer input shape (None, time_steps, features) or (time_steps, features)
    try:
        inp = _voice_model.input_shape  # tuple: (None, time_steps, features) or similar
        if isinstance(inp, tuple) and len(inp) >= 3:
            _voice_input_shape = (inp[1], inp[2])
        elif isinstance(inp, tuple) and len(inp) == 2:
            _voice_input_shape = (inp[0], inp[1])
        else:
            _voice_input_shape = (174, 40)
    except Exception:
        _voice_input_shape = (174, 40)
    print("✅ Loaded voice model. inferred input shape (time_steps,features):", _voice_input_shape)

def load_face_model():
    global _face_model, _face_input_shape
    if not FACE_MODEL_PATH or not os.path.exists(FACE_MODEL_PATH):
        raise FileNotFoundError(f"face model not found. looked at: {FACE_MODEL_PATH}")
    _face_model = load_model(FACE_MODEL_PATH, compile=False)
    try:
        inp = _face_model.input_shape  # (None, h, w, c)
        _face_input_shape = (int(inp[1]), int(inp[2]), int(inp[3]))
    except Exception:
        _face_input_shape = (48, 48, 3)
    print("✅ Loaded face model. input shape:", _face_input_shape)

def safe_decode_b64(b64str: str) -> bytes:
    try:
        return base64.b64decode(b64str)
    except Exception as e:
        raise ValueError("Invalid base64 data") from e

def extract_mfcc_from_wav_bytes(wav_bytes: bytes, target_time_steps=174, n_mfcc=40):
    # write to temp file because librosa.soundfile expects a file
    tmp = None
    try:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(wav_bytes)
        tmp.flush()
        tmp.close()
        y, sr = librosa.load(tmp.name, sr=None, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # ensure dimension (n_mfcc, time_steps)
        if mfcc.shape[1] < target_time_steps:
            pad_width = target_time_steps - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_time_steps]
        # transpose to (time_steps, n_mfcc)
        mfcc = mfcc.T.astype(np.float32)
        # normalize sample-wise
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
        return mfcc  # shape (time_steps, n_mfcc)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

def preprocess_face_image_bytes(img_bytes: bytes, target_wh=(48,48), channels=3, use_mobilenet_preprocess=False):
    # load with PIL
    img = Image.open(BytesIO(img_bytes)).convert("RGB" if channels==3 else "L")
    img = img.resize(target_wh)
    arr = np.array(img).astype(np.float32)
    if channels == 1 and arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif channels == 3 and arr.ndim == 2:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    # if mobilenet preprocessing is used, pass raw 0-255 image to preprocess_input
    if use_mobilenet_preprocess and _mobilenet_preprocess is not None:
        arr = _mobilenet_preprocess(arr)
    else:
        arr = arr / 255.0
    # expand batch
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ---------------------------
# Prediction helpers
# ---------------------------
def predict_text_from_string(text: str, top_k: int = 1):
    global _text_vectorizer, _text_clf, _text_labels
    if _text_vectorizer is None or _text_clf is None or _text_labels is None:
        raise RuntimeError("Text model not loaded")
    vec = _text_vectorizer.transform([text])
    # predict_proba for multiclass or multilabel
    try:
        probs = _text_clf.predict_proba(vec)  # shape (1, n_labels) or list-of-arrays for OneVsRest
        if isinstance(probs, list):
            # OneVsRestClassifier with many classes sometimes returns list — stack
            probs = np.vstack([p[0] if p.shape[0]==1 else p for p in probs]).T
        probs = probs.reshape(-1)
    except Exception:
        # fallback: predict (single label)
        lbl = _text_clf.predict(vec)[0]
        probs = np.zeros(len(_text_labels))
        probs[_text_labels.index(lbl)] = 1.0

    top_idx = int(np.argmax(probs))
    top_label = _text_labels[top_idx]
    top_conf = float(probs[top_idx])
    probs_map = {str(lbl): float(prob) for lbl, prob in zip(_text_labels, probs)}
    return {"label": top_label, "emotion": top_label, "confidence": top_conf, "probs_per_label": probs_map}

def predict_voice_from_bytes(wav_bytes: bytes):
    global _voice_model, _voice_input_shape
    if _voice_model is None:
        raise RuntimeError("Voice model not loaded")
    # default target steps and n_mfcc from inferred shape (fallbacks)
    time_steps, n_mfcc = _voice_input_shape if _voice_input_shape is not None else (174, 40)
    mfcc = extract_mfcc_from_wav_bytes(wav_bytes, target_time_steps=time_steps, n_mfcc=n_mfcc)
    x = np.expand_dims(mfcc, axis=0)  # (1, time_steps, n_mfcc)
    preds = _voice_model.predict(x, verbose=0)
    if preds.ndim == 2:
        probs = preds.reshape(-1)
        idx = int(np.argmax(probs))
        # we need to map idx to emotion string; assume encoder used sorted labels:
        # common voice labels (sorted): ['angry','calm','disgust','fearful','happy','neutral','sad','surprised']
        # We'll use this canonical order:
        voice_labels = ['angry','calm','disgust','fearful','happy','neutral','sad','surprised']
        label = voice_labels[idx] if idx < len(voice_labels) else str(idx)
        return {"label": label, "emotion": label, "confidence": float(probs[idx]), "probs_per_label": {lbl: float(p) for lbl,p in zip(voice_labels[:len(probs)], probs)}}
    else:
        return {"label": "unknown", "emotion": "unknown", "confidence": 0.0}

def predict_face_from_bytes(img_bytes: bytes):
    global _face_model, _face_input_shape, _face_labels
    if _face_model is None:
        raise RuntimeError("Face model not loaded")
    # infer target size and channels
    h, w, c = _face_input_shape if _face_input_shape is not None else (48,48,3)
    use_mobilenet = FACE_USE_MOBILENET_PREPROCESS and _mobilenet_preprocess is not None and c == 3
    arr = preprocess_face_image_bytes(img_bytes, target_wh=(w,h), channels=c, use_mobilenet_preprocess=use_mobilenet)
    preds = _face_model.predict(arr, verbose=0)
    probs = preds.reshape(-1)
    idx = int(np.argmax(probs))
    label = _face_labels[idx] if idx < len(_face_labels) else str(idx)
    return {"label": label, "emotion": label, "confidence": float(probs[idx]), "probs_per_label": {lbl: float(p) for lbl,p in zip(_face_labels[:len(probs)], probs)}}

# ---------------------------
# Startup: try to load models (but don't crash — give helpful message)
# ---------------------------
@app.on_event("startup")
def startup_load_models():
    errors = []
    try:
        load_text_models()
    except Exception as e:
        errors.append(f"text model load failed: {e}")
        print("WARN:", traceback.format_exc())

    try:
        load_voice_model()
    except Exception as e:
        errors.append(f"voice model load failed: {e}")
        print("WARN:", traceback.format_exc())

    try:
        load_face_model()
    except Exception as e:
        errors.append(f"face model load failed: {e}")
        print("WARN:", traceback.format_exc())

    if errors:
        print("⚠️  Some models failed to load on startup (ok for development). Errors:")
        for e in errors:
            print("   -", e)
    else:
        print("✅ All models loaded successfully.")

# ---------------------------
# API endpoints
# ---------------------------

@app.get("/health")
def health():
    return {"ok": True, "time": __import__("datetime").datetime.utcnow().isoformat() + "Z"}

class TextIn(BaseModel):
    text: str
    top_k: Optional[int] = 1

@app.post("/predict/text")
def predict_text(payload: TextIn):
    try:
        out = predict_text_from_string(payload.text, top_k=payload.top_k or 1)
        return out
    except Exception as e:
        print("predict_text error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/voice")
async def predict_voice(request: Request, audio: Optional[UploadFile] = File(None)):
    try:
        # 1) if multipart/form-data with file:
        if audio is not None:
            b = await audio.read()
            resp = predict_voice_from_bytes(b)
            return resp

        # 2) else assume JSON body with base64
        body = await request.json()
        b64 = body.get("audio_b64") or body.get("audioBase64") or body.get("audio")
        if not b64:
            raise HTTPException(status_code=400, detail="No audio provided (upload file or audio_b64 in JSON).")
        wav = safe_decode_b64(b64)
        resp = predict_voice_from_bytes(wav)
        return resp
    except HTTPException:
        raise
    except Exception as e:
        print("predict_voice error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/face")
async def predict_face(request: Request, image: Optional[UploadFile] = File(None)):
    try:
        # 1) multipart/form-data file
        if image is not None:
            b = await image.read()
            resp = predict_face_from_bytes(b)
            return resp

        # 2) JSON base64
        body = await request.json()
        b64 = body.get("image_b64") or body.get("imageBase64") or body.get("image")
        if not b64:
            raise HTTPException(status_code=400, detail="No image provided (upload file or image_b64 in JSON).")
        img = safe_decode_b64(b64)
        resp = predict_face_from_bytes(img)
        return resp
    except HTTPException:
        raise
    except Exception as e:
        print("predict_face error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Run directly
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("ML_SERVER_PORT", 9090))
    uvicorn.run("ml_server:app", host="0.0.0.0", port=port, log_level="info")
