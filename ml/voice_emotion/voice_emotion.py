# voice_emotion.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU/MPS (CPU only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TensorFlow logs

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Dataset Setup
# -----------------------------
DATA_DIR = "ravdess"   # folder containing your .wav files

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_emotion(filename):
    """Extracts emotion code from filename like 03-01-05-01-02-02-01.wav"""
    parts = filename.split("-")
    emotion_code = parts[2]
    return emotion_map.get(emotion_code, "unknown")

# -----------------------------
# 2️⃣ Feature Extraction
# -----------------------------
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs.T
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

X, y = [], []
for root, _, files in os.walk(DATA_DIR):
    for fname in files:
        if fname.endswith(".wav"):
            file_path = os.path.join(root, fname)
            label = extract_emotion(fname)
            feature = extract_features(file_path)
            if feature is not None:
                X.append(feature)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("✅ Features shape:", X.shape)
print("✅ Labels shape:", y.shape)

# -----------------------------
# 3️⃣ Encode labels
# -----------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize MFCCs
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# -----------------------------
# 4️⃣ Build Improved Model
# -----------------------------
model = Sequential([
    Conv1D(128, 5, activation="relu", input_shape=(174, 40)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(256, 5, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=False)),
    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 5️⃣ Train
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -----------------------------
# 6️⃣ Save Model
# -----------------------------
model.save("voice_emotion_model.keras")
print("🎉 Improved model saved as voice_emotion_model.keras")

# -----------------------------
# 7️⃣ Plot Training History
# -----------------------------
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
