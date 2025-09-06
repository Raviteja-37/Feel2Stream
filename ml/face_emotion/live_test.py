# live_test.py
import os

# ✅ Fix threading issues (Mac TensorFlow + OpenCV crash fix)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("face_emotion_model.keras")

# Emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ Force OpenCV single-threaded mode
cv2.setNumThreads(1)

# Open webcam
cap = cv2.VideoCapture(0)
print("📸 Live Emotion Detection Started (press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)  # (48,48,1)
        roi = np.expand_dims(roi, axis=0)   # (1,48,48,1)

        preds = model.predict(roi, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds)

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0,255,0), 2)

        print(f"Detected Emotion: {emotion} ({confidence:.2f})")

    cv2.imshow("Live Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
