"""
main.py
-------
Real-time Face Emotion Detection using OpenCV (face detection via Haar
Cascade) and a Keras CNN (emotion classification trained on FER2013).

Controls:
    q  -> quit the application
    s  -> save a screenshot of the current frame to assets/screenshot.png

Usage:
    python main.py

Requirements:
    - A trained model at model/emotion_model.h5 (run train_model.py first,
      or drop in your own compatible .h5 file).
    - A connected webcam.

Developer : MD. Atiqul Islam (Atik)
Email     : atik.cmttiu1001@gmail.com
"""

import os
import sys
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import (
    MODEL_PATH,
    HAAR_CASCADE_PATH,
    IMG_SIZE,
    EMOTION_LABELS,
    EMOTION_COLORS,
    CAMERA_INDEX,
    HAAR_SCALE_FACTOR,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
)


def load_face_detector():
    """Loads the Haar Cascade face detector, falling back to OpenCV's
    bundled copy if a local one isn't found in model/."""
    if os.path.exists(HAAR_CASCADE_PATH):
        cascade_path = HAAR_CASCADE_PATH
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise IOError(f"Failed to load Haar Cascade from: {cascade_path}")
    return detector


def load_emotion_model():
    """Loads the trained Keras emotion classification model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'.\n"
            "Run 'python train_model.py' first, or place a compatible "
            "'emotion_model.h5' file inside the 'model/' folder."
        )
    return load_model(MODEL_PATH)


def preprocess_face(face_roi_gray):
    """
    Prepares a cropped grayscale face region for the CNN:
    resize -> normalize -> reshape to (1, IMG_SIZE, IMG_SIZE, 1).
    """
    face = cv2.resize(face_roi_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)   # add channel dim
    face = np.expand_dims(face, axis=0)    # add batch dim
    return face


def draw_label(frame, text, x, y, color):
    """Draws a filled label background with text above a bounding box."""
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - text_h - 14), (x + text_w + 10, y), color, -1)
    cv2.putText(
        frame, text, (x + 5, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def main():
    print("[INFO] Loading face detector...")
    face_detector = load_face_detector()

    print("[INFO] Loading emotion recognition model...")
    emotion_model = load_emotion_model()

    print("[INFO] Starting webcam stream...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam. Check CAMERA_INDEX in config.py.")
        sys.exit(1)

    os.makedirs("assets", exist_ok=True)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            processed = preprocess_face(face_roi)

            predictions = emotion_model.predict(processed, verbose=0)[0]
            best_idx = int(np.argmax(predictions))
            emotion = EMOTION_LABELS[best_idx]
            confidence = float(predictions[best_idx]) * 100

            color = EMOTION_COLORS.get(emotion, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_label(frame, f"{emotion} ({confidence:.1f}%)", x, y, color)

        # FPS counter (useful for performance tuning on weaker machines)
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        cv2.imshow("Face Emotion Detection - by MD. Atiqul Islam (Atik)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_path = os.path.join("assets", "screenshot.png")
            cv2.imwrite(out_path, frame)
            print(f"[INFO] Screenshot saved to {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
