"""
image_emotion.py
-----------------
Runs face emotion detection on a single static image instead of a live
webcam feed. Useful for quick testing without a camera attached.

Usage:
    python image_emotion.py path/to/image.jpg

Developer : MD. Atiqul Islam (Atik)
Email     : atik.cmttiu1001@gmail.com
"""

import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import MODEL_PATH, HAAR_CASCADE_PATH, IMG_SIZE, EMOTION_LABELS, EMOTION_COLORS
from main import load_face_detector, preprocess_face, draw_label  # reuse logic


def run_on_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    face_detector = load_face_detector()
    emotion_model = load_model(MODEL_PATH)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        print("[INFO] No faces detected in the image.")

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
        print(f"[RESULT] Detected: {emotion} ({confidence:.1f}% confidence)")

    output_path = "assets/image_result.png"
    os.makedirs("assets", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"[INFO] Annotated image saved to {output_path}")

    cv2.imshow("Emotion Detection Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_emotion.py path/to/image.jpg")
        sys.exit(1)
    run_on_image(sys.argv[1])
