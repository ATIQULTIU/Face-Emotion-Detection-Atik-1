"""
config.py
---------
Centralized configuration for the Face Emotion Detection project.
Keeping constants in one place makes the project easier to maintain
and avoids "magic numbers/strings" scattered across files.

Developer : MD. Atiqul Islam (Atik)
Email     : atik.cmttiu1001@gmail.com
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")
HAAR_CASCADE_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
DATASET_CSV_PATH = os.path.join(BASE_DIR, "dataset", "fer2013.csv")

# ---------------------------------------------------------------------------
# Model / Image parameters
# ---------------------------------------------------------------------------
IMG_SIZE = 48          # FER2013 images are 48x48 grayscale
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 40

# Emotion labels in the exact order the model was trained on (FER2013 standard)
EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# BGR colors (OpenCV uses BGR, not RGB) used to draw a box per emotion
EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 128, 0),
    "Fear": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Sad": (255, 0, 0),
    "Surprise": (0, 165, 255),
    "Neutral": (200, 200, 200),
}

# ---------------------------------------------------------------------------
# Webcam / detection parameters
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (60, 60)
