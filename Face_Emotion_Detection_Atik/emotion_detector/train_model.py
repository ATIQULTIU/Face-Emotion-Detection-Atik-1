"""
train_model.py
---------------
Trains the facial emotion recognition CNN on the FER2013 dataset and
saves the resulting weights to model/emotion_model.h5 for use by
main.py during real-time webcam inference.

Dataset:
    Download "fer2013.csv" from Kaggle:
    https://www.kaggle.com/datasets/msambare/fer2013
    (or any FER2013 CSV mirror) and place it at: dataset/fer2013.csv

Usage:
    python train_model.py

Developer : MD. Atiqul Islam (Atik)
Email     : atik.cmttiu1001@gmail.com
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    DATASET_CSV_PATH,
    MODEL_PATH,
    MODEL_DIR,
    IMG_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
)
from model_builder import build_emotion_model


def load_fer2013(csv_path):
    """
    Loads and parses the FER2013 CSV file into image/label numpy arrays.

    The FER2013 CSV has three columns: 'emotion', 'pixels', 'Usage'.
    Each row's 'pixels' column is a space-separated string of 48*48
    grayscale pixel intensities.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find dataset at '{csv_path}'.\n"
            "Download fer2013.csv from "
            "https://www.kaggle.com/datasets/msambare/fer2013 "
            "and place it inside the 'dataset/' folder."
        )

    df = pd.read_csv(csv_path)

    pixels = df["pixels"].apply(lambda p: np.array(p.split(), dtype="float32"))
    X = np.stack(pixels.values)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # normalize to [0, 1]

    y = to_categorical(df["emotion"].values, num_classes=NUM_CLASSES)

    return X, y


def main():
    print("[INFO] Loading dataset...")
    X, y = load_fer2013(DATASET_CSV_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"[INFO] Train samples: {len(X_train)} | Validation samples: {len(X_val)}")

    # Light augmentation helps generalization on the relatively small FER2013 set
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    train_datagen.fit(X_train)

    print("[INFO] Building model...")
    model = build_emotion_model()
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("[INFO] Starting training...")
    model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"[INFO] Training complete. Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
