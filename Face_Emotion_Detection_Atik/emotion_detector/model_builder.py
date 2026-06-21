"""
model_builder.py
-----------------
Defines the CNN architecture used for facial emotion classification.
The network follows a classic VGG-style block pattern (Conv -> Conv ->
Pool -> Dropout) which performs well on the small 48x48 grayscale
FER2013 images while staying light enough to train on a single GPU/CPU.

Developer : MD. Atiqul Islam (Atik)
Email     : atik.cmttiu1001@gmail.com
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam

from config import IMG_SIZE, NUM_CLASSES


def build_emotion_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Builds and compiles the CNN used for emotion classification.

    Args:
        input_shape (tuple): Shape of the input image (H, W, Channels).
        num_classes (int): Number of emotion classes to predict.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    model = Sequential(name="emotion_cnn")

    # Block 1
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classifier head
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check: print the model summary when this file is run directly.
    m = build_emotion_model()
    m.summary()
