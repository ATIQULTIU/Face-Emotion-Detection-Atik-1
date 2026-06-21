# 🎭 Face Emotion Detection (Real-Time, Webcam-Based)

A real-time facial emotion recognition system built with **Python, OpenCV,
and TensorFlow/Keras**. It detects faces from a live webcam feed (or a
static image) and classifies the dominant emotion as one of **7 classes**:
`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`.

---

## 👨‍💻 Developer

| | |
|---|---|
| **Name**  | MD. Atiqul Islam (Atik) |
| **Email** | atik.cmttiu1001@gmail.com |

---

## 🧠 How It Works

1. **Face Detection** — OpenCV's Haar Cascade classifier locates faces in
   each video frame.
2. **Preprocessing** — Each detected face is cropped, converted to
   grayscale, resized to 48×48 pixels, and normalized.
3. **Emotion Classification** — A custom CNN (trained on the FER2013
   dataset) predicts probabilities across 7 emotion classes.
4. **Visualization** — The bounding box, predicted emotion, and confidence
   score are drawn live on the video feed.

```
Webcam Frame → Haar Cascade (face) → CNN (emotion) → Annotated Output
```

---

## 📁 Project Structure

```
emotion_detector/
├── main.py                 # Real-time webcam emotion detection (entry point)
├── image_emotion.py         # Run detection on a single static image
├── train_model.py           # Train the CNN on the FER2013 dataset
├── model_builder.py         # CNN architecture definition
├── config.py                 # Centralized paths & hyperparameters
├── requirements.txt          # Python dependencies
├── dataset/                  # Place fer2013.csv here (not included)
├── model/                    # Trained model (emotion_model.h5) saved here
├── assets/                   # Screenshots / output images saved here
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / Extract the project
```bash
cd emotion_detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

The repository does **not** include a pre-trained model — you train it
yourself (one-time step) so you control the dataset and weights.

1. Download **fer2013.csv** from Kaggle:
   https://www.kaggle.com/datasets/msambare/fer2013
2. Place it at: `dataset/fer2013.csv`
3. Run:
   ```bash
   python train_model.py
   ```
4. The best-performing weights are automatically saved to
   `model/emotion_model.h5`.

> 💡 Training on CPU works but is slow (~30–60 min for 40 epochs on the
> ~35k sample dataset). A GPU is recommended for faster iteration.

---

## ▶️ Run Real-Time Detection

```bash
python main.py
```

**Controls:**
| Key | Action |
|---|---|
| `q` | Quit the application |
| `s` | Save a screenshot to `assets/screenshot.png` |

---

## 🖼️ Run on a Single Image

```bash
python image_emotion.py path/to/your/photo.jpg
```
The annotated result is saved to `assets/image_result.png`.

---

## 🔧 Configuration

All tunable parameters live in `config.py`:
- `CAMERA_INDEX` — change if you have multiple webcams (default `0`)
- `HAAR_SCALE_FACTOR`, `HAAR_MIN_NEIGHBORS`, `HAAR_MIN_SIZE` — tune face
  detection sensitivity
- `EPOCHS`, `BATCH_SIZE` — training hyperparameters

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** — face detection & video I/O
- **TensorFlow / Keras** — CNN model training & inference
- **NumPy / Pandas** — data handling
- **scikit-learn** — train/validation split

---

## 🚀 Possible Improvements

- Swap Haar Cascade for a DNN-based face detector (e.g., MTCNN, RetinaFace)
  for better accuracy on angled/occluded faces.
- Add multi-face emotion logging to CSV for analytics dashboards.
- Deploy as a Flask/FastAPI web app with browser webcam access.
- Convert the model to TensorFlow Lite for mobile/edge deployment.

---

## 📄 License

This project is released under the MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- FER2013 dataset (ICML 2013 Workshop on Challenges in Representation Learning)
- OpenCV's pre-trained Haar Cascade classifiers

---

**Made with ❤️ by MD. Atiqul Islam (Atik)**
📧 atik.cmttiu1001@gmail.com
