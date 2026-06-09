<div align="center">

# 🎓 Student Engagement Detection

**Real-time classroom engagement analysis using CNN + facial landmark fusion**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00897B?logo=google&logoColor=white)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning pipeline that classifies student engagement states — **Engaged**, **Not Engaged**, and **Drowsy** — from images, video files, or live webcam feeds. The system combines a fine-tuned CNN backbone with MediaPipe facial landmark heuristics for robust, real-time predictions.

[Features](#-features) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Usage](#-usage) · [Project Structure](#-project-structure)

</div>

---

## ✨ Features

| Capability | Description |
|---|---|
| **Multi-Backbone Support** | Transfer learning with **MobileNetV2** (fast) or **ResNet50** (accurate) |
| **Landmark Fusion** | MediaPipe Face Mesh for eye openness, gaze ratio, and head pose estimation |
| **Prediction Smoothing** | Temporal smoothing with configurable window size and confidence floor |
| **Test-Time Augmentation** | Flip-based TTA for more robust inference |
| **Class-Weighted Training** | Automatic class weight computation to handle imbalanced datasets |
| **Label Smoothing** | Configurable label smoothing with early stopping for better generalization |
| **Real-Time Detection** | Live webcam inference with OpenCV or browser-based WebRTC streaming |
| **Streamlit Dashboard** | Polished dark-themed UI with model controls, metrics, and video download |
| **Engagement Scoring** | Composite engagement score computed from per-class frame counts |
| **CSV Logging** | Automatic session-level engagement logs for analytics |
| **FER2013 Support** | Built-in FER2013 CSV loader with emotion-to-engagement class mapping |
| **Graceful Fallbacks** | CNN-only mode when MediaPipe is unavailable; center-crop fallback when no face is detected |

---

## 🚀 Quick Start

### Prerequisites

- Python **3.10** or higher
- CUDA-capable GPU *(optional, for faster training and inference)*
- Webcam *(optional, for live detection)*

### 1. Clone the Repository

```bash
git clone https://github.com/NithinPranav-007/Student-Engagement-Detection.git
cd Student-Engagement-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

**Option A — Quick test with synthetic data:**

```bash
python generate_sample_data.py --output-dir dataset --num-per-class 50
```

**Option B — Real dataset (recommended for training):**

Download [FER2013 from Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) and place the CSV at `dataset/fer2013.csv`, or organize face images into the folder structure:

```
dataset/
├── train/
│   ├── Engaged/
│   ├── Not_Engaged/
│   └── Drowsy/
└── val/
    ├── Engaged/
    ├── Not_Engaged/
    └── Drowsy/
```

### 5. Train the Model

```bash
python train.py --data-dir dataset --model-name mobilenet_v2 --epochs 20 --batch-size 32
```

### 6. Launch the App

```bash
streamlit run app.py
```

---

## 🏗 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Sources                            │
│         Webcam  ·  Video File  ·  Streamlit Upload              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Face Detection Layer                          │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │  MediaPipe Face   │    │  Haar Cascade    │    │  Center   │  │
│  │  Mesh (Primary)   │───▶│  (Fallback #1)   │───▶│  Crop     │  │
│  └──────────────────┘    └──────────────────┘    │ (Fallback) │  │
│                                                   └───────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐  ┌──────────────────────┐
│   CNN Backbone       │  │  Landmark Features   │
│                      │  │                      │
│  MobileNetV2 /       │  │  • Eye Openness      │
│  ResNet50            │  │  • Gaze Ratio         │
│                      │  │  • Head Pitch/Yaw     │
│  + Flip TTA          │  │  • Attention Score    │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └──────────┬──────────────┘
                      ▼
        ┌──────────────────────────┐
        │    Prediction Fusion     │
        │                          │
        │  CNN + Heuristic Rules   │
        │  Confidence Thresholds   │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  Temporal Smoothing      │
        │                          │
        │  Exponential decay       │
        │  weighted voting         │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │     Final Output         │
        │                          │
        │  Engaged · Not_Engaged   │
        │  Drowsy · Engagement %   │
        └──────────────────────────┘
```

### Engagement Classes

| Class | Description | Visual Cues |
|---|---|---|
| **Engaged** | Actively attentive, eyes open, facing forward | High eye openness, centered gaze, neutral head pose |
| **Not_Engaged** | Distracted, looking away, or disinterested | Off-center gaze, significant head yaw/roll |
| **Drowsy** | Sleepy, eyes closing, or head drooping | Low eye openness, elevated head pitch |

### Fusion Strategy

The system uses a confidence-gated fusion approach:

1. **High CNN confidence** → Trust the CNN prediction, unless strong physiological cues (e.g., eyes nearly closed) contradict it.
2. **Low CNN confidence** → Rely on landmark heuristics only when physiological signals are very strong, preventing false overrides.
3. **No landmarks available** → Conservative CNN-only mode with stricter thresholds for the "Drowsy" class.

---

## 📖 Usage

### Training

```bash
# Basic training with MobileNetV2
python train.py --data-dir dataset --model-name mobilenet_v2 --epochs 20

# Advanced training with ResNet50, frozen backbone, and custom hyperparameters
python train.py \
  --data-dir dataset \
  --model-name resnet50 \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --freeze-backbone \
  --label-smoothing 0.05 \
  --patience 5 \
  --image-size 224 \
  --val-split 0.2 \
  --output-dir models
```

**Training Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `dataset` | Path to the dataset directory |
| `--model-name` | `mobilenet_v2` | Backbone architecture (`mobilenet_v2` or `resnet50`) |
| `--epochs` | `20` | Maximum number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `1e-4` | Learning rate (AdamW optimizer) |
| `--weight-decay` | `1e-4` | L2 regularization weight |
| `--freeze-backbone` | `false` | Freeze pretrained backbone; train only the classifier head |
| `--label-smoothing` | `0.05` | Label smoothing factor |
| `--patience` | `5` | Early stopping patience (epochs without improvement) |
| `--image-size` | `224` | Input image resolution |
| `--val-split` | `0.2` | Validation split ratio (when val set not pre-split) |
| `--output-dir` | `models` | Directory for checkpoints and evaluation artifacts |

**Training Outputs:**

```
models/
├── best_model.pt           # Best checkpoint (by validation accuracy)
├── training_curves.png     # Loss and accuracy plots
├── confusion_matrix.png    # Confusion matrix visualization
├── evaluation.txt          # Classification report with metrics
└── class_names.json        # Ordered class label list
```

### Real-Time Detection (CLI)

```bash
# Webcam mode (press 'q' to quit)
python detect.py --model-path models/best_model.pt --source 0

# Video file analysis
python detect.py --model-path models/best_model.pt --source classroom_video.mp4

# Customized detection
python detect.py \
  --model-path models/best_model.pt \
  --source 0 \
  --confidence-threshold 0.6 \
  --smoothing-window 15 \
  --smoothing-floor 0.5 \
  --log-csv models/engagement_log.csv
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides:

- **Video upload** — Drag-and-drop MP4/AVI/MOV/MKV files for batch analysis
- **Webcam mode** — Real-time browser-based detection via WebRTC
- **Model controls** — Adjust confidence threshold, smoothing window, and smoothing floor
- **Live metrics** — Per-class frame counts and composite engagement score
- **Annotated download** — Export the processed video with bounding boxes and labels

---

## 📁 Project Structure

```
Student-Engagement-Detection/
│
├── app.py                      # Streamlit web dashboard
├── train.py                    # Model training pipeline
├── detect.py                   # CLI-based real-time detection
├── generate_sample_data.py     # Synthetic dataset generator (for testing)
├── test_inference.py           # Inference pipeline smoke test
├── test_predict.py             # Single-image prediction test
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── utils/                      # Core library modules
│   ├── __init__.py
│   ├── config.py               # Constants, class names, FER2013 mapping
│   ├── data.py                 # Dataset loaders and augmentation transforms
│   ├── modeling.py             # Model architecture builder and checkpointing
│   ├── inference.py            # Inference engine, fusion, smoothing, scoring
│   ├── landmarks.py            # MediaPipe face mesh and landmark features
│   └── visualization.py       # Confusion matrix plotting
│
├── dataset/                    # Training and validation data
│   ├── train/
│   │   ├── Engaged/
│   │   ├── Not_Engaged/
│   │   └── Drowsy/
│   └── val/
│       ├── Engaged/
│       ├── Not_Engaged/
│       └── Drowsy/
│
└── models/                     # Trained models and evaluation artifacts
    ├── best_model.pt
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── evaluation.txt
    └── class_names.json
```

---

## ⚙️ Configuration

Key parameters are centralized in [`utils/config.py`](utils/config.py):

| Parameter | Value | Description |
|---|---|---|
| `CLASS_NAMES` | `["Engaged", "Not_Engaged", "Drowsy"]` | Target engagement classes |
| `DEFAULT_IMAGE_SIZE` | `224` | Input resolution for the CNN |
| `DEFAULT_MODEL_NAME` | `mobilenet_v2` | Default backbone architecture |
| `DEFAULT_SEED` | `42` | Random seed for reproducibility |

### FER2013 Emotion Mapping

When using the FER2013 dataset, emotions are automatically mapped to engagement classes:

| FER2013 Emotion | → Engagement Class |
|---|---|
| Happy, Surprise, Neutral | **Engaged** |
| Angry, Disgust, Fear | **Not_Engaged** |
| Sad | **Drowsy** |

---

## 🔧 Technical Details

### Data Augmentation (Training)

- Random resized crop (scale 0.85–1.0)
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random autocontrast (p=0.2)
- Random erasing (p=0.15)

### Inference Pipeline

1. **Face detection** — MediaPipe Face Mesh → Haar Cascade → Center Crop
2. **Feature extraction** — Eye aspect ratio, iris-based gaze ratio, head pose via PnP
3. **CNN prediction** — Forward pass + optional flip-based TTA
4. **Fusion** — Confidence-gated merging of CNN output and landmark heuristics
5. **Smoothing** — Exponential-decay weighted temporal smoothing over a sliding window
6. **Scoring** — Weighted engagement score: Engaged × 1.0 + Not_Engaged × 0.4 + Drowsy × 0.0

### Landmark Features

| Feature | Method | Key Thresholds |
|---|---|---|
| Eye Openness | Eye aspect ratio (EAR) | < 0.15 → Drowsy signal |
| Gaze Ratio | Iris position relative to eye corners | < 0.32 or > 0.68 → Not_Engaged |
| Head Pitch | solvePnP + Rodrigues decomposition | > 18° → Drowsy signal |
| Head Yaw | solvePnP + Rodrigues decomposition | > 20° → Not_Engaged signal |

---

## 📝 Notes

- **GPU acceleration** is used automatically when PyTorch detects a CUDA-capable device.
- **MobileNetV2** is recommended for smaller datasets and faster inference; **ResNet50** offers higher capacity for larger datasets.
- The **sample data generator** creates synthetic face-like images for smoke testing only — always train on real face images (e.g., FER2013) for production.
- **WebRTC webcam mode** requires the `streamlit-webrtc` and `av` packages. If unavailable, use `detect.py` for webcam inference directly.
- The system **gracefully falls back** to CNN-only predictions when MediaPipe is not installed or encounters errors.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using PyTorch, MediaPipe, and Streamlit**

</div>