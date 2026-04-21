# Student Engagement Detection

End-to-end computer vision project for classifying student engagement from images, video, or webcam input.

## Features

- Face preprocessing with OpenCV and torchvision
- Landmark-based head pose, gaze, and eye openness analysis using MediaPipe
- Transfer learning with ResNet50 or MobileNetV2
- Class-weighted training with label smoothing and early stopping
- Flip-based test-time augmentation during inference
- Training, validation, confusion matrix, and classification report
- Real-time webcam/video inference with engagement logging
- Streamlit app with a polished dashboard, threshold control, and annotated-video download

## Folder Structure

```text
dataset/
models/
utils/
app.py
train.py
detect.py
```

## Dataset Options

### Option 1: Folder dataset

Place images in class folders:

```text
dataset/
  train/
    Engaged/
    Not_Engaged/
    Drowsy/
  val/
    Engaged/
    Not_Engaged/
    Drowsy/
  test/
    Engaged/
    Not_Engaged/
    Drowsy/
```

### Option 2: FER2013 CSV

Put a CSV in `dataset/fer2013.csv` with columns like `emotion` and `pixels`.
The training script maps FER2013 emotions to the three engagement classes.

## Install

```bash
pip install -r requirements.txt
```

## Prepare Data

### Option A: Use Sample Data (for testing)

```bash
python generate_sample_data.py --output-dir dataset --num-per-class 20
```

This creates synthetic images in the proper folder structure for quick testing.

### Option B: Use Real Data

Download a dataset like [FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013) and place the CSV in `dataset/fer2013.csv`, or arrange images in the folder structure below.

## Train

```bash
python train.py --data-dir dataset --model-name mobilenet_v2 --epochs 20 --batch-size 32
```

Useful training flags:

```bash
python train.py --freeze-backbone --label-smoothing 0.05 --patience 5 --weight-decay 1e-4
```

## Detect from webcam or video

```bash
python detect.py --model-path models/best_model.pt --source 0
python detect.py --model-path models/best_model.pt --source path/to/video.mp4
```

## Run Streamlit app

```bash
streamlit run app.py
```

The app includes a styled dashboard, live model status, confidence threshold control, and a download button for annotated videos.

## Notes

- The realtime detector supports two modes:
  - **With MediaPipe** (if compatible version available): Uses face landmark fusion for gaze, head pose, and eye openness
  - **CNN-only mode** (fallback): Uses just the trained CNN for predictions, gracefully handles situations where MediaPipe isn't available
- GPU is used automatically if PyTorch detects CUDA.
- If your dataset is small, start with MobileNetV2 for faster training and inference.
- The provided sample data generation script creates synthetic images for quick testing; train on real faces from datasets like FER2013 for production use.
```