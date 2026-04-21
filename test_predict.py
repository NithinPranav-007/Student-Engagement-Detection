#!/usr/bin/env python
"""Debug inference pipeline"""
import sys
sys.path.insert(0, '.')
import torch
import cv2
from utils.inference import load_model_bundle, predict_face

# Load model
bundle = load_model_bundle('models/best_model.pt')
print('[INFO] Model loaded')
print(f'[INFO] Device: {bundle.device}')
print(f'[INFO] Class names: {bundle.class_names}')
print(f'[INFO] Image size: {bundle.image_size}')

# Load a sample image
img_path = 'dataset/val/Engaged/engaged_000.png'
frame = cv2.imread(img_path)
print(f'[INFO] Image shape: {frame.shape}')

# Test prediction
try:
    label, confidence = predict_face(bundle, frame)
    print(f'[SUCCESS] Prediction: {label} ({confidence:.4f})')
except Exception as e:
    print(f'[ERROR] Prediction failed: {e}')
    import traceback
    traceback.print_exc()
