#!/usr/bin/env python
"""Test inference pipeline"""
import sys
sys.path.insert(0, '.')
import cv2
from utils.inference import load_model_bundle, process_frame
from utils.landmarks import get_face_mesh

# Load model
bundle = load_model_bundle('models/best_model.pt')
print('[INFO] Model loaded successfully')

# Load a sample image
img_path = 'dataset/val/Engaged/engaged_000.png'
frame = cv2.imread(img_path)
print(f'[INFO] Image loaded: {img_path}, shape: {frame.shape}')

# Get face mesh (may be None if MediaPipe not available)
face_mesh = get_face_mesh()
status = "available" if face_mesh else "not available (using CNN-only)"
print(f'[INFO] Face mesh: {status}')

# Process frame
annotated, stats = process_frame(frame, bundle, face_mesh)
print('[INFO] Frame processed successfully')
print(f'[RESULT] Predictions: {stats}')
print(f'[SUCCESS] Inference pipeline working!')
