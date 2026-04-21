from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from PIL import Image, ImageOps
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import CLASS_NAMES, DEFAULT_IMAGE_SIZE
from .landmarks import LandmarkFeatures, compute_landmark_features, infer_heuristic_label
from .modeling import build_model


@dataclass
class ModelBundle:
    model: torch.nn.Module
    class_names: List[str]
    image_size: int
    device: torch.device


class PredictionSmoother:
    def __init__(self, window_size: int = 10, confidence_floor: float = 0.45) -> None:
        self.window_size = max(3, int(window_size))
        self.confidence_floor = float(confidence_floor)
        self._history: deque[tuple[str, float]] = deque(maxlen=self.window_size)
        self._last_label: str | None = None

    def reset(self) -> None:
        self._history.clear()
        self._last_label = None

    def configure(self, window_size: int | None = None, confidence_floor: float | None = None) -> None:
        if window_size is not None:
            self.window_size = max(3, int(window_size))
            self._history = deque(self._history, maxlen=self.window_size)
        if confidence_floor is not None:
            self.confidence_floor = float(confidence_floor)

    def smooth(self, label: str, confidence: float) -> str:
        # Avoid hard-locking to a previous class on low-confidence frames.
        label_to_add = label
        confidence_to_add = max(0.1, confidence)

        self._history.append((label_to_add, confidence_to_add))
        scores = defaultdict(float)
        for step, (history_label, history_confidence) in enumerate(reversed(self._history), start=1):
            # Favor recent frames while preserving short-term history.
            weight = (0.9 ** (step - 1)) * history_confidence
            scores[history_label] += weight

        stable_label = max(scores, key=scores.get)
        self._last_label = stable_label
        return stable_label


def _build_inference_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _face_to_tensor(face_bgr: np.ndarray, image_size: int):
    transform = _build_inference_transform(image_size)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(face_rgb)
    return transform(image_pil)


@lru_cache(maxsize=1)
def _get_face_cascade():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(cascade_path))


def _detect_largest_face(frame: np.ndarray) -> Tuple[np.ndarray | None, Tuple[int, int, int, int] | None]:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    # Try a few detector settings and contrast-normalized grayscale to improve robustness.
    detection_frames = [gray_frame, cv2.equalizeHist(gray_frame)]
    detector_configs = [
        {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (60, 60)},
        {"scaleFactor": 1.08, "minNeighbors": 4, "minSize": (50, 50)},
        {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (40, 40)},
    ]

    faces = ()
    for candidate_frame in detection_frames:
        for config in detector_configs:
            detected = cascade.detectMultiScale(candidate_frame, **config)
            if len(detected) > 0:
                faces = detected
                break
        if len(faces) > 0:
            break

    if len(faces) == 0:
        return None, None

    x, y, width, height = max(faces, key=lambda face_box: face_box[2] * face_box[3])
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.15)
    x_min = max(0, x - margin_x)
    y_min = max(0, y - margin_y)
    x_max = min(frame.shape[1], x + width + margin_x)
    y_max = min(frame.shape[0], y + height + margin_y)
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def _extract_center_crop(frame: np.ndarray, crop_ratio: float = 0.75) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    image_height, image_width = frame.shape[:2]
    ratio = max(0.4, min(1.0, float(crop_ratio)))
    crop_width = int(image_width * ratio)
    crop_height = int(image_height * ratio)
    x_min = max(0, (image_width - crop_width) // 2)
    y_min = max(0, (image_height - crop_height) // 2)
    x_max = min(image_width, x_min + crop_width)
    y_max = min(image_height, y_min + crop_height)
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def _predict_probabilities(bundle: ModelBundle, face_bgr: np.ndarray, use_tta: bool = True) -> torch.Tensor:
    tensors = [_face_to_tensor(face_bgr, bundle.image_size)]
    if use_tta:
        flipped = ImageOps.mirror(Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)))
        tensors.append(_build_inference_transform(bundle.image_size)(flipped))

    batch = torch.stack(tensors).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(batch)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.mean(dim=0)


def compute_engagement_score(stats: Dict[str, int]) -> float:
    engaged = float(stats.get("Engaged", 0))
    not_engaged = float(stats.get("Not_Engaged", 0))
    drowsy = float(stats.get("Drowsy", 0))
    total = engaged + not_engaged + drowsy
    if total <= 0:
        return 0.0
    score = (engaged * 1.0 + not_engaged * 0.4 + drowsy * 0.0) / total
    return float(score * 100.0)


def load_model_bundle(model_path: str) -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    model_name = checkpoint.get("model_name", "mobilenet_v2")
    model = build_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return ModelBundle(model=model, class_names=class_names, image_size=image_size, device=device)


def crop_face_from_landmarks(frame: np.ndarray, landmarks) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    image_height, image_width = frame.shape[:2]
    xs = [landmark.x for landmark in landmarks.landmark]
    ys = [landmark.y for landmark in landmarks.landmark]
    x_min = max(0, int(min(xs) * image_width) - 20)
    y_min = max(0, int(min(ys) * image_height) - 20)
    x_max = min(image_width, int(max(xs) * image_width) + 20)
    y_max = min(image_height, int(max(ys) * image_height) + 20)

    face = frame[y_min:y_max, x_min:x_max]
    if face.size == 0:
        face = frame
        x_min, y_min, x_max, y_max = 0, 0, image_width, image_height
    return face, (x_min, y_min, x_max, y_max)


def predict_face(bundle: ModelBundle, face_bgr: np.ndarray, use_tta: bool = True) -> Tuple[str, float]:
    probabilities = _predict_probabilities(bundle, face_bgr, use_tta=use_tta)
    confidence, class_index = torch.max(probabilities, dim=0)
    return bundle.class_names[int(class_index)], float(confidence.item())


def predict_face_without_landmarks(bundle: ModelBundle, face_bgr: np.ndarray, use_tta: bool = True) -> Tuple[str, float]:
    """Predict label in CNN-only mode with a conservative Drowsy guard.

    Without landmarks, Drowsy tends to be over-predicted on noisy webcam frames.
    Require stronger confidence + margin before accepting Drowsy.
    """
    probabilities = _predict_probabilities(bundle, face_bgr, use_tta=use_tta)
    top_confidence, top_index = torch.max(probabilities, dim=0)
    top_label = bundle.class_names[int(top_index)]

    if top_label != "Drowsy":
        return top_label, float(top_confidence.item())

    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    drowsy_confidence = float(sorted_probabilities[0].item())
    second_confidence = float(sorted_probabilities[1].item())
    second_label = bundle.class_names[int(sorted_indices[1].item())]

    # Accept Drowsy only when clearly dominant.
    if drowsy_confidence >= 0.72 and (drowsy_confidence - second_confidence) >= 0.12:
        return "Drowsy", drowsy_confidence

    return second_label, second_confidence


def fuse_prediction(model_label: str, confidence: float, features: LandmarkFeatures | None, confidence_threshold: float) -> str:
    if features is None:
        return model_label

    heuristic_label = infer_heuristic_label(features)
    strong_drowsy_signal = features.eye_openness < 0.10 or features.head_pitch > 30.0
    strong_not_engaged_signal = features.gaze_ratio < 0.28 or features.gaze_ratio > 0.72 or abs(features.head_yaw) > 25.0

    # At low confidence, only trust heuristic when physiological cues are very strong.
    if confidence < confidence_threshold:
        if heuristic_label == "Drowsy" and strong_drowsy_signal:
            return heuristic_label
        if heuristic_label == "Not_Engaged" and strong_not_engaged_signal:
            return heuristic_label
        return model_label

    # Only force Drowsy on strong physiological evidence.
    if heuristic_label == "Drowsy" and strong_drowsy_signal:
        return heuristic_label

    # Override Engaged with Not_Engaged only when model confidence is not decisive.
    if heuristic_label == "Not_Engaged" and model_label == "Engaged" and confidence < max(0.65, confidence_threshold + 0.1):
        return heuristic_label
    return model_label


def process_frame(
    frame: np.ndarray,
    bundle: ModelBundle,
    face_mesh,
    confidence_threshold: float = 0.5,
    prediction_smoother: PredictionSmoother | None = None,
):
    annotated = frame.copy()
    frame_stats: Dict[str, int] = {name: 0 for name in bundle.class_names}
    
    # If face_mesh is available, use landmark-based detection
    if face_mesh is not None:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    face_crop, (x_min, y_min, x_max, y_max) = crop_face_from_landmarks(frame, landmarks)
                    model_label, confidence = predict_face(bundle, face_crop)
                    features = compute_landmark_features(frame, landmarks)
                    final_label = fuse_prediction(model_label, confidence, features, confidence_threshold)
                    stable_label = prediction_smoother.smooth(final_label, confidence) if prediction_smoother else final_label
                    frame_stats[stable_label] = frame_stats.get(stable_label, 0) + 1
                    annotate_frame(annotated, (x_min, y_min, x_max, y_max), stable_label, confidence, features)
        except Exception as e:
            # Fallback to simple CNN-only prediction if landmarks fail
            pass
    
    # If no faces were detected with landmarks, try a direct face detector fallback.
    if sum(frame_stats.values()) == 0:
        try:
            face_crop, face_box = _detect_largest_face(frame)
            if face_crop is not None and face_box is not None:
                model_label, confidence = predict_face_without_landmarks(bundle, face_crop)
                final_label = fuse_prediction(model_label, confidence, None, confidence_threshold)
                stable_label = prediction_smoother.smooth(final_label, confidence) if prediction_smoother else final_label
                frame_stats[stable_label] = frame_stats.get(stable_label, 0) + 1
                annotate_frame(annotated, face_box, stable_label, confidence, None)
            else:
                # Final fallback: center crop prediction when detector misses (common on low-quality frames).
                center_crop, center_box = _extract_center_crop(frame)
                model_label, confidence = predict_face_without_landmarks(bundle, center_crop)
                accept_threshold = max(0.80, confidence_threshold + 0.2)
                # Keep this conservative to avoid biased fallback predictions (often Drowsy).
                if confidence >= accept_threshold and model_label != "Drowsy":
                    final_label = fuse_prediction(model_label, confidence, None, confidence_threshold)
                    stable_label = prediction_smoother.smooth(final_label, confidence) if prediction_smoother else final_label
                    frame_stats[stable_label] = frame_stats.get(stable_label, 0) + 1
                    annotate_frame(annotated, center_box, stable_label, confidence, None)
                else:
                    cv2.putText(
                        annotated,
                        "No face detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
        except Exception:
            cv2.putText(
                annotated,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

    return annotated, frame_stats


def annotate_frame(frame: np.ndarray, box: Tuple[int, int, int, int], label: str, confidence: float, features: LandmarkFeatures | None):
    x_min, y_min, x_max, y_max = box
    color_map = {
        "Engaged": (0, 200, 0),
        "Not_Engaged": (0, 165, 255),
        "Drowsy": (0, 0, 255),
    }
    color = color_map.get(label, (255, 255, 255))
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (x_min, max(20, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if features is not None:
        detail = f"Score {features.attention_score:.2f}"
        cv2.putText(frame, detail, (x_min, min(frame.shape[0] - 10, y_max + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
