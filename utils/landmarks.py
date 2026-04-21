from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Try to import MediaPipe, fallback if not available or incompatible
MEDIAPIPE_AVAILABLE = False
try:
    try:
        # Try newer MediaPipe API (0.10+)
        from mediapipe.tasks import vision
        MEDIAPIPE_AVAILABLE = True
    except (ImportError, AttributeError):
        # Try older MediaPipe API  
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            from mediapipe.python.solutions import face_mesh as face_mesh_module
            MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LandmarkFeatures:
    eye_openness: float
    gaze_ratio: float
    head_pitch: float
    head_yaw: float
    head_roll: float
    attention_score: float


_FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)

_LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
_LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
_RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


def get_face_mesh():
    """
    Initialize face mesh detector. Returns None if MediaPipe is not properly installed.
    The system will fall back to CNN-only predictions without landmark features.
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            return mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
    except Exception:
        pass
    
    return None


def _landmarks_to_points(landmarks, indices, image_width: int, image_height: int):
    points = []
    for index in indices:
        landmark = landmarks.landmark[index]
        points.append((int(landmark.x * image_width), int(landmark.y * image_height)))
    return points


def _eye_aspect_ratio(points) -> float:
    vertical_1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vertical_2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    if horizontal == 0:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def _iris_center(landmarks, indices, image_width: int, image_height: int):
    points = _landmarks_to_points(landmarks, indices, image_width, image_height)
    return np.mean(np.asarray(points, dtype=np.float32), axis=0)


def _gaze_ratio(landmarks, image_width: int, image_height: int) -> float:
    left_eye = _landmarks_to_points(landmarks, _LEFT_EYE_INDICES, image_width, image_height)
    right_eye = _landmarks_to_points(landmarks, _RIGHT_EYE_INDICES, image_width, image_height)
    left_iris = _iris_center(landmarks, _LEFT_IRIS_INDICES, image_width, image_height)
    right_iris = _iris_center(landmarks, _RIGHT_IRIS_INDICES, image_width, image_height)

    def _ratio(eye_points, iris_center):
        left_corner = np.array(eye_points[0], dtype=np.float32)
        right_corner = np.array(eye_points[3], dtype=np.float32)
        width = np.linalg.norm(right_corner - left_corner)
        if width == 0:
            return 0.5
        return float(np.linalg.norm(iris_center - left_corner) / width)

    return float((_ratio(left_eye, left_iris) + _ratio(right_eye, right_iris)) / 2.0)


def _head_pose(landmarks, image_width: int, image_height: int):
    image_points = np.array(
        [
            (landmarks.landmark[1].x * image_width, landmarks.landmark[1].y * image_height),
            (landmarks.landmark[152].x * image_width, landmarks.landmark[152].y * image_height),
            (landmarks.landmark[33].x * image_width, landmarks.landmark[33].y * image_height),
            (landmarks.landmark[263].x * image_width, landmarks.landmark[263].y * image_height),
            (landmarks.landmark[61].x * image_width, landmarks.landmark[61].y * image_height),
            (landmarks.landmark[291].x * image_width, landmarks.landmark[291].y * image_height),
        ],
        dtype=np.float64,
    )

    focal_length = image_width
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        _FACE_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    pitch, yaw, roll = [float(angle) for angle in euler_angles]
    return pitch, yaw, roll


def compute_landmark_features(frame_bgr: np.ndarray, landmarks) -> Optional[LandmarkFeatures]:
    if landmarks is None:
        return None

    image_height, image_width = frame_bgr.shape[:2]
    left_eye = _landmarks_to_points(landmarks, _LEFT_EYE_INDICES, image_width, image_height)
    right_eye = _landmarks_to_points(landmarks, _RIGHT_EYE_INDICES, image_width, image_height)
    left_ear = _eye_aspect_ratio(left_eye)
    right_ear = _eye_aspect_ratio(right_eye)
    eye_openness = float((left_ear + right_ear) / 2.0)
    gaze_ratio = _gaze_ratio(landmarks, image_width, image_height)
    head_pitch, head_yaw, head_roll = _head_pose(landmarks, image_width, image_height)

    attention_score = 0.0
    if eye_openness > 0.18:
        attention_score += 0.4
    if 0.35 <= gaze_ratio <= 0.65:
        attention_score += 0.3
    if abs(head_yaw) < 18.0 and abs(head_pitch) < 18.0:
        attention_score += 0.3

    return LandmarkFeatures(
        eye_openness=eye_openness,
        gaze_ratio=gaze_ratio,
        head_pitch=head_pitch,
        head_yaw=head_yaw,
        head_roll=head_roll,
        attention_score=attention_score,
    )


def infer_heuristic_label(features: LandmarkFeatures) -> str:
    if features.eye_openness < 0.15 or features.head_pitch > 18.0:
        return "Drowsy"
    if features.gaze_ratio < 0.32 or features.gaze_ratio > 0.68 or abs(features.head_yaw) > 20.0:
        return "Not_Engaged"
    return "Engaged"
