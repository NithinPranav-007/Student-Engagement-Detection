import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.config import CLASS_NAMES, DEFAULT_MODEL_PATH
from utils.inference import PredictionSmoother, compute_engagement_score, load_model_bundle, process_frame
from utils.landmarks import get_face_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime student engagement detection")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--log-csv", type=str, default="models/engagement_log.csv")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument("--smoothing-floor", type=float, default=0.45)
    return parser.parse_args()


def resolve_video_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def init_writer(output_path: str, frame_width: int, frame_height: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 20.0, (frame_width, frame_height))


def write_stats_log(log_csv_path: str, source: str, stats: dict[str, int]) -> None:
    ensure_parent_dir(log_csv_path)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "Engaged": int(stats.get("Engaged", 0)),
        "Not_Engaged": int(stats.get("Not_Engaged", 0)),
        "Drowsy": int(stats.get("Drowsy", 0)),
        "engagement_score": round(compute_engagement_score(stats), 2),
    }
    file_exists = Path(log_csv_path).exists()
    with open(log_csv_path, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_webcam_stream(bundle, confidence_threshold: float = 0.5, smoothing_window: int = 10, smoothing_floor: float = 0.45):
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Unable to open webcam")

    stats = {name: 0 for name in CLASS_NAMES}
    face_mesh = get_face_mesh()
    smoother = PredictionSmoother(window_size=smoothing_window, confidence_floor=smoothing_floor)
    start_time = time.time()

    while True:
        success, frame = capture.read()
        if not success:
            break

        annotated_frame, frame_stats = process_frame(
            frame=frame,
            bundle=bundle,
            face_mesh=face_mesh,
            confidence_threshold=confidence_threshold,
            prediction_smoother=smoother,
        )
        for label, count in frame_stats.items():
            stats[label] = stats.get(label, 0) + count

        cv2.imshow("Student Engagement Detection", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if time.time() - start_time > 60 * 30:
            break

    capture.release()
    cv2.destroyAllWindows()
    return stats


def process_video_file(
    video_path: str,
    bundle,
    confidence_threshold: float = 0.5,
    smoothing_window: int = 10,
    smoothing_floor: float = 0.45,
):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = str(Path("models") / f"annotated_{Path(video_path).stem}.mp4")
    ensure_parent_dir(output_path)
    writer = init_writer(output_path, frame_width, frame_height, fps)
    face_mesh = get_face_mesh()
    smoother = PredictionSmoother(window_size=smoothing_window, confidence_floor=smoothing_floor)
    stats = {name: 0 for name in CLASS_NAMES}

    while True:
        success, frame = capture.read()
        if not success:
            break

        annotated_frame, frame_stats = process_frame(
            frame=frame,
            bundle=bundle,
            face_mesh=face_mesh,
            confidence_threshold=confidence_threshold,
            prediction_smoother=smoother,
        )
        for label, count in frame_stats.items():
            stats[label] = stats.get(label, 0) + count
        writer.write(annotated_frame)

    capture.release()
    writer.release()
    return output_path, stats


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    bundle = load_model_bundle(args.model_path)
    source = resolve_video_source(args.source)

    if isinstance(source, int):
        stats = process_webcam_stream(
            bundle=bundle,
            confidence_threshold=args.confidence_threshold,
            smoothing_window=args.smoothing_window,
            smoothing_floor=args.smoothing_floor,
        )
        print(stats)
        print(f"Engagement score: {compute_engagement_score(stats):.1f}%")
        write_stats_log(args.log_csv, "webcam", stats)
    else:
        output_path, stats = process_video_file(
            video_path=source,
            bundle=bundle,
            confidence_threshold=args.confidence_threshold,
            smoothing_window=args.smoothing_window,
            smoothing_floor=args.smoothing_floor,
        )
        print(f"Saved to {output_path}")
        print(stats)
        print(f"Engagement score: {compute_engagement_score(stats):.1f}%")
        write_stats_log(args.log_csv, str(source), stats)


if __name__ == "__main__":
    main()
