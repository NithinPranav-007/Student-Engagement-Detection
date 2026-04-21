import os
import tempfile
from pathlib import Path

import streamlit as st

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:  # pragma: no cover - optional dependency
    av = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None

from detect import process_video_file
from utils.config import CLASS_NAMES, DEFAULT_MODEL_PATH
from utils.inference import PredictionSmoother, compute_engagement_score, load_model_bundle, process_frame
from utils.landmarks import get_face_mesh

st.set_page_config(page_title="Student Engagement Detection", layout="wide")

ACTIVE_BUNDLE = None
ACTIVE_CONFIDENCE_THRESHOLD = 0.5


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18, 94, 138, 0.18), transparent 34%),
                radial-gradient(circle at top right, rgba(241, 196, 15, 0.14), transparent 28%),
                linear-gradient(180deg, #08111f 0%, #0d1727 55%, #111d30 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            padding: 1.5rem 1.6rem;
            border-radius: 24px;
            background: linear-gradient(145deg, rgba(10, 18, 33, 0.96), rgba(25, 46, 71, 0.92));
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 18px 50px rgba(4, 10, 20, 0.28);
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.09);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: 2.5rem;
            line-height: 1.05;
            margin: 0;
            font-weight: 800;
        }
        .hero-subtitle {
            margin-top: 0.8rem;
            max-width: 720px;
            color: rgba(255, 255, 255, 0.84);
            font-size: 1rem;
        }
        .panel-card {
            background: linear-gradient(165deg, rgba(9, 25, 45, 0.9), rgba(18, 42, 68, 0.86));
            border: 1px solid rgba(147, 197, 253, 0.22);
            border-radius: 22px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 14px 34px rgba(3, 10, 20, 0.32);
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.8rem;
        }
        .pill {
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.09);
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.82rem;
            color: rgba(255, 255, 255, 0.9);
        }
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(9, 18, 34, 0.08);
            border-radius: 18px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 12px 30px rgba(16, 24, 40, 0.05);
        }
        .section-label {
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #c7defd;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .text-card {
            color: #f3f8ff;
            font-size: 0.95rem;
            line-height: 1.65;
        }
        .text-card ul {
            margin: 0.5rem 0 0 1.15rem;
            padding: 0;
        }
        .text-card li {
            margin: 0.2rem 0;
        }
        .text-card li::marker {
            color: #93c5fd;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


class EngagementVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = get_face_mesh()
        self.stats = {name: 0 for name in CLASS_NAMES}
        self.last_summary = "Waiting for frames..."
        self.smoother = PredictionSmoother(window_size=10, confidence_floor=0.45)

    def reset(self):
        self.stats = {name: 0 for name in CLASS_NAMES}
        self.last_summary = "Waiting for frames..."
        self.smoother.reset()

    def recv(self, frame):
        if ACTIVE_BUNDLE is None or av is None:
            return frame

        frame_bgr = frame.to_ndarray(format="bgr24")
        annotated_frame, frame_stats = process_frame(
            frame_bgr,
            ACTIVE_BUNDLE,
            self.face_mesh,
            confidence_threshold=ACTIVE_CONFIDENCE_THRESHOLD,
            prediction_smoother=self.smoother,
        )
        for label, count in frame_stats.items():
            self.stats[label] = self.stats.get(label, 0) + count
        active_labels = [f"{name}: {count}" for name, count in self.stats.items()]
        engagement_score = compute_engagement_score(self.stats)
        self.last_summary = f"{' | '.join(active_labels)} | Score: {engagement_score:.1f}%"
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


@st.cache_resource
def load_bundle(model_path: str):
    return load_model_bundle(model_path)


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Classroom vision system</div>
        <h1 class="hero-title">Student Engagement Detection</h1>
        <p class="hero-subtitle">
            Upload a classroom clip or open the webcam to classify engagement in real time with CNN + landmark fusion.
        </p>
        <div class="pill-row">
            <span class="pill">Engaged</span>
            <span class="pill">Not_Engaged</span>
            <span class="pill">Drowsy</span>
            <span class="pill">Annotated export</span>
            <span class="pill">Webcam mode</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    source_mode = st.radio("Input source", ["Upload video", "Webcam"], index=0)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    smoothing_window = st.slider("Smoothing window", 3, 20, 10, 1)
    smoothing_floor = st.slider("Smoothing confidence floor", 0.2, 0.8, 0.45, 0.05)
    st.caption("Higher thresholds favor landmark heuristics when the CNN is uncertain.")

bundle = None
ACTIVE_CONFIDENCE_THRESHOLD = confidence_threshold
if os.path.exists(model_path):
    bundle = load_bundle(model_path)
    ACTIVE_BUNDLE = bundle
    st.success("Model loaded successfully.")
else:
    st.warning("Model file not found. Train the model first or update the path.")

status_col, metric_col, class_col = st.columns(3)
with status_col:
    st.metric("Model status", "Ready" if bundle is not None else "Missing")
with metric_col:
    st.metric("Confidence threshold", f"{confidence_threshold:.2f}")
with class_col:
    st.metric("Classes", str(len(CLASS_NAMES)))

left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    if source_mode == "Upload video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_video_path = temp_file.name

            if bundle is not None:
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.video(temp_video_path)
                analyze_clicked = st.button("Analyze video", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if analyze_clicked:
                    output_path = None
                    try:
                        output_path, stats = process_video_file(
                            video_path=temp_video_path,
                            bundle=bundle,
                            confidence_threshold=confidence_threshold,
                            smoothing_window=smoothing_window,
                            smoothing_floor=smoothing_floor,
                        )
                    finally:
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)

                    st.success(f"Saved annotated video to {output_path}")
                    st.video(output_path)
                    download_bytes = Path(output_path).read_bytes()
                    st.download_button(
                        "Download annotated video",
                        data=download_bytes,
                        file_name=Path(output_path).name,
                        mime="video/mp4",
                        use_container_width=True,
                    )
                    stat_columns = st.columns(len(CLASS_NAMES))
                    for column, class_name in zip(stat_columns, CLASS_NAMES):
                        with column:
                            st.metric(class_name, str(stats.get(class_name, 0)))
                    st.metric("Engagement score", f"{compute_engagement_score(stats):.1f}%")
            else:
                st.info("Load a trained model to analyze the video.")
    else:
        st.info("Webcam mode uses OpenCV + MediaPipe. If browser access is limited, run detect.py directly.")
        if bundle is not None:
            if webrtc_streamer is not None and av is not None:
                webrtc_ctx = webrtc_streamer(
                    key="student-engagement-webcam",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=EngagementVideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.smoother.configure(
                        window_size=int(smoothing_window),
                        confidence_floor=float(smoothing_floor),
                    )
                    if st.button("Reset webcam stats", use_container_width=True):
                        webrtc_ctx.video_processor.reset()
                    st.caption(webrtc_ctx.video_processor.last_summary)
                    stats_columns = st.columns(len(CLASS_NAMES))
                    for column, class_name in zip(stats_columns, CLASS_NAMES):
                        with column:
                            st.metric(class_name, str(webrtc_ctx.video_processor.stats.get(class_name, 0)))
                    st.metric("Engagement score", f"{compute_engagement_score(webrtc_ctx.video_processor.stats):.1f}%")
            else:
                st.info("Install streamlit-webrtc to enable webcam inside the app.")
        else:
            st.info("Load a trained model first.")

with right_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-label">Engagement classes</div>
            <div class="text-card">
                <ul>
                    <li>Engaged</li>
                    <li>Not_Engaged</li>
                    <li>Drowsy</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="panel-card" style="margin-top: 1rem;">
            <div class="section-label">Pipeline</div>
            <div class="text-card">
                <ul>
                    <li>Face detection</li>
                    <li>Landmark extraction</li>
                    <li>CNN prediction</li>
                    <li>TTA and heuristic fusion</li>
                    <li>Engagement summary</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="panel-card" style="margin-top: 1rem;">
            <div class="section-label">Accuracy boosters</div>
            <div class="text-card">
                <ul>
                    <li>Class-weighted training</li>
                    <li>Stronger augmentation</li>
                    <li>Label smoothing and early stopping</li>
                    <li>Flip-based test-time augmentation</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
