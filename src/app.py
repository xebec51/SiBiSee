from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import streamlit as st

from sibisee.config import AppSettings, load_settings
from sibisee.domain.detection import select_primary_detection
from sibisee.domain.transcript import TranscriptBuilder
from sibisee.inference.detector import YoloDetector
from sibisee.inference.model_loader import ModelLoadError, load_encrypted_model
from sibisee.inference.preprocessing import ImageValidationError, ImageValidationSettings, validate_image_bytes
from sibisee.inference.temporal_decoder import TemporalDecoder
from sibisee.logging_config import configure_logging
from sibisee.services.gesture_guide import discover_guide_items
from sibisee.services.ice_servers import get_ice_servers
from sibisee.ui.footer import render_footer
from sibisee.ui.gesture_guide import render_guide
from sibisee.ui.sidebar import render_sidebar
from sibisee.ui.transcript import get_transcript, render_transcript_panel

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def cached_detector(settings: AppSettings, confidence_threshold: float) -> YoloDetector:
    model_security = st.secrets.get("model_security", {}) if hasattr(st.secrets, "get") else {}
    key = model_security.get("ENCRYPTION_KEY") or os.getenv("SIBISEE_MODEL_ENCRYPTION_KEY")
    inference_settings = settings.inference.__class__(
        image_size=settings.inference.image_size,
        infer_every_n_frames=settings.inference.infer_every_n_frames,
        confidence_threshold=confidence_threshold,
        iou_threshold=settings.inference.iou_threshold,
        max_detections=settings.inference.max_detections,
        primary_detection_strategy=settings.inference.primary_detection_strategy,
        class_allowlist=settings.inference.class_allowlist,
    )
    model = load_encrypted_model(
        settings.security.encrypted_model_path,
        key,
        expected_sha256=settings.security.model_sha256,
    )
    return YoloDetector(model, inference_settings)


@st.cache_data(show_spinner=False)
def cached_guide_items(guide_dir: str):
    return discover_guide_items(Path(guide_dir))


class VideoRecognitionProcessor:
    def __init__(
        self,
        detector: YoloDetector,
        decoder: TemporalDecoder,
        transcript: TranscriptBuilder,
        infer_every_n_frames: int,
    ) -> None:
        self.detector = detector
        self.decoder = decoder
        self.transcript = transcript
        self.infer_every_n_frames = infer_every_n_frames
        self.frame_index = 0
        self.last_token: str | None = None
        self.last_confidence = 0.0
        self.last_latency_ms = 0.0
        self._lock = threading.Lock()

    def snapshot(self) -> tuple[str | None, float, float]:
        with self._lock:
            return self.last_token, self.last_confidence, self.last_latency_ms

    def recv(self, frame):
        import av

        image = frame.to_ndarray(format="bgr24")
        self.frame_index += 1
        if self.frame_index % self.infer_every_n_frames == 0:
            result = self.detector.predict(image, annotate=True)
            state = self.decoder.update(result.detections)
            with self._lock:
                self.last_token = state.stable_token
                self.last_confidence = state.stable_confidence
                self.last_latency_ms = result.latency_ms
                self.transcript.append(state.confirmed_token)
            return av.VideoFrame.from_ndarray(result.annotated_frame, format="bgr24")
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def _render_static_image(st, detector: YoloDetector, transcript: TranscriptBuilder, settings: AppSettings) -> None:
    st.subheader("Gambar statis")
    source = st.radio("Sumber gambar", ("Upload file", "Ambil foto"), horizontal=True)
    uploaded = (
        st.file_uploader("Upload JPEG/PNG", type=["jpg", "jpeg", "png"])
        if source == "Upload file"
        else st.camera_input("Ambil foto gestur")
    )

    if uploaded is None:
        st.info("Pilih gambar untuk menjalankan pengenalan gestur.")
        return

    try:
        image = validate_image_bytes(
            uploaded.getvalue(),
            getattr(uploaded, "type", None),
            ImageValidationSettings(
                max_upload_mb=settings.security.max_upload_mb,
                max_image_pixels=settings.security.max_image_pixels,
            ),
        )
    except ImageValidationError as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    left.image(image, caption="Gambar input", use_container_width=True)
    if st.button("Deteksi sekarang", type="primary"):
        with st.spinner("Memproses gambar..."):
            result = detector.predict(image)
            primary = apply_static_prediction(result, transcript, settings)
            right.write("Hasil deteksi")
            if result.detections:
                for detection in result.detections:
                    right.info(f"{detection.label} ({detection.confidence:.2f})")
            else:
                right.warning("Tidak ada gestur yang terdeteksi.")
            render_transcript_panel(
                st,
                transcript,
                primary.label if primary else None,
                primary.confidence if primary else 0.0,
            )


def apply_static_prediction(result, transcript: TranscriptBuilder, settings: AppSettings):
    primary = select_primary_detection(result.detections, settings.inference.primary_detection_strategy)
    transcript.append(primary.label if primary else None)
    return primary


def _render_live_camera(
    st, detector: YoloDetector, decoder: TemporalDecoder, transcript: TranscriptBuilder, settings: AppSettings
) -> None:
    st.subheader("Live kamera")
    st.caption(
        "Pengenalan gestur SIBI terisolasi secara real-time. Gestur dinamis dan tata bahasa belum sepenuhnya ditangani."
    )
    try:
        from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("Dependency streamlit-webrtc belum terpasang.")
        return

    ice_servers = get_ice_servers(st.secrets)
    rtc_configuration = RTCConfiguration({"iceServers": ice_servers})
    ctx = webrtc_streamer(
        key="sibisee-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: VideoRecognitionProcessor(
            detector,
            decoder,
            transcript,
            settings.inference.infer_every_n_frames,
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    processor = ctx.video_processor
    if processor:

        def render_live_status() -> None:
            last_token, confidence, latency_ms = processor.snapshot()
            render_transcript_panel(st, transcript, last_token, confidence)
            st.caption(f"Latency inferensi terakhir: {latency_ms:.1f} ms")

        if hasattr(st, "fragment"):
            st.fragment(run_every="1s")(render_live_status)()
        else:
            render_live_status()


def get_temporal_decoder_for_settings(session_state, settings):
    fingerprint = (
        settings.temporal.window_size,
        settings.temporal.min_matching_frames,
        settings.temporal.min_average_confidence,
        settings.temporal.confirmation_cooldown_seconds,
    )
    if "temporal_decoder" not in session_state or session_state.get("temporal_settings_fingerprint") != fingerprint:
        session_state.temporal_decoder = TemporalDecoder(settings.temporal)
        session_state.temporal_settings_fingerprint = fingerprint
    return session_state.temporal_decoder


def main() -> None:
    configure_logging()
    settings = load_settings()
    st.set_page_config(page_title="SiBiSee", layout="wide")
    st.title("SiBiSee")
    st.write("Pengenalan gestur SIBI terisolasi secara real-time dengan YOLO.")

    mode, inference_settings, temporal_settings = render_sidebar(st, settings)
    settings = AppSettings(
        project_root=settings.project_root,
        assets_dir=settings.assets_dir,
        guide_dir=settings.guide_dir,
        inference=inference_settings,
        temporal=temporal_settings,
        security=settings.security,
    )
    render_guide(st, cached_guide_items(str(settings.guide_dir)))

    try:
        detector = cached_detector(settings, settings.inference.confidence_threshold)
    except ModelLoadError as exc:
        LOGGER.exception("Model startup failed.")
        st.error(str(exc))
        st.stop()

    transcript = get_transcript(st)
    decoder = get_temporal_decoder_for_settings(st.session_state, settings)

    if mode == "Live kamera":
        _render_live_camera(st, detector, decoder, transcript, settings)
    else:
        _render_static_image(st, detector, transcript, settings)

    render_footer(st)


if __name__ == "__main__":
    main()
