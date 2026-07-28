from __future__ import annotations

from sibisee.config import AppSettings, InferenceSettings, TemporalSettings


def render_sidebar(st, settings: AppSettings) -> tuple[str, InferenceSettings, TemporalSettings]:
    st.sidebar.header("Pengaturan")
    mode = st.sidebar.radio(
        "Mode",
        ["Live kamera", "Gambar statis"],
    )
    confidence = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.inference.confidence_threshold),
        step=0.05,
    )
    infer_every = st.sidebar.number_input(
        "Inferensi setiap N frame",
        min_value=1,
        max_value=30,
        value=settings.inference.infer_every_n_frames,
    )
    temporal_window = st.sidebar.number_input(
        "Temporal window",
        min_value=1,
        max_value=30,
        value=settings.temporal.window_size,
    )
    min_frames = st.sidebar.number_input(
        "Frame cocok minimum",
        min_value=1,
        max_value=int(temporal_window),
        value=min(settings.temporal.min_matching_frames, int(temporal_window)),
    )

    inference = InferenceSettings(
        image_size=settings.inference.image_size,
        infer_every_n_frames=int(infer_every),
        confidence_threshold=float(confidence),
        iou_threshold=settings.inference.iou_threshold,
        max_detections=settings.inference.max_detections,
        primary_detection_strategy=settings.inference.primary_detection_strategy,
        class_allowlist=settings.inference.class_allowlist,
    )
    temporal = TemporalSettings(
        window_size=int(temporal_window),
        min_matching_frames=int(min_frames),
        min_average_confidence=settings.temporal.min_average_confidence,
        confirmation_cooldown_seconds=settings.temporal.confirmation_cooldown_seconds,
    )
    return mode, inference, temporal
