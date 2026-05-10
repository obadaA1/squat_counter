from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

from fastapi import HTTPException, status

from squat_counter_api.api.schemas import RepMetric, SquatPredictionResponse
from squat_counter_api.core.artifacts import ArtifactStatus
from squat_counter_api.ml.annotator import render_annotated_video
from squat_counter_api.ml.mediapipe_runner import extract_mediapipe_keypoints
from squat_counter_api.ml.signal import metrics_and_peaks_from_keypoints

logger = logging.getLogger("squat_counter_api.analyzer")

MAX_VIDEO_DURATION_SECONDS = 90
MAX_VIDEO_FRAMES = 3000


def _probe_video(path: Path) -> tuple[float, int]:
    """Cheap header read; returns (fps, frame_count)."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required.") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not open uploaded video.",
        )
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return fps, frames


class SquatAnalyzer:
    def __init__(self, model_root: Path, status: ArtifactStatus) -> None:
        self.model_root = model_root
        self.status = status
        self.model_path = model_root / str(status.metadata.get("model_file", "pose_landmarker.task"))

    def analyze(self, video_bytes: bytes, annotate: bool = False) -> SquatPredictionResponse:
        if not self.status.ready:
            raise RuntimeError(self.status.message)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)
        try:
            fps, frame_count = _probe_video(tmp_path)
            duration = (frame_count / fps) if fps > 0 else float("inf")
            if frame_count > MAX_VIDEO_FRAMES or duration > MAX_VIDEO_DURATION_SECONDS:
                logger.warning(
                    "rejected oversized video",
                    extra={"frames": frame_count, "fps": fps, "duration": duration},
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Video must be {MAX_VIDEO_DURATION_SECONDS}s or shorter.",
                )

            keypoints = extract_mediapipe_keypoints(tmp_path, self.model_path)
            rep_count, degradation_start_rep, rows, peaks = metrics_and_peaks_from_keypoints(keypoints)
            annotated_b64: str | None = None
            if annotate:
                video_bytes_out = render_annotated_video(tmp_path, keypoints, peaks, degradation_start_rep)
                annotated_b64 = base64.b64encode(video_bytes_out).decode("ascii")
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        return SquatPredictionResponse(
            model_version=self.status.version or "unknown",
            rep_count=rep_count,
            degradation_start_rep=degradation_start_rep,
            per_rep_metrics=[RepMetric(**row) for row in rows],
            annotated_frames=[],
            annotated_video_b64=annotated_b64,
        )
