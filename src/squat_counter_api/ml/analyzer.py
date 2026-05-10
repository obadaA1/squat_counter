from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from squat_counter_api.api.schemas import RepMetric, SquatPredictionResponse
from squat_counter_api.core.artifacts import ArtifactStatus
from squat_counter_api.ml.annotator import render_annotated_video
from squat_counter_api.ml.mediapipe_runner import extract_mediapipe_keypoints
from squat_counter_api.ml.signal import metrics_and_peaks_from_keypoints


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
