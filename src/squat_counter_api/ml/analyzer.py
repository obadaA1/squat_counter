from __future__ import annotations

import tempfile
from pathlib import Path

from squat_counter_api.api.schemas import RepMetric, SquatPredictionResponse
from squat_counter_api.core.artifacts import ArtifactStatus
from squat_counter_api.ml.mediapipe_runner import extract_mediapipe_keypoints
from squat_counter_api.ml.signal import metrics_from_keypoints


class SquatAnalyzer:
    def __init__(self, model_root: Path, status: ArtifactStatus) -> None:
        self.model_root = model_root
        self.status = status
        self.model_path = model_root / str(status.metadata.get("model_file", "pose_landmarker.task"))

    def analyze(self, video_bytes: bytes) -> SquatPredictionResponse:
        if not self.status.ready:
            raise RuntimeError(self.status.message)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            keypoints = extract_mediapipe_keypoints(Path(tmp.name), self.model_path)
        rep_count, degradation_start_rep, rows = metrics_from_keypoints(keypoints)
        return SquatPredictionResponse(
            model_version=self.status.version or "unknown",
            rep_count=rep_count,
            degradation_start_rep=degradation_start_rep,
            per_rep_metrics=[RepMetric(**row) for row in rows],
            annotated_frames=[],
        )

