import json
import tempfile
from pathlib import Path

from .mediapipe_runner import extract_mediapipe_keypoints
from .schemas import ModelInfoResponse, RepMetric, SquatPredictionResponse
from .signal import metrics_from_keypoints

SAMPLE_METRICS = [
    RepMetric(rep=1, depth_ratio=0.9913, torso_lean_degrees=51.2, z_depth=-0.60, z_lean=-1.28, flagged=False),
    RepMetric(rep=2, depth_ratio=1.0083, torso_lean_degrees=55.5, z_depth=1.41, z_lean=0.12, flagged=False),
    RepMetric(rep=3, depth_ratio=0.9896, torso_lean_degrees=58.7, z_depth=-0.81, z_lean=1.16, flagged=False),
    RepMetric(rep=4, depth_ratio=0.9980, torso_lean_degrees=60.0, z_depth=0.19, z_lean=1.57, flagged=False),
    RepMetric(rep=5, depth_ratio=0.9625, torso_lean_degrees=59.0, z_depth=-4.00, z_lean=1.27, flagged=True),
    RepMetric(rep=6, depth_ratio=0.9681, torso_lean_degrees=60.3, z_depth=-3.34, z_lean=1.70, flagged=True),
    RepMetric(rep=7, depth_ratio=0.9914, torso_lean_degrees=62.4, z_depth=-0.59, z_lean=2.38, flagged=True),
]


class SquatAnalyzerService:
    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self.metadata = self._load_metadata()
        self.model_path = model_root / str(self.metadata.get("model_file", "pose_landmarker.task"))

    def _load_metadata(self) -> dict:
        metadata_path = self.model_root / "metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    @property
    def model_loaded(self) -> bool:
        return bool(self.metadata) and self.model_path.exists()

    @property
    def version(self) -> str | None:
        value = self.metadata.get("version")
        return str(value) if value else None

    def info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_loaded=self.model_loaded,
            version=self.version,
            architecture=(
                self.metadata.get("architecture", "MediaPipe Pose Landmarker squat analyzer") if self.metadata else None
            ),
            training_date=self.metadata.get("training_date"),
            metrics=self.metadata.get("metrics", {}),
        )

    def analyze(self, video_bytes: bytes) -> SquatPredictionResponse:
        if not self.model_loaded:
            raise RuntimeError("Model artifacts are not mounted or loaded.")

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            keypoints = extract_mediapipe_keypoints(Path(tmp.name), self.model_path)

        rep_count, degradation_start_rep, rows = metrics_from_keypoints(keypoints)

        return SquatPredictionResponse(
            model_version=self.version or "unknown",
            rep_count=rep_count,
            degradation_start_rep=degradation_start_rep,
            per_rep_metrics=[RepMetric(**row) for row in rows],
            annotated_frames=[],
        )
