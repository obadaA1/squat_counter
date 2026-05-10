from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def _status(
    ready: bool,
    version: str | None,
    architecture: str | None,
    message: str,
    metadata: dict,
    manifest: dict,
) -> ArtifactStatus:
    return ArtifactStatus(ready, version, architecture, message, metadata, manifest)


@dataclass(frozen=True)
class ArtifactStatus:
    ready: bool
    version: str | None
    architecture: str | None
    message: str
    metadata: dict
    manifest: dict


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts(model_root: Path) -> ArtifactStatus:
    metadata = _read_json(model_root / "metadata.json")
    manifest = _read_json(model_root / "manifest.json")
    version = str(metadata.get("version")) if metadata.get("version") else None
    architecture = metadata.get("architecture")

    if not metadata:
        return _status(False, None, None, "metadata.json is missing or invalid.", metadata, manifest)
    if not manifest:
        return _status(False, version, architecture, "manifest.json is missing or invalid.", metadata, manifest)

    model_file = str(metadata.get("model_file", "pose_landmarker.task"))
    if not (model_root / model_file).is_file():
        return _status(False, version, architecture, "MediaPipe model file is missing.", metadata, manifest)

    expected_type = manifest.get("model_type")
    if expected_type and expected_type != "mediapipe_pose_landmarker":
        return _status(False, version, architecture, "manifest model_type is not supported.", metadata, manifest)

    for artifact in manifest.get("artifacts", []):
        relative_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not relative_path:
            return _status(False, version, architecture, "manifest artifact entry is missing path.", metadata, manifest)
        artifact_path = model_root / str(relative_path)
        if artifact.get("required", True) and not artifact_path.is_file():
            return _status(
                False,
                version,
                architecture,
                f"artifact file is missing: {relative_path}",
                metadata,
                manifest,
            )
        if expected_sha and artifact_path.is_file() and _sha256(artifact_path) != expected_sha:
            return _status(
                False,
                version,
                architecture,
                f"artifact checksum mismatch: {relative_path}",
                metadata,
                manifest,
            )

    return _status(True, version, architecture, "Artifacts validated.", metadata, manifest)
