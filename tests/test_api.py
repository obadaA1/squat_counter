from fastapi.testclient import TestClient

from squat_counter_api.api import app as app_module
from squat_counter_api.api.schemas import ModelInfoResponse
from squat_counter_api.core.artifacts import validate_artifacts


def valid_mp4_bytes() -> bytes:
    return b"\x00\x00\x00\x18ftypmp42" + b"0" * 16


def test_health_live_is_always_ok():
    client = TestClient(app_module.create_app())

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_missing_artifacts():
    app_module.get_analyzer_service.cache_clear()
    client = TestClient(app_module.create_app())

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "request_error"


def test_artifact_validation_requires_manifest(tmp_path):
    (tmp_path / "metadata.json").write_text('{"version":"v1","model_file":"pose_landmarker.task"}', encoding="utf-8")
    (tmp_path / "pose_landmarker.task").write_bytes(b"model")

    status = validate_artifacts(tmp_path)

    assert status.ready is False
    assert "manifest.json" in status.message


def test_rejects_invalid_magic_bytes():
    client = TestClient(app_module.create_app())

    response = client.post(
        "/predict",
        files={"file": ("fake.mp4", b"not-a-video", "video/mp4")},
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_model_info_with_mocked_service(monkeypatch):
    class Service:
        model_loaded = True
        ready = True
        version = "v1-test"

        def info(self):
            return ModelInfoResponse(model_loaded=True, version="v1-test", architecture="MediaPipe")

    monkeypatch.setattr(app_module, "get_analyzer_service", lambda: Service())
    client = TestClient(app_module.create_app())

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["version"] == "v1-test"
