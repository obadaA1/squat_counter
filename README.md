# Squat Counter API

Pose-estimation squat analysis service for live portfolio demos. The original notebook remains in this repository as research provenance under `research/`; production serving lives under `src/squat_counter_api`.

## API

- `GET /health` - reports service and model-artifact status
- `GET /health/live` - process liveness check
- `GET /health/ready` - readiness check; returns 503 until artifacts validate and the model is loadable
- `GET /model-info` - reports active model metadata
- `POST /predict` - accepts one MP4 file up to 50 MB and returns rep count, degradation start rep, per-rep metrics, and annotated frame placeholders

Structured errors use one shape:

```json
{
  "error": {
    "code": "request_error",
    "message": "Uploaded file does not look like a valid MP4 video.",
    "request_id": "2f8c8a2a-5c5d-47f1-8f9a-7ec5f4e2d6d9"
  }
}
```

Example request:

```bash
curl -fsS http://127.0.0.1:8012/health/live
curl -fsS http://127.0.0.1:8012/model-info
curl -fsS -X POST http://127.0.0.1:8012/predict \
  -F "file=@sample-squat.mp4;type=video/mp4"
```

Example response:

```json
{
  "model_version": "v1",
  "rep_count": 21,
  "degradation_start_rep": 5,
  "per_rep_metrics": [
    {
      "rep": 1,
      "depth_ratio": 0.9913,
      "torso_lean_degrees": 51.2,
      "z_depth": -0.6,
      "z_lean": -1.28,
      "flagged": false
    }
  ],
  "annotated_frames": []
}
```

## Runtime artifacts

Large artifacts are mounted on the UN1290 and are not committed to git:

```text
/models/squat/v1/metadata.json
/models/squat/v1/manifest.json
/models/squat/v1/pose_landmarker.task
/models/squat/current -> /models/squat/v1
/data/squat/samples
```

Example `metadata.json`:

```json
{
  "version": "v1",
  "architecture": "MediaPipe Pose Landmarker squat analyzer",
  "model_file": "pose_landmarker.task",
  "metrics": {
    "rep_count_accuracy": "21/21"
  }
}
```

The notebook implementation has been converted into production modules:

- `src/squat_counter_api/ml/mediapipe_runner.py` performs frame-by-frame MediaPipe
  Pose Landmarker inference.
- `src/squat_counter_api/ml/signal.py` contains the notebook's angle, smoothing,
  peak-counting, depth-ratio, torso-lean, and z-score degradation logic.
- `src/squat_counter_api/ml/analyzer.py` connects upload bytes to the inference and
  signal-processing pipeline.

## Local development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
SQUAT_MODEL_ROOT=/models/squat/current uvicorn squat_counter_api.api.app:app --host 127.0.0.1 --port 8012
```

## Docker

```bash
docker compose -f docker-compose.example.yml up -d --build
```

The compose example follows the production security baseline:

- binds only to `127.0.0.1`
- runs as the non-root `app` user from the image
- drops Linux capabilities
- enables `no-new-privileges`
- mounts model and sample artifacts read-only
- uses a read-only root filesystem with `/tmp` tmpfs for uploaded video processing
- exposes healthchecks for Docker and Cloudflare

Cloudflare Tunnel route:

```text
squat-api.obadaalsehli.com -> http://127.0.0.1:8012
```

Recommended Cloudflare rule: 10 requests/minute per IP for this hostname.

## Operations

CI/CD is defined in `.github/workflows/ci.yml`:

- Ruff linting.
- Mypy type checking.
- Pytest with coverage threshold.
- Dependency audit with `pip-audit`.
- Secret scanning with Gitleaks.
- Docker build on PR/push.
- GHCR publish and Trivy image scan on `main`/version tags.

Tests live in `tests/` and cover health/readiness behavior, artifact validation, upload validation, structured error responses, and mocked prediction contracts. The production signal-processing code lives in `src/squat_counter_api/ml/signal.py`; add more focused unit tests there as the metric logic evolves.

CORS is configured through `SQUAT_CORS_ORIGINS` and should be restricted in production to:

```text
https://obadaalsehli.com
```
