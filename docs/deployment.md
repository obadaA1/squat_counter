# Squat API Deployment

The API is designed to run on the UN1290 behind Cloudflare Tunnel. Bind the
container to localhost only and let Cloudflare provide public HTTPS.

## Artifact layout

Runtime artifacts are mounted read-only and are not committed to git:

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

Example `manifest.json`:

```json
{
  "model_version": "v1",
  "model_type": "mediapipe_pose_landmarker",
  "created_at": "2026-04-01T00:00:00Z",
  "artifacts": [
    {"path": "pose_landmarker.task", "sha256": "<sha256>"}
  ]
}
```

## Docker run

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
