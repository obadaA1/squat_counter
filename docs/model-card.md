# Squat Counter Model Card

## Intended Use

Analyze one MP4 squat video for rep count and simple degradation indicators in a public portfolio demo.

## Inputs and Outputs

- Input: one MP4 video, maximum 50 MB.
- Output: rep count, first degradation rep, per-rep depth and torso lean metrics.

## Runtime Artifacts

Artifacts are mounted read-only at `/models/squat/current`:

- `metadata.json`
- `manifest.json`
- `pose_landmarker.task`

`manifest.json` records artifact names, SHA256 checksums, model type, version, and architecture.

## Limitations

The v1 implementation assumes a visible single-person side-view squat video. It can fail on low light, multiple people, occlusion, unusual camera angles, or partial-body framing.

## Operational Notes

The API refuses readiness until metadata, manifest, and MediaPipe task artifacts validate successfully.

