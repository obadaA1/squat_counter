"""HUD video annotation matching the original research notebook output.

Renders the MediaPipe skeleton, torso lean line, and a translucent stats
panel onto each frame of the uploaded video. Returns H.264 mp4 bytes.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("squat_counter_api.annotator")

from squat_counter_api.ml.signal import (
    BASELINE_REPS,
    MP_LEFT_ANKLE,
    MP_LEFT_HIP,
    MP_LEFT_KNEE,
    MP_LEFT_SHOULDER,
    Z_THRESHOLD,
    knee_angles_from_mediapipe,
    smooth_angles,
    z_scores,
)

POSE_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]
SKELETON_COLOR = (0, 255, 0)
JOINT_COLOR = (0, 255, 0)
JOINT_OUTLINE = (255, 255, 255)
LEAN_COLOR = (0, 255, 255)
HUD_COLOR = (0, 0, 0)
HUD_ALPHA = 0.6
TEXT_WHITE = (255, 255, 255)
TEXT_YELLOW = (0, 255, 255)
TEXT_LEAN = (255, 200, 0)
DEPTH_OK = (0, 255, 0)
DEPTH_DEGRADED = (0, 100, 255)


def _torso_lean_per_frame(keypoints):
    import numpy as np

    leans = np.zeros(len(keypoints), dtype=float)
    for i, frame in enumerate(keypoints):
        sx, sy = frame[MP_LEFT_SHOULDER, 0], frame[MP_LEFT_SHOULDER, 1]
        hx, hy = frame[MP_LEFT_HIP, 0], frame[MP_LEFT_HIP, 1]
        sv = frame[MP_LEFT_SHOULDER, 3] if frame.shape[1] > 3 else 1.0
        hv = frame[MP_LEFT_HIP, 3] if frame.shape[1] > 3 else 1.0
        if sv > 0.5 and hv > 0.5:
            dx = sx - hx
            dy = sy - hy
            leans[i] = abs(np.degrees(np.arctan2(dx, -dy)))
    return leans


def _build_rep_map(peaks, num_frames):
    import numpy as np

    rep_map = np.zeros(num_frames, dtype=int)
    for r, peak in enumerate(peaks):
        end = peaks[r + 1] if r + 1 < len(peaks) else num_frames
        rep_map[peak:end] = r + 1
    return rep_map


def _draw_hud(cv2, frame, knee_angle, lean_angle, current_rep, total_reps,
              first_degraded, frame_idx, total_frames):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 195), HUD_COLOR, -1)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

    y = 38
    cv2.putText(frame, "MediaPipe Pose", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SKELETON_COLOR, 2)
    y += 32
    cv2.putText(frame, f"Knee: {knee_angle:.1f} deg", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_YELLOW, 2)
    y += 28
    cv2.putText(frame, f"Lean: {lean_angle:.1f} deg", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_LEAN, 2)
    y += 28
    cv2.putText(frame, f"Rep: {current_rep}/{total_reps}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_WHITE, 2)
    y += 28
    if first_degraded and current_rep >= first_degraded:
        text = f"Form: DEGRADED (Rep {first_degraded}+)"
        color = DEPTH_DEGRADED
    elif first_degraded and current_rep > 0:
        text = "Form: OK"
        color = DEPTH_OK
    else:
        text = "Form: ---"
        color = (200, 200, 200)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def _draw_pose(cv2, frame, kp_frame):
    h, w = frame.shape[:2]
    for s, e in POSE_CONNECTIONS:
        if kp_frame[s, 3] < 0.5 or kp_frame[e, 3] < 0.5:
            continue
        x1, y1 = int(kp_frame[s, 0] * w), int(kp_frame[s, 1] * h)
        x2, y2 = int(kp_frame[e, 0] * w), int(kp_frame[e, 1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), SKELETON_COLOR, 2)

    for j in (MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, MP_LEFT_SHOULDER):
        if kp_frame[j, 3] < 0.5:
            continue
        cx, cy = int(kp_frame[j, 0] * w), int(kp_frame[j, 1] * h)
        cv2.circle(frame, (cx, cy), 6, JOINT_COLOR, -1)
        cv2.circle(frame, (cx, cy), 6, JOINT_OUTLINE, 2)

    if kp_frame[MP_LEFT_SHOULDER, 3] > 0.5 and kp_frame[MP_LEFT_HIP, 3] > 0.5:
        sx = int(kp_frame[MP_LEFT_SHOULDER, 0] * w)
        sy = int(kp_frame[MP_LEFT_SHOULDER, 1] * h)
        hx = int(kp_frame[MP_LEFT_HIP, 0] * w)
        hy = int(kp_frame[MP_LEFT_HIP, 1] * h)
        cv2.line(frame, (sx, sy), (hx, hy), LEAN_COLOR, 3)
        cv2.line(frame, (hx, hy), (hx, max(0, hy - 80)), TEXT_WHITE, 1, cv2.LINE_AA)


def render_annotated_video(
    video_path: Path,
    keypoints,
    peaks,
    degradation_start: int | None,
) -> bytes:
    """Read the source video frame-by-frame, overlay HUD, write H.264 mp4."""
    import cv2
    import numpy as np

    if len(keypoints) == 0:
        raise RuntimeError("Cannot annotate without keypoints.")

    angles_smooth = smooth_angles(knee_angles_from_mediapipe(keypoints))
    lean_per_frame = _torso_lean_per_frame(keypoints)
    if len(lean_per_frame) >= 7:
        try:
            from scipy.signal import savgol_filter
            lean_per_frame = savgol_filter(lean_per_frame, 7, 2)
        except Exception:
            pass
    rep_map = _build_rep_map(np.asarray(peaks, dtype=int), len(keypoints))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video for annotation.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale very large frames so the response stays small.
    max_dim = 720
    scale = min(1.0, max_dim / max(width, height))
    out_w = int(width * scale) if scale < 1.0 else width
    out_h = int(height * scale) if scale < 1.0 else height
    if out_w % 2:
        out_w -= 1
    if out_h % 2:
        out_h -= 1

    # OpenCV writes mp4v reliably from the pip wheel; we transcode to H.264 below.
    fd, raw_path = tempfile.mkstemp(suffix=".mp4", prefix="annotated_raw_")
    os.close(fd)
    raw_path_obj = Path(raw_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path_obj), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for annotation output.")

    total = len(keypoints)
    total_reps = int(len(peaks))

    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx >= total:
                # No keypoints for this frame; skip drawing pose but keep HUD.
                kp_frame = None
            else:
                kp_frame = keypoints[frame_idx]

            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            if kp_frame is not None:
                _draw_pose(cv2, frame, kp_frame)

            knee_angle = float(angles_smooth[frame_idx]) if frame_idx < len(angles_smooth) else 0.0
            lean_angle = float(lean_per_frame[frame_idx]) if frame_idx < len(lean_per_frame) else 0.0
            current_rep = int(rep_map[frame_idx]) if frame_idx < len(rep_map) else 0
            _draw_hud(cv2, frame, knee_angle, lean_angle, current_rep, total_reps,
                      degradation_start, frame_idx, total)
            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()
        capture.release()

    # Transcode mp4v -> H.264 (avc1) for universal browser playback and smaller size.
    final_path_obj = raw_path_obj
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found in PATH; returning raw mp4v output (large, limited browser support)")
    else:
        fd2, h264_path = tempfile.mkstemp(suffix=".mp4", prefix="annotated_h264_")
        os.close(fd2)
        h264_obj = Path(h264_path)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(raw_path_obj),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    str(h264_obj),
                ],
                check=True,
                timeout=60,
            )
            final_path_obj = h264_obj
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg transcode timed out after 60s; falling back to raw mp4v")
            try:
                h264_obj.unlink()
            except FileNotFoundError:
                pass
        except subprocess.CalledProcessError as exc:
            logger.error("ffmpeg transcode failed (rc=%s); falling back to raw mp4v", exc.returncode)
            try:
                h264_obj.unlink()
            except FileNotFoundError:
                pass

    try:
        return final_path_obj.read_bytes()
    finally:
        for path in {raw_path_obj, final_path_obj}:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
