from __future__ import annotations

from pathlib import Path


def extract_mediapipe_keypoints(video_path: Path, model_path: Path):
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("MediaPipe inference requires opencv-python-headless and mediapipe.") from exc

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open uploaded MP4 video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    keypoints = []

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((frame_index / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    keypoints.append(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility or 0.0, landmark.presence or 0.0]
                            for landmark in result.pose_landmarks[0]
                        ]
                    )
                frame_index += 1
    finally:
        capture.release()

    if not keypoints:
        raise RuntimeError("No pose landmarks were detected in the uploaded video.")
    return np.asarray(keypoints, dtype=float)

