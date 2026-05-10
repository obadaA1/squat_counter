from __future__ import annotations

MP_LEFT_SHOULDER = 11
MP_LEFT_HIP = 23
MP_LEFT_KNEE = 25
MP_LEFT_ANKLE = 27

SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2
PEAK_HEIGHT = 140
PEAK_DISTANCE = 30
PEAK_PROMINENCE = 30
BASELINE_REPS = 3
Z_THRESHOLD = 2.0


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Squat analysis requires numpy.") from exc
    return np


def _find_peaks():
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise RuntimeError("Squat analysis requires scipy.") from exc
    return find_peaks


def _savgol_filter():
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise RuntimeError("Squat analysis requires scipy.") from exc
    return savgol_filter


def compute_angle(a, b, c) -> float:
    np = _np()
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def smooth_angles(angles):
    np = _np()
    savgol_filter = _savgol_filter()
    angles = np.asarray(angles, dtype=float)
    if len(angles) < SAVGOL_WINDOW:
        return angles
    window = min(SAVGOL_WINDOW, len(angles) if len(angles) % 2 == 1 else len(angles) - 1)
    if window < 3:
        return angles
    return savgol_filter(angles, window_length=window, polyorder=min(SAVGOL_POLYORDER, window - 1))


def knee_angles_from_mediapipe(keypoints):
    np = _np()
    return np.array(
        [
            compute_angle(frame[MP_LEFT_HIP, :2], frame[MP_LEFT_KNEE, :2], frame[MP_LEFT_ANKLE, :2])
            for frame in keypoints
        ],
        dtype=float,
    )


def count_reps(smoothed_angles):
    find_peaks = _find_peaks()
    peaks, _ = find_peaks(
        smoothed_angles,
        height=PEAK_HEIGHT,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
    )
    return peaks


def find_bottom_frames(smoothed_angles):
    find_peaks = _find_peaks()
    valleys, _ = find_peaks(
        -smoothed_angles,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
    )
    return valleys


def compute_depth_ratio(keypoints, valley_frames):
    np = _np()
    ratios = []
    for frame_index in valley_frames:
        hip_y = keypoints[frame_index, MP_LEFT_HIP, 1]
        knee_y = keypoints[frame_index, MP_LEFT_KNEE, 1]
        ratios.append(float(hip_y / knee_y) if knee_y > 0.01 else 0.0)
    return np.array(ratios, dtype=float)


def compute_torso_lean(keypoints, valley_frames):
    np = _np()
    leans = []
    vertical = np.array([0, -1])
    for frame_index in valley_frames:
        shoulder = keypoints[frame_index, MP_LEFT_SHOULDER, :2]
        hip = keypoints[frame_index, MP_LEFT_HIP, :2]
        torso = shoulder - hip
        cos_angle = np.dot(torso, vertical) / (np.linalg.norm(torso) * np.linalg.norm(vertical) + 1e-8)
        leans.append(float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))))
    return np.array(leans, dtype=float)


def z_scores(metric_values, baseline_n: int = BASELINE_REPS):
    np = _np()
    values = np.asarray(metric_values, dtype=float)
    if len(values) == 0:
        return values
    if len(values) <= baseline_n:
        baseline_n = max(1, len(values) - 1)
    mean = np.mean(values[:baseline_n])
    std = np.std(values[:baseline_n]) + 1e-8
    return (values - mean) / std


def first_flagged_rep(*score_arrays, threshold: float = Z_THRESHOLD) -> int | None:
    if not score_arrays:
        return None
    end = min(len(scores) for scores in score_arrays)
    start = min(BASELINE_REPS, max(0, end - 1))
    for index in range(start, end):
        if any(abs(float(scores[index])) > threshold for scores in score_arrays):
            return index + 1
    return None


def metrics_from_keypoints(keypoints):
    angles = knee_angles_from_mediapipe(keypoints)
    smoothed = smooth_angles(angles)
    peaks = count_reps(smoothed)
    valleys = find_bottom_frames(smoothed)
    depth = compute_depth_ratio(keypoints, valleys)
    lean = compute_torso_lean(keypoints, valleys)
    z_depth = z_scores(depth)
    z_lean = z_scores(lean)
    degradation_start = first_flagged_rep(z_depth, z_lean)

    rows = []
    for index in range(min(len(depth), len(lean))):
        rows.append(
            {
                "rep": index + 1,
                "depth_ratio": float(depth[index]),
                "torso_lean_degrees": float(lean[index]),
                "z_depth": float(z_depth[index]),
                "z_lean": float(z_lean[index]),
                "flagged": abs(float(z_depth[index])) > Z_THRESHOLD or abs(float(z_lean[index])) > Z_THRESHOLD,
            }
        )

    return len(peaks), degradation_start, rows
