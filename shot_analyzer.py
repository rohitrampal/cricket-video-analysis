import numpy as np
import logging
import cv2
from config import FIELD_ZONES, MIN_SHOT_ANGLE_CHANGE, SHOT_COLORS, DEFAULT_SHOT_COLOR
from angle_utils import normalize_cricket_angle

log = logging.getLogger(__name__)


def normalize_angle(raw_angle_deg: float, batsman_facing: str = "right") -> float:
    # Reconstruct an image-space unit vector and normalize centrally.
    theta = np.radians(float(raw_angle_deg))
    dx = float(np.sin(theta))
    dy = float(-np.cos(theta))
    return normalize_cricket_angle(dx=dx, dy=dy, batsman_facing=batsman_facing)


def get_field_zone(angle_deg: float) -> str:
    for zone, (min_a, max_a) in FIELD_ZONES.items():
        if min_a <= angle_deg <= max_a:
            return zone
    if angle_deg > 160 or angle_deg < -160:
        return "Fine Leg"
    return "Unknown"


def estimate_shot_length(bat_vector: dict, scale: float = 1.0) -> float:
    """
    Improved length estimation — scaled for typical phone/broadcast video.
    Now uses a lower divisor so small wrist movements still show good length.
    """
    dx, dy    = bat_vector["dx"], bat_vector["dy"]
    magnitude = np.sqrt(dx**2 + dy**2)

    # Lower divisor = longer lines on wagon wheel
    # 60 works well for phone video, increase to 100 for broadcast
    normalized = np.clip(magnitude / 60.0, 0.35, 1.0)
    return round(float(normalized), 3)


def classify_shot_type(angle_deg: float, length: float) -> str:
    """
    Improved classifier — lowered defensive threshold from 0.45 to 0.38.
    """
    a = angle_deg
    l = length

    if l < 0.38:
        return "Defensive"
    if 60 <= a <= 120:
        return "Pull / Hook"
    if 20 <= a < 60:
        return "Flick / Glance"
    if -20 <= a < 20:
        return "Drive (Straight)"
    if -60 <= a < -20:
        return "Cover Drive"
    if -120 <= a < -60:
        return "Cut"
    if a > 120 or a < -120:
        return "Sweep / Ramp"
    return "Drive"


def classify_shot_type_advanced(angle_deg: float, lofted: bool, height_class: str) -> str:
    a = float(angle_deg)
    h = 6.0  # hysteresis band to reduce boundary flips
    bins = [
        ("Sweep / Ramp", -180.0, -120.0),
        ("Cut Shot", -120.0, -60.0),
        ("Cover Drive", -60.0, -20.0),
        ("Straight Drive", -20.0, 20.0),
        ("Flick", 20.0, 60.0),
        ("Pull / Hook", 60.0, 120.0),
        ("Sweep / Ramp", 120.0, 180.0),
    ]
    centers = {
        "Sweep / Ramp": 160.0 if a >= 0 else -160.0,
        "Cut Shot": -90.0,
        "Cover Drive": -40.0,
        "Straight Drive": 0.0,
        "Flick": 40.0,
        "Pull / Hook": 90.0,
    }
    base = "Straight Drive"
    for idx, (name, lo, hi) in enumerate(bins):
        core_lo = lo + h
        core_hi = hi - h
        if core_lo <= a <= core_hi:
            base = name
            break
        if lo <= a <= hi:
            # boundary zone: choose nearest class center among this and neighbors.
            cand = [name]
            if idx > 0:
                cand.append(bins[idx - 1][0])
            if idx + 1 < len(bins):
                cand.append(bins[idx + 1][0])
            base = min(cand, key=lambda c: abs(a - centers[c]))
            break
    if lofted:
        if height_class == "Six-level":
            return f"Lofted {base} (Six-level)"
        return f"Lofted {base}"
    return base


def _post_impact_points(raw_shot: dict) -> np.ndarray:
    pts = raw_shot.get("ball_filtered_points") or []
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.array(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr


def _rectify_points(points: np.ndarray, raw_shot: dict) -> np.ndarray:
    """
    Lightweight rectification. Uses optional manual 4-point mapping if provided:
    raw_shot["rectify_src_points"], raw_shot["rectify_dst_points"].
    Falls back to identity transform if unavailable.
    """
    if len(points) < 2:
        return points
    src = raw_shot.get("rectify_src_points")
    dst = raw_shot.get("rectify_dst_points")
    if not src or not dst or len(src) != 4 or len(dst) != 4:
        return points
    try:
        src_np = np.array(src, dtype=np.float32).reshape(4, 2)
        dst_np = np.array(dst, dtype=np.float32).reshape(4, 2)
        h, _ = cv2.findHomography(src_np, dst_np, method=0)
        if h is None:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
        return out.astype(np.float32)
    except Exception:
        return points


def _trajectory_features(points: np.ndarray) -> dict:
    if len(points) < 4:
        return {
            "lofted": False,
            "height_score": 0.0,
            "trajectory_type": "unknown",
            "peak_height_px": 0.0,
            "curvature_a": 0.0,
            "span_n": 0.0,
            "prominence": 0.0,
            "residual_norm": 1.0,
        }
    mu = np.mean(points, axis=0)
    centered = points - mu
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        t_hat = vt[0]
    except np.linalg.LinAlgError:
        t_hat = np.array([1.0, 0.0], dtype=np.float32)
    n_hat = np.array([-t_hat[1], t_hat[0]], dtype=np.float32)
    s = centered @ t_hat
    n = centered @ n_hat
    # Curvature-sign normalization: enforce early "up" component positive in n-axis.
    if len(n) >= 2 and float(n[1] - n[0]) < 0.0:
        n_hat = -n_hat
        n = centered @ n_hat
    if float(np.std(s)) < 1e-6:
        return {
            "lofted": False,
            "height_score": 0.0,
            "trajectory_type": "unknown",
            "peak_height_px": 0.0,
            "curvature_a": 0.0,
            "span_n": 0.0,
            "prominence": 0.0,
            "residual_norm": 1.0,
        }
    a, b, c = np.polyfit(s, n, 2)
    n_hat_fit = a * s * s + b * s + c
    residual = float(np.sqrt(np.mean((n - n_hat_fit) ** 2)))
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    s_peak = -float(b) / (2.0 * float(a)) if abs(float(a)) > 1e-9 else s[np.argmax(n_hat_fit)]
    s_peak = float(np.clip(s_peak, s_min, s_max))
    n_peak = float(a * s_peak * s_peak + b * s_peak + c)
    left_n = float(a * s_min * s_min + b * s_min + c)
    right_n = float(a * s_max * s_max + b * s_max + c)
    left_drop = n_peak - left_n
    right_drop = n_peak - right_n
    prominence = float(min(left_drop, right_drop))
    span_n = float(np.max(n_hat_fit) - np.min(n_hat_fit))
    span_min = 6.0
    prom_min = 4.0
    lofted = bool((float(a) < 0.0) and (span_n >= span_min) and (prominence >= prom_min))

    # ROI-based normalization from trajectory inlier box.
    roi_w = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    roi_h = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    roi_scale = max(12.0, float(np.sqrt(max(1.0, roi_w * roi_h))))
    height_score = float(np.clip(n_peak / max(1.0, 0.35 * roi_scale), 0.0, 1.0))
    residual_norm = float(np.clip(residual / max(6.0, 0.15 * max(1.0, s_max - s_min)), 0.0, 1.0))

    if lofted:
        traj_type = "parabolic"
    else:
        traj_type = "linear"
    return {
        "lofted": lofted,
        "height_score": round(height_score, 3),
        "trajectory_type": traj_type,
        "peak_height_px": round(float(max(0.0, n_peak)), 2),
        "curvature_a": round(float(a), 6),
        "span_n": round(span_n, 3),
        "prominence": round(prominence, 3),
        "residual_norm": round(residual_norm, 3),
        "span_min": span_min,
        "roi_scale": round(roi_scale, 3),
    }


def classify_shot_height(height_score: float, lofted: bool, num_points: int) -> str:
    if lofted and height_score > 0.65 and num_points >= 6:
        return "Six-level"
    if lofted or height_score >= 0.35:
        return "Lofted"
    return "Ground"


def analyze_shot(raw_shot: dict, runs: int = 0,
                 batsman_facing: str = "right") -> dict:
    direction_angle = raw_shot.get("ball_angle_deg", raw_shot["raw_angle_deg"])
    angle   = normalize_angle(direction_angle, batsman_facing)
    zone    = get_field_zone(angle)
    length  = estimate_shot_length(raw_shot["bat_vector"])
    pts_all = _post_impact_points(raw_shot)
    pts_all = _rectify_points(pts_all, raw_shot)
    # Adaptive post-impact analysis windows: [+2:+10], [+3:+11], [+4:+12]
    window_specs = [("w2_10", 0, 8), ("w3_11", 1, 9), ("w4_12", 2, 10)]
    best_name = "full"
    best_pts = pts_all
    best_traj = _trajectory_features(pts_all)
    best_res = float(best_traj.get("residual_norm", 1.0))
    for name, st, en in window_specs:
        if len(pts_all) < st + 4:
            continue
        sub = pts_all[st:en]
        traj_sub = _trajectory_features(sub)
        res = float(traj_sub.get("residual_norm", 1.0))
        if len(sub) >= 4 and res <= best_res:
            best_name = name
            best_pts = sub
            best_traj = traj_sub
            best_res = res
    pts = best_pts
    traj = best_traj
    inlier_ratio = float(raw_shot.get("inlier_ratio", 0.0))
    residual = float(raw_shot.get("residual_error", 1.0))
    span_n = float(traj.get("span_n", 0.0))
    span_min = float(traj.get("span_min", 6.0))
    res_th = 0.65 + (0.15 if len(pts) < 6 else 0.0) + (0.1 if span_n < span_min else 0.0)
    stability_fail = (inlier_ratio < 0.6) or (residual > res_th)

    # Six-level requires sufficient post-impact support points.
    support_points = int(len(pts))
    lofted = bool(traj["lofted"])
    height_class = classify_shot_height(float(traj["height_score"]), lofted, support_points)
    s_type = classify_shot_type_advanced(angle, lofted, height_class)
    trajectory_type = str(traj["trajectory_type"])
    if inlier_ratio < 0.5 and span_n < span_min:
        s_type = "Edge/Glance"
        trajectory_type = "edge"
    if stability_fail:
        trajectory_type = "unknown"
        s_type = "Unknown"
    color   = SHOT_COLORS.get(runs, DEFAULT_SHOT_COLOR)

    enriched = {
        **raw_shot,
        "angle_deg":   angle,
        "field_zone":  zone,
        "shot_length": length,
        "shot_type":   s_type,
        "lofted":      lofted,
        "height_score": float(traj["height_score"]),
        "shot_height": height_class,
        "trajectory_type": trajectory_type,
        "peak_height_px": float(traj["peak_height_px"]),
        "trajectory_curvature_a": float(traj["curvature_a"]),
        "trajectory_span_n": float(traj["span_n"]),
        "trajectory_prominence": float(traj["prominence"]),
        "trajectory_residual_norm": float(traj["residual_norm"]),
        "chosen_window": best_name,
        "roi_scale": float(traj.get("roi_scale", 0.0)),
        "dynamic_residual_threshold": round(float(res_th), 3),
        "runs":        runs,
        "color":       color,
    }

    enriched.pop("keypoints", None)
    enriched.pop("frame",     None)

    log.info(
        f"Shot {raw_shot['shot_id']} | "
        f"Zone: {zone} | Angle: {angle}° | "
        f"Type: {s_type} | Length: {length} | "
        f"Lofted: {enriched['lofted']} | Height: {enriched['height_score']} | "
        f"a={enriched['trajectory_curvature_a']} span={enriched['trajectory_span_n']} "
        f"prom={enriched['trajectory_prominence']} resid={residual} inlier={inlier_ratio} "
        f"window={best_name} roi_scale={enriched['roi_scale']} res_th={enriched['dynamic_residual_threshold']}"
    )
    return enriched


def smooth_shot_classes_temporal(shots: list[dict], max_dt_sec: float = 0.7, max_bin_gap: int = 1) -> list[dict]:
    """
    Temporal class smoothing: if neighboring shots are close in time and angle bins,
    keep previous class to reduce flicker.
    """
    if not shots:
        return shots
    ordered = sorted(shots, key=lambda s: float(s.get("timestamp_sec", 0.0)))
    bin_size = 20.0
    prev = ordered[0]
    for cur in ordered[1:]:
        dt = float(cur.get("timestamp_sec", 0.0)) - float(prev.get("timestamp_sec", 0.0))
        a0 = float(prev.get("angle_deg", 0.0))
        a1 = float(cur.get("angle_deg", 0.0))
        b0 = int(np.floor((a0 + 180.0) / bin_size))
        b1 = int(np.floor((a1 + 180.0) / bin_size))
        if dt <= max_dt_sec and abs(b1 - b0) <= max_bin_gap and cur.get("shot_type") != "Unknown":
            cur["shot_type"] = prev.get("shot_type", cur.get("shot_type"))
        prev = cur
    return shots


def summarize_innings(shots: list[dict]) -> dict:
    if not shots:
        return {}

    zones     = [s["field_zone"]  for s in shots]
    types     = [s["shot_type"]   for s in shots]
    angles    = [s["angle_deg"]   for s in shots]
    runs_list = [s["runs"]        for s in shots]

    zone_counts = {}
    for z in zones:
        zone_counts[z] = zone_counts.get(z, 0) + 1

    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_shots":         len(shots),
        "total_runs":          sum(runs_list),
        "avg_angle_deg":       round(float(np.mean(angles)), 2),
        "leg_side_pct":        round(sum(1 for a in angles if a > 0) / len(angles) * 100, 1),
        "off_side_pct":        round(sum(1 for a in angles if a < 0) / len(angles) * 100, 1),
        "zone_breakdown":      zone_counts,
        "shot_type_breakdown": type_counts,
    }
