import numpy as np
import logging
from config import FIELD_ZONES, MIN_SHOT_ANGLE_CHANGE, SHOT_COLORS, DEFAULT_SHOT_COLOR

log = logging.getLogger(__name__)


def normalize_angle(raw_angle_deg: float, batsman_facing: str = "right") -> float:
    facing = (batsman_facing or "right").strip().lower()
    angle = raw_angle_deg if facing == "right" else -raw_angle_deg
    angle = (angle + 180.0) % 360.0 - 180.0
    return round(angle, 2)


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


def analyze_shot(raw_shot: dict, runs: int = 0,
                 batsman_facing: str = "right") -> dict:
    angle   = normalize_angle(raw_shot["raw_angle_deg"], batsman_facing)
    zone    = get_field_zone(angle)
    length  = estimate_shot_length(raw_shot["bat_vector"])
    s_type  = classify_shot_type(angle, length)
    color   = SHOT_COLORS.get(runs, DEFAULT_SHOT_COLOR)

    enriched = {
        **raw_shot,
        "angle_deg":   angle,
        "field_zone":  zone,
        "shot_length": length,
        "shot_type":   s_type,
        "runs":        runs,
        "color":       color,
    }

    enriched.pop("keypoints", None)
    enriched.pop("frame",     None)

    log.info(
        f"Shot {raw_shot['shot_id']} | "
        f"Zone: {zone} | Angle: {angle}° | "
        f"Type: {s_type} | Length: {length}"
    )
    return enriched


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
