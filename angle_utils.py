import logging
import numpy as np

log = logging.getLogger(__name__)


def normalize_cricket_angle(dx: float, dy: float, batsman_facing: str = "right") -> float:
    """
    Central cricket-angle normalization from image-space vector.

    Steps:
    - invert image Y exactly once
    - angle = atan2(dx, dy_inverted) so 0° is straight down ground (image-up)
    - normalize to [-180, 180]
    - handedness flip for left-handed batters
    """
    dy_inverted = -float(dy)
    angle_raw = float(np.degrees(np.arctan2(float(dx), dy_inverted)))
    angle_norm = ((angle_raw + 180.0) % 360.0) - 180.0

    facing = (batsman_facing or "right").strip().lower()
    angle_facing = -angle_norm if facing == "left" else angle_norm
    angle_facing = ((angle_facing + 180.0) % 360.0) - 180.0

    log.debug(
        "normalize_cricket_angle | dx=%.3f dy=%.3f raw=%.2f facing=%s adjusted=%.2f",
        float(dx),
        float(dy),
        angle_norm,
        facing,
        angle_facing,
    )
    return round(angle_facing, 2)
