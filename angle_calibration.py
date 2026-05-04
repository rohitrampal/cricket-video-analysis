import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import ANGLE_CALIBRATION_FILE, FIELD_ZONES
from shot_analyzer import get_field_zone


def wrap_angle(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def circular_diff(a: float, b: float) -> float:
    return abs(wrap_angle(float(a) - float(b)))


def circular_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    import math

    s = 0.0
    c = 0.0
    for v in values:
        r = math.radians(float(v))
        s += math.sin(r)
        c += math.cos(r)
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return 0.0
    return wrap_angle(math.degrees(math.atan2(s, c)))


def zone_to_center_angle(zone: str) -> float:
    if zone in FIELD_ZONES:
        lo, hi = FIELD_ZONES[zone]
        return wrap_angle((float(lo) + float(hi)) / 2.0)
    if zone == "Fine Leg":
        return 140.0
    return 0.0


@dataclass
class CalibrationParams:
    mirror: bool = False
    offset_deg: float = 0.0

    def apply(self, angle: float) -> float:
        base = -float(angle) if self.mirror else float(angle)
        return wrap_angle(base + float(self.offset_deg))

    def to_dict(self) -> dict[str, Any]:
        return {"mirror": bool(self.mirror), "offset_deg": round(float(self.offset_deg), 3)}

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "CalibrationParams":
        if not isinstance(data, dict):
            return CalibrationParams()
        return CalibrationParams(
            mirror=bool(data.get("mirror", False)),
            offset_deg=float(data.get("offset_deg", 0.0)),
        )


class AngleCalibrator:
    def __init__(
        self,
        enabled: bool = False,
        global_params: CalibrationParams | None = None,
        by_source: dict[str, CalibrationParams] | None = None,
    ):
        self.enabled = bool(enabled)
        self.global_params = global_params or CalibrationParams()
        self.by_source = by_source or {}

    def params_for_source(self, source: str) -> CalibrationParams:
        src = str(source or "").strip().lower()
        return self.by_source.get(src, self.global_params)

    def apply(self, angle: float, source: str = "") -> float:
        if not self.enabled:
            return wrap_angle(angle)
        params = self.params_for_source(source)
        return params.apply(angle)

    def apply_to_shot(self, shot: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return shot
        old_angle = float(shot.get("angle_deg", shot.get("raw_angle_deg", 0.0)))
        src = str(shot.get("direction_source", "")).lower()
        new_angle = self.apply(old_angle, source=src)
        shot["angle_uncalibrated_deg"] = round(old_angle, 2)
        shot["angle_deg"] = round(new_angle, 2)
        shot["field_zone"] = get_field_zone(float(shot["angle_deg"]))
        shot["calibration_applied"] = True
        return shot

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "global": self.global_params.to_dict(),
            "by_source": {k: v.to_dict() for k, v in self.by_source.items()},
        }

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "AngleCalibrator":
        if not isinstance(data, dict):
            return AngleCalibrator(enabled=False)
        by_source_raw = data.get("by_source", {}) if isinstance(data.get("by_source"), dict) else {}
        by_source = {str(k).lower(): CalibrationParams.from_dict(v) for k, v in by_source_raw.items()}
        return AngleCalibrator(
            enabled=bool(data.get("enabled", False)),
            global_params=CalibrationParams.from_dict(data.get("global", {})),
            by_source=by_source,
        )

    @staticmethod
    def load(path: str | Path = ANGLE_CALIBRATION_FILE) -> "AngleCalibrator":
        p = Path(path)
        if not p.exists():
            return AngleCalibrator(enabled=False)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AngleCalibrator.from_dict(data)
        except Exception:
            return AngleCalibrator(enabled=False)

    def save(self, path: str | Path = ANGLE_CALIBRATION_FILE, metadata: dict[str, Any] | None = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if metadata:
            data["metadata"] = metadata
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return p


def fit_params(samples: list[dict[str, Any]]) -> CalibrationParams:
    """
    Fit mirror + circular offset from samples.
    Each sample must contain:
      - pred_angle (deg)
      - target_angle (deg)
    """
    if not samples:
        return CalibrationParams()
    best = CalibrationParams(mirror=False, offset_deg=0.0)
    best_err = float("inf")
    for mirror in (False, True):
        deltas = []
        for s in samples:
            pred = float(s["pred_angle"])
            target = float(s["target_angle"])
            base = -pred if mirror else pred
            deltas.append(wrap_angle(target - base))
        offset = circular_mean(deltas)
        errs = []
        for s in samples:
            pred = float(s["pred_angle"])
            target = float(s["target_angle"])
            calibrated = wrap_angle((-pred if mirror else pred) + offset)
            errs.append(circular_diff(calibrated, target))
        mae = float(sum(errs) / max(1, len(errs)))
        if mae < best_err:
            best_err = mae
            best = CalibrationParams(mirror=mirror, offset_deg=offset)
    return best

