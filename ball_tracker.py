import logging
from pathlib import Path

import cv2
import numpy as np

from detector import PoseDetector

log = logging.getLogger(__name__)


class BallTracker:
    """
    Ball-first direction estimator around detected shot impact frame.
    - Preferred detector: YOLO (if ultralytics is installed + model path provided)
    - Fallback detector: HSV + contour (white-ball heuristic)
    """

    def __init__(self, debug_dir: str | Path = "output/ball_debug", yolo_model_path: str | None = None):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.pose = PoseDetector()
        self.yolo = None
        if yolo_model_path:
            try:
                from ultralytics import YOLO  # optional dependency
                self.yolo = YOLO(yolo_model_path)
                log.info("BallTracker YOLO loaded: %s", yolo_model_path)
            except Exception as e:  # pragma: no cover
                log.warning("BallTracker YOLO unavailable, using HSV fallback: %s", e)

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        if angle_deg > 180.0:
            angle_deg -= 360.0
        if angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    @staticmethod
    def _point_angle(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        dx = float(p2[0] - p1[0])
        dy = -float(p2[1] - p1[1])  # invert image Y-axis
        return BallTracker._normalize_angle(float(np.degrees(np.arctan2(dy, dx))))

    def _detect_ball_yolo(self, frame: np.ndarray, frame_index: int) -> dict | None:
        if self.yolo is None:
            return None
        try:
            result = self.yolo.predict(frame, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                return None
            best = None
            best_conf = 0.0
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)
                    best_conf = conf
                    best = (cx, cy)
            if best is None:
                return None
            return {
                "frame_index": frame_index,
                "ball_center": best,
                "confidence": best_conf,
            }
        except Exception:
            return None

    @staticmethod
    def _detect_ball_hsv(frame: np.ndarray, frame_index: int) -> dict | None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 180], dtype=np.uint8)
        upper = np.array([180, 70, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_conf = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 8 or area > 250:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = float(4.0 * np.pi * area / (peri * peri))
            if circularity < 0.45:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            conf = float(min(1.0, 0.5 * circularity + 0.5 * min(1.0, area / 80.0)))
            if conf > best_conf:
                best_conf = conf
                best = (cx, cy)
        if best is None:
            return None
        return {
            "frame_index": frame_index,
            "ball_center": best,
            "confidence": best_conf,
        }

    def _detect_ball(self, frame: np.ndarray, frame_index: int) -> dict | None:
        det = self._detect_ball_yolo(frame, frame_index)
        if det is not None:
            return det
        return self._detect_ball_hsv(frame, frame_index)

    @staticmethod
    def _filter_outliers(detections: list[dict], max_step_px: float = 120.0) -> list[dict]:
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d["frame_index"])
        kept = [detections[0]]
        for det in detections[1:]:
            prev = kept[-1]["ball_center"]
            cur = det["ball_center"]
            if float(np.hypot(cur[0] - prev[0], cur[1] - prev[1])) <= max_step_px:
                kept.append(det)
        return kept

    def _fallback_bat_angle(self, frame: np.ndarray) -> float | None:
        keypoints = self.pose.detect(frame)
        if not keypoints:
            return None

        def arm_vis(side: str) -> float:
            w = keypoints[f"{side}_wrist"]["visibility"]
            e = keypoints[f"{side}_elbow"]["visibility"]
            s = keypoints[f"{side}_shoulder"]["visibility"]
            return float((w + e + s) / 3.0)

        side = "left" if arm_vis("left") >= arm_vis("right") else "right"
        if arm_vis(side) < 0.3:
            return None

        wrist = keypoints[f"{side}_wrist"]
        elbow = keypoints[f"{side}_elbow"]
        p1 = (float(elbow["x"]), float(elbow["y"]))
        p2 = (float(wrist["x"]), float(wrist["y"]))
        return self._point_angle(p1, p2)

    def _save_debug_overlay(
        self,
        frame: np.ndarray,
        trajectory: list[tuple[float, float]],
        shot_id: int,
        frame_index: int,
        video_stem: str,
    ) -> str:
        canvas = frame.copy()
        if len(trajectory) >= 2:
            pts = np.array([[int(x), int(y)] for x, y in trajectory], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
            cv2.circle(canvas, tuple(pts[0]), 5, (0, 255, 0), -1)   # start
            cv2.circle(canvas, tuple(pts[-1]), 5, (0, 0, 255), -1)  # end
        out_path = self.debug_dir / f"{video_stem}_shot_{shot_id}_f{frame_index}.png"
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)

    def estimate_shot_direction(
        self,
        frames: list[dict],
        impact_frame_idx: int,
        shot_id: int,
        video_stem: str,
    ) -> dict:
        window_start = impact_frame_idx - 2
        window_end = impact_frame_idx + 8
        window_frames = [
            f for f in frames
            if window_start <= int(f["frame_idx"]) <= window_end
        ]

        detections = []
        for fd in window_frames:
            det = self._detect_ball(fd["frame"], int(fd["frame_idx"]))
            if det and float(det["confidence"]) >= 0.2:
                detections.append(det)

        detections = self._filter_outliers(detections)
        post = [d for d in detections if int(d["frame_index"]) >= impact_frame_idx]
        points = [d["ball_center"] for d in post]
        if len(points) < 2 and len(detections) >= 2:
            points = [detections[0]["ball_center"], detections[-1]["ball_center"]]

        impact_frame = None
        for fd in window_frames:
            if int(fd["frame_idx"]) == impact_frame_idx:
                impact_frame = fd["frame"]
                break
        if impact_frame is None and window_frames:
            impact_frame = window_frames[0]["frame"]

        if len(detections) >= 3 and len(points) >= 2:
            angle = self._point_angle(points[0], points[-1])
            debug_path = self._save_debug_overlay(impact_frame, points, shot_id, impact_frame_idx, video_stem) if impact_frame is not None else ""
            return {
                "angle": round(angle, 2),
                "confidence": "high",
                "source": "ball",
                "ball_detections": len(detections),
                "debug_image": debug_path,
            }

        bat_angle = self._fallback_bat_angle(impact_frame) if impact_frame is not None else None
        if bat_angle is not None:
            return {
                "angle": round(bat_angle, 2),
                "confidence": "low",
                "source": "bat",
                "ball_detections": len(detections),
                "debug_image": "",
            }

        return {
            "angle": 0.0,
            "confidence": "low",
            "source": "bat",
            "ball_detections": len(detections),
            "debug_image": "",
        }

    def close(self):
        self.pose.close()
