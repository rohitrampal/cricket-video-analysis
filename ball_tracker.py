import logging
from pathlib import Path

import cv2
import numpy as np

from detector import PoseDetector
from angle_utils import normalize_cricket_angle

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
        self.min_post_impact_points = 4
        self.max_track_jump_px = 600.0
        self.max_missing_frames = 3
        self.expected_post_impact_points = 6
        self.mahalanobis_gate_chi2 = 25.0
        self.noise_continuity_px = 150.0
        self.prev_output_angle: float | None = None

    @staticmethod
    def _wrap_angle_diff(curr: float, prev: float) -> float:
        return abs(((float(curr) - float(prev) + 180.0) % 360.0) - 180.0)

    def _apply_angle_smoothing(self, current_angle: float, max_diff_for_smooth: float = 40.0) -> float:
        if self.prev_output_angle is None:
            self.prev_output_angle = float(current_angle)
            return float(current_angle)
        diff = self._wrap_angle_diff(current_angle, self.prev_output_angle)
        if diff < max_diff_for_smooth:
            out = 0.7 * float(current_angle) + 0.3 * float(self.prev_output_angle)
        else:
            out = float(current_angle)
        self.prev_output_angle = float(out)
        return out

    @staticmethod
    def _point_angle(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])  # image-space dy; inversion handled centrally
        return normalize_cricket_angle(dx=dx, dy=dy, batsman_facing="right")

    def _init_kalman(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        # State: [x, y, vx, vy, ax, ay]
        state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cov = np.eye(6, dtype=np.float32) * 120.0
        return state, cov

    @staticmethod
    def _kalman_predict(state: np.ndarray, cov: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt * dt, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt * dt],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        q_pos = 9.0
        q_vel = 18.0
        q_acc = 14.0
        q = np.array(
            [
                [q_pos, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, q_pos, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, q_vel, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, q_vel, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, q_acc, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, q_acc],
            ],
            dtype=np.float32,
        )
        s = f @ state
        p = f @ cov @ f.T + q
        return s, p

    @staticmethod
    def _kalman_update(
        pred_state: np.ndarray,
        pred_cov: np.ndarray,
        meas_x: float,
        meas_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        r = np.array([[36.0, 0.0], [0.0, 36.0]], dtype=np.float32)
        z = np.array([meas_x, meas_y], dtype=np.float32)
        y = z - (h @ pred_state)
        s = h @ pred_cov @ h.T + r
        k = pred_cov @ h.T @ np.linalg.inv(s)
        new_state = pred_state + (k @ y)
        i = np.eye(6, dtype=np.float32)
        new_cov = (i - k @ h) @ pred_cov
        return new_state, new_cov

    @staticmethod
    def _innovation_mahalanobis(
        pred_state: np.ndarray,
        pred_cov: np.ndarray,
        meas_x: float,
        meas_y: float,
    ) -> float:
        h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        r = np.array([[36.0, 0.0], [0.0, 36.0]], dtype=np.float32)
        z = np.array([meas_x, meas_y], dtype=np.float32)
        y = z - (h @ pred_state)
        s = h @ pred_cov @ h.T + r
        try:
            d2 = float(y.T @ np.linalg.inv(s) @ y)
        except np.linalg.LinAlgError:
            d2 = 1e9
        return d2

    @staticmethod
    def _estimate_global_motion(window_frames: list[dict]) -> dict[int, tuple[float, float]]:
        """
        Estimate camera motion via sparse optical flow on background points.
        Returns cumulative dx,dy per frame relative to first frame in window.
        """
        if len(window_frames) < 2:
            return {int(window_frames[0]["frame_idx"]): (0.0, 0.0)} if window_frames else {}
        frames = sorted(window_frames, key=lambda x: int(x["frame_idx"]))
        motion: dict[int, tuple[float, float]] = {int(frames[0]["frame_idx"]): (0.0, 0.0)}
        prev_gray = cv2.cvtColor(frames[0]["frame"], cv2.COLOR_BGR2GRAY)
        h, w = prev_gray.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        cx1, cy1 = int(w * 0.30), int(h * 0.22)
        cx2, cy2 = int(w * 0.72), int(h * 0.92)
        cv2.rectangle(mask, (cx1, cy1), (cx2, cy2), 0, -1)  # suppress batsman area
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=120, qualityLevel=0.02, minDistance=7, blockSize=7, mask=mask
        )
        cum_dx = 0.0
        cum_dy = 0.0
        for fd in frames[1:]:
            idx = int(fd["frame_idx"])
            gray = cv2.cvtColor(fd["frame"], cv2.COLOR_BGR2GRAY)
            if prev_pts is None or len(prev_pts) < 8:
                prev_pts = cv2.goodFeaturesToTrack(
                    prev_gray, maxCorners=120, qualityLevel=0.02, minDistance=7, blockSize=7, mask=mask
                )
            if prev_pts is None or len(prev_pts) < 4:
                motion[idx] = (cum_dx, cum_dy)
                prev_gray = gray
                continue
            next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=2
            )
            if next_pts is None or st is None:
                motion[idx] = (cum_dx, cum_dy)
                prev_gray = gray
                prev_pts = None
                continue
            good_prev = prev_pts[st.flatten() == 1]
            good_next = next_pts[st.flatten() == 1]
            if len(good_prev) < 4:
                motion[idx] = (cum_dx, cum_dy)
                prev_gray = gray
                prev_pts = None
                continue
            flow = np.asarray(good_next - good_prev, dtype=np.float32).reshape(-1, 2)
            if flow.shape[0] == 0:
                motion[idx] = (cum_dx, cum_dy)
                prev_gray = gray
                prev_pts = None
                continue
            dx = float(np.median(flow[:, 0]))
            dy = float(np.median(flow[:, 1]))
            cum_dx += dx
            cum_dy += dy
            motion[idx] = (cum_dx, cum_dy)
            prev_gray = gray
            prev_pts = good_next.reshape(-1, 1, 2)
        return motion

    def _detect_ball_yolo(self, frame: np.ndarray, frame_index: int) -> list[dict]:
        if self.yolo is None:
            return []
        try:
            result = self.yolo.predict(frame, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []
            candidates: list[dict] = []
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                candidates.append(
                    {
                        "frame_index": frame_index,
                        "ball_center": (cx, cy),
                        "confidence": conf,
                    }
                )
            return sorted(candidates, key=lambda d: float(d["confidence"]), reverse=True)
        except Exception:
            return []

    @staticmethod
    def _detect_ball_hsv(frame: np.ndarray, frame_index: int) -> list[dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 180], dtype=np.uint8)
        upper = np.array([180, 70, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
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
            candidates.append(
                {
                    "frame_index": frame_index,
                    "ball_center": (cx, cy),
                    "confidence": conf,
                }
            )
        return sorted(candidates, key=lambda d: float(d["confidence"]), reverse=True)

    def _detect_ball(self, frame: np.ndarray, frame_index: int) -> list[dict]:
        det = self._detect_ball_yolo(frame, frame_index)
        if det:
            return det
        return self._detect_ball_hsv(frame, frame_index)

    @staticmethod
    def _smooth_points(points: list[tuple[float, float]], window: int = 3) -> list[tuple[float, float]]:
        if len(points) < 3:
            return points
        half = max(1, window // 2)
        out = []
        for i in range(len(points)):
            lo = max(0, i - half)
            hi = min(len(points), i + half + 1)
            block = np.array(points[lo:hi], dtype=np.float32)
            out.append((float(np.mean(block[:, 0])), float(np.mean(block[:, 1]))))
        return out

    @staticmethod
    def _line_from_regression(points: list[tuple[float, float]], frame_idx: list[int]) -> tuple[float, float] | None:
        if len(points) < 2:
            return None
        t = np.array(frame_idx, dtype=np.float32)
        x = np.array([p[0] for p in points], dtype=np.float32)
        y = np.array([p[1] for p in points], dtype=np.float32)
        if float(np.std(t)) < 1e-6:
            return None
        vx, _ = np.polyfit(t, x, 1)
        vy, _ = np.polyfit(t, y, 1)
        return float(vx), float(vy)

    def _trajectory_checks(self, points: list[tuple[float, float]]) -> tuple[bool, float, float, float]:
        """
        Returns: is_valid, dir_variance_norm, vel_variance_norm, monotonic_ratio
        """
        if len(points) < 4:
            return False, 1.0, 1.0, 0.0
        arr = np.array(points, dtype=np.float32)
        d = np.diff(arr, axis=0)
        if len(d) < 2:
            return False, 1.0, 1.0, 0.0
        vel = np.linalg.norm(d, axis=1)
        vel_mean = float(np.mean(vel))
        vel_std = float(np.std(vel))
        vel_variance_norm = float(np.clip(vel_std / max(1e-6, vel_mean), 0.0, 1.0))

        seg_angles = np.array([np.degrees(np.arctan2(di[1], di[0])) for di in d], dtype=np.float32)
        dir_var_deg = float(np.std(seg_angles))
        dir_variance_norm = float(np.clip(dir_var_deg / 45.0, 0.0, 1.0))

        dominant_axis = 0 if float(np.std(arr[:, 0])) >= float(np.std(arr[:, 1])) else 1
        steps = d[:, dominant_axis]
        valid_steps = steps[np.abs(steps) > 1.0]
        if len(valid_steps) == 0:
            monotonic_ratio = 0.0
        else:
            main_sign = 1.0 if float(np.sum(valid_steps >= 0)) >= len(valid_steps) / 2 else -1.0
            monotonic_ratio = float(np.mean(np.sign(valid_steps) == main_sign))

        is_valid = (dir_variance_norm <= 0.72) and (vel_variance_norm <= 0.80) and (monotonic_ratio >= 0.60)
        return is_valid, dir_variance_norm, vel_variance_norm, monotonic_ratio

    @staticmethod
    def _circular_mean(angles_deg: list[float]) -> float | None:
        if not angles_deg:
            return None
        rad = np.deg2rad(np.array(angles_deg, dtype=np.float32))
        s = float(np.mean(np.sin(rad)))
        c = float(np.mean(np.cos(rad)))
        return float(np.degrees(np.arctan2(s, c)))

    def _fit_direction_model(
        self,
        comp_points: list[tuple[float, float]],
        screen_points: list[tuple[float, float]],
        frame_idx: list[int],
        impact_idx: int,
    ) -> dict:
        if len(comp_points) < 4:
            return {"ok": False}
        comp = np.array(comp_points, dtype=np.float32)
        scr = np.array(screen_points, dtype=np.float32)
        mean = np.mean(comp, axis=0)
        centered = comp - mean
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return {"ok": False}
        v = vt[0]
        net = comp[-1] - comp[0]
        if float(np.dot(v, net)) < 0:
            v = -v
        line_angle = self._point_angle((0.0, 0.0), (float(v[0]), float(v[1])))
        ortho = np.array([-v[1], v[0]], dtype=np.float32)
        line_resid = float(np.sqrt(np.mean((centered @ ortho) ** 2)))
        span = float(np.linalg.norm(net))
        line_resid_norm = float(np.clip(line_resid / max(8.0, 0.2 * span + 8.0), 0.0, 1.0))

        axis_x = float(np.std(comp[:, 0])) >= float(np.std(comp[:, 1]))
        quad_resid_norm = 1.0
        curve_metric = 0.0
        parabola_angle = line_angle
        if axis_x:
            x = comp[:, 0]
            y = comp[:, 1]
            if float(np.std(x)) > 1e-3:
                a, b, c = np.polyfit(x, y, 2)
                y_hat = a * x * x + b * x + c
                quad_resid = float(np.sqrt(np.mean((y - y_hat) ** 2)))
                quad_resid_norm = float(np.clip(quad_resid / max(8.0, 0.2 * span + 8.0), 0.0, 1.0))
                curve_metric = float(abs(a) * max(1.0, float(np.max(x) - np.min(x))))
                x0 = float(comp[max(0, min(len(comp) - 1, impact_idx)), 0])
                slope = 2.0 * float(a) * x0 + float(b)
                parabola_angle = self._point_angle((0.0, 0.0), (1.0, slope))
        else:
            y = comp[:, 1]
            x = comp[:, 0]
            if float(np.std(y)) > 1e-3:
                a, b, c = np.polyfit(y, x, 2)
                x_hat = a * y * y + b * y + c
                quad_resid = float(np.sqrt(np.mean((x - x_hat) ** 2)))
                quad_resid_norm = float(np.clip(quad_resid / max(8.0, 0.2 * span + 8.0), 0.0, 1.0))
                curve_metric = float(abs(a) * max(1.0, float(np.max(y) - np.min(y))))
                y0 = float(comp[max(0, min(len(comp) - 1, impact_idx)), 1])
                slope = 2.0 * float(a) * y0 + float(b)  # dx/dy
                parabola_angle = self._point_angle((0.0, 0.0), (slope, 1.0))

        use_parabola = (curve_metric > 0.020) and (quad_resid_norm < line_resid_norm * 0.85)
        final_angle = parabola_angle if use_parabola else line_angle
        residual_norm = quad_resid_norm if use_parabola else line_resid_norm

        seg = np.diff(comp, axis=0)
        seg_angles = [self._point_angle((0.0, 0.0), (float(d[0]), float(d[1]))) for d in seg if float(np.linalg.norm(d)) > 1e-6]
        angle_stability = 0.0
        if len(seg_angles) >= 2:
            angle_stability = float(np.clip(1.0 - (np.std(np.array(seg_angles, dtype=np.float32)) / 35.0), 0.0, 1.0))

        p1 = (float(scr[0, 0]), float(scr[0, 1]))
        p2 = (float(scr[-1, 0]), float(scr[-1, 1]))
        return {
            "ok": True,
            "angle": float(final_angle),
            "model": "parabola" if use_parabola else "pca_line",
            "residual_norm": float(residual_norm),
            "angle_stability": float(angle_stability),
            "regression_line": (p1, p2),
        }

    def _refine_impact_frame(
        self,
        tracked: list[dict],
        window_frames: list[dict],
        base_impact_idx: int,
    ) -> int:
        lo = base_impact_idx - 5
        hi = base_impact_idx + 5
        cand = [d for d in tracked if lo <= int(d["frame_index"]) <= hi]
        if not cand:
            return base_impact_idx
        cand = sorted(cand, key=lambda d: int(d["frame_index"]))
        best_speed_idx = int(cand[0]["frame_index"])
        if len(cand) >= 3:
            speeds = []
            for i in range(1, len(cand)):
                p0 = np.array(cand[i - 1]["ball_center"], dtype=np.float32)
                p1 = np.array(cand[i]["ball_center"], dtype=np.float32)
                dt = max(1, int(cand[i]["frame_index"]) - int(cand[i - 1]["frame_index"]))
                speeds.append(float(np.linalg.norm(p1 - p0) / dt))
            if len(speeds) >= 2:
                accel = np.diff(np.array(speeds, dtype=np.float32))
                j = int(np.argmax(accel)) + 1
                best_speed_idx = int(cand[j]["frame_index"])
        frame_map = {int(f["frame_idx"]): f["frame"] for f in window_frames}
        best_dist_idx = None
        best_dist = 1e9
        for d in cand:
            fi = int(d["frame_index"])
            frame = frame_map.get(fi)
            if frame is None:
                continue
            kp = self.pose.detect(frame)
            if not kp:
                continue
            lw = kp["left_wrist"]["visibility"]
            rw = kp["right_wrist"]["visibility"]
            side = "left" if lw >= rw else "right"
            w = kp[f"{side}_wrist"]
            wrist = np.array([float(w["x"]), float(w["y"])], dtype=np.float32)
            ball_screen = np.array(d.get("screen_center", d["ball_center"]), dtype=np.float32)
            dist = float(np.linalg.norm(ball_screen - wrist))
            if dist < best_dist:
                best_dist = dist
                best_dist_idx = fi
        return int(best_dist_idx if best_dist_idx is not None else best_speed_idx)

    def _track_ball_path(
        self,
        window_frames: list[dict],
        camera_motion: dict[int, tuple[float, float]],
        conf_min: float = 0.2,
    ) -> tuple[list[dict], list[dict], list[tuple[float, float]], list[tuple[float, float]], dict]:
        """
        Track ball using Kalman predict/update and prediction-based association.
        Returns accepted, rejected, raw_top, and full kalman path points.
        """
        accepted: list[dict] = []
        rejected: list[dict] = []
        raw_top_points: list[tuple[float, float]] = []
        kalman_path: list[tuple[float, float]] = []
        state: np.ndarray | None = None
        cov: np.ndarray | None = None
        missing_streak = 0
        gating_rejections = 0
        total_candidates = 0
        used_detections = 0

        for fd in sorted(window_frames, key=lambda x: int(x["frame_idx"])):
            frame_idx = int(fd["frame_idx"])
            mdx, mdy = camera_motion.get(frame_idx, (0.0, 0.0))
            candidates = [
                c for c in self._detect_ball(fd["frame"], frame_idx)
                if float(c["confidence"]) >= conf_min
            ]
            total_candidates += len(candidates)
            if candidates:
                raw_top_points.append(tuple(candidates[0]["ball_center"]))

            if state is None:
                if not candidates:
                    continue
                first = candidates[0]
                x0, y0 = first["ball_center"]
                x = float(x0 - mdx)
                y = float(y0 - mdy)
                state, cov = self._init_kalman(float(x), float(y))
                accepted.append(
                    {
                        "frame_index": frame_idx,
                        "ball_center": (float(x), float(y)),  # camera-motion compensated
                        "screen_center": (float(x + mdx), float(y + mdy)),
                        "confidence": float(first["confidence"]),
                        "kind": "detected",
                    }
                )
                used_detections += 1
                kalman_path.append((float(state[0] + mdx), float(state[1] + mdy)))
                continue

            pred_state, pred_cov = self._kalman_predict(state, cov, dt=1.0)
            chosen = None
            if candidates:
                scored = []
                for cand in candidates:
                    cx, cy = cand["ball_center"]
                    ccomp_x = float(cx - mdx)
                    ccomp_y = float(cy - mdy)
                    d2 = self._innovation_mahalanobis(pred_state, pred_cov, ccomp_x, ccomp_y)
                    scored.append((d2, ccomp_x, ccomp_y, cand))
                scored.sort(key=lambda x: x[0])
                best_d2, bx, by, best = scored[0]
                dist = float(np.linalg.norm(np.array([bx, by], dtype=np.float32) - np.array([pred_state[0], pred_state[1]], dtype=np.float32)))
                continuity_jump = dist > self.noise_continuity_px
                if best_d2 <= self.mahalanobis_gate_chi2 and dist <= self.max_track_jump_px and (not continuity_jump):
                    chosen = (best, bx, by)
                    for _, _, _, other in scored[1:]:
                        rejected.append({**other, "reason": "not_nearest"})
                else:
                    gating_rejections += len(scored)
                    reason = "tracking_jump" if continuity_jump else "jump"
                    for _, _, _, cand in scored:
                        rejected.append({**cand, "reason": reason})

            if chosen is not None:
                det, mx, my = chosen
                state, cov = self._kalman_update(pred_state, pred_cov, float(mx), float(my))
                accepted.append(
                    {
                        "frame_index": frame_idx,
                        "ball_center": (float(state[0]), float(state[1])),  # compensated
                        "screen_center": (float(state[0] + mdx), float(state[1] + mdy)),
                        "confidence": float(det["confidence"]),
                        "kind": "detected",
                    }
                )
                used_detections += 1
                missing_streak = 0
            else:
                state, cov = pred_state, pred_cov
                missing_streak += 1
                if missing_streak <= self.max_missing_frames:
                    accepted.append(
                        {
                            "frame_index": frame_idx,
                            "ball_center": (float(state[0]), float(state[1])),  # compensated
                            "screen_center": (float(state[0] + mdx), float(state[1] + mdy)),
                            "confidence": 0.0,
                            "kind": "predicted",
                        }
                    )
                else:
                    state = None
                    cov = None
                    missing_streak = 0
                    continue
            kalman_path.append((float(state[0] + mdx), float(state[1] + mdy)))
        stats = {
            "gating_rejections": int(gating_rejections),
            "total_candidates": int(total_candidates),
            "used_detections": int(used_detections),
        }
        return accepted, rejected, raw_top_points, kalman_path, stats

    def _fallback_bat_angle(
        self,
        window_frames: list[dict],
        impact_frame_idx: int,
    ) -> tuple[float | None, tuple[float, float] | None, tuple[float, float] | None]:
        lo = impact_frame_idx - 1
        hi = impact_frame_idx + 1
        sample_angles: list[float] = []
        impact_wrist = None
        impact_tip = None
        for fd in window_frames:
            fi = int(fd["frame_idx"])
            if fi < lo or fi > hi:
                continue
            keypoints = self.pose.detect(fd["frame"])
            if not keypoints:
                continue
            left_vis = float((keypoints["left_wrist"]["visibility"] + keypoints["left_shoulder"]["visibility"]) / 2.0)
            right_vis = float((keypoints["right_wrist"]["visibility"] + keypoints["right_shoulder"]["visibility"]) / 2.0)
            side = "left" if left_vis >= right_vis else "right"
            if max(left_vis, right_vis) < 0.3:
                continue
            wrist = keypoints[f"{side}_wrist"]
            shoulder = keypoints[f"{side}_shoulder"]
            wrist_pt = np.array([float(wrist["x"]), float(wrist["y"])], dtype=np.float32)
            shoulder_pt = np.array([float(shoulder["x"]), float(shoulder["y"])], dtype=np.float32)
            arm = wrist_pt - shoulder_pt
            norm = float(np.linalg.norm(arm))
            if norm < 1e-6:
                continue
            unit = arm / norm
            bat_tip = wrist_pt + unit * (0.9 * norm)
            ang = self._point_angle(tuple(wrist_pt), tuple(bat_tip))
            sample_angles.append(float(ang))
            if fi == impact_frame_idx:
                impact_wrist = tuple(wrist_pt)
                impact_tip = tuple(bat_tip)
        mean_angle = self._circular_mean(sample_angles)
        if mean_angle is None:
            return None, None, None
        return float(normalize_cricket_angle(np.sin(np.radians(mean_angle)), -np.cos(np.radians(mean_angle)), "right")), impact_wrist, impact_tip

    def _save_debug_overlay(
        self,
        frame: np.ndarray,
        trajectory: list[tuple[float, float]],
        kalman_path: list[tuple[float, float]],
        rejected_points: list[tuple[float, float]],
        regression_line: tuple[tuple[float, float], tuple[float, float]] | None,
        final_source: str,
        final_vector: tuple[tuple[float, float], tuple[float, float]] | None,
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
        if len(kalman_path) >= 2:
            kpts = np.array([[int(x), int(y)] for x, y in kalman_path], dtype=np.int32)
            cv2.polylines(canvas, [kpts], False, (180, 0, 255), 1)
        for px, py in rejected_points:
            cv2.circle(canvas, (int(px), int(py)), 4, (0, 0, 200), -1)
        if regression_line is not None:
            rp1, rp2 = regression_line
            cv2.line(canvas, (int(rp1[0]), int(rp1[1])), (int(rp2[0]), int(rp2[1])), (255, 255, 0), 2)
        if final_vector is not None:
            p1, p2 = final_vector
            cv2.arrowedLine(
                canvas,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (255, 120, 0),
                2,
                tipLength=0.16,
            )
        cv2.putText(
            canvas,
            f"source={final_source}",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
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
        rejected = []
        raw_points = []
        kalman_path = []
        track_stats = {"gating_rejections": 0, "total_candidates": 0, "used_detections": 0}
        camera_motion = self._estimate_global_motion(window_frames) if window_frames else {}
        if window_frames:
            detections, rejected, raw_points, kalman_path, track_stats = self._track_ball_path(
                window_frames,
                camera_motion=camera_motion,
                conf_min=0.2,
            )
        if rejected:
            jump_rej = sum(1 for r in rejected if r.get("reason") in ("jump", "tracking_jump"))
            if jump_rej > 0:
                log.info("Shot %s rejection_reason=tracking_jump count=%s", shot_id, jump_rej)
        refined_impact_idx = self._refine_impact_frame(detections, window_frames, impact_frame_idx)
        refined_impact_idx = max(window_start, refined_impact_idx - 1)
        det_sorted = sorted(detections, key=lambda d: int(d["frame_index"]))
        last_valid_idx = int(det_sorted[-1]["frame_index"]) if det_sorted else refined_impact_idx
        post_start_idx = refined_impact_idx + 2  # skip impact and immediate unstable frame
        post_end_idx = min(refined_impact_idx + 10, last_valid_idx)
        post = [
            d for d in det_sorted
            if post_start_idx <= int(d["frame_index"]) <= post_end_idx
        ]
        filtered_points = [d["ball_center"] for d in post]  # compensated coordinates
        filtered_screen_points = [d.get("screen_center", d["ball_center"]) for d in post]
        post_idx = [int(d["frame_index"]) for d in post]
        if len(filtered_points) < 2:
            refined_impact_idx = max(window_start, refined_impact_idx - 1)
            post_start_idx = refined_impact_idx + 2
            post_end_idx = min(refined_impact_idx + 10, last_valid_idx)
            post = [
                d for d in det_sorted
                if post_start_idx <= int(d["frame_index"]) <= post_end_idx
            ]
            filtered_points = [d["ball_center"] for d in post]
            filtered_screen_points = [d.get("screen_center", d["ball_center"]) for d in post]
            post_idx = [int(d["frame_index"]) for d in post]
        smoothed_points = self._smooth_points(filtered_points, window=3)
        smoothed_screen_points = self._smooth_points(filtered_screen_points, window=3)

        impact_frame = None
        for fd in window_frames:
            if int(fd["frame_idx"]) == refined_impact_idx:
                impact_frame = fd["frame"]
                break
        if impact_frame is None and window_frames:
            impact_frame = window_frames[0]["frame"]

        is_valid, dir_var_norm, vel_var_norm, monotonic_ratio = self._trajectory_checks(smoothed_points)
        impact_local_idx = 0
        if post_idx:
            impact_local_idx = int(np.argmin(np.abs(np.array(post_idx, dtype=np.int32) - refined_impact_idx)))
        fit = self._fit_direction_model(smoothed_points, smoothed_screen_points, post_idx, impact_local_idx)
        # If curvature is present, prefer later post-impact subset (impact+4 -> impact+10).
        if bool(fit.get("ok", False)) and fit.get("model") == "parabola":
            late_start_idx = refined_impact_idx + 4
            late_post = [
                d for d in det_sorted
                if late_start_idx <= int(d["frame_index"]) <= post_end_idx
            ]
            late_points = [d["ball_center"] for d in late_post]
            late_screen_points = [d.get("screen_center", d["ball_center"]) for d in late_post]
            late_idx = [int(d["frame_index"]) for d in late_post]
            late_smooth = self._smooth_points(late_points, window=3)
            late_smooth_screen = self._smooth_points(late_screen_points, window=3)
            if len(late_smooth) >= 4:
                late_valid, late_dir_var, late_vel_var, late_mono = self._trajectory_checks(late_smooth)
                late_impact_local_idx = 0
                if late_idx:
                    late_impact_local_idx = int(
                        np.argmin(np.abs(np.array(late_idx, dtype=np.int32) - refined_impact_idx))
                    )
                late_fit = self._fit_direction_model(
                    late_smooth,
                    late_smooth_screen,
                    late_idx,
                    late_impact_local_idx,
                )
                if late_valid and bool(late_fit.get("ok", False)):
                    post = late_post
                    filtered_points = late_points
                    filtered_screen_points = late_screen_points
                    post_idx = late_idx
                    smoothed_points = late_smooth
                    smoothed_screen_points = late_smooth_screen
                    fit = late_fit
                    is_valid = late_valid
                    dir_var_norm = late_dir_var
                    vel_var_norm = late_vel_var
                    monotonic_ratio = late_mono
        if (len(smoothed_points) >= 3) and bool(fit.get("ok", False)):
            angle = float(fit["angle"])
            vec = (smoothed_screen_points[0], smoothed_screen_points[-1])
            tracked_count = len(smoothed_points)
            raw_count = len([p for p in raw_points if p is not None])
            continuity_ratio = float(np.clip(tracked_count / max(1.0, float(len(window_frames))), 0.0, 1.0))
            point_ratio = float(np.clip(tracked_count / float(self.expected_post_impact_points), 0.0, 1.0))
            trajectory_variance = float(np.clip((dir_var_norm + vel_var_norm + (1.0 - monotonic_ratio)) / 3.0, 0.0, 1.0))
            inlier_ratio = float(np.clip(track_stats["used_detections"] / max(1.0, float(track_stats["total_candidates"])), 0.0, 1.0))
            residual_score = float(np.clip(1.0 - float(fit["residual_norm"]), 0.0, 1.0))
            angle_stability = float(np.clip(fit.get("angle_stability", 0.0), 0.0, 1.0))
            confidence_score = float(
                np.clip(
                    point_ratio
                    * (1.0 - trajectory_variance)
                    * continuity_ratio
                    * (0.55 + 0.25 * residual_score + 0.20 * angle_stability)
                    * (0.60 + 0.40 * inlier_ratio),
                    0.0,
                    1.0,
                )
            )
            if confidence_score >= 0.7:
                conf_label = "high"
            elif confidence_score >= 0.4:
                conf_label = "medium"
            else:
                conf_label = "low"
            if abs(angle) > 175 or float(fit.get("angle_stability", 0.0)) < 0.15:
                conf_label = "low"
                confidence_score = min(confidence_score, 0.35)
            angle = self._apply_angle_smoothing(angle, max_diff_for_smooth=40.0)
            debug_path = ""
            if impact_frame is not None:
                debug_path = self._save_debug_overlay(
                    impact_frame,
                    trajectory=smoothed_screen_points,
                    kalman_path=kalman_path,
                    rejected_points=[tuple(r["ball_center"]) for r in rejected],
                    regression_line=fit.get("regression_line"),
                    final_source="ball",
                    final_vector=vec,
                    shot_id=shot_id,
                    frame_index=refined_impact_idx,
                    video_stem=video_stem,
                )
            log.info(
                "Shot %s source_type=ball_strong | ball_pts_raw=%s ball_pts_tracked=%s kalman_used=%s regression_angle=%.2f final_angle=%.2f confidence_score=%.3f",
                shot_id,
                raw_count,
                tracked_count,
                True,
                float(fit["angle"]),
                angle,
                confidence_score,
            )
            log.info(
                "Shot %s quality | inliers=%s residual_error=%.3f angle_stability=%.3f gating_rejections=%s model=%s",
                shot_id,
                track_stats["used_detections"],
                float(fit["residual_norm"]),
                float(fit.get("angle_stability", 0.0)),
                track_stats["gating_rejections"],
                fit.get("model", "unknown"),
            )
            log.info(
                "Shot %s trajectory | raw=%s filtered=%s",
                shot_id,
                raw_points,
                filtered_points,
            )
            return {
                "angle": round(angle, 2),
                "regression_angle": round(float(fit["angle"]), 2),
                "confidence": conf_label,
                "confidence_score": round(confidence_score, 3),
                "source": "ball_strong",
                "ball_detections": tracked_count,
                "ball_pts_raw": raw_count,
                "ball_pts_tracked": tracked_count,
                "kalman_used": True,
                "inlier_ratio": round(inlier_ratio, 3),
                "residual_error": round(float(fit["residual_norm"]), 3),
                "angle_stability": round(float(fit.get("angle_stability", 0.0)), 3),
                "gating_rejections": int(track_stats["gating_rejections"]),
                "raw_points": raw_points,
                "filtered_points": filtered_points,
                "debug_image": debug_path,
                "skip_wagon_wheel": False,
            }
        # Weak trajectory fallback: keep approximate ball direction when >=2 points.
        if len(smoothed_screen_points) >= 2:
            p1 = smoothed_screen_points[0]
            p2 = smoothed_screen_points[-1]
            weak_mag = float(np.hypot(float(p2[0] - p1[0]), float(p2[1] - p1[1])))
            if weak_mag < 5.0:
                log.info("Shot %s source_type=ball_weak | reason=movement_too_small_fallback_to_bat mag=%.2f", shot_id, weak_mag)
            else:
                weak_angle = self._point_angle(p1, p2)
                weak_angle = self._apply_angle_smoothing(weak_angle, max_diff_for_smooth=40.0)
                tracked_count = len(smoothed_points)
                raw_count = len([p for p in raw_points if p is not None])
                debug_path = ""
                if impact_frame is not None:
                    debug_path = self._save_debug_overlay(
                        impact_frame,
                        trajectory=smoothed_screen_points,
                        kalman_path=kalman_path,
                        rejected_points=[tuple(r["ball_center"]) for r in rejected],
                        regression_line=fit.get("regression_line") if bool(fit.get("ok", False)) else None,
                        final_source="ball",
                        final_vector=(p1, p2),
                        shot_id=shot_id,
                        frame_index=refined_impact_idx,
                        video_stem=video_stem,
                    )
                log.info(
                    "Shot %s source_type=ball_weak | reason=low_confidence_but_used | ball_pts_raw=%s ball_pts_tracked=%s final_angle=%.2f",
                    shot_id,
                    raw_count,
                    tracked_count,
                    weak_angle,
                )
                return {
                    "angle": round(weak_angle, 2),
                    "regression_angle": round(float(fit.get("angle", weak_angle)), 2),
                    "confidence": "low",
                    "confidence_score": 0.35,
                    "source": "ball_weak",
                    "ball_detections": tracked_count,
                    "ball_pts_raw": raw_count,
                    "ball_pts_tracked": tracked_count,
                    "kalman_used": True,
                    "inlier_ratio": round(float(track_stats["used_detections"] / max(1.0, float(track_stats["total_candidates"]))), 3),
                    "residual_error": round(float(fit.get("residual_norm", 1.0)), 3),
                    "angle_stability": round(float(fit.get("angle_stability", 0.0)), 3),
                    "gating_rejections": int(track_stats["gating_rejections"]),
                    "raw_points": raw_points,
                    "filtered_points": filtered_points,
                    "debug_image": debug_path,
                    "skip_wagon_wheel": False,
                }

        bat_angle, wrist_pt, bat_tip_pt = self._fallback_bat_angle(window_frames, refined_impact_idx)
        if bat_angle is not None:
            bat_angle = self._apply_angle_smoothing(bat_angle, max_diff_for_smooth=40.0)
            tracked_count = len(smoothed_points)
            raw_count = len([p for p in raw_points if p is not None])
            debug_path = ""
            if impact_frame is not None:
                debug_path = self._save_debug_overlay(
                    impact_frame,
                    trajectory=smoothed_screen_points,
                    kalman_path=kalman_path,
                    rejected_points=[tuple(r["ball_center"]) for r in rejected],
                    regression_line=fit.get("regression_line") if bool(fit.get("ok", False)) else None,
                    final_source="bat",
                    final_vector=((wrist_pt[0], wrist_pt[1]), (bat_tip_pt[0], bat_tip_pt[1])) if wrist_pt and bat_tip_pt else None,
                    shot_id=shot_id,
                    frame_index=refined_impact_idx,
                    video_stem=video_stem,
                )
            log.info(
                "Shot %s source_type=bat | reason=insufficient_points | ball_pts_raw=%s ball_pts_tracked=%s kalman_used=%s final_angle=%.2f confidence_score=%.3f",
                shot_id,
                raw_count,
                tracked_count,
                True,
                bat_angle,
                0.2,
            )
            log.info(
                "Shot %s trajectory | raw=%s filtered=%s",
                shot_id,
                raw_points,
                filtered_points,
            )
            return {
                "angle": round(bat_angle, 2),
                "confidence": "low",
                "confidence_score": 0.2,
                "source": "bat",
                "ball_detections": tracked_count,
                "ball_pts_raw": raw_count,
                "ball_pts_tracked": tracked_count,
                "kalman_used": True,
                "inlier_ratio": round(float(track_stats["used_detections"] / max(1.0, float(track_stats["total_candidates"]))), 3),
                "residual_error": 1.0,
                "angle_stability": 0.0,
                "gating_rejections": int(track_stats["gating_rejections"]),
                "raw_points": raw_points,
                "filtered_points": filtered_points,
                "debug_image": debug_path,
                "skip_wagon_wheel": False,
            }

        tracked_count = len(smoothed_points)
        raw_count = len([p for p in raw_points if p is not None])
        fallback_angle = 0.0
        if self.prev_output_angle is not None:
            fallback_angle = float(self.prev_output_angle)
        fallback_angle = self._apply_angle_smoothing(fallback_angle, max_diff_for_smooth=40.0)
        log.info(
            "Shot %s source_type=fallback | reason=no_direction | ball_pts_raw=%s ball_pts_tracked=%s kalman_used=%s final_angle=%.2f confidence_score=%.3f",
            shot_id,
            raw_count,
            tracked_count,
            True,
            fallback_angle,
            0.0,
        )
        log.info(
            "Shot %s trajectory | raw=%s filtered=%s",
            shot_id,
            raw_points,
            filtered_points,
        )
        return {
            "angle": round(fallback_angle, 2),
            "confidence": "low",
            "confidence_score": 0.0,
            "source": "fallback",
            "ball_detections": tracked_count,
            "ball_pts_raw": raw_count,
            "ball_pts_tracked": tracked_count,
            "kalman_used": True,
            "inlier_ratio": round(float(track_stats["used_detections"] / max(1.0, float(track_stats["total_candidates"]))), 3),
            "residual_error": 1.0,
            "angle_stability": 0.0,
            "gating_rejections": int(track_stats["gating_rejections"]),
            "raw_points": raw_points,
            "filtered_points": filtered_points,
            "debug_image": "",
            "skip_wagon_wheel": False,
        }

    def close(self):
        self.pose.close()
