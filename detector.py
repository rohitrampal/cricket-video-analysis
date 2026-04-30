
import cv2
import numpy as np
import mediapipe as mp
import logging
from config import POSE_CONFIDENCE
try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - keep pipeline alive if scipy missing
    savgol_filter = None

log = logging.getLogger(__name__)
mp_pose = mp.solutions.pose


class PoseDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=POSE_CONFIDENCE,
            min_tracking_confidence=POSE_CONFIDENCE
        )

    def detect(self, frame: np.ndarray) -> dict | None:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        lm   = result.pose_landmarks.landmark

        def pt(idx):
            p = lm[idx]
            return {"x": round(p.x*w, 2),
                    "y": round(p.y*h, 2),
                    "visibility": round(p.visibility, 3)}

        return {
            "right_wrist":    pt(mp_pose.PoseLandmark.RIGHT_WRIST),
            "right_elbow":    pt(mp_pose.PoseLandmark.RIGHT_ELBOW),
            "right_shoulder": pt(mp_pose.PoseLandmark.RIGHT_SHOULDER),
            "right_hip":      pt(mp_pose.PoseLandmark.RIGHT_HIP),
            "right_knee":     pt(mp_pose.PoseLandmark.RIGHT_KNEE),
            "right_ankle":    pt(mp_pose.PoseLandmark.RIGHT_ANKLE),
            "left_wrist":     pt(mp_pose.PoseLandmark.LEFT_WRIST),
            "left_elbow":     pt(mp_pose.PoseLandmark.LEFT_ELBOW),
            "left_shoulder":  pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
            "left_hip":       pt(mp_pose.PoseLandmark.LEFT_HIP),
            "left_knee":      pt(mp_pose.PoseLandmark.LEFT_KNEE),
            "left_ankle":     pt(mp_pose.PoseLandmark.LEFT_ANKLE),
            "nose":           pt(mp_pose.PoseLandmark.NOSE),
        }

    def close(self):
        self.pose.close()


class MotionShotDetector:
    """
    Simplified, reliable shot detector for umpire-side close-up videos.

    Strategy:
    - Use whole-frame motion score (fast, no ball tracking needed)
    - Find the single peak motion window = the shot
    - Extract wrist vector at peak motion for direction
    - Ignore everything else
    """

    # ── Tuning ────────────────────────────────────────────
    MOTION_THRESHOLD    = 0.010  # normalized active-pixel ratio
    MOTION_BINARY_TH    = 25     # binary threshold for frame diff mask
    PEAK_HOLD_FRAMES    = 8      # frames around peak to average wrist vector
    MIN_GAP_FRAMES      = 45     # minimum frames between two shots
    END_OF_SHOT_QUIET   = 12     # consecutive quiet frames = shot ended
    MIN_VISIBILITY      = 0.3
    MIN_VECTOR_MAGNITUDE = 0.01  # in body-relative coordinates
    MAX_POSE_JUMP_PX    = 100.0  # reject sudden wrist jumps
    VERTICAL_MOTION_RATIO_MAX = 1.8  # reject mostly-vertical vectors
    ENABLE_VERTICAL_FILTER = False   # disabled to prioritize recall
    CALIBRATION_FRAMES  = 90
    MID_CALIBRATION_START = 180
    MID_CALIBRATION_FRAMES = 90
    MIN_CONFIDENCE_SCORE = 0.20
    DEDUP_WINDOW_SEC    = 0.5
    DEDUP_ANGLE_DIFF_MAX = 25.0
    VERTICAL_REJECT_MAGNITUDE = 0.12
    MIN_ARM_ANGLE_CHANGE = 15.0   # minimum arm-angle change to confirm shot
    MIN_WRIST_TRAVEL_PX = 15.0    # reject tiny wrist-travel windows
    IMPACT_FORWARD_FRAMES = 4      # use post-impact direction (N in [3,5])
    UNSTABLE_ANGLE_ABS_DEG = 110.0

    def __init__(self, debug_metrics: bool = False):
        self.pose_detector   = PoseDetector()
        self.prev_gray       = None
        self.last_shot_frame = -999
        self.shots_detected  = 0
        self.debug_metrics = debug_metrics

        # Shot window state
        self.in_shot_window  = False
        self.shot_start_frame = -1
        self.quiet_count     = 0
        self.peak_motion     = 0.0
        self.peak_frame_data = None   # frame_data at peak motion

        # Arm positions during shot window
        self.wrist_positions = []     # body-relative (x, y) during active window
        self.wrist_frame_indices = []
        self.arm_angles = []
        self.shot_arm_side = None
        self.max_wrist_velocity = 0.0
        self.impact_sample_idx = -1
        self.last_frame_idx = -1
        self.last_timestamp_sec = 0.0
        self.pending_shot = None

        # Adaptive calibration state (early frames only)
        self.calibration_done = False
        self.calibration_motion_scores = []
        self.calibration_wrist_velocities = []
        self.calib_prev_wrist = None
        self.calib_prev_frame_idx = -1
        self.dynamic_motion_threshold = self.MOTION_THRESHOLD
        self.dynamic_min_impact_velocity = 0.0
        self.mid_calibration_done = False
        self.mid_calibration_motion_scores = []
        self.mid_calibration_wrist_velocities = []

        # Percentile-based confidence gating.
        self.shot_confidence_history = []

    def _motion_score(self, frame: np.ndarray) -> float:
        """Thresholded frame-diff score using normalized active pixels."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        diff = cv2.absdiff(self.prev_gray, gray)
        _, mask = cv2.threshold(diff, self.MOTION_BINARY_TH, 255, cv2.THRESH_BINARY)
        active_pixels = float(np.count_nonzero(mask))
        total_pixels = float(mask.shape[0] * mask.shape[1])
        score = active_pixels / max(1.0, total_pixels)
        self.prev_gray = gray
        return score

    @staticmethod
    def _point_xy(point: dict) -> np.ndarray:
        return np.array([point["x"], point["y"]], dtype=np.float32)

    def _arm_visibility(self, keypoints: dict, side: str) -> float:
        wrist = keypoints[f"{side}_wrist"]["visibility"]
        elbow = keypoints[f"{side}_elbow"]["visibility"]
        shoulder = keypoints[f"{side}_shoulder"]["visibility"]
        return float((wrist + elbow + shoulder) / 3.0)

    def _hip_center(self, keypoints: dict) -> np.ndarray:
        left_hip = self._point_xy(keypoints["left_hip"])
        right_hip = self._point_xy(keypoints["right_hip"])
        left_vis = keypoints["left_hip"]["visibility"]
        right_vis = keypoints["right_hip"]["visibility"]
        if left_vis >= self.MIN_VISIBILITY and right_vis >= self.MIN_VISIBILITY:
            return (left_hip + right_hip) / 2.0
        if left_vis >= right_vis:
            return left_hip
        return right_hip

    def _best_visible_arm(self, keypoints: dict) -> str | None:
        left_vis = self._arm_visibility(keypoints, "left")
        right_vis = self._arm_visibility(keypoints, "right")
        best_vis = max(left_vis, right_vis)
        if best_vis < self.MIN_VISIBILITY:
            return None
        return "left" if left_vis >= right_vis else "right"

    def _extract_bat_sample(self, keypoints: dict, side: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        side_vis = self._arm_visibility(keypoints, side)
        if side_vis < self.MIN_VISIBILITY:
            return None, None

        wrist = self._point_xy(keypoints[f"{side}_wrist"])
        shoulder = self._point_xy(keypoints[f"{side}_shoulder"])
        hip_center = self._hip_center(keypoints)

        wrist_rel = wrist - hip_center
        shoulder_rel = shoulder - hip_center

        shoulder_vis = keypoints[f"{side}_shoulder"]["visibility"]
        if shoulder_vis < self.MIN_VISIBILITY:
            return None, None
        # Use shoulder-based arm vector for more stable shot motion cue.
        bat_vec = wrist_rel - shoulder_rel
        return wrist_rel, bat_vec

    def _smooth_trajectory(self, positions: np.ndarray) -> np.ndarray:
        n = len(positions)
        if n < 5:
            return positions
        window = min(9, n if n % 2 == 1 else n - 1)
        if window < 5:
            return positions
        if savgol_filter is None:
            return positions
        smoothed = positions.copy()
        smoothed[:, 0] = savgol_filter(positions[:, 0], window_length=window, polyorder=2, mode="interp")
        smoothed[:, 1] = savgol_filter(positions[:, 1], window_length=window, polyorder=2, mode="interp")
        return smoothed

    @staticmethod
    def _wrap_angle_deg(angle: float) -> float:
        return ((angle + 180.0) % 360.0) - 180.0

    def _detect_impact_idx(self, smooth_positions: np.ndarray, frame_indices: np.ndarray) -> tuple[int, np.ndarray]:
        """Detect impact via rise-then-drop velocity pattern (not absolute max)."""
        if len(smooth_positions) < 3:
            return 0, np.zeros(0, dtype=np.float32)
        diffs = np.diff(smooth_positions, axis=0)
        frame_deltas = np.diff(frame_indices).astype(np.float32)
        frame_deltas = np.maximum(1.0, frame_deltas)
        velocities = np.linalg.norm(diffs, axis=1) / frame_deltas
        if len(velocities) < 3:
            return int(np.argmax(velocities)) + 1, velocities

        best_i = -1
        best_score = -1.0
        for i in range(1, len(velocities) - 1):
            left = velocities[i] - velocities[i - 1]
            right = velocities[i] - velocities[i + 1]
            if left > 0 and right > 0:
                prominence = min(left, right)
                local_level = max(1e-6, (velocities[i - 1] + velocities[i] + velocities[i + 1]) / 3.0)
                score = prominence / local_level
                if score > best_score:
                    best_score = score
                    best_i = i

        if best_i >= 0:
            return best_i + 1, velocities
        return int(np.argmax(velocities)) + 1, velocities

    def _adaptive_window(self, n: int, impact_idx: int) -> tuple[int, int]:
        """Dynamic pre/post frame window around impact based on trajectory length."""
        radius = max(2, min(6, n // 5))
        start_idx = max(0, impact_idx - radius)
        end_idx = min(n - 1, impact_idx + radius)
        return start_idx, end_idx

    def _stable_angle_from_impact(self, smooth_positions: np.ndarray, impact_idx: int) -> float:
        """Average local angles around impact to reduce temporal spikes."""
        n = len(smooth_positions)
        angle_values = []
        for c in range(max(1, impact_idx - 1), min(n - 1, impact_idx + 2)):
            vec = smooth_positions[c] - smooth_positions[c - 1]
            mag = float(np.linalg.norm(vec))
            if mag < 1e-6:
                continue
            unit = vec / mag
            angle = float(np.degrees(np.arctan2(-unit[1], unit[0])))
            angle_values.append(np.deg2rad(self._wrap_angle_deg(angle)))
        if not angle_values:
            return 0.0
        mean_sin = float(np.mean(np.sin(angle_values)))
        mean_cos = float(np.mean(np.cos(angle_values)))
        return self._wrap_angle_deg(float(np.degrees(np.arctan2(mean_sin, mean_cos))))

    def _median_recent_vector(self, smooth_positions: np.ndarray, impact_idx: int) -> np.ndarray:
        """Median local motion vector around impact for unstable-angle fallback."""
        n = len(smooth_positions)
        diffs = []
        for i in range(max(1, impact_idx - 2), min(n, impact_idx + 3)):
            d = smooth_positions[i] - smooth_positions[i - 1]
            if float(np.linalg.norm(d)) > 1e-6:
                diffs.append(d)
        if not diffs:
            return np.array([0.0, 0.0], dtype=np.float32)
        arr = np.array(diffs, dtype=np.float32)
        return np.median(arr, axis=0)

    def _update_calibration(self, frame_idx: int, motion_score: float, keypoints: dict | None):
        def trimmed_mean(values: list[float], trim_ratio: float = 0.15) -> float:
            if not values:
                return 0.0
            arr = np.sort(np.array(values, dtype=np.float32))
            n = len(arr)
            if n < 7:
                return float(np.mean(arr))
            k = int(n * trim_ratio)
            if 2 * k >= n:
                return float(np.mean(arr))
            return float(np.mean(arr[k:n - k]))

        collect_early = not self.calibration_done
        collect_mid = (frame_idx >= self.MID_CALIBRATION_START) and (not self.mid_calibration_done)
        if not collect_early and not collect_mid:
            return
        if collect_early:
            self.calibration_motion_scores.append(float(motion_score))
        if collect_mid:
            self.mid_calibration_motion_scores.append(float(motion_score))

        if keypoints:
            side = self._best_visible_arm(keypoints)
            if side:
                wrist_rel, _ = self._extract_bat_sample(keypoints, side)
                if wrist_rel is not None:
                    curr = np.array([float(wrist_rel[0]), float(wrist_rel[1])], dtype=np.float32)
                    if self.calib_prev_wrist is not None:
                        frame_delta = max(1, frame_idx - self.calib_prev_frame_idx)
                        vel = float(np.linalg.norm(curr - self.calib_prev_wrist) / frame_delta)
                        if collect_early:
                            self.calibration_wrist_velocities.append(vel)
                        if collect_mid:
                            self.mid_calibration_wrist_velocities.append(vel)
                    self.calib_prev_wrist = curr
                    self.calib_prev_frame_idx = frame_idx

        if (not self.calibration_done) and len(self.calibration_motion_scores) >= self.CALIBRATION_FRAMES:
            mean_motion = trimmed_mean(self.calibration_motion_scores)
            mean_vel = trimmed_mean(self.calibration_wrist_velocities)
            early_motion_th = float(np.clip(mean_motion * 2.2 * 0.6, 0.0025, 0.06))
            early_vel_th = float(max(0.0, mean_vel * 2.0))
            self.dynamic_motion_threshold = early_motion_th
            self.dynamic_min_impact_velocity = early_vel_th
            self.calibration_done = True
            log.info(
                "Calibration complete | motion_th=%.4f impact_vel_th=%.3f",
                self.dynamic_motion_threshold,
                self.dynamic_min_impact_velocity,
            )
        if (not self.mid_calibration_done) and len(self.mid_calibration_motion_scores) >= self.MID_CALIBRATION_FRAMES:
            mid_motion = trimmed_mean(self.mid_calibration_motion_scores)
            mid_vel = trimmed_mean(self.mid_calibration_wrist_velocities)
            mid_motion_th = float(np.clip(mid_motion * 2.2 * 0.6, 0.0025, 0.06))
            mid_vel_th = float(max(0.0, mid_vel * 2.0))
            if self.calibration_done:
                self.dynamic_motion_threshold = 0.6 * self.dynamic_motion_threshold + 0.4 * mid_motion_th
                self.dynamic_min_impact_velocity = 0.6 * self.dynamic_min_impact_velocity + 0.4 * mid_vel_th
            else:
                self.dynamic_motion_threshold = mid_motion_th
                self.dynamic_min_impact_velocity = mid_vel_th
                self.calibration_done = True
            self.mid_calibration_done = True
            log.info(
                "Mid calibration blended | motion_th=%.4f impact_vel_th=%.3f",
                self.dynamic_motion_threshold,
                self.dynamic_min_impact_velocity,
            )

    def _queue_or_emit_shot(self, shot: dict, current_ts: float) -> dict | None:
        if self.pending_shot is None:
            self.pending_shot = shot
            return None
        prev_ts = float(self.pending_shot["timestamp_sec"])
        if current_ts - prev_ts <= self.DEDUP_WINDOW_SEC:
            prev_angle = float(self.pending_shot.get("raw_angle_deg", 0.0))
            curr_angle = float(shot.get("raw_angle_deg", 0.0))
            angle_diff = abs(self._wrap_angle_deg(curr_angle - prev_angle))
            if angle_diff <= self.DEDUP_ANGLE_DIFF_MAX:
                prev_conf = float(self.pending_shot.get("confidence_score", 0.0))
                curr_conf = float(shot.get("confidence_score", 0.0))
                if curr_conf >= prev_conf:
                    self.pending_shot = shot
                return None
        emit = self.pending_shot
        self.pending_shot = shot
        return emit

    def process_frame(self, frame_data: dict,
                      min_gap: int = 45) -> dict | None:
        """
        Process one frame. Returns shot dict only when a complete
        shot window (rise → peak → quiet) has been detected.
        """
        frame     = frame_data["frame"]
        frame_idx = frame_data["frame_idx"]
        ts        = frame_data["timestamp_sec"]
        self.last_frame_idx = frame_idx
        self.last_timestamp_sec = ts

        motion = self._motion_score(frame)
        keypoints = self.pose_detector.detect(frame)
        self._update_calibration(frame_idx, motion, keypoints)
        log.debug(
            "frame=%s motion=%.5f threshold=%.5f in_window=%s quiet=%s",
            frame_idx,
            motion,
            self.dynamic_motion_threshold,
            self.in_shot_window,
            self.quiet_count,
        )

        # Emit pending shot once dedup window elapsed.
        if self.pending_shot is not None:
            pending_ts = float(self.pending_shot["timestamp_sec"])
            if ts - pending_ts > self.DEDUP_WINDOW_SEC:
                emit = self.pending_shot
                self.pending_shot = None
                return emit

        # ── Detect shot window opening ────────────────────
        if not self.in_shot_window:
            if motion >= self.dynamic_motion_threshold:
                # Check minimum gap from last shot
                if frame_idx - self.last_shot_frame >= min_gap:
                    self.in_shot_window   = True
                    self.shot_start_frame = frame_idx
                    self.quiet_count      = 0
                    self.peak_motion      = motion
                    self.peak_frame_data  = frame_data
                    self.wrist_positions  = []
                    self.wrist_frame_indices = []
                    self.arm_angles = []
                    self.shot_arm_side = None
                    self.max_wrist_velocity = 0.0
                    self.impact_sample_idx = -1
                    log.debug(f"Shot window opened at frame {frame_idx}")
            return None

        # ── Inside shot window ────────────────────────────
        # Track peak motion frame
        if motion > self.peak_motion:
            self.peak_motion     = motion
            self.peak_frame_data = frame_data

        # Collect wrist positions during window
        if keypoints:
            if self.shot_arm_side is None:
                self.shot_arm_side = self._best_visible_arm(keypoints)
            if self.shot_arm_side:
                wrist_rel, arm_vec = self._extract_bat_sample(keypoints, self.shot_arm_side)
                if wrist_rel is not None:
                    curr = np.array([float(wrist_rel[0]), float(wrist_rel[1])], dtype=np.float32)
                    curr_angle = float(np.degrees(np.arctan2(-arm_vec[1], arm_vec[0])))
                    if self.wrist_positions:
                        prev = np.array(self.wrist_positions[-1], dtype=np.float32)
                        jump = float(np.linalg.norm(curr - prev))
                        if jump > self.MAX_POSE_JUMP_PX:
                            curr = None
                    if curr is not None:
                        self.wrist_positions.append((float(curr[0]), float(curr[1])))
                        self.wrist_frame_indices.append(frame_idx)
                        self.arm_angles.append(curr_angle)
                        curr_idx = len(self.wrist_positions) - 1
                        if curr_idx > 0 and len(self.arm_angles) > 1:
                            prev_idx = self.wrist_frame_indices[curr_idx - 1]
                            curr_frame = self.wrist_frame_indices[curr_idx]
                            frame_delta = max(1, curr_frame - prev_idx)
                            prev_angle = self.arm_angles[-2]
                            angle_delta = abs(self._wrap_angle_deg(curr_angle - prev_angle)) / frame_delta
                            if angle_delta > self.max_wrist_velocity:
                                self.max_wrist_velocity = angle_delta
                                self.impact_sample_idx = curr_idx

        # Detect end of shot (motion drops back to quiet)
        if motion < self.dynamic_motion_threshold:
            self.quiet_count += 1
        else:
            self.quiet_count = 0

        # ── Shot window closed ────────────────────────────
        if self.quiet_count >= self.END_OF_SHOT_QUIET:
            self.in_shot_window = False
            shot = self._finalize_shot(frame_idx, ts)
            if shot is None:
                return None
            return self._queue_or_emit_shot(shot, ts)

        return None

    def _finalize_shot(self, end_frame_idx: int, end_timestamp_sec: float) -> dict | None:
        """
        Called when a shot window closes.
        Compute bat vector from wrist trajectory during the window.
        """
        if len(self.wrist_positions) < 3:
            # Fallback to motion-only shot so real shots are not fully missed.
            peak_fd = self.peak_frame_data
            if not peak_fd:
                log.debug("Shot window closed but not enough wrist data")
                return None
            self.shots_detected += 1
            self.last_shot_frame = end_frame_idx
            peak_frame_idx = peak_fd["frame_idx"]
            peak_ts = peak_fd["timestamp_sec"]
            fallback_conf = max(self.MIN_CONFIDENCE_SCORE, 0.22)
            shot_data = {
                "shot_id": self.shots_detected,
                "frame_idx": peak_frame_idx,
                "timestamp_sec": peak_ts,
                "keypoints": {},
                "bat_vector": {"dx": 0.0, "dy": 0.0},
                "raw_angle_deg": 0.0,
                "peak_motion": round(self.peak_motion, 2),
                "wrist_samples": len(self.wrist_positions),
                "confidence_score": round(fallback_conf, 3),
            }
            if self.debug_metrics:
                shot_data["debug"] = {
                    "velocity_peak": 0.0,
                    "velocity_variance": 0.0,
                    "vector_magnitude": 0.0,
                    "vertical_ratio": 0.0,
                    "confidence_score": round(fallback_conf, 4),
                    "angle_stability_score": 0.0,
                    "confidence_gate": round(self.MIN_CONFIDENCE_SCORE, 4),
                }
            log.info(
                "Fallback motion-only shot emitted | frame=%s ts=%.3f peak_motion=%.4f",
                peak_frame_idx,
                peak_ts,
                self.peak_motion,
            )
            return shot_data

        positions = np.array(self.wrist_positions, dtype=np.float32)
        smooth_positions = self._smooth_trajectory(positions)
        wrist_travel = float(np.linalg.norm(smooth_positions[-1] - smooth_positions[0]))
        if wrist_travel < self.MIN_WRIST_TRAVEL_PX:
            log.debug("Shot rejected due to low wrist travel (%.2f px)", wrist_travel)
            return None

        frame_indices = np.array(self.wrist_frame_indices, dtype=np.int32)
        if len(self.arm_angles) >= 2:
            angle_deltas = [
                abs(self._wrap_angle_deg(self.arm_angles[i] - self.arm_angles[i - 1]))
                for i in range(1, len(self.arm_angles))
            ]
            max_angle_change = float(max(angle_deltas)) if angle_deltas else 0.0
            impact_idx = int(np.argmax(angle_deltas)) + 1 if angle_deltas else 0
            velocities = np.array(angle_deltas, dtype=np.float32)
        else:
            impact_idx, velocities = self._detect_impact_idx(smooth_positions, frame_indices)
            max_angle_change = 0.0
        self.impact_sample_idx = impact_idx
        if max_angle_change < self.MIN_ARM_ANGLE_CHANGE:
            log.debug("Shot rejected due to low arm angle change (%.2f deg)", max_angle_change)
            return None
        n = len(smooth_positions)
        start_idx = int(np.clip(self.impact_sample_idx, 0, n - 1))
        end_idx = int(np.clip(self.impact_sample_idx + self.IMPACT_FORWARD_FRAMES, 0, n - 1))
        if end_idx <= start_idx:
            end_idx = min(n - 1, start_idx + 1)
        start = smooth_positions[start_idx]
        end = smooth_positions[end_idx]
        vec = end - start

        dx, dy = float(vec[0]), float(vec[1])
        total_dist = float(np.linalg.norm(vec))

        if total_dist < self.MIN_VECTOR_MAGNITUDE:
            log.debug(f"Shot window closed but wrist barely moved ({total_dist:.3f} rel-units)")
            return None
        if len(self.arm_angles) < 2:
            h, w = self.peak_frame_data["frame"].shape[:2] if self.peak_frame_data else (1, 1)
            frame_scale = max(1.0, float(np.sqrt(w * h)))
            max_wrist_velocity_norm = float(self.max_wrist_velocity / frame_scale)
            dynamic_impact_velocity_norm = float(self.dynamic_min_impact_velocity / frame_scale)
            if max_wrist_velocity_norm < dynamic_impact_velocity_norm:
                log.debug("Shot rejected due to low impact velocity (%.3f)", self.max_wrist_velocity)
                return None

        # Reject mostly vertical movement to reduce false shot directions.
        vertical_ratio = float(abs(dy) / max(1e-6, abs(dx)))
        if self.ENABLE_VERTICAL_FILTER and vertical_ratio > self.VERTICAL_MOTION_RATIO_MAX and total_dist < self.VERTICAL_REJECT_MAGNITUDE:
            log.debug("Shot rejected due to dominant vertical motion")
            return None

        unit_vec = vec / max(1e-6, total_dist)
        raw_angle = float(np.degrees(np.arctan2(-unit_vec[1], unit_vec[0])))
        raw_angle = self._wrap_angle_deg(raw_angle)

        # Fallback for unstable/high-elevation style directions.
        if abs(raw_angle) > self.UNSTABLE_ANGLE_ABS_DEG:
            med_vec = self._median_recent_vector(smooth_positions, self.impact_sample_idx)
            med_mag = float(np.linalg.norm(med_vec))
            if med_mag > 1e-6:
                med_unit = med_vec / med_mag
                raw_angle = float(np.degrees(np.arctan2(-med_unit[1], med_unit[0])))
                raw_angle = self._wrap_angle_deg(raw_angle)

        # Multi-factor confidence: magnitude + peak sharpness + trajectory smoothness.
        magnitude_score = min(1.0, total_dist / 0.35)
        peak_sharpness_score = 0.0
        if len(velocities) >= 3 and 1 <= self.impact_sample_idx - 1 < len(velocities) - 1:
            v_i = velocities[self.impact_sample_idx - 1]
            v_l = velocities[self.impact_sample_idx - 2]
            v_r = velocities[self.impact_sample_idx]
            peak_sharpness_score = float(max(0.0, v_i - 0.5 * (v_l + v_r)) / max(1e-6, v_i))

        smooth_diffs = np.diff(smooth_positions, axis=0)
        smooth_vel = np.linalg.norm(smooth_diffs, axis=1)
        if len(smooth_vel) >= 2:
            var = float(np.std(smooth_vel))
            mean_v = float(np.mean(smooth_vel))
            trajectory_smoothness_score = float(max(0.0, 1.0 - (var / max(1e-6, mean_v + 1e-6))))
        else:
            trajectory_smoothness_score = 0.5
        local_angles = []
        n = len(smooth_positions)
        for c in range(max(1, self.impact_sample_idx - 2), min(n - 1, self.impact_sample_idx + 3)):
            dvec = smooth_positions[c] - smooth_positions[c - 1]
            dmag = float(np.linalg.norm(dvec))
            if dmag < 1e-6:
                continue
            ang = float(np.degrees(np.arctan2(-dvec[1], dvec[0])))
            local_angles.append(np.deg2rad(self._wrap_angle_deg(ang)))
        if len(local_angles) >= 2:
            mean_sin = float(np.mean(np.sin(local_angles)))
            mean_cos = float(np.mean(np.cos(local_angles)))
            resultant = float(np.sqrt(mean_sin * mean_sin + mean_cos * mean_cos))
            angle_stability_score = float(np.clip(resultant, 0.0, 1.0))
        else:
            angle_stability_score = 0.5
        confidence_score = float(
            0.38 * magnitude_score +
            0.24 * peak_sharpness_score +
            0.20 * trajectory_smoothness_score +
            0.18 * angle_stability_score
        )
        confidence_score = float(np.clip(confidence_score, 0.0, 1.0))
        self.shot_confidence_history.append(confidence_score)
        p30 = float(np.percentile(self.shot_confidence_history, 30)) if len(self.shot_confidence_history) >= 3 else self.MIN_CONFIDENCE_SCORE
        confidence_gate = max(self.MIN_CONFIDENCE_SCORE, p30)
        if confidence_score < confidence_gate:
            log.debug("Shot rejected due to low confidence (%.3f)", confidence_score)
            self.shot_confidence_history.pop()
            return None

        velocity_peak = float(np.max(velocities)) if len(velocities) > 0 else 0.0
        velocity_variance = float(np.var(velocities)) if len(velocities) > 1 else 0.0

        # Lightweight aerial correction: when high-length shots have sign mismatch
        # between angle and post-impact lateral motion, trust post-impact direction.
        shot_length_proxy = float(np.clip(total_dist / 60.0, 0.35, 1.0))
        if shot_length_proxy > 0.75:
            lateral_dx = float(end[0] - start[0])
            if (raw_angle > 0.0 and lateral_dx < 0.0) or (raw_angle < 0.0 and lateral_dx > 0.0):
                raw_angle = -raw_angle
                raw_angle = self._wrap_angle_deg(raw_angle)

        self.shots_detected  += 1
        self.last_shot_frame  = end_frame_idx

        peak_fd = self.peak_frame_data
        peak_frame_idx = peak_fd["frame_idx"] if peak_fd else end_frame_idx
        peak_ts = peak_fd["timestamp_sec"] if peak_fd else end_timestamp_sec
        shot_data = {
            "shot_id":       self.shots_detected,
            "frame_idx":     peak_frame_idx,
            "timestamp_sec": peak_ts,
            "keypoints":     {},
            "bat_vector":    {"dx": round(dx, 2), "dy": round(dy, 2)},
            "raw_angle_deg": round(raw_angle, 2),
            "peak_motion":   round(self.peak_motion, 2),
            "wrist_samples": len(self.wrist_positions),
            "confidence_score": round(confidence_score, 3),
        }
        if self.debug_metrics:
            shot_data["debug"] = {
                "velocity_peak": round(velocity_peak, 4),
                "velocity_variance": round(velocity_variance, 6),
                "vector_magnitude": round(total_dist, 4),
                "vertical_ratio": round(vertical_ratio, 4),
                "confidence_score": round(confidence_score, 4),
                "angle_stability_score": round(angle_stability_score, 4),
                "confidence_gate": round(confidence_gate, 4),
            }

        log.info(
            "✅ Shot %s finalized | frame %s | angle %.1f° | dist %.3f | "
            "v_peak %.3f | v_var %.5f | v_ratio %.3f | conf %.3f",
            self.shots_detected,
            peak_frame_idx,
            raw_angle,
            total_dist,
            velocity_peak,
            velocity_variance,
            vertical_ratio,
            confidence_score,
        )
        print(f"   🏏 Shot {self.shots_detected} at "
              f"{peak_ts}s | "
              f"angle {raw_angle:.1f}° | "
              f"wrist travel {total_dist:.1f}px")
        return shot_data

    def flush(self) -> dict | None:
        """Call after last frame to catch any open shot window."""
        if self.in_shot_window and self.wrist_positions:
            self.in_shot_window = False
            shot = self._finalize_shot(self.last_frame_idx, self.last_timestamp_sec)
            if shot:
                queued = self._queue_or_emit_shot(shot, self.last_timestamp_sec)
                if queued:
                    return queued
        if self.pending_shot is not None:
            final = self.pending_shot
            self.pending_shot = None
            return final
        return None

    def close(self):
        self.pose_detector.close()


# Keep ShotDetector as alias so pipeline.py works unchanged
ShotDetector = MotionShotDetector
