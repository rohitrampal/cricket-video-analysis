
import cv2
import numpy as np
import mediapipe as mp
import logging
from config import POSE_CONFIDENCE

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
    MOTION_THRESHOLD    = 6.5    # mean pixel diff to count as active motion
    PEAK_HOLD_FRAMES    = 8      # frames around peak to average wrist vector
    MIN_GAP_FRAMES      = 45     # minimum frames between two shots
    END_OF_SHOT_QUIET   = 12     # consecutive quiet frames = shot ended

    def __init__(self):
        self.pose_detector   = PoseDetector()
        self.prev_gray       = None
        self.last_shot_frame = -999
        self.shots_detected  = 0

        # Shot window state
        self.in_shot_window  = False
        self.shot_start_frame = -1
        self.quiet_count     = 0
        self.peak_motion     = 0.0
        self.peak_frame_data = None   # frame_data at peak motion

        # Wrist positions during shot window
        self.wrist_positions = []     # (x, y) during active window

    def _motion_score(self, frame: np.ndarray) -> float:
        """Simple frame-diff motion score."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        diff  = cv2.absdiff(self.prev_gray, gray)
        score = float(diff.mean())
        self.prev_gray = gray
        return score

    def process_frame(self, frame_data: dict,
                      min_gap: int = 45) -> dict | None:
        """
        Process one frame. Returns shot dict only when a complete
        shot window (rise → peak → quiet) has been detected.
        """
        frame     = frame_data["frame"]
        frame_idx = frame_data["frame_idx"]
        ts        = frame_data["timestamp_sec"]

        motion = self._motion_score(frame)

        # ── Detect shot window opening ────────────────────
        if not self.in_shot_window:
            if motion >= self.MOTION_THRESHOLD:
                # Check minimum gap from last shot
                if frame_idx - self.last_shot_frame >= min_gap:
                    self.in_shot_window   = True
                    self.shot_start_frame = frame_idx
                    self.quiet_count      = 0
                    self.peak_motion      = motion
                    self.peak_frame_data  = frame_data
                    self.wrist_positions  = []
                    log.debug(f"Shot window opened at frame {frame_idx}")
            return None

        # ── Inside shot window ────────────────────────────
        # Track peak motion frame
        if motion > self.peak_motion:
            self.peak_motion     = motion
            self.peak_frame_data = frame_data

        # Collect wrist positions during window
        keypoints = self.pose_detector.detect(frame)
        if keypoints:
            rw = keypoints["right_wrist"]
            lw = keypoints["left_wrist"]
            wrist = rw if rw["visibility"] >= lw["visibility"] else lw
            if wrist["visibility"] > 0.3:
                self.wrist_positions.append((wrist["x"], wrist["y"]))

        # Detect end of shot (motion drops back to quiet)
        if motion < self.MOTION_THRESHOLD:
            self.quiet_count += 1
        else:
            self.quiet_count = 0

        # ── Shot window closed ────────────────────────────
        if self.quiet_count >= self.END_OF_SHOT_QUIET:
            self.in_shot_window = False
            shot = self._finalize_shot(frame_idx)
            return shot

        return None

    def _finalize_shot(self, end_frame_idx: int) -> dict | None:
        """
        Called when a shot window closes.
        Compute bat vector from wrist trajectory during the window.
        """
        if len(self.wrist_positions) < 3:
            log.debug("Shot window closed but not enough wrist data")
            return None

        positions = np.array(self.wrist_positions)

        # Use first 30% vs last 30% of positions for stable vector
        n     = len(positions)
        start = positions[:max(1, n//3)].mean(axis=0)
        end   = positions[min(n-1, 2*n//3):].mean(axis=0)
        vec   = end - start

        dx, dy = float(vec[0]), float(vec[1])
        total_dist = np.sqrt(dx**2 + dy**2)

        if total_dist < 8:
            log.debug(f"Shot window closed but wrist barely moved ({total_dist:.1f}px)")
            return None

        raw_angle = np.degrees(np.arctan2(-dy, dx))

        self.shots_detected  += 1
        self.last_shot_frame  = end_frame_idx

        peak_fd = self.peak_frame_data
        shot_data = {
            "shot_id":       self.shots_detected,
            "frame_idx":     peak_fd["frame_idx"],
            "timestamp_sec": peak_fd["timestamp_sec"],
            "keypoints":     {},
            "bat_vector":    {"dx": round(dx, 2), "dy": round(dy, 2)},
            "raw_angle_deg": round(raw_angle, 2),
            "peak_motion":   round(self.peak_motion, 2),
            "wrist_samples": len(self.wrist_positions),
        }

        log.info(
            f"✅ Shot {self.shots_detected} finalized | "
            f"frame {peak_fd['frame_idx']} | "
            f"angle {raw_angle:.1f}° | dist {total_dist:.1f}px"
        )
        print(f"   🏏 Shot {self.shots_detected} at "
              f"{peak_fd['timestamp_sec']}s | "
              f"angle {raw_angle:.1f}° | "
              f"wrist travel {total_dist:.1f}px")
        return shot_data

    def flush(self) -> dict | None:
        """Call after last frame to catch any open shot window."""
        if self.in_shot_window and self.wrist_positions:
            self.in_shot_window = False
            return self._finalize_shot(self.last_shot_frame + 50)
        return None

    def close(self):
        self.pose_detector.close()


# Keep ShotDetector as alias so pipeline.py works unchanged
ShotDetector = MotionShotDetector
