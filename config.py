import os
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    pass
VIDEOS_DIR      = BASE_DIR / "videos"
OUTPUT_DIR      = BASE_DIR / "output"
WAGON_WHEEL_DIR = OUTPUT_DIR / "wagon_wheels"
JSON_DIR        = OUTPUT_DIR / "json"
LOGS_DIR        = BASE_DIR / "logs"
MODELS_DIR      = BASE_DIR / "models"
CALIBRATION_DIR = BASE_DIR / "calibration"
ANGLE_CALIBRATION_FILE = CALIBRATION_DIR / "angle_calibration.json"

# ─── Batsman default facing
DEFAULT_FACING = "left"   # change per video

# ─── Device config (flip to 'cuda' on GPU server) ─────────
DEVICE          = "cpu"
BATCH_MODE      = "sequential"   # change to "concurrent" on server

# ─── Frame extraction ─────────────────────────────────────
FRAME_SAMPLE_RATE   = 2          # extract every Nth frame (higher temporal fidelity)
MIN_SHOT_GAP_FRAMES = 12         # minimum frames between two shots

# ─── Detection thresholds ─────────────────────────────────
POSE_CONFIDENCE     = 0.6
BALL_CONFIDENCE     = 0.4
MIN_SHOT_ANGLE_CHANGE = 8       # degrees, to confirm a real shot

# ─── Ball detection ROI (HSV / candidate gating) ───────────
# Restricts ball search to near the batsman to cut sky/net/ground false positives.
BALL_ROI_ENABLED = True
BALL_ROI_POSE_MARGIN_FRAC = 0.18   # expand tight pose bbox by this × max(frame w, h)
BALL_ROI_MIN_WIDTH_FRAC = 0.28    # minimum ROI width as fraction of frame width
BALL_ROI_MIN_HEIGHT_FRAC = 0.30   # minimum ROI height as fraction of frame height
BALL_ROI_KEYPOINT_MIN_VIS = 0.22   # include landmark if visibility >= this
# Fallback when pose is missing or weak (normalized 0..1, inclusive-ish edges)
BALL_ROI_FALLBACK_X0 = 0.06
BALL_ROI_FALLBACK_X1 = 0.94
BALL_ROI_FALLBACK_Y0 = 0.10
BALL_ROI_FALLBACK_Y1 = 0.93
# Hard strip at top of frame (sky / sun / tree highlights); applied to every ROI.
BALL_ROI_EXCLUDE_TOP_FRAC = 0.22
# Intersect pose ROI with fallback so pose margin cannot reopen full sky / corners.
BALL_ROI_INTERSECT_FALLBACK = True

# ─── Ball YOLO (Ultralytics) ───────────────────────────────
# Default: project-local path under models/. If that file is missing, resolver falls
# back to the same basename on the Ultralytics hub (e.g. yolov8n.pt → auto-download).
# For a cricket-specific detector, replace with models/cricket_ball.pt (your weights).
# Override order: CLI --yolo-model → env BALL_YOLO_MODEL_PATH → BALL_YOLO_MODEL_PATH.
BALL_YOLO_MODEL_PATH = "models/yolov8n.pt"
BALL_YOLO_CONF = 0.25      # minimum detection confidence (Ultralytics predict threshold)
BALL_YOLO_IOU = 0.45       # NMS IoU during predict
BALL_YOLO_IMGSZ = 640     # inference size (set 0 to let Ultralytics auto)
# Hybrid tracking: subsample YOLO, high-confidence gate, temporal / prediction gates
BALL_YOLO_RUN_STRIDE = 3           # run YOLO every Nth frame in the tracking window (0-based index)
BALL_YOLO_STRONG_CONF = 0.4        # treat as “strong” YOLO only at or above this (after predict)
BALL_YOLO_PRED_GATE_PX = 72.0      # accept strong YOLO vs Kalman prediction (pixel distance, compensated)
BALL_YOLO_COHERENCE_PX = 88.0     # accept YOLO if consecutive YOLO runs agree within this distance
BALL_YOLO_INIT_SINGLE_CONF = 0.55  # without a second YOLO run, allow one-shot init if this confident


def _yolo_hub_weight_ref(raw: str) -> bool:
    """Bare Ultralytics weight id (e.g. yolov8n.pt) — no path separators."""
    s = str(raw).strip()
    return bool(s) and "/" not in s and "\\" not in s and s.lower().endswith(".pt")


def resolve_ball_yolo_model_path(cli_override: str | None = None) -> str | None:
    """
    Return a path to local YOLO weights, or a hub-style weight id (e.g. yolov8n.pt).
    Returns None if nothing is configured or local paths are missing.
    If cli_override is non-empty, only that value is tried.
    """
    candidates: list[str] = []
    if cli_override and str(cli_override).strip():
        candidates.append(str(cli_override).strip())
    else:
        env = os.environ.get("BALL_YOLO_MODEL_PATH", "").strip()
        if env:
            candidates.append(env)
        cfg = str(BALL_YOLO_MODEL_PATH or "").strip()
        if cfg:
            candidates.append(cfg)
    for raw in candidates:
        if _yolo_hub_weight_ref(raw):
            return raw.strip()
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.is_file():
            return str(p.resolve())
        # Prefer local models/<name>.pt; if missing, try hub id <name>.pt
        if p.parent == MODELS_DIR and p.suffix.lower() == ".pt" and _yolo_hub_weight_ref(p.name):
            return p.name
    return None


# ─── Cricket field zones (angle in degrees from straight) ──
# 0° = straight down the ground (long on/off axis)
# Positive = leg side, Negative = off side
FIELD_ZONES = {
    "Fine Leg":    (120,  160),
    "Square Leg":  (80,   120),
    "Mid Wicket":  (40,    80),
    "Long On":     (10,    40),
    "Straight":    (-10,   10),
    "Long Off":    (-40,  -10),
    "Cover":       (-80,  -40),
    "Point":       (-120, -80),
    "Third Man":   (-160,-120),
}

# ─── Wagon wheel visual settings ──────────────────────────
SHOT_COLORS = {
    0:  "#4FC3F7",   # dot ball  → light blue
    1:  "#FFFFFF",   # 1 run     → white
    2:  "#81C784",   # 2 runs    → green
    3:  "#FFD54F",   # 3 runs    → yellow
    4:  "#FF8A65",   # 4 runs    → orange
    6:  "#E040FB",   # 6 runs    → purple
}
DEFAULT_SHOT_COLOR = "#FFFFFF"

# ─── Output settings ──────────────────────────────────────
SAVE_ANNOTATED_FRAMES = False    # set True for debugging
WAGON_WHEEL_DPI       = 150
WAGON_WHEEL_SIZE      = (8, 8)   # inches

# Ensure all directories exist
for d in [VIDEOS_DIR, WAGON_WHEEL_DIR, JSON_DIR, LOGS_DIR, MODELS_DIR, CALIBRATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)
