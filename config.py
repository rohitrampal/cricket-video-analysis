import os
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
VIDEOS_DIR      = BASE_DIR / "videos"
OUTPUT_DIR      = BASE_DIR / "output"
WAGON_WHEEL_DIR = OUTPUT_DIR / "wagon_wheels"
JSON_DIR        = OUTPUT_DIR / "json"
LOGS_DIR        = BASE_DIR / "logs"
MODELS_DIR      = BASE_DIR / "models"

# ─── Batsman default facing
DEFAULT_FACING = "left"   # change per video

# ─── Device config (flip to 'cuda' on GPU server) ─────────
DEVICE          = "cpu"
BATCH_MODE      = "sequential"   # change to "concurrent" on server

# ─── Frame extraction ─────────────────────────────────────
FRAME_SAMPLE_RATE   = 5          # extract every Nth frame (CPU friendly)
MIN_SHOT_GAP_FRAMES = 30         # minimum frames between two shots

# ─── Detection thresholds ─────────────────────────────────
POSE_CONFIDENCE     = 0.6
BALL_CONFIDENCE     = 0.4
MIN_SHOT_ANGLE_CHANGE = 8       # degrees, to confirm a real shot

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
for d in [VIDEOS_DIR, WAGON_WHEEL_DIR, JSON_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
