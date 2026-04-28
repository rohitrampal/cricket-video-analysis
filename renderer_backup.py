import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for CPU/server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
from pathlib import Path
import logging
from config import (
    WAGON_WHEEL_DIR, WAGON_WHEEL_DPI,
    WAGON_WHEEL_SIZE, SHOT_COLORS, FIELD_ZONES
)

log = logging.getLogger(__name__)

# ─── Field zone label positions (angle in radians, radius) ─
ZONE_LABEL_POSITIONS = {
    "Fine Leg":   (np.radians(140),  0.88),
    "Square Leg": (np.radians(100),  0.88),
    "Mid Wicket": (np.radians(60),   0.88),
    "Long On":    (np.radians(25),   0.88),
    "Straight":   (np.radians(0),    0.92),
    "Long Off":   (np.radians(-25),  0.88),
    "Cover":      (np.radians(-60),  0.88),
    "Point":      (np.radians(-100), 0.88),
    "Third Man":  (np.radians(-140), 0.88),
}


def _draw_field(ax):
    """Draw the cricket field background — circles, pitch, zones."""

    # ── Outer boundary ────────────────────────────────────
    outer = plt.Circle((0, 0), 1.0,
                        color="#1a5c1a", zorder=0)
    ax.add_patch(outer)

    # ── 30-yard circle ────────────────────────────────────
    inner = plt.Circle((0, 0), 0.55,
                        color="#1f6e1f", zorder=1)
    ax.add_patch(inner)

    # Boundary ring
    boundary = plt.Circle((0, 0), 1.0, fill=False,
                           edgecolor="#ffffff", linewidth=2.5,
                           linestyle="-", zorder=5, alpha=0.9)
    ax.add_patch(boundary)

    # 30-yard ring
    ring30 = plt.Circle((0, 0), 0.55, fill=False,
                         edgecolor="#ffffff", linewidth=1.0,
                         linestyle="--", zorder=5, alpha=0.5)
    ax.add_patch(ring30)

    # ── Pitch rectangle ───────────────────────────────────
    pitch = plt.Rectangle((-0.05, -0.22), 0.10, 0.44,
                           color="#c8a96e", zorder=2, alpha=0.9)
    ax.add_patch(pitch)

    # Crease lines
    ax.plot([-0.07, 0.07], [ 0.18,  0.18],
            color="white", lw=1.0, zorder=3, alpha=0.8)
    ax.plot([-0.07, 0.07], [-0.18, -0.18],
            color="white", lw=1.0, zorder=3, alpha=0.8)

    # ── Stumps (3 lines) ─────────────────────────────────
    for x in [-0.015, 0, 0.015]:
        ax.plot([x, x], [0.18, 0.21],
                color="white", lw=1.5, zorder=4)
        ax.plot([x, x], [-0.18, -0.21],
                color="white", lw=1.5, zorder=4)

    # ── Zone divider lines ────────────────────────────────
    zone_angles = [
        160, 120, 80, 40, 10, -10, -40, -80, -120, -160
    ]
    for a in zone_angles:
        rad = np.radians(a)
        ax.plot([0, np.cos(rad)], [0, np.sin(rad)],
                color="white", lw=0.4, alpha=0.25,
                linestyle="--", zorder=3)

    # ── Zone labels ───────────────────────────────────────
    for zone, (rad, r) in ZONE_LABEL_POSITIONS.items():
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        ax.text(x, y, zone,
                color="white", fontsize=6.5,
                ha="center", va="center",
                fontweight="bold",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#000000",
                          alpha=0.45,
                          edgecolor="none"))

    # ── Batsman marker ────────────────────────────────────
    ax.plot(0, 0, "o",
            color="white", markersize=5,
            zorder=10, alpha=0.9)


def _angle_to_plot_coords(angle_deg: float, length: float):
    """
    Convert cricket angle + length to matplotlib x,y.

    Cricket angle convention:
      0°   = top of plot (straight)
      +90° = right (leg side)
      -90° = left (off side)

    We rotate so 0° points UP (north = straight drive).
    """
    # Rotate: cricket 0° = straight = top of circle = 90° in math
    math_angle = np.radians(90 - angle_deg)
    x = length * np.cos(math_angle)
    y = length * np.sin(math_angle)
    return x, y


def draw_wagon_wheel(shots: list[dict],
                     video_name: str,
                     summary: dict,
                     batsman_name: str = "Batsman") -> Path:
    """
    Render the wagon wheel for a list of analyzed shots.
    Saves PNG to WAGON_WHEEL_DIR and returns the file path.
    """
    fig, ax = plt.subplots(1, 1, figsize=WAGON_WHEEL_SIZE,
                           facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Draw field ────────────────────────────────────────
    _draw_field(ax)

    # ── Draw shot lines ───────────────────────────────────
    if not shots:
        ax.text(0, 0, "No shots detected",
                color="white", ha="center", va="center",
                fontsize=12)
    else:
        for shot in shots:
            angle  = shot.get("angle_deg", 0)
            length = shot.get("shot_length", 0.7)
            color  = shot.get("color", "#FFFFFF")
            runs   = shot.get("runs", 0)

            x, y = _angle_to_plot_coords(angle, length)

            # Line width scales with runs
            lw = 1.2 + runs * 0.3

            ax.plot([0, x], [0, y],
                    color=color,
                    linewidth=lw,
                    alpha=0.85,
                    solid_capstyle="round",
                    zorder=6)

            # Dot at end of shot line
            ax.plot(x, y, "o",
                    color=color,
                    markersize=2.5 + runs * 0.3,
                    zorder=7,
                    alpha=0.9)

    # ── Title ─────────────────────────────────────────────
    total_shots = summary.get("total_shots", len(shots))
    total_runs  = summary.get("total_runs", 0)
    leg_pct     = summary.get("leg_side_pct", 0)
    off_pct     = summary.get("off_side_pct", 0)

    fig.text(0.5, 0.96,
             f"{batsman_name} — Wagon Wheel",
             color="white", fontsize=13,
             fontweight="bold", ha="center", va="top")

    fig.text(0.5, 0.925,
             f"{total_shots} shots  |  {total_runs} runs  |  "
             f"Leg {leg_pct}%  •  Off {off_pct}%",
             color="#aaaaaa", fontsize=8.5,
             ha="center", va="top")

    # ── Legend ────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=SHOT_COLORS[0], label="Dot"),
        mpatches.Patch(color=SHOT_COLORS[1], label="1 run"),
        mpatches.Patch(color=SHOT_COLORS[2], label="2 runs"),
        mpatches.Patch(color=SHOT_COLORS[3], label="3 runs"),
        mpatches.Patch(color=SHOT_COLORS[4], label="4 runs"),
        mpatches.Patch(color=SHOT_COLORS[6], label="6 runs"),
    ]
    ax.legend(handles=legend_items,
              loc="lower center",
              bbox_to_anchor=(0.5, -0.08),
              ncol=6,
              fontsize=7,
              framealpha=0.3,
              facecolor="#111111",
              edgecolor="#444444",
              labelcolor="white")

    # ── Zone breakdown text ───────────────────────────────
    zone_text = "  ".join(
        f"{z[:4]}: {c}"
        for z, c in summary.get("zone_breakdown", {}).items()
    )
    fig.text(0.5, 0.04,
             zone_text,
             color="#888888", fontsize=6.5,
             ha="center", va="bottom")

    # ── Save ──────────────────────────────────────────────
    out_path = WAGON_WHEEL_DIR / f"{Path(video_name).stem}_wagon_wheel.png"
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(out_path, dpi=WAGON_WHEEL_DPI,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info(f"Wagon wheel saved → {out_path}")
    print(f"\n   🎯 Wagon wheel saved → {out_path}")
    return out_path
