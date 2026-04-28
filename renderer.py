
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from pathlib import Path
import logging
from config import WAGON_WHEEL_DIR, WAGON_WHEEL_DPI, SHOT_COLORS

log = logging.getLogger(__name__)

ZONE_LABELS = {
    "Fine Leg":   (140, 0.94),
    "Square Leg": (100, 0.94),
    "Mid Wicket": (60,  0.94),
    "Long On":    (25,  0.94),
    "Straight":   (0,   0.97),
    "Long Off":   (-25, 0.94),
    "Cover":      (-60, 0.94),
    "Point":      (-100,0.94),
    "Third Man":  (-140,0.94),
}

def _draw_field(ax, fig):
    outer_ring = Circle((0,0), 1.08, color="#111111", zorder=0)
    ax.add_patch(outer_ring)

    grass_colors = ["#2d7a2d","#318c31","#2d7a2d","#318c31","#2d7a2d","#318c31","#2d7a2d"]
    ring_edges   = [1.0, 0.86, 0.72, 0.58, 0.44, 0.30, 0.16, 0.0]
    for i in range(len(grass_colors)):
        ax.add_patch(Circle((0,0), ring_edges[i], color=grass_colors[i], zorder=1))

    ax.add_patch(Circle((0,0), 0.58, fill=False,
                         edgecolor="white", linewidth=1.2,
                         linestyle="--", zorder=4, alpha=0.6))
    ax.add_patch(Circle((0,0), 1.0, fill=False,
                         edgecolor="white", linewidth=2.5,
                         zorder=4, alpha=0.95))

    for a in [160,120,80,40,10,-10,-40,-80,-120,-160]:
        rad = np.radians(90 - a)
        ax.plot([0, 1.08*np.cos(rad)], [0, 1.08*np.sin(rad)],
                color="white", lw=0.5, alpha=0.3,
                linestyle="--", zorder=3)

    for zone, (angle, r) in ZONE_LABELS.items():
        rad = np.radians(90 - angle)
        x   = r * 1.08 * np.cos(rad)
        y   = r * 1.08 * np.sin(rad)
        rot = angle if -90 <= angle <= 90 else angle + 180
        ax.text(x, y, zone, color="white", fontsize=5.8,
                ha="center", va="center", fontweight="bold",
                rotation=rot, rotation_mode="anchor", zorder=10)

    pitch = plt.Rectangle((-0.045,-0.20), 0.09, 0.40,
                           color="#c8a96e", zorder=5, alpha=0.95)
    ax.add_patch(pitch)
    for y in [0.16, -0.16]:
        ax.plot([-0.065, 0.065], [y, y], color="white", lw=1.0, zorder=6)
    for x in [-0.012, 0, 0.012]:
        ax.plot([x,x], [ 0.16,  0.20], color="white", lw=1.8, zorder=7)
        ax.plot([x,x], [-0.16, -0.20], color="white", lw=1.8, zorder=7)
    ax.plot(0, 0, "o", color="white", markersize=4, zorder=8)


def _angle_to_xy(angle_deg, length):
    math_angle = np.radians(90 - angle_deg)
    return length * np.cos(math_angle), length * np.sin(math_angle)


def draw_wagon_wheel(shots, video_name, summary,
                     batsman_name="Batsman") -> Path:

    fig = plt.figure(figsize=(9, 9), facecolor="#111111")
    ax  = fig.add_subplot(111, aspect="equal")
    ax.set_facecolor("#111111")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")

    _draw_field(ax, fig)

    if not shots:
        ax.text(0, 0, "No shots detected", color="white",
                ha="center", va="center", fontsize=12)
    else:
        for shot in shots:
            angle  = shot.get("angle_deg", 0)
            length = min(shot.get("shot_length", 0.7), 1.0)
            color  = shot.get("color", "#5B9BD5")
            runs   = shot.get("runs", 0)
            x, y   = _angle_to_xy(angle, length)
            lw     = 1.5 + runs * 0.35
            ax.plot([0,x],[0,y], color=color, linewidth=lw+1.5,
                    alpha=0.25, solid_capstyle="round", zorder=8)
            ax.plot([0,x],[0,y], color=color, linewidth=lw,
                    alpha=0.92, solid_capstyle="round", zorder=9)
            ax.plot(x, y, "o", color=color,
                    markersize=3.0+runs*0.4, alpha=0.95, zorder=10)

    total_shots = summary.get("total_shots", len(shots))
    total_runs  = summary.get("total_runs", 0)
    leg_pct     = summary.get("leg_side_pct", 0)
    off_pct     = summary.get("off_side_pct", 0)

    fig.text(0.5, 0.97, batsman_name,
             color="white", fontsize=15, fontweight="bold",
             ha="center", va="top")
    fig.text(0.5, 0.935,
             f"Innings  |  {total_shots} Balls  |  {total_runs} Runs",
             color="#cccccc", fontsize=9, ha="center", va="top")

    for (label, val), px in zip(
        [("LEG SIDE",f"{leg_pct}%"),("OFF SIDE",f"{off_pct}%"),
         ("SHOTS",str(total_shots)),("RUNS",str(total_runs))],
        [0.18, 0.38, 0.62, 0.82]
    ):
        fig.text(px, 0.905, val, color="white", fontsize=11,
                 fontweight="bold", ha="center")
        fig.text(px, 0.885, label, color="#888888",
                 fontsize=6.5, ha="center")

    legend_items = [
        mpatches.Patch(color=SHOT_COLORS[0], label="Dot"),
        mpatches.Patch(color=SHOT_COLORS[1], label="1"),
        mpatches.Patch(color=SHOT_COLORS[2], label="2"),
        mpatches.Patch(color=SHOT_COLORS[3], label="3"),
        mpatches.Patch(color=SHOT_COLORS[4], label="4"),
        mpatches.Patch(color=SHOT_COLORS[6], label="6"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5,-0.07), ncol=6, fontsize=7.5,
              framealpha=0.25, facecolor="#222222",
              edgecolor="#555555", labelcolor="white", handlelength=1.5)

    zone_parts = "  |  ".join(
        f"{z}: {c}" for z,c in summary.get("zone_breakdown",{}).items()
    )
    fig.text(0.5, 0.055, zone_parts, color="#777777",
             fontsize=6.5, ha="center", va="bottom")

    out_path = WAGON_WHEEL_DIR / f"{Path(video_name).stem}_wagon_wheel.png"
    plt.tight_layout(rect=[0, 0.07, 1, 0.88])
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Wagon wheel saved → {out_path}")
    print(f"\n   🎯 Wagon wheel saved → {out_path}")
    return out_path
