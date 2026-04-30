import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse
import matplotlib.patheffects as pe
from pathlib import Path
import logging
from config import WAGON_WHEEL_DIR, SHOT_COLORS

log = logging.getLogger(__name__)

ZONE_LABELS = [
    ( 140, 1.13, "Fine Leg"),
    ( 100, 1.13, "Square Leg"),
    (  60, 1.13, "Mid Wicket"),
    (  20, 1.13, "Long On"),
    (   0, 1.16, "Straight"),
    ( -20, 1.13, "Long Off"),
    ( -60, 1.13, "Cover"),
    (-100, 1.13, "Point"),
    (-140, 1.13, "Third Man"),
]

def cricket_to_xy(angle_deg, radius=1.0):
    # Angle mapping for wagon wheel rendering.
    math_rad = np.radians(90 + angle_deg)
    return radius * np.cos(math_rad), radius * np.sin(math_rad)

def draw_3d_field(ax, fig):
    ax.add_patch(Circle((0,0), 1.22, color="#0a0a0a", zorder=0))
    ax.add_patch(Ellipse((0.03,-0.06), 2.08, 1.85,
                          color="#000000", alpha=0.35, zorder=1))

    grass = ["#1e6b1e","#237023","#1e6b1e","#237023",
             "#1e6b1e","#237023","#1e6b1e","#237023"]
    edges = [1.0,0.875,0.75,0.625,0.50,0.375,0.25,0.125,0.0]
    for c, r in zip(grass, edges):
        ax.add_patch(Circle((0,0), r, color=c, zorder=2))

    for r, alpha, lw in [(1.0,0.9,3.0),(1.0,0.3,6.0)]:
        ax.add_patch(Circle((0,0), r, fill=False,
                            edgecolor="white", lw=lw,
                            alpha=alpha, zorder=5))

    ax.add_patch(Circle((0,0), 0.58, fill=False,
                         edgecolor="white", lw=1.0,
                         linestyle="--", alpha=0.45, zorder=5))

    for angle in [160,120,80,40,10,-10,-40,-80,-120,-160]:
        x, y = cricket_to_xy(angle, 1.0)
        ax.plot([0,x],[0,y], color="white", lw=0.4,
                alpha=0.2, linestyle="--", zorder=4)

    ax.add_patch(Circle((0,0), 1.22, fill=False,
                         edgecolor="#111111", lw=18, zorder=6))

    for angle, r, label in ZONE_LABELS:
        x, y = cricket_to_xy(angle, r)
        ax.text(x, y, label, color="white", fontsize=6.2,
                ha="center", va="center", fontweight="bold",
                rotation=0, zorder=12,
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground="#111111")])

    ax.add_patch(plt.Polygon(
        [[-0.042,-0.21],[0.058,-0.21],[0.058,0.21],[-0.042,0.21]],
        color="black", alpha=0.4, zorder=6))

    ax.add_patch(plt.Rectangle((-0.048,-0.205), 0.096, 0.41,
                                facecolor="#c8a96e", zorder=7,
                                linewidth=1, edgecolor="#b89a5e"))
    ax.fill_between([-0.048,0.048],[0.205,0.205],[0.215,0.215],
                    color="#dfc07e", zorder=8)

    for yc in [0.155,-0.155]:
        ax.plot([-0.065,0.065],[yc,yc],
                color="white", lw=1.2, zorder=9, alpha=0.9)

    stump_colors = ["#d4a853","#e8c070","#d4a853"]
    for i, sx in enumerate([-0.014,0,0.014]):
        ax.plot([sx,sx],[ 0.155, 0.21], color=stump_colors[i],
                lw=2.2, zorder=10, solid_capstyle="round")
        ax.plot([sx,sx],[-0.155,-0.21], color=stump_colors[i],
                lw=2.2, zorder=10, solid_capstyle="round")
    ax.plot([-0.014,0.014],[0.21,0.21],  color="#f0d080", lw=1.5, zorder=11)
    ax.plot([-0.014,0.014],[-0.21,-0.21],color="#f0d080", lw=1.5, zorder=11)
    ax.plot(0, 0, "o", color="white", markersize=5, zorder=12, alpha=0.95,
            path_effects=[pe.withStroke(linewidth=3, foreground="#000")])


def draw_wagon_wheel(shots, video_name, summary,
                     batsman_name="Batsman") -> Path:
    fig = plt.figure(figsize=(10,11), facecolor="#0d0d0d")
    ax  = fig.add_axes([0.05,0.08,0.90,0.78], aspect="equal")
    ax.set_facecolor("#0d0d0d")
    ax.set_xlim(-1.28,1.28)
    ax.set_ylim(-1.28,1.28)
    ax.axis("off")

    draw_3d_field(ax, fig)

    for shot in shots:
        angle  = shot.get("angle_deg", 0)
        length = min(shot.get("shot_length", 0.7), 1.0)
        color  = shot.get("color", "#4FC3F7")
        runs   = shot.get("runs", 0)
        x, y   = cricket_to_xy(angle, length)
        lw     = 1.6 + runs * 0.4
        ax.plot([0,x],[0,y], color="black",   lw=lw+3,   alpha=0.3,
                solid_capstyle="round", zorder=13)
        ax.plot([0,x],[0,y], color=color,     lw=lw+2.5, alpha=0.18,
                solid_capstyle="round", zorder=14)
        ax.plot([0,x],[0,y], color=color,     lw=lw,     alpha=0.95,
                solid_capstyle="round", zorder=15)
        ax.plot(x, y, "o",  color=color,
                markersize=3.5+runs*0.5, zorder=16, alpha=1.0,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    total_shots = summary.get("total_shots", len(shots))
    total_runs  = summary.get("total_runs", 0)
    leg_pct     = summary.get("leg_side_pct", 0)
    off_pct     = summary.get("off_side_pct", 0)

    fig.text(0.5, 0.985, batsman_name,
             color="white", fontsize=17, fontweight="bold",
             ha="center", va="top",
             path_effects=[pe.withStroke(linewidth=3, foreground="#333")])
    fig.text(0.5, 0.952,
             f"Innings Analysis  ·  {total_shots} Deliveries  ·  {total_runs} Runs",
             color="#aaaaaa", fontsize=9, ha="center", va="top")

    stats = [("LEG SIDE",f"{leg_pct}%","#4FC3F7"),
             ("OFF SIDE", f"{off_pct}%","#FF8A65"),
             ("SHOTS",    str(total_shots),"#81C784"),
             ("RUNS",     str(total_runs),"#FFD54F")]
    for i,(label,val,col) in enumerate(stats):
        px = 0.12 + i*0.25
        fig.patches.append(mpatches.FancyBboxPatch(
            (px-0.09,0.908),0.18,0.034,
            boxstyle="round,pad=0.005",
            facecolor="#1a1a1a", edgecolor=col,
            linewidth=1.0, transform=fig.transFigure, zorder=20))
        fig.text(px, 0.934, val,   color=col,     fontsize=12,
                 fontweight="bold", ha="center", va="top")
        fig.text(px, 0.916, label, color="#777777", fontsize=6.5,
                 ha="center", va="top")

    legend_items = [
        mpatches.Patch(facecolor=SHOT_COLORS[c], edgecolor="#555",
                       label=l)
        for c,l in [(0,"Dot"),(1,"1 run"),(2,"2 runs"),
                    (3,"3 runs"),(4,"4 runs"),(6,"6 runs")]
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5,-0.10), ncol=6, fontsize=7.5,
              framealpha=0.3, facecolor="#111111",
              edgecolor="#444444", labelcolor="white", handlelength=1.8)

    zone_str = "   ".join(
        f"{z}: {c}" for z,c in summary.get("zone_breakdown",{}).items()
    )
    fig.text(0.5, 0.035, zone_str, color="#666666",
             fontsize=6.5, ha="center", va="bottom")

    out_path = WAGON_WHEEL_DIR / f"{Path(video_name).stem}_wagon_wheel.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Wagon wheel saved → {out_path}")
    print(f"\n   🎯 Wagon wheel saved → {out_path}")
    return out_path
