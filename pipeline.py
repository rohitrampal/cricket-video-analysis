"""
Cricket Wagon Wheel Analysis Pipeline
======================================
Usage:
    python pipeline.py --video videos/match1.mp4
    python pipeline.py --video videos/match1.mp4 --batsman "Virat Kohli" --facing right
    python pipeline.py --all          ← process all videos in videos/

Runs one video at a time (CPU safe).
Server-ready: set BATCH_MODE=concurrent in config.py for GPU server.
"""

import argparse
import json
import time
import logging
from pathlib import Path

from config      import (VIDEOS_DIR, JSON_DIR, BATCH_MODE,
                          MIN_SHOT_GAP_FRAMES)
from extractor   import extract_frames, get_video_metadata
from detector    import ShotDetector
from shot_analyzer import analyze_shot, summarize_innings
from renderer    import draw_wagon_wheel
from ball_tracker import BallTracker

log = logging.getLogger(__name__)


def process_video(video_path: str,
                  batsman_name: str = "Batsman",
                  batsman_facing: str = "right",
                  runs_map: dict = None) -> dict:
    """
    Full pipeline for one video.

    Args:
        video_path:     path to .mp4 / .mov file
        batsman_name:   display name for the wagon wheel title
        batsman_facing: 'right' or 'left' (which way batsman faces camera)
        runs_map:       optional dict {shot_id: runs} to assign run values
                        e.g. {1: 4, 2: 0, 3: 6}  — if None, all shots = 0

    Returns:
        result dict with shots, summary, output paths
    """
    video_path = Path(video_path)
    runs_map   = runs_map or {}

    print(f"\n{'='*55}")
    print(f"  Processing: {video_path.name}")
    print(f"  Batsman:    {batsman_name}")
    print(f"  Facing:     {batsman_facing}")
    print(f"{'='*55}")

    start_time = time.time()

    # ── Step 1: Extract frames ────────────────────────────
    print("\n[1/4] Extracting frames...")
    frames = extract_frames(str(video_path))

    # ── Step 2: Detect shots ──────────────────────────────
    print("\n[2/4] Detecting shots...")
    detector     = ShotDetector()
    raw_shots    = []

    for frame_data in frames:
        shot = detector.process_frame(
            frame_data,
            min_gap=MIN_SHOT_GAP_FRAMES
        )
        if shot:
            raw_shots.append(shot)
            print(f"   → Shot {shot['shot_id']} at "
                  f"{shot['timestamp_sec']}s | "
                  f"angle {shot['raw_angle_deg']}°")

    # detector.close()
    # flush any open shot window at end of video
    final = detector.flush()
    if final:
        raw_shots.append(final)
        print(f"   → Shot {final['shot_id']} (flushed) at {final['timestamp_sec']}s | angle {final['raw_angle_deg']}°")
    detector.close()
    print(f"   ✅ {len(raw_shots)} shots detected")

    # ── Step 3: Analyze shots ─────────────────────────────
    print("\n[3/4] Estimating ball-based directions...")
    tracker = BallTracker(debug_dir="output/ball_debug")
    for raw in raw_shots:
        direction = tracker.estimate_shot_direction(
            frames=frames,
            impact_frame_idx=int(raw["frame_idx"]),
            shot_id=int(raw["shot_id"]),
            video_stem=video_path.stem,
        )
        raw["raw_angle_deg"] = float(direction["angle"])
        raw["direction_source"] = direction["source"]
        raw["direction_confidence"] = direction["confidence"]
        raw["ball_detections"] = int(direction["ball_detections"])
        raw["ball_debug_image"] = direction["debug_image"]
        print(
            f"   Shot {raw['shot_id']:>2} | impact {raw['frame_idx']:>5} | "
            f"ball_pts {raw['ball_detections']:>2} | angle {raw['raw_angle_deg']:>7.1f}° | "
            f"source {raw['direction_source']:<4} | conf {raw['direction_confidence']}"
        )
    tracker.close()

    print("\n[4/4] Analyzing shot directions...")
    analyzed_shots = []

    for raw in raw_shots:
        runs    = runs_map.get(raw["shot_id"], 0)
        enriched = analyze_shot(raw, runs=runs,
                                batsman_facing=batsman_facing)
        analyzed_shots.append(enriched)
        confidence = float(enriched.get("confidence_score", 0.0))
        print(f"   {enriched['shot_id']} | {enriched['angle_deg']:.1f} | "
              f"{enriched['field_zone']} | {confidence:.3f}")
        print(f"   Shot {enriched['shot_id']:>2} | "
              f"{enriched['field_zone']:<12} | "
              f"{enriched['angle_deg']:>7.1f}° | "
              f"{enriched['shot_type']}")

    # ── Step 4: Render wagon wheel ────────────────────────
    print("\n[5/5] Rendering wagon wheel...")
    summary   = summarize_innings(analyzed_shots)
    wheel_path = draw_wagon_wheel(
        shots        = analyzed_shots,
        video_name   = video_path.name,
        summary      = summary,
        batsman_name = batsman_name
    )

    # ── Save JSON results ─────────────────────────────────
    result = {
        "video":        video_path.name,
        "batsman":      batsman_name,
        "facing":       batsman_facing,
        "shots":        analyzed_shots,
        "summary":      summary,
        "wagon_wheel":  str(wheel_path),
    }

    json_path = JSON_DIR / f"{video_path.stem}_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = round(time.time() - start_time, 1)
    print(f"\n{'='*55}")
    print(f"  ✅ Done in {elapsed}s")
    print(f"  📊 Shots:      {summary.get('total_shots', 0)}")
    print(f"  🏏 Zones:      {summary.get('zone_breakdown', {})}")
    print(f"  📁 JSON:       {json_path}")
    print(f"  🖼️  Wagon Wheel: {wheel_path}")
    print(f"{'='*55}\n")

    return result


def process_all_videos(batsman_name: str = "Batsman",
                       batsman_facing: str = "right") -> list[dict]:
    """Process every video in the videos/ directory sequentially."""
    videos = list(VIDEOS_DIR.glob("*.mp4")) + \
             list(VIDEOS_DIR.glob("*.mov")) + \
             list(VIDEOS_DIR.glob("*.avi"))

    if not videos:
        print(f"❌ No videos found in {VIDEOS_DIR}")
        print(f"   Drop your .mp4 / .mov files into the videos/ folder")
        return []

    print(f"\n🎬 Found {len(videos)} video(s) to process")
    results = []

    for i, v in enumerate(sorted(videos), 1):
        print(f"\n── Video {i}/{len(videos)} ──")
        try:
            r = process_video(
                video_path     = str(v),
                batsman_name   = batsman_name,
                batsman_facing = batsman_facing
            )
            results.append(r)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            log.error(f"Failed on {v.name}: {e}", exc_info=True)

    print(f"\n🏁 All done. Processed {len(results)}/{len(videos)} videos.")
    return results


# ─── CLI entry point ──────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cricket Wagon Wheel Analyzer"
    )
    parser.add_argument("--video",   type=str,
                        help="Path to a single video file")
    parser.add_argument("--all",     action="store_true",
                        help="Process all videos in videos/ folder")
    parser.add_argument("--batsman", type=str, default="Batsman",
                        help="Batsman display name")
    parser.add_argument("--facing",  type=str, default="right",
                        choices=["right", "left"],
                        help="Direction batsman faces on screen")

    args = parser.parse_args()

    if args.all:
        process_all_videos(
            batsman_name   = args.batsman,
            batsman_facing = args.facing
        )
    elif args.video:
        process_video(
            video_path     = args.video,
            batsman_name   = args.batsman,
            batsman_facing = args.facing
        )
    else:
        print("Usage:")
        print("  python pipeline.py --video videos/match1.mp4")
        print("  python pipeline.py --all --batsman 'Player Name'")
