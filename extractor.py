import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from config import FRAME_SAMPLE_RATE, LOGS_DIR, SAVE_ANNOTATED_FRAMES

logging.basicConfig(
    filename=LOGS_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def extract_frames(video_path: str) -> list[dict]:
    """
    Extract frames from video at configured sample rate.
    Returns list of {frame_idx, timestamp_sec, frame (np.array)}
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0

    log.info(f"Video: {video_path.name} | Frames: {total_frames} | FPS: {fps:.1f} | Duration: {duration_sec:.1f}s")
    print(f"\n📹 Video: {video_path.name}")
    print(f"   Frames: {total_frames} | FPS: {fps:.1f} | Duration: {duration_sec:.1f}s")
    print(f"   Sampling every {FRAME_SAMPLE_RATE} frames → ~{total_frames // FRAME_SAMPLE_RATE} frames to process")

    frames = []
    frame_idx = 0

    with tqdm(total=total_frames // FRAME_SAMPLE_RATE, desc="   Extracting") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SAMPLE_RATE == 0:
                frames.append({
                    "frame_idx":     frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3) if fps > 0 else 0,
                    "frame":         frame
                })
                pbar.update(1)

            frame_idx += 1

    cap.release()
    log.info(f"Extracted {len(frames)} frames from {video_path.name}")
    print(f"   ✅ Extracted {len(frames)} frames")
    return frames


def get_video_metadata(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    meta = {
        "filename":     video_path.name,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration_sec"] = round(meta["total_frames"] / meta["fps"], 2) if meta["fps"] > 0 else 0
    cap.release()
    return meta
