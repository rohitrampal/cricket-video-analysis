import argparse
import json
from pathlib import Path

from angle_calibration import AngleCalibrator, fit_params, zone_to_center_angle, wrap_angle
from config import ANGLE_CALIBRATION_FILE, JSON_DIR
from shot_analyzer import get_field_zone


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_samples(gt_path: Path, pred_dir: Path) -> list[dict]:
    gt = load_json(gt_path)
    if not isinstance(gt, list):
        raise ValueError("Ground truth must be a list of objects.")
    samples: list[dict] = []
    for item in gt:
        video = str(item["video"])
        shot_id = int(item["shot_id"])
        zone = str(item["zone"])
        pred_json = pred_dir / f"{Path(video).stem}_results.json"
        if not pred_json.exists():
            continue
        pred = load_json(pred_json)
        shots = pred.get("shots", [])
        shot = next((s for s in shots if int(s.get("shot_id", -1)) == shot_id), None)
        if shot is None:
            continue
        samples.append(
            {
                "video": video,
                "shot_id": shot_id,
                "source": str(shot.get("direction_source", "")).lower(),
                "pred_angle": float(shot.get("angle_deg", shot.get("raw_angle_deg", 0.0))),
                "target_zone": zone,
                "target_angle": float(zone_to_center_angle(zone)),
            }
        )
    return samples


def zone_accuracy(samples: list[dict], calibrator: AngleCalibrator | None = None) -> float:
    if not samples:
        return 0.0
    hits = 0
    for s in samples:
        angle = float(s["pred_angle"])
        source = str(s["source"])
        if calibrator is not None and calibrator.enabled:
            angle = calibrator.apply(angle, source=source)
        pred_zone = get_field_zone(wrap_angle(angle))
        if pred_zone == str(s["target_zone"]):
            hits += 1
    return float(hits / len(samples))


def fit_calibrator(samples: list[dict], min_source_samples: int = 3) -> AngleCalibrator:
    global_params = fit_params(samples)
    by_source = {}
    source_groups: dict[str, list[dict]] = {}
    for s in samples:
        source_groups.setdefault(str(s["source"]), []).append(s)
    for source, group in source_groups.items():
        if len(group) >= min_source_samples:
            by_source[source] = fit_params(group)
    return AngleCalibrator(enabled=True, global_params=global_params, by_source=by_source)


def main():
    parser = argparse.ArgumentParser(description="Fit generic angle calibration (mirror + offset).")
    parser.add_argument("--gt", required=True, help="Path to GT shots JSON.")
    parser.add_argument("--pred-dir", default=str(JSON_DIR), help="Directory containing <video>_results.json files.")
    parser.add_argument("--out", default=str(ANGLE_CALIBRATION_FILE), help="Output calibration JSON path.")
    parser.add_argument("--min-source-samples", type=int, default=3, help="Min samples for per-source calibration.")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_dir = Path(args.pred_dir)
    out_path = Path(args.out)

    samples = build_samples(gt_path, pred_dir)
    if not samples:
        raise ValueError("No calibration samples found. Check GT and prediction paths.")

    before_acc = zone_accuracy(samples, calibrator=None)
    calibrator = fit_calibrator(samples, min_source_samples=int(args.min_source_samples))
    after_acc = zone_accuracy(samples, calibrator=calibrator)

    metadata = {
        "num_samples": len(samples),
        "before_zone_accuracy": round(before_acc, 4),
        "after_zone_accuracy": round(after_acc, 4),
        "min_source_samples": int(args.min_source_samples),
        "sources": sorted(list({str(s["source"]) for s in samples})),
    }
    calibrator.save(path=out_path, metadata=metadata)

    print("Calibration saved:", out_path)
    print("Samples:", len(samples))
    print("Zone accuracy before:", f"{before_acc * 100:.1f}%")
    print("Zone accuracy after: ", f"{after_acc * 100:.1f}%")
    print("Global params:", calibrator.global_params.to_dict())
    if calibrator.by_source:
        print("Per-source params:")
        for k, v in calibrator.by_source.items():
            print(f"  {k}: {v.to_dict()}")


if __name__ == "__main__":
    main()

