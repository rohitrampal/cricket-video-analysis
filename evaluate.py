import argparse
import json
from pathlib import Path


MATCH_TOLERANCE_SEC = 0.6


def circular_diff(a: float, b: float) -> float:
    """Return shortest absolute angle difference in degrees."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _load_json(path: str) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_predictions(pred_json_path: str) -> list[dict]:
    data = _load_json(pred_json_path)
    if isinstance(data, dict) and "shots" in data and isinstance(data["shots"], list):
        return data["shots"]
    if isinstance(data, list):
        return data
    raise ValueError("Prediction JSON must be a result object with 'shots' or a shots list.")


def _load_ground_truth(gt_path: str) -> list[dict]:
    gt = _load_json(gt_path)
    if not isinstance(gt, list):
        raise ValueError("Ground-truth file must contain a list of shots.")
    for i, item in enumerate(gt):
        if not isinstance(item, dict):
            raise ValueError(f"Ground-truth item #{i} is not an object.")
        for key in ("time", "angle", "zone"):
            if key not in item:
                raise ValueError(f"Ground-truth item #{i} missing required key: '{key}'")
    return gt


def _best_match(gt_shot: dict, predictions: list[dict], used_pred_indices: set[int]) -> tuple[int | None, dict | None]:
    best_idx = None
    best_pred = None
    best_time_diff = float("inf")
    gt_t = float(gt_shot["time"])

    for idx, pred in enumerate(predictions):
        if idx in used_pred_indices:
            continue
        pred_t = float(pred.get("timestamp_sec", -1))
        dt = abs(pred_t - gt_t)
        if dt <= MATCH_TOLERANCE_SEC and dt < best_time_diff:
            best_time_diff = dt
            best_idx = idx
            best_pred = pred
    return best_idx, best_pred


def evaluate(pred_json_path: str, ground_truth: list[dict]) -> dict:
    predictions = _load_predictions(pred_json_path)
    used_pred_indices = set()
    rows = []

    total_gt = len(ground_truth)
    matched = 0
    zone_hits = 0
    angle_errors = []

    for gt in ground_truth:
        pred_idx, pred = _best_match(gt, predictions, used_pred_indices)
        if pred is None:
            rows.append({
                "gt_time": float(gt["time"]),
                "gt_angle": float(gt["angle"]),
                "gt_zone": str(gt["zone"]),
                "pred_time": None,
                "pred_angle": None,
                "pred_zone": None,
                "angle_error": None,
                "zone_match": False,
                "confidence_score": None,
                "matched": False,
            })
            continue

        used_pred_indices.add(pred_idx)
        matched += 1

        gt_angle = float(gt["angle"])
        pred_angle = float(pred.get("angle_deg", pred.get("raw_angle_deg", 0.0)))
        angle_err = circular_diff(pred_angle, gt_angle)
        angle_errors.append(angle_err)

        gt_zone = str(gt["zone"])
        pred_zone = str(pred.get("field_zone", "Unknown"))
        zone_match = pred_zone == gt_zone
        if zone_match:
            zone_hits += 1

        rows.append({
            "gt_time": float(gt["time"]),
            "gt_angle": gt_angle,
            "gt_zone": gt_zone,
            "pred_time": float(pred.get("timestamp_sec", 0.0)),
            "pred_angle": pred_angle,
            "pred_zone": pred_zone,
            "angle_error": angle_err,
            "zone_match": zone_match,
            "confidence_score": pred.get("confidence_score"),
            "matched": True,
        })

    avg_angle_error = sum(angle_errors) / len(angle_errors) if angle_errors else None
    zone_accuracy = (zone_hits / matched) if matched else 0.0
    detection_recall = (matched / total_gt) if total_gt else 0.0

    return {
        "matched_shots": matched,
        "total_ground_truth": total_gt,
        "total_predictions": len(predictions),
        "avg_angle_error_deg": avg_angle_error,
        "zone_accuracy": zone_accuracy,
        "shot_detection_recall": detection_recall,
        "details": rows,
    }


def _print_report(report: dict):
    print("\n=== Cricket Shot Evaluation ===")
    print(f"Matched Shots:        {report['matched_shots']}/{report['total_ground_truth']}")
    print(f"Total Predictions:    {report['total_predictions']}")
    if report["avg_angle_error_deg"] is None:
        print("Avg Angle Error (°):  N/A")
    else:
        print(f"Avg Angle Error (°):  {report['avg_angle_error_deg']:.2f}")
    print(f"Zone Accuracy:        {report['zone_accuracy'] * 100:.1f}%")
    print(f"Detection Recall:     {report['shot_detection_recall'] * 100:.1f}%")

    print("\nShot | GT Angle | Pred Angle | Error | Zone Match | Confidence")
    print("-" * 66)
    for i, row in enumerate(report["details"], start=1):
        if not row["matched"]:
            print(f"{i:>4} | {row['gt_angle']:>8.1f} | {'-':>10} | {'-':>5} | {'MISS':>10} | {'-':>10}")
            continue
        conf = row["confidence_score"]
        conf_txt = f"{float(conf):.3f}" if conf is not None else "N/A"
        zone_txt = "YES" if row["zone_match"] else "NO"
        print(
            f"{i:>4} | "
            f"{row['gt_angle']:>8.1f} | "
            f"{row['pred_angle']:>10.1f} | "
            f"{row['angle_error']:>5.1f} | "
            f"{zone_txt:>10} | "
            f"{conf_txt:>10}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate cricket shot prediction JSON against ground truth.")
    parser.add_argument("--json", required=True, help="Path to prediction JSON (pipeline result file).")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON list.")
    parser.add_argument("--out", default="", help="Optional output path to save evaluation report JSON.")
    args = parser.parse_args()

    pred_path = Path(args.json)
    gt_path = Path(args.gt)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_path}")

    gt = _load_ground_truth(str(gt_path))
    report = evaluate(str(pred_path), gt)
    _print_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report JSON: {out_path}")


if __name__ == "__main__":
    main()
