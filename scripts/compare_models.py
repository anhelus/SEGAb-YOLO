"""Train + compare YOLOv26 models (baseline vs enhanced) or validate existing weights.

Typical usage — train and compare::

    python -m scripts.compare_models --data data/dataset.yaml --scales n s --epochs 50

Validate only::

    python -m scripts.compare_models --validate-only --weights \\
        runs/baseline_n/weights/best.pt runs/enhanced_n/weights/best.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from segab_yolo import YOLO


def _metrics_dict(model, imgsz: int, batch: int, device: str) -> dict:
    """Run validation and return a dict of scalar metrics.

    Args:
        model: Loaded YOLO model.
        imgsz: Inference image size.
        batch: Batch size.
        device: Device string.

    Returns:
        Dict with keys *map50*, *map50_95*, *map75*, *precision*,
        *recall*, *f1*.
    """
    metrics = model.val(
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=0.001,
        iou=0.7,
        plots=False,
        save_json=False,
        verbose=False,
    )
    box = metrics.box
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "map75": float(box.map75),
        "precision": (
            float(box.p[0])
            if hasattr(box, "p") and box.p is not None and len(box.p) > 0
            else None
        ),
        "recall": (
            float(box.r[0])
            if hasattr(box, "r") and box.r is not None and len(box.r) > 0
            else None
        ),
        "f1": (
            float(box.f1[0])
            if hasattr(box, "f1") and box.f1 is not None and len(box.f1) > 0
            else None
        ),
    }


def train_and_val(
    yaml_path: str,
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name_prefix: str,
    scale: str,
) -> dict:
    """Train a model from a YAML config and return validation metrics.

    Args:
        yaml_path: Path to the model YAML.
        data: Dataset YAML path.
        epochs: Number of training epochs.
        imgsz: Image size.
        batch: Batch size.
        device: Device string.
        project: Project directory.
        name_prefix: Experiment name prefix.
        scale: Model scale (n/s/m/l/x).

    Returns:
        Metric dict from :func:`_metrics_dict`.
    """
    name = f"{name_prefix}_{scale}"
    model = YOLO(str(yaml_path))
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        verbose=False,
        fraction=0.01,
    )
    return _metrics_dict(model, imgsz, batch, device)


def _fmt(val: Optional[float]) -> str:
    """Format an optional float for table display."""
    return f"{val:.4f}" if val is not None else ""


def _delta(a: Optional[float], b: Optional[float]) -> str:
    """Format the signed delta between two optional values."""
    if a is None or b is None:
        return ""
    return f"{b - a:<+8.4f}"


def print_table(entries: List[Tuple[str, str, Optional[dict]]], title: str = "COMPARISON TABLE") -> None:
    """Print a formatted comparison table.

    Args:
        entries: List of ``(group_label, model_label, metrics_dict_or_None)``.
        title: Table title.
    """
    print(f"\n\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")
    header = (
        f"{'Group':<12} {'Model':<20} {'mAP50':<9} {'mAP50-95':<11} "
        f"{'mAP75':<9} {'Prec':<8} {'Recall':<8} {'F1':<8}"
    )
    print(header)
    print("-" * 75)

    prev_group: Optional[str] = None
    prev_m: Optional[dict] = None

    for group, tag, m in entries:
        if group != prev_group:
            if prev_group is not None:
                print()
            prev_group = group

        if m is None:
            print(f"{group:<12} {tag:<20} {'ERROR':<9}")
        else:
            print(
                f"{group:<12} {tag:<20} {m['map50']:<9.4f} {m['map50_95']:<11.4f} "
                f"{m['map75']:<9.4f} {_fmt(m['precision']):<8} "
                f"{_fmt(m['recall']):<8} {_fmt(m['f1']):<8}"
            )

        # Delta line
        if m is not None and prev_m is not None:
            print(
                f"{'':<12} {'Δ':<20} {m['map50'] - prev_m['map50']:<+9.4f} "
                f"{m['map50_95'] - prev_m['map50_95']:<+11.4f} "
                f"{m['map75'] - prev_m['map75']:<+9.4f} "
                f"{_delta(prev_m['precision'], m['precision']):<8} "
                f"{_delta(prev_m['recall'], m['recall']):<8} "
                f"{_delta(prev_m['f1'], m['f1']):<8}"
            )
        prev_m = m


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train + compare YOLOv26 models, or validate already-trained weights"
    )
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip training; evaluate existing weights",
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        help="Paths to trained .pt files (requires --validate-only)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Display names for --weights (default: filenames)",
    )
    parser.add_argument(
        "--baseline-yaml",
        type=str,
        default="yolo26{scale}.yaml",
        help="Baseline YAML pattern, {scale} replaced with n/s/m/l/x",
    )
    parser.add_argument(
        "--enhanced-yaml",
        type=str,
        default="yolo26{scale}-mod.yaml",
        help="Enhanced YAML pattern, {scale} replaced with n/s/m/l/x",
    )
    parser.add_argument(
        "--scales",
        type=str,
        nargs="+",
        default=["n", "s", "m", "l", "x"],
        help="Scales to train and evaluate (ignored with --validate-only)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/compare",
        help="Project directory for training runs",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results JSON (optional)"
    )
    return parser.parse_args()


def _validate_only(args: argparse.Namespace) -> None:
    """Validate already-trained weights and print a comparison table."""
    if args.labels and len(args.labels) != len(args.weights):
        raise ValueError("--labels count must match --weights count")

    labels = args.labels or [Path(w).stem for w in args.weights]
    entries: list = []
    results: Dict = {"data": args.data, "weights": {}}

    for label, w in zip(labels, args.weights):
        print(f"\n  --- {label} ({w}) ---")
        try:
            model = YOLO(str(w))
            m = _metrics_dict(model, args.imgsz, args.batch, args.device)
            entries.append(("", label, m))
            results["weights"][label] = m
            print(f"  mAP50={m['map50']:.4f}  mAP50-95={m['map50_95']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            entries.append(("", label, None))
            results["weights"][label] = None

    print_table(entries, title="VALIDATION COMPARISON")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


def _train_and_compare(args: argparse.Namespace) -> None:
    """Train baseline and enhanced models for each scale and compare."""
    results: Dict = {"data": args.data, "epochs": args.epochs, "scales": {}}
    entries: list = []

    for scale in args.scales:
        print(f"\n{'=' * 60}")
        print(f"  Scale: {scale}")
        print(f"{'=' * 60}")

        for label, yaml_pat in [
            ("baseline", args.baseline_yaml),
            ("enhanced", args.enhanced_yaml),
        ]:
            yaml_path = yaml_pat.replace("{scale}", scale)
            print(f"\n  --- {label} ({yaml_path}) ---")
            try:
                m = train_and_val(
                    yaml_path,
                    args.data,
                    args.epochs,
                    args.imgsz,
                    args.batch,
                    args.device,
                    args.project,
                    label,
                    scale,
                )
                results["scales"].setdefault(scale, {})[label] = m
                entries.append((scale, label, m))
                print(f"  mAP50={m['map50']:.4f}  mAP50-95={m['map50_95']:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results["scales"].setdefault(scale, {})[label] = None
                entries.append((scale, label, None))

    print_table(entries, title="TRAIN + COMPARE TABLE")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


def main() -> None:
    """Entry point: dispatch to validate-only or train + compare."""
    args = parse_args()
    if args.validate_only:
        if not args.weights:
            raise ValueError("--validate-only requires --weights")
        _validate_only(args)
    else:
        _train_and_compare(args)


if __name__ == "__main__":
    main()
