"""Evaluate a trained YOLO model on corrupted validation data.

For each corruption type the validation images are corrupted on-the-fly and
metrics are collected.  A delta table and JSON result file are produced.

Typical usage::

    python -m scripts.evaluate_corruptions --model runs/train/weights/best.pt \\
        --data data/lettuce/dataset.yaml --severity 2
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from segab_yolo import YOLO

from segab_yolo.utils.corruptions import (
    SEVERITY_PARAMS,
    apply_corruption,
    image_generator,
)

# Re-export function map for CLI choices
CORRUPTION_NAMES = [
    "darken",
    "blur",
    "motion_blur",
    "noise",
    "occlusion",
    "salt_pepper",
    "brightness",
    "jpeg",
    "hue",
]


def corrupt_split(
    source_dir: Path,
    output_dir: Path,
    corruption: str,
    severity: int,
    ext: Optional[str] = None,
    label_dir: Optional[Path] = None,
) -> Path:
    """Apply a single corruption to all images in *source_dir*.

    Args:
        source_dir: Directory containing the original images.
        output_dir: Root directory for the corrupted output.
        corruption: Corruption name (key in :data:`CORRUPTIONS`).
        severity: Severity level (1-3).
        ext: Optional image extension filter.
        label_dir: Optional labels directory; labels are copied alongside
            images when provided.

    Returns:
        Path to the output images directory.
    """
    out_img_dir = output_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    if label_dir:
        out_label_dir = output_dir / "labels"
        out_label_dir.mkdir(parents=True, exist_ok=True)

    sp = SEVERITY_PARAMS[severity]
    images = list(image_generator(str(source_dir), ext))

    for img_path in tqdm(images, desc=f"  Corrupting {corruption}", leave=False):
        rel = (
            img_path.relative_to(source_dir)
            if source_dir.is_dir()
            else img_path.name
        )
        out_path = out_img_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = apply_corruption(img, corruption, sp)
        cv2.imwrite(str(out_path), img)

        if label_dir:
            label_src = label_dir / img_path.with_suffix(".txt").name
            if label_src.exists():
                shutil.copy2(label_src, out_label_dir / img_path.with_suffix(".txt").name)

    return out_img_dir


def make_val_yaml(
    cfg: dict, new_img_dir: Path, suffix: str, orig_yaml_path: Path, output_dir: Path
) -> Path:
    """Create a temporary YAML pointing *val* to *new_img_dir*.

    Args:
        cfg: Original dataset configuration dict.
        new_img_dir: New validation image directory.
        suffix: Suffix for the temporary YAML filename.
        orig_yaml_path: Path to the original dataset YAML.
        output_dir: Directory for the temporary YAML.

    Returns:
        Path to the newly created YAML file.
    """
    cfg = cfg.copy()
    cfg["val"] = str(new_img_dir.resolve())
    new_yaml_path = output_dir / f"{orig_yaml_path.stem}_{suffix}.yaml"
    with open(new_yaml_path, "w") as f:
        yaml.dump(cfg, f)
    return new_yaml_path


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on corrupted datasets."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Trained YOLO model (.pt)"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Original dataset YAML (val split used)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/corrupt_eval",
        help="Base output directory",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default=CORRUPTION_NAMES,
        choices=CORRUPTION_NAMES,
        help="Corruption types to evaluate",
    )
    parser.add_argument("--severity", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument(
        "--ext", type=str, default=None, help="Image extension filter"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep corrupted images after evaluation",
    )
    return parser.parse_args()


def print_results(results: Dict[str, dict], corruptions):
    """Print a formatted delta comparison table.

    Args:
        results: Dict mapping label (``'baseline'`` or corruption name) to
            metric dict with keys *map50*, *map50_95*, *map75*.
        corruptions: List of corruption names evaluated.
    """
    print(f"\n\n{'=' * 75}")
    print("  CORRUPTION ROBUSTNESS COMPARISON")
    print(f"{'=' * 75}")
    header = (
        f"{'Corruption':<18} {'mAP50':<10} {'ΔmAP50':<10} "
        f"{'mAP50-95':<12} {'ΔmAP50-95':<12} {'mAP75':<10}"
    )
    print(header)
    print("-" * 75)

    base = results.get("baseline", {})
    for key in ["baseline"] + corruptions:
        r = results.get(key)
        if r is None:
            continue
        d50 = r["map50"] - base["map50"] if key != "baseline" else 0.0
        d95 = r["map50_95"] - base["map50_95"] if key != "baseline" else 0.0
        print(
            f"{key:<18} {r['map50']:<10.4f} {d50:<+10.4f} "
            f"{r['map50_95']:<12.4f} {d95:<+12.4f} {r['map75']:<10.4f}"
        )


def main() -> None:
    """Entry point: run baseline validation, then each corruption."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    corrupt_base = output_dir / "corrupted"

    # Parse original YAML
    with open(args.data) as f:
        orig_cfg = yaml.safe_load(f)
    orig_path = Path(orig_cfg["path"]) if "path" in orig_cfg else Path.cwd()
    val_rel = orig_cfg.get("val", "val/images")
    val_dir = orig_path / val_rel if not Path(val_rel).is_absolute() else Path(val_rel)
    if not val_dir.exists():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    # Derive label directory
    label_rel = orig_cfg.get(
        "label", val_rel.replace("images", "labels").replace("img", "label")
    )
    label_dir = (
        orig_path / label_rel if not Path(label_rel).is_absolute() else Path(label_rel)
    )
    if not label_dir.exists():
        label_dir = val_dir.parent / "labels"
    if not label_dir.exists():
        print("  Warning: no labels directory found, metrics will be 0")
        label_dir = None

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    results: Dict[str, dict] = {}

    # Baseline
    print(f"\n{'=' * 60}")
    print("  Baseline (no corruption)")
    print(f"{'=' * 60}")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=False,
        save_json=False,
        verbose=False,
    )
    box = metrics.box
    results["baseline"] = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "map75": float(box.map75),
    }
    print(
        f"  mAP50={results['baseline']['map50']:.4f}  "
        f"mAP50-95={results['baseline']['map50_95']:.4f}"
    )

    # Per-corruption evaluation
    for corr in args.corruptions:
        print(f"\n{'=' * 60}")
        print(f"  Corruption: {corr} (severity {args.severity})")
        print(f"{'=' * 60}")

        out_dir = corrupt_base / corr
        if out_dir.exists():
            shutil.rmtree(out_dir)

        img_dir = corrupt_split(
            val_dir, out_dir, corr, args.severity, args.ext, label_dir=label_dir
        )
        suffix = f"corrupt_{corr}"
        val_yaml = make_val_yaml(
            orig_cfg, img_dir, suffix, Path(args.data), output_dir
        )
        print(f"  Temp YAML: {val_yaml}")
        print(f"  Images dir: {img_dir.resolve()}")

        try:
            metrics = model.val(
                data=str(val_yaml),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                plots=False,
                save_json=False,
                verbose=False,
            )
            box = metrics.box
            results[corr] = {
                "map50": float(box.map50),
                "map50_95": float(box.map),
                "map75": float(box.map75),
            }
            print(
                f"  mAP50={results[corr]['map50']:.4f}  "
                f"mAP50-95={results[corr]['map50_95']:.4f}"
            )
        finally:
            val_yaml.unlink(missing_ok=True)

        if not args.no_cleanup:
            shutil.rmtree(out_dir)

    print_results(results, args.corruptions)

    out_json = output_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_json}")


if __name__ == "__main__":
    main()
