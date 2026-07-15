"""Batch inference benchmark over multiple models.

Reads configuration from infer_config.yaml, runs YOLO predict on all
images in a source directory for each model, measures inference time
per image, and prints a comparison report.

Typical usage::

    python scripts/batch_infer.py --config infer_config.yaml
    python scripts/batch_infer.py --config infer_config.yaml --model lettuces/yolo11n
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from segab_yolo import YOLO


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict) -> None:
    required_keys = {"models", "source", "common"}
    if not all(k in config for k in required_keys):
        raise ValueError(f"Config must contain keys: {required_keys}")
    if not config["models"]:
        raise ValueError("No models defined in config.")
    for i, m in enumerate(config["models"]):
        if "path" not in m:
            raise ValueError(f"Model {i} missing 'path'.")
        if not Path(m["path"]).exists():
            raise ValueError(f"Model path not found: {m['path']}")
    if not Path(config["source"]).exists():
        raise ValueError(f"Source directory not found: {config['source']}")


def collect_images(source: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    images = sorted(
        p for p in Path(source).iterdir()
        if p.suffix.lower() in exts
    )
    if not images:
        raise ValueError(f"No images found in {source}")
    return images


def time_predict(model, img_path: Path, device: Optional[str], **kwargs) -> float:
    """Run predict and return inference time in ms."""
    results = model.predict(
        source=str(img_path),
        device=device,
        verbose=False,
        **kwargs,
    )
    # results[0].speed is a dict: {'preprocess': ms, 'inference': ms, 'postprocess': ms}
    return results[0].speed["inference"]


def benchmark_model(
    model_path: str,
    model_name: str,
    images: List[Path],
    device: Optional[str],
    warmup: int,
    verbose: bool,
    **kwargs,
) -> Tuple[str, List[float]]:
    if verbose:
        print(f"  Loading model...")
    model = YOLO(model_path)

    # Warmup
    if warmup > 0 and images:
        if verbose:
            print(f"  Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            model.predict(source=str(images[0]), device=device, verbose=False, **kwargs)

    times = []
    for img in images:
        t = time_predict(model, img, device, **kwargs)
        times.append(t)

    return model_name, times


def print_report(results: List[Tuple[str, List[float]]]) -> None:
    print()
    print(f"{'=' * 90}")
    print(f"{'Model':<30} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'FPS':>10}")
    print(f"{'-' * 90}")

    for name, times in results:
        arr = np.array(times)
        mean_ms = arr.mean()
        std_ms = arr.std()
        min_ms = arr.min()
        max_ms = arr.max()
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        print(
            f"{name:<30} {mean_ms:>10.2f} {std_ms:>10.2f} {min_ms:>10.2f} {max_ms:>10.2f} {fps:>10.1f}"
        )

    print(f"{'=' * 90}")


def batch_run(
    config: Dict,
    model_filter: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    models = config["models"]
    source = config["source"]
    common = config["common"]

    device = common.get("device", None)
    warmup = common.get("warmup", 3)
    conf = common.get("conf", 0.25)
    imgsz = common.get("imgsz", 640)
    iou = common.get("iou", 0.45)

    images = collect_images(source)
    n_images = len(images)
    if verbose:
        print(f"Found {n_images} images in {source}")

    # Filter models
    filtered = []
    for m in models:
        name = m.get("name", Path(m["path"]).stem)
        if model_filter and name != model_filter:
            continue
        filtered.append(m)

    if not filtered:
        print("No models match the filter.")
        return

    results = []
    for m_cfg in filtered:
        model_path = m_cfg["path"]
        model_name = m_cfg.get("name", Path(model_path).stem)

        header = f"[{model_name}]  {n_images} images"
        print(f"\n{header}")
        print("-" * len(header))

        if dry_run:
            print(f"  [DRY RUN] Would benchmark {model_path}")
            continue

        try:
            name, times = benchmark_model(
                model_path, model_name, images,
                device=device, warmup=warmup,
                verbose=verbose, conf=conf, imgsz=imgsz, iou=iou,
            )
            results.append((name, times))
            if verbose:
                arr = np.array(times)
                print(f"  Mean: {arr.mean():.2f} ms  |  FPS: {1000.0/arr.mean():.1f}")
        except Exception as e:
            print(f"  Error: {e}")

    if not dry_run and results:
        print_report(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference benchmark over multiple models."
    )
    parser.add_argument(
        "--config", type=str, default="infer_config.yaml",
        help="Path to inference configuration YAML file.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Only benchmark this model (name from config).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    print(f"\n{'=' * 60}")
    print(f"Batch Inference Benchmark")
    print(f"Config: {args.config}")
    print(f"{'=' * 60}")

    batch_run(
        config,
        model_filter=args.model,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
