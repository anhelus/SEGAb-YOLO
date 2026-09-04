"""Batch inference benchmark over multiple models."""

import argparse
import json
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
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Config must contain 'datasets'.")
    for ds_name, ds_cfg in config["datasets"].items():
        if "source" not in ds_cfg:
            raise ValueError(f"Dataset '{ds_name}' missing 'source'.")
        if "models" not in ds_cfg or not ds_cfg["models"]:
            raise ValueError(f"Dataset '{ds_name}' missing 'models' list.")
        # Skip existence check if source will be overridden via CLI
        if "source" in ds_cfg and ds_cfg["source"] and not ds_cfg.get("_source_override"):
            if not Path(ds_cfg["source"]).exists():
                raise ValueError(f"Source not found: {ds_cfg['source']}")


def collect_images(source: str, limit: Optional[int] = None) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    images = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in exts)
    if not images:
        raise ValueError(f"No images found in {source}")
    if limit:
        images = images[:limit]
    return images


def save_detections(results_json: str, model_name: str, ds_name: str, images: List[Path], model, device: Optional[str], **kwargs) -> None:
    """Run inference on images, save bbox detections to JSON and annotated images."""
    import cv2
    base_dir = Path(results_json).parent
    annotated_dir = base_dir / model_name
    original_dir = base_dir / model_name / "original"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    detections = []
    for img in images:
        res = model.predict(source=str(img), device=device, verbose=False, **kwargs)
        r = res[0]
        boxes = []
        if r.boxes is not None:
            for box in r.boxes:
                boxes.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": round(box.conf[0].item(), 4),
                    "class": int(box.cls[0].item()),
                })
        detections.append({
            "file": img.name,
            "width": r.orig_shape[1] if hasattr(r, 'orig_shape') else None,
            "height": r.orig_shape[0] if hasattr(r, 'orig_shape') else None,
            "detections": boxes,
        })
        plotted = r.plot()
        stem = img.stem
        cv2.imwrite(str(annotated_dir / f"{stem}.jpg"), plotted)
        orig = cv2.imread(str(img))
        if orig is not None:
            cv2.imwrite(str(original_dir / f"{stem}.jpg"), orig)
    output = {
        "dataset": ds_name,
        "model": model_name,
        "total_images": len(images),
        "images": detections,
    }
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(output, f, indent=2)
    n_boxes = sum(len(d["detections"]) for d in detections)
    print(f"  Saved: {results_json} ({n_boxes} boxes, {len(images)} images + annotated)")


def time_predict(model, img_path: Path, device: Optional[str], **kwargs) -> float:
    results = model.predict(
        source=str(img_path), device=device, verbose=False, **kwargs,
    )
    return results[0].speed["inference"]


def benchmark_model(
    model_path: str, model_name: str, images: List[Path],
    device: Optional[str], warmup: int, verbose: bool, **kwargs,
) -> Tuple[str, List[float]]:
    if verbose:
        print(f"  Loading model...")
    model = YOLO(model_path)
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
        print(f"{name:<30} {mean_ms:>10.2f} {arr.std():>10.2f} {arr.min():>10.2f} {arr.max():>10.2f} {1000.0/mean_ms if mean_ms > 0 else 0.0:>10.1f}")
    print(f"{'=' * 90}")


def batch_run(
    config: Dict,
    dataset_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
    limit: Optional[int] = None,
    save_results: Optional[str] = None,
) -> None:
    common = config.get("common", {})
    device = common.get("device")
    warmup = common.get("warmup", 3)
    conf = common.get("conf", 0.25)
    imgsz = common.get("imgsz", 640)
    iou = common.get("iou", 0.45)

    all_results = []

    for ds_name, ds_cfg in config["datasets"].items():
        if dataset_filter and ds_name != dataset_filter:
            continue

        source = ds_cfg["source"]
        model_base = Path(ds_cfg.get("model_base", f"runs/{ds_name}"))
        model_names = ds_cfg["models"]

        images = collect_images(source, limit=limit)
        if verbose:
            print(f"\nDataset: {ds_name}  |  {len(images)} images  |  source: {source}")

        for model_name in model_names:
            if model_filter and model_name != model_filter:
                continue

            model_path = model_base / model_name / "weights" / "best.pt"
            if not model_path.exists():
                print(f"  Skipping {ds_name}/{model_name}: model not found at {model_path}")
                continue

            label = f"{ds_name}/{model_name}"
            header = f"[{label}]  {len(images)} images"
            print(f"\n{header}")
            print("-" * len(header))

            if dry_run:
                print(f"  [DRY RUN] Would benchmark {model_path}")
                continue

            if save_results:
                model = YOLO(str(model_path))
                save_detections(
                    str(Path(save_results) / ds_name / f"{model_name}.json"),
                    model_name, ds_name, images, model,
                    device=device, conf=conf, imgsz=imgsz, iou=iou,
                )

            try:
                name, times = benchmark_model(
                    str(model_path), label, images,
                    device=device, warmup=warmup,
                    verbose=verbose, conf=conf, imgsz=imgsz, iou=iou,
                )
                all_results.append((name, times))
                if verbose:
                    arr = np.array(times)
                    print(f"  Mean: {arr.mean():.2f} ms  |  FPS: {1000.0/arr.mean():.1f}")
            except Exception as e:
                print(f"  Error: {e}")

    if not dry_run and all_results:
        print_report(all_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference benchmark over multiple models.")
    parser.add_argument("--config", type=str, default="infer_config.yaml", help="Path to inference configuration YAML file.")
    parser.add_argument("--dataset", type=str, default=None, help="Only benchmark this dataset.")
    parser.add_argument("--model", type=str, default=None, help="Only benchmark this model name.")
    parser.add_argument("--source", type=str, default=None, help="Override source path for the dataset (images directory).")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N images per dataset.")
    parser.add_argument("--save-results", type=str, default=None, help="Directory to save per-image detection JSONs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Override source if provided via CLI
    if args.source and args.dataset:
        if args.dataset in config.get("datasets", {}):
            config["datasets"][args.dataset]["_source_override"] = True
            config["datasets"][args.dataset]["source"] = args.source
        else:
            raise ValueError(f"Dataset '{args.dataset}' not found in config.")
    
    validate_config(config)

    print(f"\n{'=' * 60}")
    print(f"Batch Inference Benchmark")
    print(f"Config: {args.config}")
    print(f"{'=' * 60}")

    batch_run(
        config,
        dataset_filter=args.dataset,
        model_filter=args.model,
        dry_run=args.dry_run,
        verbose=args.verbose,
        limit=args.limit,
        save_results=args.save_results,
    )


if __name__ == "__main__":
    main()
