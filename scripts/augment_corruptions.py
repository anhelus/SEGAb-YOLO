"""Split a dataset into N partitions and apply a different corruption to each.

The resulting dataset can be used for robust training.  An optional
``--keep-clean`` flag includes the original (uncorrupted) images as well.

Typical usage::

    python -m scripts.augment_corruptions --source data/lettuce \\
        --output data/lettuce_corrupted
"""

import argparse
import math
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from segab_yolo.utils.corruptions import (
    IMAGE_EXTENSIONS,
    SEVERITY_PARAMS,
    apply_corruption,
    image_generator,
)

# Re-export for CLI choices
CORRUPTION_NAMES: List[str] = [
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


def find_label(img_path: Path, label_dir: Path) -> Optional[Path]:
    """Find the label file matching an image path by stem.

    Args:
        img_path: Path to the image.
        label_dir: Directory containing label files.

    Returns:
        Path to the label file, or None.
    """
    stem = img_path.stem
    for ext in [".txt"]:
        p = label_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split a dataset into N partitions, apply one corruption "
        "per partition, and merge into a new dataset."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source dataset directory (must contain images/ and labels/ "
        "subdirectories)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output dataset directory"
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default=CORRUPTION_NAMES,
        choices=CORRUPTION_NAMES,
        help="Corruption types (N = number of partitions)",
    )
    parser.add_argument(
        "--severity", type=int, default=2, choices=[1, 2, 3]
    )
    parser.add_argument(
        "--keep-clean",
        action="store_true",
        help="Also include the original clean images in the output",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        dest="yaml_path",
        help="Path to save the dataset YAML (default: output/dataset.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible partition split",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, split, corrupt, and merge."""
    args = parse_args()
    src = Path(args.source)
    out = Path(args.output)
    img_src = src / "images"
    label_src = src / "labels"

    if not img_src.is_dir():
        raise FileNotFoundError(f"Source images directory not found: {img_src}")
    if not label_src.is_dir():
        print(f"  Warning: labels directory not found at {label_src}")

    images = sorted(image_generator(str(img_src), ext=None))
    if not images:
        raise FileNotFoundError(f"No images found in {img_src}")

    # Clean output
    if out.exists():
        shutil.rmtree(out)
    out_img_dir = out / "images"
    out_label_dir = out / "labels"
    out_img_dir.mkdir(parents=True)
    out_label_dir.mkdir(parents=True)

    corruptions = args.corruptions
    N = len(corruptions)
    sp = SEVERITY_PARAMS[args.severity]

    # Deterministic shuffle
    rng = np.random.RandomState(args.seed)
    rng.shuffle(images)
    bin_size = math.ceil(len(images) / N)
    partitions = [images[i * bin_size : (i + 1) * bin_size] for i in range(N)]

    total_partitioned = sum(len(p) for p in partitions)
    print(f"Splitting {len(images)} images into {N} partitions ({bin_size} per bin)")
    for i, (corr, part) in enumerate(zip(corruptions, partitions)):
        print(f"  Partition {i}: {corr} -> {len(part)} images")

    for i, (corr, part) in enumerate(zip(corruptions, partitions)):
        desc = f"[{i + 1}/{N}] {corr}"
        for img_path in tqdm(part, desc=desc):
            img = cv2.imread(str(img_path))
            if img is None:
                tqdm.write(f"  Skipping {img_path}: cannot read")
                continue
            img = apply_corruption(img, corr, sp)
            out_path = out_img_dir / img_path.name
            cv2.imwrite(str(out_path), img)

            label_path = find_label(img_path, label_src)
            if label_path:
                shutil.copy2(label_path, out_label_dir / label_path.name)

    # Optional clean originals
    if args.keep_clean:
        print("\nCopying clean originals...")
        for img_path in tqdm(images, desc="Clean originals"):
            out_path = out_img_dir / img_path.name
            if not out_path.exists():
                shutil.copy2(img_path, out_path)
                label_path = find_label(img_path, label_src)
                if label_path:
                    shutil.copy2(label_path, out_label_dir / label_path.name)

    n_imgs = len(list(out_img_dir.iterdir()))
    n_labels = len(list(out_label_dir.iterdir()))
    print(f"\nDone: {n_imgs} images, {n_labels} labels -> {out}")

    # Dataset YAML
    yaml_path = Path(args.yaml_path) if args.yaml_path else out / "dataset.yaml"
    orig_yaml = src / "dataset.yaml"
    if orig_yaml.exists():
        with open(orig_yaml) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"names": {0: "object"}}

    cfg["path"] = str(out.resolve())
    cfg["train"] = "images"
    cfg["val"] = "images"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Dataset YAML: {yaml_path}")
    print(f"\nTrain with:\n  yolo train model=yolo26n.pt data={yaml_path} epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
