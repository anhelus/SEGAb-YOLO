"""Apply image corruptions to datasets for robustness testing.

Supports 9 corruption types across 3 severity levels. Labels can be
copied alongside the corrupted images while preserving directory structure.

Typical usage::

    python -m scripts.corrupt_dataset --source data/train/images \\
        --output data/train_corrupted --labels data/train/labels \\
        --corruptions darken blur --severity 2
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

from segab_yolo.utils.corruptions import (
    CORRUPTIONS,
    SEVERITY_PARAMS,
    apply_corruption,
    image_generator,
)


def find_labels(img_path: Path, label_dir: Path) -> Optional[Path]:
    """Find the label file corresponding to an image path.

    Args:
        img_path: Path to the image file.
        label_dir: Directory to search for labels.

    Returns:
        Path to the label file, or None if not found.
    """
    for suffix in {".txt"}:
        candidate = label_dir / img_path.with_suffix(suffix).name
        if candidate.exists():
            return candidate
    stem = img_path.stem
    for f in label_dir.iterdir():
        if f.stem == stem:
            return f
    return None


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Corrupt dataset images for robustness testing."
    )
    parser.add_argument("--source", type=str, required=True, help="Source image file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--labels", type=str, default=None, help="Labels directory (copied alongside images, preserving structure)"
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default=["darken", "blur"],
        choices=list(CORRUPTIONS.keys()),
        help="Corruption types to apply (composable, applied in order)",
    )
    parser.add_argument("--ext", type=str, default=None, help="Image extension filter (e.g. jpg, png). Default: all")
    parser.add_argument(
        "--severity", type=int, default=1, choices=[1, 2, 3], help="Severity level (1=mild, 2=moderate, 3=severe)"
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and apply corruptions to all images."""
    args = parse_args()

    source = Path(args.source)
    output = Path(args.output)
    sp = SEVERITY_PARAMS[args.severity]

    images = list(image_generator(source, args.ext))
    if not images:
        print(f"No images found in {source}")
        return

    suffix = "_".join(args.corruptions)
    label_src = Path(args.labels) if args.labels else None

    for img_path in tqdm(images, desc=f"Corrupting ({suffix}, sev={args.severity})"):
        rel = img_path.relative_to(source) if source.is_dir() else img_path.name
        out_img_path = output / rel
        out_img_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"  Skipping {img_path}: cannot read")
            continue

        for corr in args.corruptions:
            img = apply_corruption(img, corr, sp)

        cv2.imwrite(str(out_img_path), img)

        if label_src:
            label_path = find_labels(img_path, label_src)
            if label_path:
                out_label_path = output / label_path.relative_to(label_src)
                out_label_path.parent.mkdir(parents=True, exist_ok=True)
                out_label_path.write_bytes(label_path.read_bytes())

    print(f"Done. Output: {output}")


if __name__ == "__main__":
    main()
