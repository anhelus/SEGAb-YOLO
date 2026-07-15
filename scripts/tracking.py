"""Run YOLO tracking on videos or images with optional display and save.

Typical usage::

    python -m scripts.tracking --source data/videos --model yolo26n.pt --show --save
"""

import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm
from segab_yolo import YOLO


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO tracking on videos or images."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=".",
        help="Input file or directory to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="federated.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="*.mp4",
        help="File extension pattern when source is a directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display tracking results in a window",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save tracking results to disk",
    )
    return parser.parse_args()


def collect_files(source: Path, ext: str) -> List[Path]:
    """Collect input files from a file or directory path.

    Args:
        source: Path to a single file or a directory.
        ext: Glob pattern used when *source* is a directory.

    Returns:
        A sorted list of file paths matching the criteria.
    """
    if source.is_file():
        return [source]
    return sorted(source.rglob(ext))


def track_file(model: YOLO, file_path: Path, show: bool, save: bool) -> None:
    """Run YOLO tracking on a single file.

    Args:
        model: Loaded YOLO model instance.
        file_path: Path to the input video or image.
        show: Whether to display results in a window.
        save: Whether to save results to disk.
    """
    model.track(str(file_path), show=show, save=save)


def main() -> None:
    """Entry point: parse args, collect files, and run tracking on each."""
    args = parse_args()
    model = YOLO(args.model)
    source = Path(args.source)

    files = collect_files(source, args.ext)
    if not files:
        print(f"No files matching '{args.ext}' found in {source}")
        return

    for f in tqdm(files, desc="Tracking"):
        track_file(model, f, args.show, args.save)

    print(f"Done. Processed {len(files)} file(s).")


if __name__ == "__main__":
    main()
