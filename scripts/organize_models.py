"""Copy model .pt files from models/{dataset}/ to runs/{dataset}/{model}/weights/best.pt."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize model files into runs directory structure."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. tomatoes)")
    parser.add_argument("--source", type=str, default=None, help="Source dir (default: models/{dataset})")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Base runs directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Base models directory")
    args = parser.parse_args()

    source_dir = Path(args.source) if args.source else Path(args.models_dir) / args.dataset
    runs_dir = Path(args.runs_dir) / args.dataset

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    pt_files = sorted(source_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {source_dir}")
        return

    for pt in pt_files:
        model_name = pt.stem.removesuffix("_best").removesuffix("-best")
        dest = runs_dir / model_name / "weights" / "best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pt, dest)
        print(f"  {pt.name} -> {dest}")


if __name__ == "__main__":
    main()
