"""Batch training over multiple YOLO model variants.

Reads configuration from train_config.yaml and runs training
for each model variant.

Typical usage::

    python scripts/batch_train.py --config train_config.yaml
    python scripts/batch_train.py --config train_config.yaml --model yolo11n-ema
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from segab_yolo import YOLO


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict) -> None:
    """Validate configuration structure."""
    if "models" not in config or not config["models"]:
        raise ValueError("Config must contain at least one model under 'models'.")
    if "dataset" not in config:
        raise ValueError("Config must contain 'dataset'.")
    if "training" not in config:
        raise ValueError("Config must contain 'training'.")


def batch_run(
    config: Dict,
    model_filter: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Run training over all specified model variants."""
    models = config["models"]
    dataset = config["dataset"]
    training_cfg = config["training"]

    project = training_cfg.get("project", "runs/train")
    epochs = training_cfg.get("epochs", 100)
    batch = training_cfg.get("batch", 16)
    imgsz = training_cfg.get("imgsz", 640)
    fraction = training_cfg.get("fraction", 1.0)
    device = training_cfg.get("device", None)
    patience = training_cfg.get("patience", 50)

    os.makedirs(project, exist_ok=True)

    from tqdm import tqdm as tqdm_bar

    # Count matches
    total = sum(
        1 for m in models
        if not model_filter or (m.get("name") or m) == model_filter
    )
    if total == 0:
        print("No models match the filter.")
        return

    pbar = tqdm_bar(total=total, desc="Batch Train", disable=verbose)
    trained = 0

    for entry in models:
        if isinstance(entry, str):
            model_name = entry
        elif isinstance(entry, dict):
            model_name = entry.get("name", entry.get("model", str(entry)))
        else:
            raise ValueError(f"Unexpected model entry type: {type(entry)}")

        if model_filter and model_name != model_filter:
            continue

        trained += 1

        if verbose:
            print(f"\n[{trained}/{total}] Training: {model_name}")

        if dry_run:
            if verbose:
                yaml_path = f"{model_name}.yaml"
                print(f"  [DRY RUN] YOLO('{yaml_path}').train(...)")
            pbar.update(1)
            continue

        try:
            model = YOLO(f"{model_name}.yaml")
            model.train(
                data=dataset,
                project=project,
                name=model_name,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                fraction=fraction,
                device=device,
                patience=patience,
                exist_ok=True,
            )
            if verbose:
                print(f"  \u2713 Completed: {model_name}")
        except Exception as e:
            if verbose:
                print(f"  \u2717 Error: {e}")

        pbar.update(1)

    pbar.close()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Batch training completed: {trained} model(s).")
        print(f"Results saved to: {project}")
        print(f"{'=' * 60}")
    else:
        print(f"Batch completed: {trained} model(s) trained. -> {project}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch training over multiple YOLO model variants."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Only train this model (name). E.g. 'yolo11n-ema'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output instead of progress bar",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    batch_run(
        config,
        model_filter=args.model,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
