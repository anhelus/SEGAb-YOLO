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
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Config must contain 'datasets' list.")
    if "training" not in config:
        raise ValueError("Config must contain 'training'.")


def batch_run(
    config: Dict,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Run training over all specified model variants and datasets."""
    models = config["models"]
    datasets = config["datasets"]
    training_cfg = config["training"]

    epochs = training_cfg.get("epochs", 100)
    batch = training_cfg.get("batch", 16)
    imgsz = training_cfg.get("imgsz", 640)
    fraction = training_cfg.get("fraction", 1.0)
    device = training_cfg.get("device", None)
    patience = training_cfg.get("patience", 50)

    from tqdm import tqdm as tqdm_bar

    filtered_datasets = [
        d for d in datasets
        if not dataset_filter or d.get("name") == dataset_filter
    ]
    if not filtered_datasets:
        print("No datasets match the filter.")
        return

    for ds in filtered_datasets:
        ds_name = ds["name"]
        ds_path = ds["path"]
        project = ds.get("project", f"runs/{ds_name}")
        abs_project = str(Path(project).resolve())
        os.makedirs(abs_project, exist_ok=True)

        def _model_name(m):
            return m if isinstance(m, str) else m.get("name", m.get("model", str(m)))
        total = sum(
            1 for m in models
            if not model_filter or _model_name(m) == model_filter
        )
        if total == 0:
            continue

        pbar = tqdm_bar(total=total, desc=f"Train {ds_name}", disable=verbose)
        trained = 0

        for entry in models:
            model_name = _model_name(entry)

            if model_filter and model_name != model_filter:
                continue

            trained += 1

            sep = "=" * 70
            if verbose:
                print(f"\n{sep}", flush=True)
                print(f"  [{trained}/{total}] Training: dataset={ds_name}  model={model_name}", flush=True)
                print(f"  YOLO('{model_name}.yaml').train(data={ds_path}, project={project})", flush=True)
                print(f"{sep}\n", flush=True)

            if dry_run:
                if verbose:
                    print(f"  [DRY RUN] (would train)", flush=True)
                pbar.update(1)
                continue

            try:
                model = YOLO(f"{model_name}.yaml")
                model.train(
                    data=ds_path,
                    project=abs_project,
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
                    print(f"  ok {ds_name}/{model_name}", flush=True)
            except Exception as e:
                if verbose:
                    print(f"  fail {ds_name}/{model_name}: {e}", flush=True)

            pbar.update(1)

        pbar.close()
        print(f"Done: {trained} model(s) trained for {ds_name} -> {project}")


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
        help="Only train this model (name). E.g. 'yolo11n'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only train on this dataset (name). E.g. 'tomatoes'.",
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
        dataset_filter=args.dataset,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
