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
    data_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    imgsz_override: Optional[int] = None,
    device_override: Optional[str] = None,
    name_override: Optional[str] = None,
) -> None:
    """Run training over all specified model variants and datasets."""
    models = config["models"]
    datasets = config["datasets"]
    training_cfg = config["training"]

    epochs = epochs_override if epochs_override is not None else training_cfg.get("epochs", 100)
    batch = batch_override if batch_override is not None else training_cfg.get("batch", 16)
    imgsz = imgsz_override if imgsz_override is not None else training_cfg.get("imgsz", 640)
    fraction = training_cfg.get("fraction", 1.0)
    device = device_override if device_override is not None else training_cfg.get("device", None)
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
        # Apply data_override only if dataset_filter is set (single dataset mode)
        # Otherwise use each dataset's own path
        if data_override is not None and dataset_filter is not None:
            ds_path = data_override
        else:
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
                run_name = name_override if name_override is not None else model_name
                model.train(
                    data=ds_path,
                    project=abs_project,
                    name=run_name,
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
        "--data",
        type=str,
        default=None,
        help="Custom data YAML path (overrides dataset path from config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config training.epochs).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (overrides config training.batch).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size (overrides config training.imgsz).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (overrides config training.device).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom run name (subdirectory under project). Defaults to model name.",
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
        data_override=args.data,
        epochs_override=args.epochs,
        batch_override=args.batch,
        imgsz_override=args.imgsz,
        device_override=args.device,
        name_override=args.name,
    )


if __name__ == "__main__":
    main()
