"""Batch XAI processing over multiple models and methods.

Reads configuration from xai_config.yaml and runs xai_predict.py
over all combinations of models and methods.

Typical usage::

    python scripts/batch_xai.py --config xai_config.yaml
    python scripts/batch_xai.py --config xai_config.yaml --model yolo11n-simam
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Ensure the repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.xai_predict import run_xai


def load_config(config_path: str) -> Dict:
    """Load XAI configuration from YAML file.

    Args:
        config_path: Path to xai_config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict) -> None:
    """Validate configuration structure.

    Args:
        config: Configuration dictionary.

    Raises:
        ValueError: If configuration is invalid.
    """
    required_keys = {"models", "methods", "common"}
    if not all(k in config for k in required_keys):
        raise ValueError(f"Config must contain keys: {required_keys}")

    if not config["models"]:
        raise ValueError("No models defined in config.")
    if not config["methods"]:
        raise ValueError("No methods defined in config.")

    for i, model_cfg in enumerate(config["models"]):
        if "path" not in model_cfg:
            raise ValueError(f"Model {i} missing 'path'.")
        if not Path(model_cfg["path"]).exists():
            raise ValueError(f"Model path does not exist: {model_cfg['path']}")


def batch_run(
    config: Dict,
    model_filter: Optional[str] = None,
    method_filter: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Run XAI processing over all model-method combinations.

    Args:
        config: Configuration dictionary.
        model_filter: Only run this model name (e.g. 'yolo11n-simam').
        method_filter: Only run this method (e.g. 'eigencam').
        dry_run: Print commands without executing.
    """
    models = config["models"]
    methods = config["methods"]
    common = config["common"]

    output_base = common["output_base"]
    conf_threshold = common.get("conf_threshold", 0.25)
    device = common.get("device", None)

    os.makedirs(output_base, exist_ok=True)

    from tqdm import tqdm as tqdm_bar
    
    # Count total combinations first
    total_combinations = sum(
        1 for m_cfg in models
        if not model_filter or m_cfg.get("name", Path(m_cfg["path"]).stem) == model_filter
        for method in methods
        if not method_filter or method == method_filter
    )
    
    if total_combinations == 0:
        print("No combinations match the filters.")
        return

    pbar = tqdm_bar(total=total_combinations, desc="Batch XAI", disable=verbose)
    total_runs = 0
    
    for model_cfg in models:
        model_path = model_cfg["path"]
        model_name = model_cfg.get("name", Path(model_path).stem)
        target_layer_index = model_cfg.get("target_layer_index", None)
        target_location = model_cfg.get("target_location", "backbone")
        source = model_cfg.get("source", common.get("source"))
        if source is None:
            raise ValueError(f"Model '{model_name}' has no source and no common source defined.")

        if not Path(source).exists():
            raise ValueError(f"Source directory does not exist for model '{model_name}': {source}")

        # Skip if model_filter is set and doesn't match
        if model_filter and model_name != model_filter:
            continue

        for method in methods:
            # Skip if method_filter is set and doesn't match
            if method_filter and method != method_filter:
                continue

            total_runs += 1
            output_dir = os.path.join(output_base, model_name, method)

            cmd_info = (
                f"Model: {model_name} | Source: {source} | Method: {method} | "
                f"Target: {target_location} | "
                f"Layer: {target_layer_index or 'auto'}"
            )
            
            if verbose:
                print(f"\n[{total_runs}] {cmd_info}")

            if dry_run:
                if verbose:
                    print(f"  [DRY RUN] Would execute xai_predict.py")
                pbar.update(1)
                continue

            try:
                run_xai(
                    model_path=model_path,
                    source=source,
                    output_dir=output_dir,
                    method=method,
                    conf_thres=conf_threshold,
                    device=device,
                    target_layer_index=target_layer_index,
                    location=target_location,
                    verbose=verbose,
                )
                if verbose:
                    print(f"  ✓ Completed")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error: {e}")
            
            pbar.update(1)
    
    pbar.close()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Batch run completed: {total_runs} combination(s) processed.")
        print(f"Results saved to: {output_base}")
        print(f"{'=' * 60}")
    else:
        print(f"Batch completed: {total_runs} combination(s) processed. -> {output_base}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch XAI processing over multiple models and methods."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="xai_config.yaml",
        help="Path to XAI configuration YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Only process this model (name from config). E.g. 'yolo11n-simam'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Only process this method. E.g. 'eigencam'.",
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

    print(f"\n{'=' * 60}")
    print(f"XAI Batch Configuration: {args.config}")
    print(f"{'=' * 60}\n")

    batch_run(
        config,
        model_filter=args.model,
        method_filter=args.method,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
