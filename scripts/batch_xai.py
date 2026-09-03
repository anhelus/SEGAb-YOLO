"""Batch XAI processing over multiple models and methods."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.xai_predict import run_xai


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict) -> None:
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Config must contain 'datasets'.")
    if "methods" not in config or not config["methods"]:
        raise ValueError("No methods defined in config.")


def batch_run(
    config: Dict,
    dataset_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    method_filter: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
    limit: Optional[int] = None,
) -> None:
    methods = config["methods"]
    common = config.get("common", {})
    output_base = common.get("output_base", "reports/xai")
    conf_threshold = common.get("conf_threshold", 0.25)
    device = common.get("device")
    target_location = common.get("target_location", "both")

    output_path = Path(output_base)
    output_path.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm as tqdm_bar

    total_runs = 0

    for ds_name, ds_cfg in config["datasets"].items():
        if dataset_filter and ds_name != dataset_filter:
            continue

        source = ds_cfg.get("source", common.get("source"))
        if not source:
            print(f"  Skipping {ds_name}: no source defined")
            continue
        if not Path(source).exists():
            print(f"  Skipping {ds_name}: source not found at {source}")
            continue

        model_base = Path(ds_cfg.get("model_base", f"runs/{ds_name}"))
        model_names = ds_cfg.get("models", [])

        # Count
        count = sum(
            1 for m in model_names
            if not model_filter or m == model_filter
            for meth in methods
            if not method_filter or meth == method_filter
        )
        if count == 0:
            continue

        pbar = tqdm_bar(total=count, desc=f"XAI {ds_name}", disable=verbose)

        for model_name in model_names:
            if model_filter and model_name != model_filter:
                continue

            model_path = model_base / model_name / "weights" / "best.pt"
            if not model_path.exists():
                if verbose:
                    print(f"  Skipping {ds_name}/{model_name}: model not found")
                continue

            for method in methods:
                if method_filter and method != method_filter:
                    continue

                total_runs += 1
                out_dir = output_path / ds_name / model_name / method

                if verbose:
                    print(f"\n[{total_runs}] {ds_name}/{model_name} / {method}")

                if dry_run:
                    if verbose:
                        print(f"  [DRY RUN] run_xai(model={model_path}, source={source}, method={method})")
                    pbar.update(1)
                    continue

                try:
                    run_xai(
                        model_path=str(model_path),
                        source=source,
                        output_dir=str(out_dir),
                        method=method,
                        conf_thres=conf_threshold,
                        device=device,
                        location=target_location,
                        verbose=verbose,
                        limit=limit,
                    )
                    if verbose:
                        print(f"  OK")
                except Exception as e:
                    if verbose:
                        print(f"  Error: {e}")

                pbar.update(1)

        pbar.close()

    if not verbose:
        print(f"Batch XAI completed: {total_runs} run(s) -> {output_base}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch XAI processing over multiple models and methods.")
    parser.add_argument("--config", type=str, default="xai_config.yaml", help="Path to XAI configuration YAML file.")
    parser.add_argument("--dataset", type=str, default=None, help="Only process this dataset.")
    parser.add_argument("--model", type=str, default=None, help="Only process this model name.")
    parser.add_argument("--method", type=str, default=None, help="Only process this method.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N images per dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    batch_run(
        config,
        dataset_filter=args.dataset,
        model_filter=args.model,
        method_filter=args.method,
        dry_run=args.dry_run,
        verbose=args.verbose,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
