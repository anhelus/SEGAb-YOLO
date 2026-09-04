"""Run model validation on all models in runs/{dataset}/ and save results.csv."""

import argparse
import csv
from pathlib import Path


def find_dataset_yaml(dataset_name: str, data_dir: Path = None) -> Path:
    base = data_dir or Path("data") / dataset_name
    if not base.exists():
        return None
    candidates = list(base.glob("*.yaml"))
    if not candidates:
        return None
    return candidates[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate trained models and save results.csv."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, default=None, help="Model filter (e.g. yolo11n)")
    parser.add_argument("--dataset-yaml", type=str, default=None, help="Path to dataset YAML")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Base runs directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir) / args.dataset
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return

    dataset_yaml = args.dataset_yaml
    if not dataset_yaml:
        yaml_path = find_dataset_yaml(args.dataset)
        if yaml_path:
            dataset_yaml = str(yaml_path)
            print(f"Auto-detected dataset YAML: {dataset_yaml}")
        else:
            print("No dataset YAML found. Use --dataset-yaml to specify.")
            return

    if args.model:
        model_dirs = [runs_dir / args.model]
    else:
        model_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    total = len(model_dirs)
    for idx, model_dir in enumerate(model_dirs, 1):
        best_pt = model_dir / "weights" / "best.pt"
        if not best_pt.exists():
            continue

        model_name = model_dir.name
        print(f"[{idx}/{total}] Validating {model_name}...")

        from segab_yolo import YOLO
        model = YOLO(str(best_pt))
        results = model.val(data=dataset_yaml, verbose=args.verbose)

        results_csv = model_dir / "results.csv"
        metrics = {
            "model": model_name,
            "metrics/precision(B)": results.box.p[0] if hasattr(results.box, 'p') else 0,
            "metrics/recall(B)": results.box.r[0] if hasattr(results.box, 'r') else 0,
            "metrics/mAP50(B)": results.box.map50,
            "metrics/mAP50-95(B)": results.box.map,
        }

        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"  Results saved: {results_csv}")


if __name__ == "__main__":
    main()
