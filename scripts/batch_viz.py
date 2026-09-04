"""Compare model training results: heatmap, barplot, CSV."""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import yaml

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_runs_dir(config: dict) -> Path:
    return Path(config.get("runs_dir", "runs"))


def get_output_base(config: dict) -> Path:
    return Path(config.get("output_base", "reports/comparison"))


def collect_results(config: dict, dataset_filter: str = None) -> pl.DataFrame:
    runs_dir = get_runs_dir(config)
    datasets = config.get("datasets", {})
    rows = []

    for ds_name, ds_cfg in datasets.items():
        if dataset_filter and ds_name != dataset_filter:
            continue
        labels = ds_cfg.get("labels", {})
        baseline_name = ds_cfg.get("baseline")
        custom_metric_names = ds_cfg.get("metrics")
        models_cfg = ds_cfg.get("models", {})

        for model_key, model_info in models_cfg.items():
            model_path = runs_dir / ds_name / model_key / "weights" / "best.pt"
            results_csv = runs_dir / ds_name / model_key / "results.csv"
            label = model_info.get("label", model_key)

            row = {
                "dataset": ds_name,
                "model_key": model_key,
                "label": label,
                "is_baseline": model_key == baseline_name,
                "model_path": str(model_path),
            }

            if results_csv.exists():
                try:
                    df = pl.read_csv(results_csv)
                    if df.height > 0:
                        best_row = df.row(df.height - 1, named=True)
                        row.update({
                            k: best_row.get(k)
                            for k in custom_metric_names or [
                                "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                                "metrics/precision(B)", "metrics/recall(B)",
                            ]
                        })
                except Exception:
                    pass

            rows.append(row)

    return pl.DataFrame(rows)


def timed_inference(model_path: str, source: str, num_warmup: int = 10, num_iter: int = 50) -> dict:
    from segab_yolo import YOLO
    model = YOLO(model_path)
    first_img = None
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files = sorted(Path(source).glob(ext))
        if files:
            first_img = str(files[0])
            break
    if not first_img:
        return {"fps": 0.0, "ms": 0.0}
    total = 0
    for _ in range(num_warmup):
        results = model(first_img, verbose=False)
        total += sum(len(r.boxes) for r in results)
    if total == 0:
        return {"fps": 0.0, "ms": 0.0}
    start = time.perf_counter()
    for _ in range(num_iter):
        results = model(first_img, verbose=False)
        total += sum(len(r.boxes) for r in results)
    elapsed = time.perf_counter() - start
    fps = num_iter / elapsed
    ms = (elapsed / num_iter) * 1000
    return {"fps": round(fps, 1), "ms": round(ms, 1)}


def add_timing(df: pl.DataFrame, config: dict, dataset_filter: str = None) -> pl.DataFrame:
    datasets = config.get("datasets", {})
    timing_rows = []
    for ds_name, ds_cfg in datasets.items():
        if dataset_filter and ds_name != dataset_filter:
            continue
        timing_cfg = ds_cfg.get("timing", {})
        if not timing_cfg.get("enabled", False):
            continue
        source = timing_cfg.get("source", "")
        if not source:
            continue
        models_cfg = ds_cfg.get("models", {})
        for model_key in models_cfg:
            model_path = get_runs_dir(config) / ds_name / model_key / "weights" / "best.pt"
            if model_path.exists():
                t = timed_inference(str(model_path), source)
                timing_rows.append({"dataset": ds_name, "model_key": model_key, **t})

    timing_df = pl.DataFrame(timing_rows) if timing_rows else pl.DataFrame()
    if timing_df.height > 0:
        df = df.join(timing_df, on=["dataset", "model_key"], how="left")
    return df


def generate_heatmap(df: pl.DataFrame, config: dict, dataset_filter: str = None):
    output = get_output_base(config)
    datasets = config.get("datasets", {})
    for ds_name, ds_cfg in datasets.items():
        if dataset_filter and ds_name != dataset_filter:
            continue
        ds_df = df.filter(pl.col("dataset") == ds_name)
        if ds_df.height == 0:
            continue
        metric_names = ds_cfg.get("metrics") or [
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/precision(B)", "metrics/recall(B)",
        ]
        timing_cfg = ds_cfg.get("timing", {})
        if timing_cfg.get("enabled", False) and "fps" in ds_df.columns:
            metric_names = [m for m in metric_names if m not in ("fps", "ms")]
        labels = ds_cfg.get("labels", {})
        baseline_name = ds_cfg.get("baseline")

        available_metrics = [m for m in metric_names if m in ds_df.columns]
        if not available_metrics:
            continue

        short_metrics = [m.split("/")[-1].replace("(B)", "") for m in available_metrics]

        model_keys = ds_df["model_key"].to_list()
        baseline_idx = None
        for i, mk in enumerate(model_keys):
            if mk == baseline_name:
                baseline_idx = i
                break

        data = np.zeros((len(available_metrics), len(model_keys)))
        for j, mk in enumerate(model_keys):
            ms = ds_df.filter(pl.col("model_key") == mk)
            for i, mn in enumerate(available_metrics):
                val = ms[mn].to_list()[0] if ms.height > 0 else None
                if val is not None:
                    data[i, j] = float(val)

        if baseline_idx is not None:
            baseline_vals = data[:, baseline_idx:baseline_idx+1].copy()
            mask = baseline_vals != 0
            pct_change = np.full_like(data, np.nan)
            pct_change[:, :] = np.where(mask, ((data - baseline_vals) / np.abs(baseline_vals)) * 100, np.nan)
            display_data = pct_change
            vmin, vmax = -15, 15
            cmap = "coolwarm"
            fmt = ".2f"
            suffix = "_pct"
            title_suffix = " (% change vs baseline)"
        else:
            display_data = data
            vmin, vmax = None, None
            cmap = "viridis"
            fmt = ".3f"
            suffix = ""
            title_suffix = ""

        fig, ax = plt.subplots(figsize=(max(6, len(model_keys) * 1.5), max(4, len(available_metrics) * 0.8)))
        im = ax.imshow(display_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(model_keys)))
        ax.set_xticklabels([labels.get(mk, mk) for mk in model_keys], rotation=45, ha="right")

        ax.set_yticks(range(len(short_metrics)))
        ax.set_yticklabels(short_metrics)

        for i in range(len(short_metrics)):
            for j in range(len(model_keys)):
                val = display_data[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 10 else "black"
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=9, color=color)

        ax.set_title(f"{ds_name}{title_suffix}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        out_path = output / f"{ds_name}_heatmap{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Heatmap saved: {out_path}")


def generate_barplot(df: pl.DataFrame, config: dict, dataset_filter: str = None):
    output = get_output_base(config)
    datasets = config.get("datasets", {})
    for ds_name, ds_cfg in datasets.items():
        if dataset_filter and ds_name != dataset_filter:
            continue
        ds_df = df.filter(pl.col("dataset") == ds_name)
        if ds_df.height == 0:
            continue
        timing_cfg = ds_cfg.get("timing", {})
        if not timing_cfg.get("enabled", False) or "fps" not in ds_df.columns:
            continue

        labels = ds_cfg.get("labels", {})
        model_keys = ds_df["model_key"].to_list()
        fps_vals = []
        for mk in model_keys:
            row = ds_df.filter(pl.col("model_key") == mk)
            fps_vals.append(row["fps"].to_list()[0] if row.height > 0 else 0)

        baseline_name = ds_cfg.get("baseline")
        colors = []
        for mk in model_keys:
            if mk == baseline_name:
                colors.append("#2ecc71")
            else:
                colors.append("#3498db")

        fig, ax = plt.subplots(figsize=(max(6, len(model_keys) * 1.2), 5))
        x = np.arange(len(model_keys))
        bars = ax.bar(x, fps_vals, color=colors, width=0.6)

        for bar, val in zip(bars, fps_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(mk, mk) for mk in model_keys], rotation=45, ha="right")
        ax.set_ylabel("FPS")
        ax.set_title(f"{ds_name} — Inference Speed (FPS)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out_path = output / f"{ds_name}_fps_barplot.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  FPS barplot saved: {out_path}")


def generate_csv(df: pl.DataFrame, config: dict, dataset_filter: str = None):
    output = get_output_base(config)
    datasets = config.get("datasets", {})
    for ds_name, ds_cfg in datasets.items():
        if dataset_filter and ds_name != dataset_filter:
            continue
        ds_df = df.filter(pl.col("dataset") == ds_name)
        if ds_df.height == 0:
            continue
        out_path = output / f"{ds_name}_comparison.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in ds_df.columns if c not in ("dataset", "model_path")]
        ds_df.select(cols).write_csv(out_path)
        print(f"  CSV saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model training results.")
    parser.add_argument("--config", type=str, default="viz_config.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset filter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = get_output_base(config)
    output.mkdir(parents=True, exist_ok=True)

    print("Collecting results...")
    df = collect_results(config, args.dataset)
    if df.height == 0:
        print("No results found.")
        return

    print("Running timing benchmarks...")
    df = add_timing(df, config, args.dataset)

    print("Generating heatmap...")
    generate_heatmap(df, config, args.dataset)

    print("Generating barplot...")
    generate_barplot(df, config, args.dataset)

    print("Generating CSV...")
    generate_csv(df, config, args.dataset)

    print("Done.")


if __name__ == "__main__":
    main()
