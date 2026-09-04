"""Orchestrate the full pipeline: infer -> XAI -> viz."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm


def resolve(path: str, root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


_STEP_TOTALS = {
    "Train": "epochs",
    "Validate": "models",
    "Infer": "images",
    "XAI": "images",
    "Viz": "models",
}


def _read_total_from_config(step_name: str, step_cfg: dict, root: Path) -> int:
    """Try to determine total progress units from the step's config file."""
    if step_name == "Train":
        cfg_file = step_cfg.get("config", "train_config.yaml")
        p = resolve(cfg_file, root)
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("training", {}).get("epochs", 100)
    if step_name == "Validate":
        cfg_file = step_cfg.get("config", "train_config.yaml")
        p = resolve(cfg_file, root)
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            return len(cfg.get("models", []))
    if step_name in ("Infer", "Viz"):
        cfg_file = step_cfg.get("config", "infer_config.yaml")
        p = resolve(cfg_file, root)
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            total = 0
            for ds in cfg.get("datasets", {}).values():
                total += len(ds.get("models", []))
            return total or 30
    if step_name == "XAI":
        cfg_file = step_cfg.get("config", "xai_config.yaml")
        p = resolve(cfg_file, root)
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            total = 0
            for ds in cfg.get("datasets", {}).values():
                total += len(ds.get("models", [])) * len(cfg.get("methods", []))
            return total or 30
    return 100


def run() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline: infer -> XAI -> viz.")
    parser.add_argument("--pipeline", type=str, default="pipeline.yaml", help="Pipeline config file.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset filter (passed to supported steps).")
    parser.add_argument("--model", type=str, default=None, help="Model filter (passed to supported steps).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N images per dataset for Infer/XAI.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = resolve(args.pipeline, root)
    with open(cfg_path) as f:
        pipeline_cfg = yaml.safe_load(f)

    steps = pipeline_cfg.get("steps", [])
    if not steps:
        print("No steps defined in pipeline config.", flush=True)
        return

    datasets_to_run = []
    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        train_cfg_path = root / "train_config.yaml"
        if train_cfg_path.exists():
            with open(train_cfg_path) as f:
                train_cfg = yaml.safe_load(f)
            datasets_to_run = [d["name"] for d in train_cfg.get("datasets", []) if "name" in d]
        if not datasets_to_run:
            tqdm.write("No datasets to run. Use --dataset or define datasets in train_config.yaml.")
            return

    n_steps = len(steps)
    n_datasets = len(datasets_to_run)
    global_pbar = tqdm(total=n_steps * n_datasets, desc="Pipeline", position=0)

    for ds_idx, ds_name in enumerate(datasets_to_run):
        tqdm.write(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

        for step_idx, step in enumerate(steps):
            step_name = step.get("name", "?")
            script = step.get("script", "")
            step_config = step.get("config", "")
            forward = step.get("forward_flags", [])

            if not script:
                tqdm.write(f"[{step_name}] No script defined, skipping.")
                global_pbar.update(1)
                continue

            script_path = resolve(script, root)
            cmd = [sys.executable, str(script_path)]

            if step_config:
                cfg_resolved = resolve(step_config, root)
                cmd += ["--config", str(cfg_resolved)]

            for flag in forward:
                if flag == "dataset":
                    cmd += ["--dataset", ds_name]
                elif flag == "model" and args.model:
                    cmd += ["--model", args.model]
                elif flag == "limit" and args.limit:
                    cmd += ["--limit", str(args.limit)]

            total_units = _read_total_from_config(step_name, step, root)

            if not args.dry_run:
                step_pbar = tqdm(
                    total=total_units,
                    desc=f"  {step_name}",
                    leave=False,
                    position=1,
                )

            flags_str = " ".join(str(c) for c in cmd[2:])
            tqdm.write(f"[{step_name}] Running: {script_path.name} {flags_str}")

            if args.dry_run:
                if args.verbose:
                    tqdm.write(f"  (dry-run) {' '.join(cmd)}")
                global_pbar.update(1)
                continue

            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                env=env,
            )

            epoch_re = re.compile(r"^\s*(\d+)/(\d+)\s+")

            for line in proc.stdout:
                stripped = line.rstrip()

                if args.verbose:
                    tqdm.write(stripped)

                if step_name == "Train":
                    m = epoch_re.match(stripped)
                    if m:
                        cur = int(m.group(1))
                        step_pbar.n = cur
                        step_pbar.refresh()

                elif step_name == "Validate":
                    if stripped.startswith("["):
                        m = re.match(r"\[(\d+)/(\d+)\]", stripped)
                        if m:
                            step_pbar.n = int(m.group(1))
                            step_pbar.refresh()

                elif step_name == "Infer":
                    if stripped.startswith("["):
                        m = re.match(r"\[(\d+)/(\d+)\]", stripped)
                        if m:
                            step_pbar.n = int(m.group(1))
                            step_pbar.refresh()

                elif step_name == "XAI":
                    if stripped.startswith("["):
                        m = re.match(r"\[(\d+)/(\d+)\]", stripped)
                        if m:
                            step_pbar.n = int(m.group(1))
                            step_pbar.refresh()

                elif step_name == "Viz":
                    if stripped.startswith("["):
                        m = re.match(r"\[(\d+)/(\d+)\]", stripped)
                        if m:
                            step_pbar.n = int(m.group(1))
                            step_pbar.refresh()

            proc.stdout.close()
            ret = proc.wait()

            step_pbar.update(0)
            step_pbar.close()

            if ret != 0:
                tqdm.write(f"[{step_name}] ERROR: exit code {ret}")
                sys.exit(ret)

            global_pbar.update(1)
            tqdm.write(f"[{step_name}] Done.")

    global_pbar.close()
    print("\nPipeline completed.", flush=True)


if __name__ == "__main__":
    run()
