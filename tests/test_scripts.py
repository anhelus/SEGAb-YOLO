"""Tests for custom pipeline scripts."""

import sys
from pathlib import Path

import pytest
import yaml

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

SCRIPTS_DIR = _repo_root / "scripts"
CONFIGS_DIR = _repo_root


# ---------------------------------------------------------------------------
# Config existence & schema
# ---------------------------------------------------------------------------

def _cfg(name: str) -> dict:
    p = CONFIGS_DIR / name
    assert p.exists(), f"Config not found: {p}"
    with open(p) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("cfg_name", [
    "train_config.yaml",
    "infer_config.yaml",
    "xai_config.yaml",
    "viz_config.yaml",
    "pipeline.yaml",
])
def test_config_exists(cfg_name):
    assert (CONFIGS_DIR / cfg_name).exists()


def test_train_config_schema():
    cfg = _cfg("train_config.yaml")
    assert "datasets" in cfg
    assert "models" in cfg
    assert "training" in cfg
    assert len(cfg["datasets"]) >= 1
    for ds in cfg["datasets"]:
        assert "name" in ds
        assert "path" in ds
        assert "project" in ds


def test_infer_config_schema():
    cfg = _cfg("infer_config.yaml")
    assert "datasets" in cfg
    assert len(cfg["datasets"]) >= 1
    for ds_name, ds_cfg in cfg["datasets"].items():
        assert "source" in ds_cfg
        assert "models" in ds_cfg
        assert len(ds_cfg["models"]) >= 1


def test_xai_config_schema():
    cfg = _cfg("xai_config.yaml")
    assert "datasets" in cfg
    assert "methods" in cfg
    assert len(cfg["datasets"]) >= 1
    assert len(cfg["methods"]) >= 1


def test_viz_config_schema():
    cfg = _cfg("viz_config.yaml")
    assert "datasets" in cfg
    assert "output_base" in cfg
    assert "runs_dir" in cfg
    for ds_name, ds_cfg in cfg["datasets"].items():
        assert "metrics" in ds_cfg, f"Missing 'metrics' in dataset '{ds_name}'"
        assert "labels" in ds_cfg, f"Missing 'labels' in dataset '{ds_name}'"
        assert "models" in ds_cfg, f"Missing 'models' in dataset '{ds_name}'"
        assert len(ds_cfg["models"]) >= 1


def test_pipeline_config_schema():
    cfg = _cfg("pipeline.yaml")
    assert "steps" in cfg
    assert len(cfg["steps"]) >= 1
    for step in cfg["steps"]:
        assert "name" in step
        assert "script" in step
        assert "config" in step


# ---------------------------------------------------------------------------
# Script imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script_name", [
    "batch_train.py",
    "batch_infer.py",
    "batch_xai.py",
    "batch_viz.py",
    "run_pipeline.py",
    "validate_models.py",
    "organize_models.py",
])
def test_script_importable(script_name):
    import importlib.util
    path = SCRIPTS_DIR / script_name
    assert path.exists(), f"Script not found: {path}"
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    assert spec is not None, f"Could not load spec for {script_name}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


# ---------------------------------------------------------------------------
# train_config model-list consistency
# ---------------------------------------------------------------------------

def test_train_config_models_list():
    """Models in train_config must be a list (not commented out)."""
    cfg = _cfg("train_config.yaml")
    models = cfg.get("models", [])
    assert isinstance(models, list)
    assert len(models) >= 1, "No active models in train_config.yaml (all commented out?)"
    for m in models:
        assert isinstance(m, str) and m.strip(), f"Invalid model entry: {m!r}"


# ---------------------------------------------------------------------------
# batch_train load_config
# ---------------------------------------------------------------------------

def test_batch_train_load_config():
    from scripts.batch_train import load_config, validate_config
    cfg = load_config(str(CONFIGS_DIR / "train_config.yaml"))
    validate_config(cfg)


# ---------------------------------------------------------------------------
# batch_infer load_config
# ---------------------------------------------------------------------------

def test_batch_infer_load_config():
    from scripts.batch_infer import load_config, validate_config
    cfg = load_config(str(CONFIGS_DIR / "infer_config.yaml"))
    validate_config(cfg)


# ---------------------------------------------------------------------------
# batch_xai load_config
# ---------------------------------------------------------------------------

def test_batch_xai_load_config():
    from scripts.batch_xai import load_config, validate_config
    cfg = load_config(str(CONFIGS_DIR / "xai_config.yaml"))
    validate_config(cfg)


# ---------------------------------------------------------------------------
# batch_viz load_config
# ---------------------------------------------------------------------------

def test_batch_viz_load_config():
    from scripts.batch_viz import load_config
    cfg = load_config(str(CONFIGS_DIR / "viz_config.yaml"))
    assert "datasets" in cfg


# ---------------------------------------------------------------------------
# Dry-run tests (fast, no GPU/models needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script_module,config_name,extra_args", [
    ("scripts.batch_train", "train_config.yaml", ["--dry-run", "--dataset", "tomatoes", "--model", "yolo11n"]),
    ("scripts.batch_infer", "infer_config.yaml", ["--dry-run", "--dataset", "tomatoes"]),
    ("scripts.batch_xai",  "xai_config.yaml",  ["--dry-run", "--dataset", "tomatoes"]),
])
def test_dry_run(script_module, config_name, extra_args):
    import importlib
    mod = importlib.import_module(script_module)
    cfg_path = str(CONFIGS_DIR / config_name)
    if hasattr(mod, "load_config"):
        cfg = mod.load_config(cfg_path)
    else:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

    if script_module == "scripts.batch_train":
        mod.batch_run(cfg, dataset_filter="tomatoes", model_filter="yolo11n", dry_run=True)
    elif script_module == "scripts.batch_infer":
        mod.batch_run(cfg, dataset_filter="tomatoes", dry_run=True)
    elif script_module == "scripts.batch_xai":
        mod.batch_run(cfg, dataset_filter="tomatoes", dry_run=True)
