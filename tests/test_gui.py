"""Smoke tests for the Gradio GUI."""

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def test_build_app():
    """GUI builds without error (does not launch)."""
    from scripts.gui import build_app
    app = build_app()
    assert app is not None
    assert hasattr(app, "blocks")


def test_tabs_exist():
    """All expected tabs are present."""
    from scripts.gui import build_app
    app = build_app()
    tab_labels = set()
    for block in app.blocks.values():
        label = getattr(block, "label", None)
        if label:
            tab_labels.add(label)
    expected = {"Pipeline", "Train", "Infer", "XAI", "Viz", "Organize", "Validate", "Results"}
    assert expected.issubset(tab_labels), f"Missing tabs: {expected - tab_labels}"


def test_scan_reports():
    """_scan_reports returns a list without crashing."""
    from scripts.gui import _scan_reports
    items = _scan_reports()
    assert isinstance(items, list)


def test_browse_reports_invalid():
    """_browse_reports handles missing path gracefully."""
    from scripts.gui import _browse_reports
    img, text = _browse_reports("nonexistent/file.png")
    assert img is None
    assert "not found" in text.lower()


def test_resolve():
    """_resolve handles absolute and relative paths."""
    from scripts.gui import _resolve
    abs_p = _resolve(str(_repo_root / "train_config.yaml"))
    assert abs_p.exists()
    rel_p = _resolve("train_config.yaml")
    assert rel_p.exists()
    assert abs_p == rel_p
