"""Gradio GUI for SEGAb-YOLO pipeline."""

import os
import subprocess
import sys
import yaml
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRIPT_MAP = {
    "train": "scripts/batch_train.py",
    "infer": "scripts/batch_infer.py",
    "xai":   "scripts/batch_xai.py",
    "viz":   "scripts/batch_viz.py",
}

CONFIG_DEFAULTS = {
    "train":    "train_config.yaml",
    "infer":    "infer_config.yaml",
    "xai":      "xai_config.yaml",
    "viz":      "viz_config.yaml",
    "pipeline": "pipeline.yaml",
}

CSS = """
.log-box { font-family: monospace; font-size: 13px; }
.fixed-preview { max-height: 400px; max-width: 600px; overflow: hidden; }
.fixed-preview img { max-height: 400px; max-width: 600px; object-fit: contain; }
.fixed-text { height: 350px !important; overflow-y: auto !important; }
"""


def _resolve(p: str) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _load_yaml(rel_path: str) -> dict:
    path = _resolve(rel_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _scan_reports() -> list:
    reports_dir = REPO_ROOT / "reports"
    if not reports_dir.exists():
        return []
    items = []
    if (reports_dir / "comparison").exists():
        for f in sorted((reports_dir / "comparison").iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".csv"):
                items.append(str(f.relative_to(REPO_ROOT)))
    if (reports_dir / "xai").exists():
        for sub in sorted((reports_dir / "xai").iterdir()):
            if sub.is_dir():
                for f in sorted(sub.iterdir()):
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".csv"):
                        items.append(str(f.relative_to(REPO_ROOT)))
    return items


def _browse_reports(path: str) -> tuple:
    if not path:
        return None, ""
    full_path = _resolve(path)
    if not full_path.exists():
        return None, f"File not found: {full_path}"
    ext = full_path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg"):
        return str(full_path), ""
    if ext == ".csv":
        try:
            with open(full_path, encoding="utf-8") as f:
                content = f.read()
            return None, content
        except Exception as e:
            return None, f"Error reading CSV: {e}"
    return None, f"Unsupported file: {full_path.name}"


def _run_direct_generator(script_name: str, args: list):
    script_path = _resolve(script_name)
    cmd = [sys.executable, str(script_path)] + [a for a in args if a]
    try:
        sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(REPO_ROOT),
            encoding="utf-8", errors="replace", env=sub_env,
        )
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip()
        proc.stdout.close()
        proc.wait()
        if proc.returncode != 0:
            yield f"\n[ERROR] Exit code {proc.returncode}"
        else:
            yield "\n[DONE]"
    except Exception as e:
        yield f"\n[ERROR] {e}"


def _run_script_generator(script_key: str, config_rel: str, extra_flags: list):
    script = SCRIPT_MAP.get(script_key, script_key)
    script_path = _resolve(script)
    config_path = _resolve(config_rel)
    cmd = [sys.executable, str(script_path), "--config", str(config_path)]
    for f in extra_flags or []:
        if f:
            cmd.append(f)
    try:
        sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(REPO_ROOT),
            encoding="utf-8", errors="replace", env=sub_env,
        )
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip()
        proc.stdout.close()
        proc.wait()
        if proc.returncode != 0:
            yield f"\n[ERROR] Exit code {proc.returncode}"
        else:
            yield "\n[DONE]"
    except Exception as e:
        yield f"\n[ERROR] {e}"


def _run_pipeline_generator(config_rel: str):
    script_path = _resolve("scripts/run_pipeline.py")
    config_path = _resolve(config_rel)
    cmd = [sys.executable, str(script_path), "--pipeline", str(config_path)]
    try:
        sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(REPO_ROOT),
            encoding="utf-8", errors="replace", env=sub_env,
        )
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip()
        proc.stdout.close()
        proc.wait()
        if proc.returncode != 0:
            yield f"\n[ERROR] Exit code {proc.returncode}"
        else:
            yield "\n[DONE]"
    except Exception as e:
        yield f"\n[ERROR] {e}"


def build_app() -> gr.Blocks:
    app = gr.Blocks(title="SEGAb-YOLO Pipeline GUI")

    with app:
        gr.Markdown("# SEGAb-YOLO Pipeline GUI")

        # --------------------------------------------------------------- Pipeline
        with gr.TabItem("Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    pipeline_config = gr.Dropdown(
                        label="Pipeline config",
                        choices=["pipeline.yaml"],
                        value="pipeline.yaml",
                    )
                    pipeline_run_btn = gr.Button("Run Pipeline", variant="primary")
                with gr.Column(scale=2):
                    pipeline_log = gr.Textbox(
                        label="Output", lines=30, max_lines=80,
                        elem_classes=["log-box"],
                    )
            pipeline_run_btn.click(
                fn=_run_pipeline_generator,
                inputs=[pipeline_config],
                outputs=[pipeline_log],
            )

        # --------------------------------------------------------------- Train
        with gr.TabItem("Train"):
            with gr.Row():
                with gr.Column(scale=1):
                    train_cfg = gr.Dropdown(
                        label="Config", choices=["train_config.yaml"],
                        value="train_config.yaml",
                    )
                    train_cfg_editor = gr.Code(
                        label="train_config.yaml",
                        value=yaml.dump(_load_yaml("train_config.yaml"), default_flow_style=False) or "",
                        language="yaml",
                    )
                    train_dataset = gr.Textbox(
                        label="Dataset filter", placeholder="e.g. tomatoes"
                    )
                    train_model = gr.Textbox(
                        label="Model filter", placeholder="e.g. yolo11n"
                    )
                    train_dry = gr.Checkbox(label="Dry run", value=False)
                    train_run_btn = gr.Button("Run Train", variant="primary")
                with gr.Column(scale=2):
                    train_log = gr.Textbox(
                        label="Output", lines=30, max_lines=80,
                        elem_classes=["log-box"],
                    )

            def do_train(dataset, model, dry):
                flags = []
                if dataset:
                    flags += ["--dataset", dataset]
                if model:
                    flags += ["--model", model]
                if dry:
                    flags.append("--dry-run")
                yield from _run_script_generator("train", "train_config.yaml", flags)
            train_run_btn.click(
                fn=do_train,
                inputs=[train_dataset, train_model, train_dry],
                outputs=[train_log],
            )

        # --------------------------------------------------------------- Infer
        with gr.TabItem("Infer"):
            with gr.Row():
                with gr.Column(scale=1):
                    infer_cfg_editor = gr.Code(
                        label="infer_config.yaml",
                        value=yaml.dump(_load_yaml("infer_config.yaml"), default_flow_style=False) or "",
                        language="yaml",
                    )
                    infer_model = gr.Textbox(
                        label="Model filter", placeholder="e.g. tomatoes/yolo11n"
                    )
                    infer_run_btn = gr.Button("Run Infer", variant="primary")
                with gr.Column(scale=2):
                    infer_log = gr.Textbox(
                        label="Output", lines=30, max_lines=80,
                        elem_classes=["log-box"],
                    )

            def do_infer(model):
                flags = []
                if model:
                    flags += ["--model", model]
                yield from _run_script_generator("infer", "infer_config.yaml", flags)
            infer_run_btn.click(
                fn=do_infer,
                inputs=[infer_model],
                outputs=[infer_log],
            )

        # --------------------------------------------------------------- XAI
        with gr.TabItem("XAI"):
            with gr.Row():
                with gr.Column(scale=1):
                    xai_cfg_editor = gr.Code(
                        label="xai_config.yaml",
                        value=yaml.dump(_load_yaml("xai_config.yaml"), default_flow_style=False) or "",
                        language="yaml",
                    )
                    xai_model = gr.Textbox(
                        label="Model filter", placeholder="e.g. yolo11n"
                    )
                    xai_run_btn = gr.Button("Run XAI", variant="primary")
                with gr.Column(scale=2):
                    xai_log = gr.Textbox(
                        label="Output", lines=30, max_lines=80,
                        elem_classes=["log-box"],
                    )

            def do_xai(model):
                flags = []
                if model:
                    flags += ["--model", model]
                yield from _run_script_generator("xai", "xai_config.yaml", flags)
            xai_run_btn.click(
                fn=do_xai,
                inputs=[xai_model],
                outputs=[xai_log],
            )

        # --------------------------------------------------------------- Viz
        with gr.TabItem("Viz"):
            with gr.Row():
                with gr.Column(scale=1):
                    viz_cfg_editor = gr.Code(
                        label="viz_config.yaml",
                        value=yaml.dump(_load_yaml("viz_config.yaml"), default_flow_style=False) or "",
                        language="yaml",
                    )
                    viz_dataset = gr.Textbox(
                        label="Dataset filter", placeholder="e.g. tomatoes"
                    )
                    viz_run_btn = gr.Button("Run Viz", variant="primary")
                with gr.Column(scale=2):
                    viz_log = gr.Textbox(
                        label="Output", lines=30, max_lines=80,
                        elem_classes=["log-box"],
                    )

            def do_viz(dataset):
                flags = []
                if dataset:
                    flags += ["--dataset", dataset]
                yield from _run_script_generator("viz", "viz_config.yaml", flags)
            viz_run_btn.click(
                fn=do_viz,
                inputs=[viz_dataset],
                outputs=[viz_log],
            )

        # --------------------------------------------------------------- Organize
        with gr.TabItem("Organize"):
            with gr.Row():
                with gr.Column(scale=1):
                    org_dataset = gr.Textbox(
                        label="Dataset name",
                        placeholder="e.g. tomatoes",
                    )
                    org_source = gr.Textbox(
                        label="Source dir (optional)",
                        placeholder="default: models/{dataset}",
                    )
                    org_run_btn = gr.Button("Run Organize", variant="primary")
                with gr.Column(scale=2):
                    org_log = gr.Textbox(
                        label="Output", lines=30, max_lines=80,
                        elem_classes=["log-box"],
                    )

            def do_organize(dataset, source):
                args = ["scripts/organize_models.py", "--dataset", dataset]
                if source:
                    args += ["--source", source]
                yield from _run_direct_generator(args[0], args[1:])
            org_run_btn.click(
                fn=do_organize,
                inputs=[org_dataset, org_source],
                outputs=[org_log],
            )

        # --------------------------------------------------------------- Validate
        with gr.TabItem("Validate"):
            with gr.Row():
                with gr.Column(scale=1):
                    val_dataset = gr.Textbox(
                        label="Dataset name",
                        placeholder="e.g. tomatoes",
                    )
                    val_model = gr.Textbox(
                        label="Model filter (optional)",
                        placeholder="e.g. yolo11n",
                    )
                    val_yaml = gr.Textbox(
                        label="Dataset YAML (optional)",
                        placeholder="Auto-detected if empty",
                    )
                    val_verbose = gr.Checkbox(label="Verbose output", value=False)
                    val_run_btn = gr.Button("Run Validate", variant="primary")
                with gr.Column(scale=2):
                    val_log = gr.Textbox(
                        label="Output", lines=30, max_lines=80,
                        elem_classes=["log-box"],
                    )

            def do_validate(dataset, model_filter, yaml_path, verbose):
                args = ["scripts/validate_models.py", "--dataset", dataset]
                if model_filter:
                    args += ["--model", model_filter]
                if yaml_path:
                    args += ["--dataset-yaml", yaml_path]
                if verbose:
                    args.append("--verbose")
                yield from _run_direct_generator(args[0], args[1:])
            val_run_btn.click(
                fn=do_validate,
                inputs=[val_dataset, val_model, val_yaml, val_verbose],
                outputs=[val_log],
            )

        # --------------------------------------------------------------- Results
        with gr.TabItem("Results"):
            with gr.Row():
                with gr.Column(scale=1):
                    choices = _scan_reports()
                    file_list = gr.Dropdown(
                        label="Files in reports/",
                        choices=choices,
                        value=choices[0] if choices else None,
                        interactive=True,
                    )
                    refresh_btn = gr.Button("Refresh")
                with gr.Column(scale=2):
                    result_image = gr.Image(
                        label="Preview", show_label=True,
                        elem_classes=["fixed-preview"],
                    )
                    result_text = gr.Textbox(
                        label="File content",
                        lines=20, max_lines=30,
                        elem_classes=["log-box", "fixed-text"],
                    )

            refresh_btn.click(
                fn=_scan_reports,
                outputs=file_list,
            )
            file_list.change(
                fn=_browse_reports,
                inputs=file_list,
                outputs=[result_image, result_text],
            )
            app.load(
                fn=_browse_reports,
                inputs=file_list,
                outputs=[result_image, result_text],
            )

    return app


if __name__ == "__main__":
    app = build_app()
    import socket
    port = 7860
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            port = 7861
    app.launch(server_name="127.0.0.1", server_port=port, show_error=True, css=CSS)
