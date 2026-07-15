"""Run XAI (EigenCAM / GradCAM / Activation) on YOLO predictions.

Supports single images or batch processing of directories.

Typical usage::

    python -m scripts.xai_predict --model yolo26n.pt --method gradcam \\
        --source data/images --output xai_output
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure the repo root is on sys.path so the local segab_yolo (with
# custom modules like GAM/SimAM) is found even when run directly.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from segab_yolo import YOLO
from segab_yolo.nn.modules import C3k2
from segab_yolo.nn.modules.attention import GAM, SimAM
from segab_yolo.utils.xai import (
    DummyTarget,
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    VanillaActivation,
    generate_cam,
    preprocess_for_cam,
    scale_cam_image,
    show_cam_on_image,
)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def image_generator(source_path: str) -> Generator[Path, None, None]:
    """Yield image file paths from a file or directory.

    Args:
        source_path: Path to a single image or a directory.

    Yields:
        Path to each supported image file.
    """
    source = Path(source_path)
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            yield source
        else:
            print(f"Warning: {source} is not a supported image format. Skipping.")
    elif source.is_dir():
        for img_path in sorted(source.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield img_path
    else:
        raise FileNotFoundError(f"Source path does not exist: {source_path}")


def _resolve_block_target(layer: torch.nn.Module) -> torch.nn.Module:
    """Resolve a high-level block to a suitable CAM target submodule.

    * ``GAM`` / ``SimAM`` → the attention module itself
    * ``SPPF`` / ``SPP`` with a ``.cv2`` conv → the ``.cv2`` submodule
    * Any other block → its last ``nn.Conv2d``
    """
    class_name = type(layer).__name__
    BACKBONE_END_CLASSES = {"GAM", "SimAM"}
    if class_name in BACKBONE_END_CLASSES:
        return layer
    if hasattr(layer, "cv2") and "SPP" in class_name:
        return layer.cv2
    convs = [m for m in layer.modules() if isinstance(m, torch.nn.Conv2d)]
    if convs:
        return convs[-1]
    return layer


def _find_backbone_end(seq) -> int:
    """Return index of the last backbone layer (before first Upsample/Detect)."""
    HEAD_NAMES = {"Detect", "Segment", "Pose", "OBB"}
    for i in range(len(seq)):
        class_name = type(seq[i]).__name__
        if class_name in HEAD_NAMES or class_name == "Upsample":
            return i - 1 if i > 0 else 0
    return len(seq) - 1


def _find_prehead(seq) -> int:
    """Return index of the last backbone/neck layer before Detect."""
    HEAD_NAMES = {"Detect", "Segment", "Pose", "OBB"}
    for i in range(len(seq) - 1, -1, -1):
        if type(seq[i]).__name__ not in HEAD_NAMES:
            return i
    return len(seq) - 2


def find_target_layers(
    model: YOLO,
    target_layer_name: Optional[str] = None,
    target_layer_index: Optional[int] = None,
    location: str = "backbone",
) -> List[torch.nn.Module]:
    """Find target layer(s) for CAM visualisation.

    Walks the top-level ``nn.Sequential`` to locate the desired layer
    and resolves it to a concrete submodule suitable for CAM hooks.

    Args:
        model: Loaded YOLO model.
        target_layer_name: Optional layer type name to use as the target.
        target_layer_index: Optional index in ``model.model`` to use as the target.
        location: ``'backbone'`` (default, end of backbone before neck),
                  ``'prehead'`` (last block before Detect head),
                  or ``'both'`` (returns both).

    Returns:
        List of target layer module(s).
    """
    if not hasattr(model.model, "model"):
        return _fallback_target(model)

    seq = model.model.model

    if target_layer_index is not None:
        idx = target_layer_index
        if idx < 0:
            idx += len(seq)
        if idx < 0 or idx >= len(seq):
            raise IndexError(
                f"Target layer index {target_layer_index} is out of range "
                f"(model.model has {len(seq)} layers)."
            )
        layer = seq[idx]
        print(f"  Target layer: model.model[{idx}] ({type(layer).__name__})")
        return [layer]

    if target_layer_name:
        for i in range(len(seq)):
            if type(seq[i]).__name__ == target_layer_name:
                print(f"  Target layer: model.model[{i}] ({target_layer_name})")
                return [seq[i]]
        print(f"  Warning: requested target layer '{target_layer_name}' not found. "
              "Falling back to location-based selection.")

    if location == "prehead":
        idx = _find_prehead(seq)
        target = _resolve_block_target(seq[idx])
        print(f"  Target layer (pre-head): model.model[{idx}] "
              f"({type(seq[idx]).__name__}) -> {type(target).__name__}")
        return [target]

    if location == "both":
        bb_idx = _find_backbone_end(seq)
        ph_idx = _find_prehead(seq)
        # Avoid duplicate if they point at the same layer
        if bb_idx == ph_idx:
            target = _resolve_block_target(seq[bb_idx])
            print(f"  Target layer (backbone==prehead): model.model[{bb_idx}] "
                  f"({type(seq[bb_idx]).__name__}) -> {type(target).__name__}")
            return [target]
        bb_target = _resolve_block_target(seq[bb_idx])
        ph_target = _resolve_block_target(seq[ph_idx])
        print(f"  Target layers: backbone model.model[{bb_idx}] "
              f"({type(seq[bb_idx]).__name__}) -> {type(bb_target).__name__}, "
              f"pre-head model.model[{ph_idx}] "
              f"({type(seq[ph_idx]).__name__}) -> {type(ph_target).__name__}")
        return [bb_target, ph_target]

    # Default: backbone
    idx = _find_backbone_end(seq)
    target = _resolve_block_target(seq[idx])
    print(f"  Target layer (backbone): model.model[{idx}] "
          f"({type(seq[idx]).__name__}) -> {type(target).__name__}")
    return [target]


def _fallback_target(model: YOLO) -> List[torch.nn.Module]:
    """Fallback: last ``nn.Conv2d`` anywhere in the model."""
    all_convs = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
    if all_convs:
        print("  Target layer: last Conv2d in model (fallback)")
        return [all_convs[-1]]
    return []


def _resolve_device(device: Optional[str]) -> str:
    """Resolve the device string, auto-detecting CUDA if available.

    Args:
        device: User-specified device, or None for auto-detect.

    Returns:
        Device string (e.g. ``'cuda:0'``, ``'cpu'``).
    """
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def setup_model(
    model_path: str,
    method: str = "eigencam",
    device: Optional[str] = None,
    target_layer_name: Optional[str] = None,
    target_layer_index: Optional[int] = None,
    location: str = "backbone",
) -> Dict:
    """Load the YOLO model, configure attention modules, and find target layers.

    Args:
        model_path: Path to the ``.pt`` weights file.
        method: XAI method name (used to pre-build model-specific objects).
        device: Device override (auto-detected if None).

    Returns:
        Context dict with keys *device*, *model*, *attention_modules*,
        *target_layers*, *eigencam*, *activation*.
    """
    device = _resolve_device(device)
    print(f"Loading model: {model_path}  (device: {device})")
    model = YOLO(model_path)
    model.to(device)
    model.model.to(device)

    attention_modules = []
    for m in model.model.modules():
        if isinstance(m, (GAM, SimAM)):
            m.save_attention = True
            attention_modules.append(m)
    print(f"Found {len(attention_modules)} attention modules (GAM/SimAM).")

    target_layers = find_target_layers(
        model,
        target_layer_name=target_layer_name,
        target_layer_index=target_layer_index,
        location=location,
    )
    if not target_layers:
        raise RuntimeError("Could not find any suitable target layers for CAM.")
    print(f"Target layers: {len(target_layers)}")

    method_key = method.lower()

    eigencam_obj = None
    if method_key == "eigencam":
        eigencam_obj = EigenCAM(model.model, target_layers)
        print("EigenCAM initialized.")

    activation_obj = None
    if method_key == "activation":
        activation_obj = VanillaActivation(model.model, target_layers)
        print("VanillaActivation hook registered.")

    return {
        "device": device,
        "model": model,
        "attention_modules": attention_modules,
        "target_layers": target_layers,
        "eigencam": eigencam_obj,
        "activation": activation_obj,
    }


def draw_detections(image: np.ndarray, result, model) -> np.ndarray:
    """Draw bounding boxes and labels on an RGB image.

    Args:
        image: RGB image (uint8) to draw on.
        result: segab_yolo ``Results`` object with ``.boxes``.
        model: YOLO model instance (provides ``.names``).

    Returns:
        The annotated image (same array, modified in-place).
    """
    print(f"  Detected {len(result.boxes)} object(s):")
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        print(f"    [{i}] {cls_name} ({conf:.2%}) | Box: {xyxy.tolist()}")

        color = (255, 0, 128)
        thickness = 5
        cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)
    return image


def save_attention_maps(
    attention_modules: List, output_dir: str, stem: str, img_rgb: np.ndarray
) -> None:
    """Save GAM/SimAM attention visualisations alongside the XAI output.

    Args:
        attention_modules: List of modules with a ``.last_attention`` attribute.
        output_dir: Root XAI output directory.
        stem: Image filename stem for sub-directory naming.
        img_rgb: Original RGB image (used as base for overlays).
    """
    for idx, m in enumerate(attention_modules):
        if m.last_attention is None:
            continue
        mod_name = f"{type(m).__name__}_{idx}"
        mod_dir = os.path.join(output_dir, stem, mod_name)
        os.makedirs(mod_dir, exist_ok=True)

        att = m.last_attention
        if isinstance(att, dict):
            for k, v in att.items():
                if k == "spatial":
                    heatmap = v[0, 0].cpu().numpy()
                    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
                    viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
                    cv2.imwrite(
                        os.path.join(mod_dir, f"{k}_attention.jpg"),
                        cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
                    )
                elif k == "channel":
                    with open(os.path.join(mod_dir, f"{k}_values.txt"), "w") as f:
                        f.write(str(v[0].squeeze().cpu().numpy().tolist()))
        else:
            heatmap = att[0].mean(dim=0).cpu().numpy()
            heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
            viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
            cv2.imwrite(
                os.path.join(mod_dir, "simam_spatial_mean.jpg"),
                cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
            )


def process_single_image(
    img_path: Path,
    ctx: Dict,
    output_dir: str,
    method: str,
    conf_thres: float = 0.25,
    save_npy: bool = False,
    act_method: str = "l2",
    n_samples: int = 15,
) -> bool:
    """Run XAI on a single image and save the result.

    Args:
        img_path: Path to the image file.
        ctx: Context dict from :func:`setup_model`.
        output_dir: Output directory for visualisations.
        method: XAI method name.
        conf_thres: Detection confidence threshold.
        save_npy: If True, save raw CAM arrays as ``.npy`` files.
        act_method: Aggregation method for activation maps.
        n_samples: Number of noisy samples for SS-GradCAM++.

    Returns:
        True on success, False on failure.
    """
    device = ctx["device"]
    model = ctx["model"]
    attention_modules = ctx["attention_modules"]
    target_layers = ctx["target_layers"]
    eigencam_obj = ctx["eigencam"]
    activation_obj = ctx["activation"]

    stem = img_path.stem

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not load image: {img_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgsz = model.overrides.get("imgsz", 640)
    imgsz = int(imgsz) if not isinstance(imgsz, (list, tuple)) else int(imgsz[0])
    stride = int(model.model.stride.max()) if hasattr(model.model, "stride") else 32

    results = model.predict(str(img_path), save=False, verbose=False, conf=conf_thres)
    result = results[0]

    img_tensor, lb_params = preprocess_for_cam(img_rgb, imgsz, stride, device)

    method_key = method.lower()
    cam = None

    if method_key == "eigencam":
        cams = eigencam_obj(img_tensor, targets=[DummyTarget()])
        cam = cams[0]
    elif method_key == "activation":
        _ = model.model(img_tensor)
        cam = activation_obj.get_heatmap(method=act_method)
    else:
        if len(result.boxes) == 0:
            print(f"  No objects detected in {img_path.name}. Cannot run {method}.")
            return False
        target_box = result.boxes[0]
        t_layer = target_layers[-1]
        with torch.inference_mode(False):
            img_tensor.requires_grad_(True)
            cam = generate_cam(
                model.model, img_tensor, t_layer, target_box,
                method=method, n_samples=n_samples,
            )

    if cam is None:
        print(f"  Failed to generate {method} for {img_path.name}.")
        return False

    # Save raw CAM as .npy
    if save_npy:
        npy_dir = os.path.join(output_dir, "npy")
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, f"{stem}_{method}.npy"), cam)

    # Crop letterbox padding
    h_unpad, w_unpad = lb_params["new_unpad"][1], lb_params["new_unpad"][0]
    t, l = lb_params["top"], lb_params["left"]
    if h_unpad < cam.shape[0] or w_unpad < cam.shape[1]:
        cam = cam[t : t + h_unpad, l : l + w_unpad]
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_viz = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)

    cam_viz = draw_detections(cam_viz, result, model)

    out_path = os.path.join(output_dir, f"{stem}_{method}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(cam_viz, cv2.COLOR_RGB2BGR))
    print(f"  Saved -> {out_path}")

    save_attention_maps(attention_modules, output_dir, stem, img_rgb)
    return True


def run_xai(
    model_path: str,
    source: str,
    output_dir: str = "xai_output",
    method: str = "eigencam",
    conf_thres: float = 0.25,
    device: Optional[str] = None,
    target_layer_name: Optional[str] = None,
    target_layer_index: Optional[int] = None,
    save_npy: bool = False,
    act_method: str = "l2",
    n_samples: int = 15,
    verbose: bool = False,
    location: str = "backbone",
) -> None:
    """Process a single image or every image in a folder.

    Args:
        model_path: Path to the ``.pt`` weights file.
        source: Image file or directory of images.
        output_dir: Output directory for visualisations.
        method: XAI method name.
        conf_thres: Detection confidence threshold.
        device: Device override (auto-detected if None).
        save_npy: If True, save raw CAM arrays as ``.npy`` files.
        act_method: Aggregation method for activation maps.
        n_samples: Number of noisy samples for SS-GradCAM++.
        location: ``'backbone'``, ``'prehead'``, or ``'both'``.
    """
    if location == "both":
        for loc in ("backbone", "prehead"):
            loc_dir = os.path.join(output_dir, loc)
            run_xai(
                model_path=model_path,
                source=source,
                output_dir=loc_dir,
                method=method,
                conf_thres=conf_thres,
                device=device,
                target_layer_name=target_layer_name,
                target_layer_index=target_layer_index,
                save_npy=save_npy,
                act_method=act_method,
                n_samples=n_samples,
                verbose=verbose,
                location=loc,
            )
        return

    os.makedirs(output_dir, exist_ok=True)
    if verbose:
        ctx = setup_model(
            model_path,
            method=method,
            device=device,
            target_layer_name=target_layer_name,
            target_layer_index=target_layer_index,
            location=location,
        )
    else:
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = setup_model(
                model_path,
                method=method,
                device=device,
                target_layer_name=target_layer_name,
                target_layer_index=target_layer_index,
                location=location,
            )

    images = list(image_generator(source))
    total = len(images)
    if total == 0:
        print(f"No images found in: {source}")
        return

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing {total} image(s) with method: {method}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {ctx['device']}")
        print(f"{'=' * 60}\n")

    success = 0
    failed = 0

    pbar = tqdm(total=total, desc=f"Processing ({method})", disable=verbose)
    for i, img_path in enumerate(images, 1):
        if verbose:
            print(f"[{i}/{total}] {img_path.name}")
        try:
            import io
            import contextlib
            if verbose:
                ok = process_single_image(
                    img_path,
                    ctx,
                    output_dir,
                    method,
                    conf_thres=conf_thres,
                    save_npy=save_npy,
                    act_method=act_method,
                    n_samples=n_samples,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    ok = process_single_image(
                        img_path,
                        ctx,
                        output_dir,
                        method,
                        conf_thres=conf_thres,
                        save_npy=save_npy,
                        act_method=act_method,
                        n_samples=n_samples,
                    )
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            failed += 1
        pbar.update(1)
    pbar.close()

    if ctx.get("activation"):
        ctx["activation"].close()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Done! {success}/{total} succeeded, {failed} failed.")
        print(f"Results saved to: {output_dir}")
        print(f"{'=' * 60}")
    else:
        print(f"Done! {success}/{total} succeeded, {failed} failed. -> {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run XAI (EigenCAM / GradCAM / Activation) on YOLO predictions."
    )
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt",
        help="Path to YOLO model weights (.pt)",
    )
    parser.add_argument(
        "--source", type=str, default="segab_yolo/assets/bus.jpg",
        help="Path to a single image or a folder of images",
    )
    parser.add_argument(
        "--output", type=str, default="xai_output",
        help="Output directory for XAI visualizations",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="eigencam",
        choices=["eigencam", "gradcam", "gradcam++", "ss-gradcam++", "activation"],
        help="XAI method to use",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (e.g. cpu, cuda:0). Default: auto",
    )
    parser.add_argument(
        "--save-npy", action="store_true",
        help="Save raw CAM/activation arrays as .npy files",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help="Explicit target layer type name for CAM hooks, e.g. C3k2.",
    )
    parser.add_argument(
        "--target-layer-index",
        type=int,
        default=None,
        help="Explicit target layer index in model.model for CAM hooks.",
    )
    parser.add_argument(
        "--act-method",
        type=str,
        default="l2",
        choices=["l2", "mean", "max"],
        help="Aggregation method for activation maps",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=15,
        help="Number of noisy samples for SS-GradCAM++",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output instead of progress bar",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args and run XAI."""
    args = parse_args()
    run_xai(
        args.model,
        args.source,
        args.output,
        args.method,
        conf_thres=args.conf,
        device=args.device,
        target_layer_name=args.target_layer,
        target_layer_index=args.target_layer_index,
        save_npy=args.save_npy,
        act_method=args.act_method,
        n_samples=args.n_samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
