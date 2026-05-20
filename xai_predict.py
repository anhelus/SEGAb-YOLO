import sys
import os
sys.path.append(os.getcwd())

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.xai import EigenCAM, GradCAM, GradCAMPlusPlus, generate_cam, show_cam_on_image, scale_cam_image
from ultralytics.nn.modules.attention import GAM, SimAM
import argparse

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class DummyTarget:
    def __call__(self, model_output):
        return torch.tensor(0.0)


def image_generator(source_path):
    """
    Generator that yields image file paths from a directory.
    If source_path is a single file, yields just that file.
    If source_path is a directory, recursively yields all image files.
    """
    source = Path(source_path)

    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            yield source
        else:
            print(f"Warning: {source} is not a supported image format. Skipping.")
    elif source.is_dir():
        # Sort for deterministic ordering
        for img_path in sorted(source.rglob('*')):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield img_path
    else:
        raise FileNotFoundError(f"Source path does not exist: {source_path}")


def setup_model(model_path):
    """Load the YOLO model and configure attention modules. Returns (model, attention_modules, target_layers)."""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Enable save_attention for GAM and SimAM modules
    attention_modules = []
    for m in model.model.modules():
        if isinstance(m, (GAM, SimAM)):
            m.save_attention = True
            attention_modules.append(m)

    print(f"Found {len(attention_modules)} attention modules (GAM/SimAM).")

    # Prepare target layers – last few Conv2d layers as a heuristic
    target_layers = []
    for m in list(model.model.modules())[-5:]:
        if isinstance(m, torch.nn.Conv2d):
            target_layers.append(m)

    if not target_layers:
        print("Warning: Could not find specific target layers in tail. Falling back to last Conv2d.")
        all_convs = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
        if all_convs:
            target_layers = [all_convs[-1]]

    print(f"Target layers: {len(target_layers)}")
    return model, attention_modules, target_layers


def process_single_image(img_path, model, attention_modules, target_layers, output_dir, method):
    """
    Run XAI on a single image and save the result.
    Returns True on success, False on failure.
    """
    img_path = Path(img_path)
    stem = img_path.stem  # filename without extension

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Could not load image: {img_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO prediction
    results = model.predict(str(img_path), save=False, verbose=False)
    result = results[0]

    # Prepare image tensor resized to 640x640 (prevents SVD memory errors)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(model.device)

    # --- CAM computation ---
    if method.lower() == 'eigencam':
        eigencam = EigenCAM(model.model, target_layers)
        cams = eigencam(img_tensor, targets=[DummyTarget()])
        cam = cams[0]
    else:
        if len(result.boxes) == 0:
            print(f"  ✗ No objects detected in {img_path.name}. Cannot run {method}.")
            return False

        target_box = result.boxes[0]
        t_layer = target_layers[-1]
        cam = generate_cam(model.model, img_tensor, t_layer, target_box, method=method)

    # --- Visualize ---
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_viz = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)

    # Print and draw predictions
    print(f"  Detected {len(result.boxes)} object(s):")
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        print(f"    [{i}] {cls_name} ({conf:.2%}) | Box: {xyxy.tolist()}")

        # Draw bounding box
        color = (255, 0, 128)  # Neon magenta (RGB)
        thickness = 5
        cv2.rectangle(cam_viz, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)

        # Label with background banner
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 3
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        text_y = xyxy[1] - 10
        if text_y - text_h < 0:
            text_y = xyxy[1] + text_h + 15

        cv2.rectangle(
            cam_viz,
            (xyxy[0], text_y - text_h - 5),
            (xyxy[0] + text_w + 10, text_y + baseline + 5),
            color, -1
        )
        cv2.putText(cam_viz, label, (xyxy[0] + 5, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Save CAM image
    out_path = os.path.join(output_dir, f"{stem}_{method}.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(cam_viz, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Saved → {out_path}")

    # --- Attention maps ---
    for idx, m in enumerate(attention_modules):
        if m.last_attention is not None:
            mod_name = f"{type(m).__name__}_{idx}"
            mod_dir = os.path.join(output_dir, stem, mod_name)
            os.makedirs(mod_dir, exist_ok=True)

            att = m.last_attention

            if isinstance(att, dict):  # GAM
                for k, v in att.items():
                    if k == 'spatial':
                        heatmap = v[0, 0].cpu().numpy()
                        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
                        viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
                        cv2.imwrite(os.path.join(mod_dir, f'{k}_attention.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    elif k == 'channel':
                        with open(os.path.join(mod_dir, f'{k}_values.txt'), 'w') as f:
                            f.write(str(v[0].squeeze().cpu().numpy().tolist()))
            else:  # SimAM
                heatmap = att[0].mean(dim=0).cpu().numpy()
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
                viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
                cv2.imwrite(os.path.join(mod_dir, 'simam_spatial_mean.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    return True


def run_xai(model_path, source, output_dir='xai_output', method='eigencam'):
    """Main entry point: process a single image or every image in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    model, attention_modules, target_layers = setup_model(model_path)

    # Count images first for progress reporting
    images = list(image_generator(source))
    total = len(images)

    if total == 0:
        print(f"No images found in: {source}")
        return

    print(f"\n{'='*60}")
    print(f"Processing {total} image(s) with method: {method}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    success = 0
    failed = 0

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{total}] {img_path.name}")
        try:
            ok = process_single_image(img_path, model, attention_modules, target_layers, output_dir, method)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
        print()

    # Summary
    print(f"{'='*60}")
    print(f"Done! {success}/{total} succeeded, {failed} failed.")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI (EigenCAM / GradCAM) on YOLO predictions.")
    parser.add_argument('--model',  type=str, default='yolo11n.pt',
                        help='Path to YOLO model weights (.pt)')
    parser.add_argument('--source', type=str, default='ultralytics/assets/bus.jpg',
                        help='Path to a single image or a folder of images')
    parser.add_argument('--output', type=str, default='xai_output',
                        help='Output directory for XAI visualizations')
    parser.add_argument('--method', type=str, default='eigencam',
                        choices=['eigencam', 'gradcam', 'gradcam++', 'ss-gradcam++'],
                        help='XAI method to use')
    args = parser.parse_args()

    run_xai(args.model, args.source, args.output, args.method)
