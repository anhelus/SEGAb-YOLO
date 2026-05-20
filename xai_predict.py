import sys
import os
sys.path.append(os.getcwd())

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.xai import EigenCAM, GradCAM, GradCAMPlusPlus, generate_cam, show_cam_on_image, scale_cam_image
from ultralytics.nn.modules.attention import GAM, SimAM
import argparse

class DummyTarget:
    def __call__(self, model_output):
        return torch.tensor(0.0)

def run_xai(model_path, source, output_dir='xai_output', method='eigencam'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    # Use a YAML to ensure we have GAM modules (weights will be random but structure correct)
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Enable save_attention for GAM and SimAM modules
    attention_modules = []
    for m in model.model.modules():
        if isinstance(m, (GAM, SimAM)):
            m.save_attention = True
            attention_modules.append(m)
            
    print(f"Found {len(attention_modules)} attention modules (GAM/SimAM).")
    
    # Prepare target layers
    # We target the last convolution layer of the backbone or head. 
    # For YOLOv8/11, let's try to target the last layer of the backbone (usually SPPF) 
    # and the Detect head layers if possible.
    # For now, let's just target the last layer of the model.module list, or generic Conv2d layers.
    # A safe bet for EigenCAM is usually the last 2D layer before the head or within the head.
    
    # Let's target all Conv2d layers in the last few modules of the model
    target_layers = []
    # Heuristic: Target the last 2 modules' Conv2d layers
    for m in list(model.model.modules())[-5:]:
        if isinstance(m, torch.nn.Conv2d):
            target_layers.append(m)
            
    if not target_layers:
        print("Warning: Could not find specific target layers. Using all Conv2d outputs might be too heavy.")
        # Fallback: traverse and find the last conv2d
        all_convs = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
        if all_convs:
            target_layers = [all_convs[-1]]
            
    print(f"Target layers: {len(target_layers)}")
    
    # Load image
    img = cv2.imread(source)
    if img is None:
        raise ValueError(f"Could not load image: {source}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess for YOLO
    results = model.predict(source, save=False, verbose=False)
    result = results[0]
    
    # Prepare image tensor resized to standard 640x640 to prevent SVD memory errors and match training scale
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(model.device)
    
    if method.lower() == 'eigencam':
        print("Running EigenCAM...")
        eigencam = EigenCAM(model.model, target_layers)
        cams = eigencam(img_tensor, targets=[DummyTarget()])
        cam = cams[0] # PCA 1
    else:
        print(f"Running {method}...")
        if len(result.boxes) == 0:
            print("No objects detected. Cannot run Grad-CAM variants.")
            return
        
        # Take the best detection
        target_box = result.boxes[0]
        
        # We need a single target layer for generate_cam as currently implemented
        # Use the last one
        t_layer = target_layers[-1]
        
        cam = generate_cam(model.model, img_tensor, t_layer, target_box, method=method)
        
    # Save Result
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_viz = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)
    
    # Print predictions and draw bounding boxes on the image
    print(f"\nDetected {len(result.boxes)} objects:")
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        print(f"  - Object {i}: {cls_name} (Class {cls_id}) | Confidence: {conf:.4f} | Box: {xyxy.tolist()}")
        
        # Draw bounding boxes and labels on the heatmap visualization
        color = (255, 0, 128)  # Vibrant magenta/pink in RGB
        thickness = 5
        cv2.rectangle(cam_viz, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)
        
        # Prepare label and font properties
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 3
        
        # Get text dimensions to render background card
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Adjust text placement so it doesn't render outside the image top boundary
        text_y = xyxy[1] - 10
        if text_y - text_height < 0:
            text_y = xyxy[1] + text_height + 15
            
        # Draw background banner for the text
        cv2.rectangle(
            cam_viz, 
            (xyxy[0], text_y - text_height - 5), 
            (xyxy[0] + text_width + 10, text_y + baseline + 5), 
            color, 
            -1 # Solid fill
        )
        
        # Draw text inside the banner (white for high contrast)
        cv2.putText(
            cam_viz, 
            label, 
            (xyxy[0] + 5, text_y), 
            font, 
            font_scale, 
            (255, 255, 255), 
            font_thickness
        )
        
    out_path = os.path.join(output_dir, f'{method}.jpg')
    cv2.imwrite(out_path, cv2.cvtColor(cam_viz, cv2.COLOR_RGB2BGR))
    print(f"Saved {method} (with bounding boxes) to {out_path}")
        
    # Process Attention Maps
    for i, m in enumerate(attention_modules):
        if m.last_attention is not None:
            # Create subfolder for this module
            mod_name = f"{type(m).__name__}_{i}"
            mod_dir = os.path.join(output_dir, mod_name)
            os.makedirs(mod_dir, exist_ok=True)
            
            att = m.last_attention
            
            if isinstance(att, dict): # GAM
                for k, v in att.items():
                    # v is attention tensor
                    # Channel attention: [B, C, 1, 1] -> Plot as bar chart? Or apply to features?
                    # Spatial attention: [B, 1, H, W] -> Heatmap
                    
                    if k == 'spatial':
                        # v shape: [B, 1, H, W]
                        heatmap = v[0, 0].cpu().numpy()
                        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                        # Normalize
                        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
                        viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
                        cv2.imwrite(os.path.join(mod_dir, f'{k}_attention.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    elif k == 'channel':
                        # Save raw values?
                         with open(os.path.join(mod_dir, f'{k}_values.txt'), 'w') as f:
                             f.write(str(v[0].squeeze().cpu().numpy().tolist()))
            else: # SimAM
                # SimAM outputs sigmoid(y) which is [B, C, H, W]
                # We can average over channels to get a spatial map
                # v shape: [B, C, H, W]
                heatmap = att[0].mean(dim=0).cpu().numpy()
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
                viz = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
                cv2.imwrite(os.path.join(mod_dir, 'simam_spatial_mean.jpg'), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    print(f"Processing complete. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to model')
    parser.add_argument('--source', type=str, default='ultralytics/assets/bus.jpg', help='Path to image')
    parser.add_argument('--output', type=str, default='xai_output', help='Output directory')
    parser.add_argument('--method', type=str, default='eigencam', help='XAI method: gradcam, gradcam++, ss-gradcam++, eigencam')
    args = parser.parse_args()
    
    run_xai(args.model, args.source, args.output, args.method)
