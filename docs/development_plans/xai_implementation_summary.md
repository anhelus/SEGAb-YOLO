# XAI Implementation Walkthrough

I have implemented Explainable AI (XAI) features for the SEGAb-YOLO repository, focusing on **EigenCAM** and **Attention Map Visualization** for GAM and SimAM modules.

## Changes

### 1. Attention Modules (`ultralytics/nn/modules/attention.py`)
- **GAM (Global Attention Mechanism):** Updated to optionally re-compute and save channel and spatial attention maps during the forward pass.
- **SimAM (Simple Attention Module):** Updated to optionally save the computed energy map (sigmoid output).
- Added `save_attention` attribute to both classes to control this behavior.

### 2. XAI Module (`ultralytics/xai/`)
- **`eigencam.py`**: Implemented the `EigenCAM` class which uses PCA on the activations of target layers (hooks) to generate a class activation map.
- **`utils.py`**: Added visualization helpers like `show_cam_on_image`.

### 3. Verification Script (`xai_predict.py`)
- Created a script to demonstrating how to run XAI on a model.
- Loading `yolo11-gam.yaml` (or other models) enables the GAM/SimAM modules, allowing visualization of their attention maps.
- Generates `EigenCAM` heatmap for the input image.

## Verification

To verify the changes, run the `xai_predict.py` script using your environment's Python:

```bash
python xai_predict.py --model ultralytics/cfg/models/11/yolo11-gam.yaml --source ultralytics/assets/bus.jpg
```

This will produce an `xai_output` directory containing:
- `eigencam.jpg`: The EigenCAM visualization.
- Subdirectories for each GAM/SimAM module with their respective attention maps.
