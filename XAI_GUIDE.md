# Explainable AI (XAI) Guide for SEGAb-YOLO

This guide explains how to use the built-in XAI mechanisms to visualize and interpret the decisions of your YOLO models.

## 🚀 Overview

The repository provides tools to generate heatmaps (Class Activation Maps) and visualize internal attention mechanisms (GAM, SimAM). These tools help you understand which parts of an image the model is focusing on when making a detection.

### Supported Methods
- **EigenCAM**: Great for visualizing principal components of activations (doesn't require a specific class target).
- **Grad-CAM / Grad-CAM++**: Visualizes gradients to show class-specific importance.
- **SmoothGrad-CAM++**: A smoother version of Grad-CAM++ using noise injection.
- **Attention Maps**: Direct visualization of GAM (Global Attention Module) and SimAM (Simple Attention Module) layers.

---

## 🛠️ Prerequisites

Ensure you have the necessary dependencies installed:

```bash
pip install grad-cam
```
*(This is already included in the `pyproject.toml` dependencies)*.

---

## 📖 Using `xai_predict.py`

The main entry point for XAI is the `xai_predict.py` script.

### Basic Usage

To run **EigenCAM** (default) on an image:

```bash
python xai_predict.py --model path/to/your/model.pt --source image.jpg --output results_folder
```

### Specifying XAI Methods

You can choose different CAM variants using the `--method` flag:

| Method | Command | Use Case |
| :--- | :--- | :--- |
| **EigenCAM** | `--method eigencam` | General localization, no class target needed. |
| **Grad-CAM** | `--method gradcam` | Basic gradient-based class activation. |
| **Grad-CAM++** | `--method gradcam++` | Better object localization than Grad-CAM. |
| **SmoothGrad** | `--method ss-gradcam++` | Reduced noise in heatmaps (uses multiple samples). |

---

## 🧠 Visualizing Attention (GAM / SimAM)

If your model architecture includes **GAM** or **SimAM** modules (like in `yolo11-simam-full.yaml`), `xai_predict.py` will automatically detect them and save their internal attention maps.

1. The script finds all `GAM` and `SimAM` modules in the model.
2. It sets `save_attention = True` for each.
3. After inference, it extracts:
   - **GAM**: Channel and Spatial attention maps.
   - **SimAM**: Mean spatial attention maps.

These results are saved in subfolders within your output directory (e.g., `xai_output/GAM_0/spatial_attention.jpg`).

---

## 🎯 Target Layers Configuration

By default, `xai_predict.py` tries to find the best layers to visualize using a heuristic:
- It targets the `Conv2d` layers in the last 5 modules of the model.
- If none are found, it falls back to the very last `Conv2d` layer.

### Customizing Layers
If the heatmaps don't look right, you might need to adjust the target layers in `xai_predict.py`. Generally, you want to target the layers just before the prediction head or the final layers of the backbone (e.g., SPPF).

```python
# In xai_predict.py (around line 40)
for m in list(model.model.modules())[-10:]: # Increase range to find earlier layers
    if isinstance(m, torch.nn.Conv2d):
        target_layers.append(m)
```

---

## 📁 Output Structure

The output directory will contain:
1. `[method].jpg`: The overlay of the CAM heatmap on the original image.
2. `[ModuleType]_[Index]/`: Folders containing internal attention maps (if applicable).
   - `spatial_attention.jpg`: Visual representation of where the module is "looking".
   - `channel_values.txt`: Numerical values of channel importance (for GAM).

---

## 💡 Tips for Better Results

- **Class Selection**: For Grad-CAM variants, the script currently targets the **highest confidence detection**. If you want to target a specific class, you would need to modify `xai_predict.py` to filter results.
- **Resolution**: Heatmaps are generated at the resolution of the target layer's feature map. Using earlier layers provides higher resolution but less semantic meaning; using later layers provides lower resolution but better class-specific localization.
- **Confidence**: If the model doesn't detect anything, Grad-CAM methods will fail. EigenCAM will still work as it doesn't depend on the output scores.
