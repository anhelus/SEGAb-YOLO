# Implementation Plan - XAI for SEGAb-YOLO

The goal is to implement Explainable AI (XAI) features, specifically EigenCAM and Attention Maps, for the SEGAb-YOLO model.

## User Review Required

> [!IMPORTANT]
> **XAI Focus:** Attention map visualization will be implemented specifically for `GAM` (Global Attention Mechanism) and `SimAM` (Simple Attention Module).
> **Modifications:** This requires modifying `ultralytics/nn/modules/attention.py` to expose internal attention weights.

## Proposed Changes

### 1. Modify Attention Modules
**File:** `ultralytics/nn/modules/attention.py`

*   **Goal:** Expose attention maps for visualization.
*   **Target Classes:** `GAM`, `SimAM`.
*   **Changes:**
    *   Add `save_attention` flag/logic to the modules.
    *   **GAM:** Capture the output of `self.channel_attention(x)` and `self.spatial_attention(x)`.
    *   **SimAM:** Capture the energy map `y` or the sigmoid output.

### 2. Implement EigenCAM and XAI Utilities
**New Directory:** `ultralytics/xai`

#### [NEW] `ultralytics/xai/__init__.py`
*   Expose `EigenCAM`.

#### [NEW] `ultralytics/xai/eigencam.py`
*   **Class:** `EigenCAM`
*   **Logic:**
    *   Hook into the last 2D layer or user-specified layers of the backbone/head.
    *   Compute PCA of activations.
    *   Return the first principal component as the CAM.

#### [NEW] `ultralytics/xai/utils.py`
*   Visualization helpers (`show_cam_on_image`, etc.).

### 3. Integration Script
**New File:** `xai_predict.py`

*   Example usage:
    ```python
    from ultralytics import YOLO
    from ultralytics.xai import EigenCAM

    model = YOLO("yolo11n.pt")
    # ... run xai ...
    ```

### 4. Decoupling from Upstream
**Files:** `ultralytics/utils/checks.py`, `ultralytics/utils/__init__.py`

*   **Goal:** Prevent automatic connections to Ultralytics servers (updates, analytics, assets).
*   **Changes:**
    *   **Disable Online Checks:** Modify `check_online()` or environment variables to default to offline mode or strictly disable "call home".
    *   **Disable Auto-Updates:** Modify `check_pip_update_available` to always return False.
    *   **Disable Model Auto-Download (Optional):** Users should provide local weights. We can warn instead of downloading.
    *   **Asset URL:** Point `ASSETS_URL` to a placeholder or local path to prevent accidental downloads.

## Verification Plan

### Decoupling Verification
1.  **Network Check:** Run inference with network disabled (or monitored) to ensure no requests to `pypi.org` or `github.com` are made.
2.  **Asset Check:** Try to load a model that usually triggers a download. It should fail gracefully or prompt for a local file.
