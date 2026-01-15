import cv2
import numpy as np
import torch

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    
    Args:
        img: The base image in RGB or BGR format.
        mask: The cam mask (2D array, 0-1 range).
        use_rgb: Whether the input image is RGB (True) or BGR (False).
        colormap: The OpenCV colormap to use.
    
    Returns:
        The image with the heatmap overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    heatmap = np.float32(heatmap) / 255
    
    if np.max(img) > 1:
        img = img / 255.0
        
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def scale_cam_image(cam, target_size=None):
    """
    Scales the CAM to the target size.
    """
    if target_size is not None:
        cam = cv2.resize(cam, target_size)
    return cam
