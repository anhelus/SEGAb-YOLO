import torch
import numpy as np
import cv2
from ultralytics.utils import LOGGER
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from torchvision.ops import box_iou

class DetCAM_Target:
    """
    Custom CAM Target for YOLO-style Object Detection models.
    
    This class defines the target to be maximized by the CAM algorithm, which is 
    typically the confidence score of a specific class for the best matching bounding box.
    """

    def __init__(self, box: torch.Tensor, cls_idx: int):
        """
        Initialize the DetCAM_Target.

        Args:
            box (torch.Tensor): Ground truth or predicted bounding box in xyxy format.
            cls_idx (int): The class index to target.
        """
        self.box = box
        self.cls_idx = cls_idx

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Extract the score for the target class from the model output.

        Args:
            model_output (torch.Tensor): The raw model output.

        Returns:
            torch.Tensor: The score of the target class at the best matching anchor.
        """
        output = model_output[0]

        # Ensure the output tensor is 3D (Batch, Channels, Predictions)
        # by adding back the batch dimension if it was squeezed.
        if output.ndim == 2:
            output = output.unsqueeze(0)
        
        # Ensure output is (Batch, Predictions, Channels)
        # YOLO outputs are sometimes (Batch, Channels, Predictions)
        if output.shape[1] < output.shape[2]:
            output = output.transpose(1, 2)
        
        boxes = output[..., :4]
        
        # Convert boxes from cxcywh to xyxy
        # cx, cy, w, h -> x1, y1, x2, y2
        boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
        
        # Find the box with the highest IoU with the target box
        ious = box_iou(self.box, boxes[0]).squeeze()
        best_match_index = ious.argmax()
        
        # Return the score for the target class
        # (4 box coords + cls_idx)
        score = output[0, best_match_index, 4 + self.cls_idx]
        return score


def generate_cam(model: torch.nn.Module, 
                 image_tensor: torch.Tensor, 
                 target_layer: torch.nn.Module, 
                 target_box, 
                 method: str = 'gradcam',
                 n_samples: int = 15,
                 noise_level: float = 0.1) -> np.ndarray:
    """
    Generates a class activation map (CAM) using the specified method.

    Args:
        model (torch.nn.Module): The YOLO model.
        image_tensor (torch.Tensor): The input image tensor.
        target_layer (torch.nn.Module): The layer to visualize.
        target_box (object): An object containing .xyxy (box) and .cls (class) attributes.
        method (str): The XAI method to use ('gradcam', 'gradcam++', 'eigencam', 'ss-gradcam++').
        n_samples (int): Number of noisy samples for smoothing (only for 'ss-' methods).
        noise_level (float): Noise level for smoothing (only for 'ss-' methods).

    Returns:
        np.ndarray: The generated CAM heatmap.
    """
    cam_methods = {
        'gradcam': GradCAM, 
        'gradcam++': GradCAMPlusPlus, 
        'eigencam': EigenCAM,
        'ss-gradcam++': GradCAMPlusPlus
    }
    
    method_key = method.lower()
    cam_constructor = cam_methods.get(method_key)
    if cam_constructor is None:
        LOGGER.error(f"Invalid XAI method '{method}'. Supported methods: {list(cam_methods.keys())}")
        return torch.zeros_like(image_tensor).squeeze().cpu().numpy()

    # Initialize the CAM object
    cam = cam_constructor(model=model, target_layers=[target_layer])

    predicted_class_index = int(target_box.cls[0])
    targets = [DetCAM_Target(target_box.xyxy, predicted_class_index)]

    if method_key.startswith('ss-'):
        # SmoothGrad implementation
        total_cam = None
        for i in range(n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(image_tensor) * noise_level * (image_tensor.max() - image_tensor.min())
            noisy_input = (image_tensor + noise).detach().requires_grad_(True)
            
            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=noisy_input, targets=targets)
                if total_cam is None:
                    total_cam = grayscale_cam[0, :]
                else:
                    total_cam += grayscale_cam[0, :]
        
        heatmap = total_cam / n_samples
    else:
        # Standard implementation
        grad_enabled_tensor = image_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=grad_enabled_tensor, targets=targets)
        heatmap = grayscale_cam[0, :]
        
    return heatmap


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays the CAM mask on the image as a heatmap.

    Args:
        img (np.ndarray): The base image in RGB or BGR format.
        mask (np.ndarray): The CAM mask (2D array, 0-1 range).
        use_rgb (bool): Whether the input image is RGB (True) or BGR (False).
        colormap (int): The OpenCV colormap to use.

    Returns:
        np.ndarray: The image with the heatmap overlay (uint8, 0-255).
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


def scale_cam_image(cam: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """
    Scales the CAM to the target size.

    Args:
        cam (np.ndarray): The CAM heatmap.
        target_size (tuple, optional): The target (width, height).

    Returns:
        np.ndarray: The resized CAM.
    """
    if target_size is not None:
        cam = cv2.resize(cam, target_size)
    return cam
