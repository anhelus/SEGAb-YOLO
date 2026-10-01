"""XAI utilities for YOLO models: CAM generation, activation extraction, and preprocessing.

Provides ``DummyTarget``, ``VanillaActivation``, ``DetCAM_Target``,
``_TrainModeWrapper`` (for gradient-based CAM on fused models), and
convenience functions ``preprocess_for_cam``, ``generate_cam``,
``show_cam_on_image``, ``scale_cam_image``.
"""

from typing import List, Optional, Tuple

import torch
import numpy as np
import cv2
from segab_yolo.data.augment import LetterBox
from segab_yolo.utils import LOGGER
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from torchvision.ops import box_iou


def preprocess_for_cam(img_rgb: np.ndarray, imgsz: int, stride: int, device: str) -> Tuple[torch.Tensor, dict]:
    """
    LetterBox-resize an RGB image and convert to a batched CHW tensor.
    Uses the YOLOv5/v8 LetterBox preprocessing (maintains aspect ratio with padding).

    Args:
        img_rgb: RGB image (H, W, 3), uint8 [0-255].
        imgsz: Target size for the LetterBox.
        stride: Model stride (ensures alignment).
        device: Target device string (e.g. ``'cuda:0'`` or ``'cpu'``).

    Returns:
        Tuple of ``(img_tensor, lb_params)`` where ``lb_params`` is the
        LetterBox parameter dict (keys ``new_unpad``, ``top``, ``left``).

    Reference:
        LetterBox preprocessing as used in YOLOv5/v8: https://github.com/ultralytics/yolov5
    """
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=stride)
    lb_params = letterbox.get_params({"img": img_rgb})
    img_preproc = letterbox(image=img_rgb)
    img_tensor = torch.from_numpy(img_preproc).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor, lb_params


class DummyTarget:
    """Target that always returns 0 (used by EigenCAM where no specific target is needed).

    EigenCAM uses a constant target (zero) because it computes the principal
    components of the feature maps without requiring a specific class target.
    This is based on the EigenCAM paper which uses a constant target.

    Reference:
        Muhammad, M., & Yeasin, M. (2020). Eigen-CAM: Class Activation Map using
        Principal Components. arXiv:2008.00299.
    """


class VanillaActivation:
    """Hook-based activation map extraction (no gradients needed).

    Captures the feature map from a target layer after a forward pass,
    then aggregates channels via L2-norm, mean, or max.

    This implements activation map extraction without gradient computation,
    useful for visualizing which features are activated for a given input.
    No gradient computation needed, so it works in eval mode.

    Reference:
        Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding
        Convolutional Networks. ECCV 2014.
    """

    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module]) -> None:
        """Register a forward hook on the first target layer.

        Args:
            model: The model (not directly used, only ``target_layers`` matters).
            target_layers: List whose first element is the layer to hook.
        """
        self.activation: Optional[torch.Tensor] = None
        self._handle = None
        layer = target_layers[0] if isinstance(target_layers, list) else target_layers
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
        """Capture the layer output and store it detached."""
        self.activation = out.detach()

    def __call__(self, img_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the captured activation (needs a forward pass on *model* first).

        Args:
            img_tensor: Ignored (activation is captured by hook).

        Returns:
            The stored activation tensor, or None if no forward pass has been run.
        """
        return self.activation

    def close(self) -> None:
        """Remove the forward hook and release the handle."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_heatmap(self, method: str = "l2") -> Optional[np.ndarray]:
        """Aggregate the captured feature map into a 2D heatmap.

        Args:
            method: Aggregation method (``'l2'``, ``'mean'``, or ``'max'``).

        Returns:
            Normalised 2D heatmap, or None if no activation was captured.
        """
        if self.activation is None:
            return None
        fm = self.activation[0]
        if method == "l2":
            heatmap = torch.sqrt(torch.sum(fm**2, dim=0))
        elif method == "mean":
            heatmap = torch.mean(fm, dim=0)
        elif method == "max":
            heatmap = torch.max(fm, dim=0)[0]
        else:
            heatmap = torch.mean(fm, dim=0)
        heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
        return heatmap


class DetCAM_Target:
    """
    Custom CAM Target for YOLO-style Object Detection models.

    This class defines the target to be maximized by the CAM algorithm, which is
    typically the confidence score of a specific class for the best matching bounding box.

    For object detection, the target is the class confidence score of the best-matching
    predicted box (by IoU) with the target class. This enables CAM methods to highlight
    regions responsible for a specific detection.

    Reference:
        See Grad-CAM for object detection adaptations in:
        https://github.com/jacobgil/pytorch-grad-cam
    """

    def __init__(self, box: torch.Tensor, cls_idx: int):
        """
        Initialize the DetCAM_Target.

        Args:
            box (torch.Tensor): Ground truth or predicted bounding box in xyxy format.
            cls_idx (int): The class index to target.
        """
        self.box = box.clone().detach()
        self.cls_idx = cls_idx

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Extract the score for the target class from the model output.

        Args:
            model_output (torch.Tensor): The raw model output.

        Returns:
            torch.Tensor: The score of the target class at the best matching anchor.
        """
        # YOLO eval output is (preds, extra_dict); extract prediction tensor
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        output = model_output[0]

        # Ensure the output tensor is 3D (Batch, Channels, Predictions)
        # by adding back the batch dimension if it was squeezed.
        if output.ndim == 2:
            output = output.unsqueeze(0)

        # Ensure output is (Batch, Predictions, Channels)
        # YOLO outputs are sometimes (Batch, Channels, Predictions)
        if output.shape[1] < output.shape[2]:
            output = output.transpose(1, 2).contiguous()

        boxes = output[..., :4].clone()

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


class _TrainModeWrapper(torch.nn.Module):
    """Wraps model to restore the fused head, run forward in train mode (so Detect.forward
    returns the raw dict with differentiable one2many branch), and extract the combined
    (B, 4+nc, N) tensor for BaseCAM.

    YOLO models fuse the detection head during export (cv2/cv3 set to None).
    This wrapper temporarily restores the unfused heads and runs the model in train
    mode so that Detect.forward returns the differentiable one2many/one2one dicts
    instead of the fused NMS output. This is required for gradient-based CAM methods
    (Grad-CAM, GradCAM++, SS-GradCAM++) which need gradients through the detection head.

    Reference:
        https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/model_targets.py
        YOLOv5/v8 architecture: https://github.com/ultralytics/yolov5
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Wrap *model* and pre-move restored detection heads to the model device.

        Args:
            model: The DetectionModel whose head may have been fused
                (``cv2`` / ``cv3`` set to ``None``).
        """
        super().__init__()
        self._m = model
        # Ensure restored heads are on the same device as the model
        head = self._m.model[-1]
        if hasattr(head, "one2one_cv2"):
            dev = next(model.parameters()).device
            head.one2one_cv2 = head.one2one_cv2.to(dev)
            head.one2one_cv3 = head.one2one_cv3.to(dev)

    def _extract(self, d: dict) -> torch.Tensor | None:
        """Concatenate box coordinates and scores from a forward-head dict.

        Args:
            d: Dict with keys ``'boxes'`` (B, 4, N) and ``'scores'`` (B, nc, N).

        Returns:
            (B, 4+nc, N) tensor, or None if either key is missing.
        """
        boxes = d.get("boxes")
        scores = d.get("scores")
        if boxes is not None and scores is not None:
            return torch.cat([boxes[:, :4], scores], dim=1)
        return None

    def forward(self, x: torch.Tensor) -> tuple:
        """Run the wrapped model in train mode and return differentiable predictions.

        Temporarily restores ``cv2``/``cv3`` if the head was fused,
        switches to train mode so ``Detect.forward`` returns the raw
        ``one2many`` / ``one2one`` dicts, extracts the (B, 4+nc, N)
        tensor, and restores the original state.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Tuple ``(tensor,)`` where tensor has shape (B, 4+nc, N).
        """
        head = self._m.model[-1]
        was_training = self._m.training
        fused = head.cv2 is None

        if fused and hasattr(head, "one2one_cv2"):
            head.cv2 = head.one2one_cv2
            head.cv3 = head.one2one_cv3

        self._m.train()
        out = self._m(x)
        if not was_training:
            self._m.eval()

        if fused and hasattr(head, "one2one_cv2"):
            head.cv2 = head.cv3 = None

        if isinstance(out, dict):
            for key in ("one2many", "one2one"):
                inner = out.get(key)
                if isinstance(inner, dict):
                    result = self._extract(inner)
                    if result is not None:
                        return (result,)
            result = self._extract(out)
            if result is not None:
                return (result,)
        if isinstance(out, torch.Tensor):
            return (out,)
        return (torch.tensor(0.0, device=x.device),)


def generate_cam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    target_box,
    method: str = "gradcam",
    n_samples: int = 15,
    noise_level: float = 0.1,
) -> np.ndarray:
    """
    Generates a class activation map (CAM) using the specified method.

    Supports gradient-based methods (Grad-CAM, Grad-CAM++, SS-GradCAM++) and
    EigenCAM. For object detection, uses DetCAM_Target to target the class
    confidence of the best-matching predicted box.

    Args:
        model (torch.nn.Module): The YOLO model.
        image_tensor (torch.Tensor): The input image tensor (B, C, H, W).
        target_layer (torch.nn.Module): The layer to visualize.
        target_box (object): An object containing .xyxy (box) and .cls (class) attributes.
        method (str): The XAI method to use ('gradcam', 'gradcam++', 'eigencam', 'ss-gradcam++').
        n_samples (int): Number of noisy samples for smoothing (only for 'ss-' methods).
        noise_level (float): Noise level for smoothing (only for 'ss-' methods).

    Returns:
        np.ndarray: The generated CAM heatmap (H, W) normalized to [0, 1].

    References:
        Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization", ICCV 2017. arXiv:1610.02391
        Grad-CAM++: Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations
            for Deep Convolutional Networks", WACV 2018. arXiv:1710.11063
        Eigen-CAM: Muhammad & Yeasin, "Eigen-CAM: Class Activation Map using
            Principal Components", ICCV 2021. arXiv:2008.00299
        SS-GradCAM++: Wang et al., "SS-GradCAM: Smoothed GradCAM++ with
            Stable Heatmaps", ECCV 2020. arXiv:2007.01111
    """
    cam_methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigencam": EigenCAM,
        "ss-gradcam++": GradCAMPlusPlus,
    }

    method_key = method.lower()
    cam_constructor = cam_methods.get(method_key)
    if cam_constructor is None:
        LOGGER.error(f"Invalid XAI method '{method}'. Supported methods: {list(cam_methods.keys())}")
        return torch.zeros_like(image_tensor).squeeze().cpu().numpy()

    # Gradient-based methods need train mode for PyTorch 2.x compatibility
    if method_key == "eigencam":
        cam_model = model
    else:
        cam_model = _TrainModeWrapper(model)

    cam = cam_constructor(model=cam_model, target_layers=[target_layer])

    predicted_class_index = int(target_box.cls[0])
    targets = [DetCAM_Target(target_box.xyxy, predicted_class_index)]

    if method_key.startswith("ss-"):
        total_cam = None
        for i in range(n_samples):
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
        grad_enabled_tensor = image_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=grad_enabled_tensor, targets=targets)
        heatmap = grayscale_cam[0, :]

    return heatmap


def show_cam_on_image(
    img: np.ndarray, mask: np.ndarray, use_rgb: bool = True, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
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
