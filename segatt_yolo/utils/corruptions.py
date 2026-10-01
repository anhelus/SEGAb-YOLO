"""Shared image corruption primitives for robustness evaluation.

Provides corruption functions, severity parameter maps, and image
discovery utilities used by the corruption-related scripts.
"""

from pathlib import Path
from typing import Dict, Generator, Optional

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def darken(img: np.ndarray, alpha: float = 0.5, beta: float = 0) -> np.ndarray:
    """Reduce image brightness."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gaussian_blur(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply Gaussian blur."""
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (ks, ks), 0)


def motion_blur(img: np.ndarray, kernel_size: int = 15, angle: float = 0) -> np.ndarray:
    """Apply directional motion blur."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    center = k // 2
    rad = np.deg2rad(angle)
    for i in range(k):
        x = int(round((i - center) * np.cos(rad)))
        y = int(round((i - center) * np.sin(rad)))
        cx, cy = center + x, center + y
        if 0 <= cx < k and 0 <= cy < k:
            kernel[cy, cx] = 1
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, std: float = 25) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.randn(*img.shape).astype(np.float32) * std
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_occlusion(img: np.ndarray, max_boxes: int = 3, max_size: float = 0.3) -> np.ndarray:
    """Add random coloured rectangles to the image."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(np.random.randint(1, max_boxes + 1)):
        bw = int(w * np.random.uniform(0.05, max_size))
        bh = int(h * np.random.uniform(0.05, max_size))
        x = np.random.randint(0, w - bw)
        y = np.random.randint(0, h - bh)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, -1)
    return out


def salt_pepper(img: np.ndarray, prob: float = 0.01) -> np.ndarray:
    """Add salt-and-pepper noise."""
    out = img.copy()
    mask = np.random.random(img.shape[:2])
    out[mask < prob / 2] = 0
    out[mask > 1 - prob / 2] = 255
    return out


def brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """Adjust brightness and contrast."""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def jpeg_compression(img: np.ndarray, quality: int = 30) -> np.ndarray:
    """Simulate JPEG compression artefacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def hue_shift(img: np.ndarray, shift: float = 30) -> np.ndarray:
    """Shift the hue channel."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


CORRUPTIONS: Dict[str, callable] = {
    "darken": darken,
    "blur": gaussian_blur,
    "motion_blur": motion_blur,
    "noise": gaussian_noise,
    "occlusion": random_occlusion,
    "salt_pepper": salt_pepper,
    "brightness": brightness_contrast,
    "jpeg": jpeg_compression,
    "hue": hue_shift,
}

SEVERITY_PARAMS: Dict[int, dict] = {
    1: {
        "alpha": 0.7,
        "kernel": 7,
        "std": 15,
        "occlusion": 0.15,
        "prob": 0.005,
        "brightness": -30,
        "contrast": 0.8,
        "quality": 50,
        "hue_shift": 15,
        "motion_kernel": 9,
    },
    2: {
        "alpha": 0.5,
        "kernel": 15,
        "std": 25,
        "occlusion": 0.3,
        "prob": 0.01,
        "brightness": -60,
        "contrast": 0.6,
        "quality": 30,
        "hue_shift": 30,
        "motion_kernel": 15,
    },
    3: {
        "alpha": 0.3,
        "kernel": 25,
        "std": 40,
        "occlusion": 0.5,
        "prob": 0.03,
        "brightness": -100,
        "contrast": 0.4,
        "quality": 15,
        "hue_shift": 60,
        "motion_kernel": 25,
    },
}


def image_generator(source_path: str, ext: Optional[str] = None) -> Generator[Path, None, None]:
    """Yield image file paths from a file or directory."""
    source = Path(source_path)
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            yield source
    else:
        pattern = f"*.{ext}" if ext else "*"
        for p in sorted(source.rglob(pattern)):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p


def apply_corruption(img: np.ndarray, corr: str, sp: dict) -> np.ndarray:
    """Apply a single corruption by name with the given severity parameters."""
    if corr == "darken":
        return darken(img, alpha=sp["alpha"])
    elif corr == "blur":
        return gaussian_blur(img, kernel_size=sp["kernel"])
    elif corr == "motion_blur":
        return motion_blur(img, kernel_size=sp["motion_kernel"], angle=np.random.randint(0, 180))
    elif corr == "noise":
        return gaussian_noise(img, std=sp["std"])
    elif corr == "occlusion":
        return random_occlusion(img, max_boxes=2, max_size=sp["occlusion"])
    elif corr == "salt_pepper":
        return salt_pepper(img, prob=sp["prob"])
    elif corr == "brightness":
        return brightness_contrast(img, brightness=sp["brightness"], contrast=sp["contrast"])
    elif corr == "jpeg":
        return jpeg_compression(img, quality=sp["quality"])
    elif corr == "hue":
        return hue_shift(img, shift=sp["hue_shift"])
    return img
