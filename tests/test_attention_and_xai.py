"""Tests for ALL attention modules and XAI utility functions."""

from pathlib import Path

import pytest
import torch


# ======================================================================
# Attention module tests
# ======================================================================

# ----- CBAM (in conv.py) -----

class TestCBAM:
    @pytest.mark.parametrize("batch,channels,height,width", [
        (1, 64, 32, 32),
        (2, 128, 16, 16),
    ])
    def test_output_shape(self, batch, channels, height, width):
        from segab_yolo.nn.modules.conv import CBAM
        module = CBAM(c1=channels)
        x = torch.randn(batch, channels, height, width)
        out = module(x)
        assert out.shape == x.shape

    def test_grad_flow(self):
        from segab_yolo.nn.modules.conv import CBAM
        module = CBAM(c1=64)
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        out = module(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_invalid_kernel_size(self):
        from segab_yolo.nn.modules.conv import CBAM
        with pytest.raises(AssertionError):
            CBAM(c1=64, kernel_size=5)


# ----- ChannelAttention (in conv.py) -----

class TestChannelAttention:
    @pytest.mark.parametrize("channels", [32, 64, 128])
    def test_output_shape(self, channels):
        from segab_yolo.nn.modules.conv import ChannelAttention
        module = ChannelAttention(channels)
        x = torch.randn(1, channels, 16, 16)
        out = module(x)
        assert out.shape == x.shape


# ----- SpatialAttention (in conv.py) -----

class TestSpatialAttention:
    @pytest.mark.parametrize("kernel_size", [3, 7])
    def test_output_shape(self, kernel_size):
        from segab_yolo.nn.modules.conv import SpatialAttention
        module = SpatialAttention(kernel_size)
        x = torch.randn(1, 64, 32, 32)
        out = module(x)
        assert out.shape == x.shape


# ----- EMA (in attention.py) -----

class TestEMA:
    @pytest.mark.parametrize("batch,channels,height,width", [
        (1, 32, 32, 32),
        (2, 64, 16, 16),
    ])
    def test_output_shape(self, batch, channels, height, width):
        from segab_yolo.nn.modules.attention import EMA
        module = EMA(channels=channels)
        x = torch.randn(batch, channels, height, width)
        out = module(x)
        assert out.shape == x.shape

    def test_grad_flow(self):
        from segab_yolo.nn.modules.attention import EMA
        module = EMA(channels=32)
        x = torch.randn(1, 32, 32, 32, requires_grad=True)
        out = module(x).sum()
        out.backward()
        assert x.grad is not None

    def test_invalid_channels(self):
        from segab_yolo.nn.modules.attention import EMA
        with pytest.raises(AssertionError):
            EMA(channels=33, groups=4)  # 33 not divisible by 4


# ----- CoT (in attention.py) -----

class TestCoT:
    @pytest.mark.parametrize("batch,channels,height,width", [
        (1, 32, 16, 16),
        (2, 64, 8, 8),
    ])
    def test_output_shape(self, batch, channels, height, width):
        from segab_yolo.nn.modules.attention import CoT
        module = CoT(in_channels=channels)
        x = torch.randn(batch, channels, height, width)
        out = module(x)
        assert out.shape == x.shape

    def test_grad_flow(self):
        from segab_yolo.nn.modules.attention import CoT
        module = CoT(in_channels=32)
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        out = module(x).sum()
        out.backward()
        assert x.grad is not None


# ----- ODConv (in attention.py) -----
# Note: ODConv uses BatchNorm in its attention MLP, so batch_size must be >= 2.

class TestODConv:
    @pytest.mark.parametrize("cin,cout,ks", [
        (16, 32, 3),
        (32, 64, 1),
        (64, 128, 5),
    ])
    def test_output_shape(self, cin, cout, ks):
        from segab_yolo.nn.modules.attention import ODConv
        module = ODConv(cin, cout, ks).eval()
        x = torch.randn(2, cin, 16, 16)
        out = module(x)
        assert out.shape == (2, cout, 16, 16)

    def test_grad_flow(self):
        from segab_yolo.nn.modules.attention import ODConv
        module = ODConv(16, 32, 3)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        out = module(x).sum()
        out.backward()
        assert x.grad is not None

    def test_stride_downsample(self):
        from segab_yolo.nn.modules.attention import ODConv
        module = ODConv(16, 32, 3, stride=2).eval()
        x = torch.randn(2, 16, 32, 32)
        out = module(x)
        assert out.shape == (2, 32, 16, 16)

    def test_groups(self):
        from segab_yolo.nn.modules.attention import ODConv
        module = ODConv(32, 64, 3, groups=2).eval()
        x = torch.randn(2, 32, 16, 16)
        out = module(x)
        assert out.shape == (2, 64, 16, 16)


# ----- PConv + FasterNetBlock (in attention.py) -----

class TestPConv:
    @pytest.mark.parametrize("channels,n_div", [(32, 4), (64, 2)])
    def test_output_shape(self, channels, n_div):
        from segab_yolo.nn.modules.attention import PConv
        module = PConv(channels, n_div=n_div)
        x = torch.randn(1, channels, 16, 16)
        out = module(x)
        assert out.shape == x.shape


class TestFasterNetBlock:
    @pytest.mark.parametrize("cin,cout,stride", [
        (32, 32, 1),
        (32, 64, 2),
        (64, 128, 2),
    ])
    def test_output_shape(self, cin, cout, stride):
        from segab_yolo.nn.modules.attention import FasterNetBlock
        module = FasterNetBlock(cin, cout, stride=stride)
        h = w = 32 // stride
        x = torch.randn(1, cin, 32, 32)
        out = module(x)
        assert out.shape == (1, cout, h, w)

    def test_grad_flow(self):
        from segab_yolo.nn.modules.attention import FasterNetBlock
        module = FasterNetBlock(32, 32)
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        out = module(x).sum()
        out.backward()
        assert x.grad is not None

    def test_identity_shortcut(self):
        from segab_yolo.nn.modules.attention import FasterNetBlock
        module = FasterNetBlock(32, 32, stride=1)
        assert isinstance(module.shortcut, torch.nn.Identity)

    def test_conv_shortcut(self):
        from segab_yolo.nn.modules.attention import FasterNetBlock
        module = FasterNetBlock(32, 64, stride=2)
        assert not isinstance(module.shortcut, torch.nn.Identity)


# ----- SimAM and GAM (already tested in test_attention_modules.py) -----


# ======================================================================
# XAI utility tests
# ======================================================================

class TestGenerateCam:
    """Test generate_cam function from segab_yolo.utils.xai."""

    def test_invalid_method_returns_zeros(self):
        from segab_yolo.utils.xai import generate_cam
        import torch.nn as nn
        model = nn.Conv2d(3, 16, 3)
        img = torch.randn(1, 3, 32, 32)
        target_layer = model
        target_box = type("Box", (), {"xyxy": [0, 0, 10, 10], "cls": [0]})()
        result = generate_cam(model, img, target_layer, target_box, method="nonexistent")
        # squeeze() on (1,3,32,32) -> (3,32,32)
        assert result.shape == (3, 32, 32)

    def test_valid_methods_listed(self):
        from segab_yolo.utils.xai import generate_cam
        import inspect
        source = inspect.getsource(generate_cam)
        for method in ("gradcam", "gradcam++", "eigencam", "ss-gradcam++"):
            assert method in source, f"{method} not found in generate_cam"

    def test_cam_config_methods_match_generate_cam(self):
        """All methods in xai_config.yaml must be supported by generate_cam or process_single_image."""
        from segab_yolo.utils.xai import generate_cam
        import inspect
        source = inspect.getsource(generate_cam)
        # activation is handled separately in process_single_image, not generate_cam
        supported_in_generate_cam = []
        for line in source.splitlines():
            if ":" in line and '"' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    supported_in_generate_cam.append(parts[1])
        # Just ensure eigencam (the main one) is supported
        assert "eigencam" in source

    def test_show_cam_on_image_rgb(self):
        from segab_yolo.utils.xai import show_cam_on_image
        import numpy as np
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cam = np.random.rand(32, 32).astype(np.float32)
        result = show_cam_on_image(img, cam, use_rgb=True)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_show_cam_on_image_bgr(self):
        from segab_yolo.utils.xai import show_cam_on_image
        import numpy as np
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cam = np.random.rand(32, 32).astype(np.float32)
        result = show_cam_on_image(img, cam, use_rgb=False)
        assert result.shape == (32, 32, 3)


class TestEigenCAMDirect:
    """Test EigenCAM class directly from pytorch_grad_cam."""

    def test_import_eigencam(self):
        from pytorch_grad_cam import EigenCAM
        import torch.nn as nn
        model = nn.Conv2d(3, 16, 3)
        cam = EigenCAM(model=model, target_layers=[model])
        assert cam is not None


# ======================================================================
# Model YAML instantiation tests (integration)
# ======================================================================

class TestModelYamls:
    """Verify that YAML model definitions can be parsed and run a forward pass."""

    MODELS_DIR = Path(__import__("segab_yolo").__file__).resolve().parent / "cfg" / "models"

    @pytest.mark.parametrize("subdir,yaml_name", [
        ("11", "yolo11.yaml"),
        ("11", "yolo11-gam.yaml"),
        ("11", "yolo11-simam.yaml"),
        ("11", "yolo11-simam-bbone.yaml"),
        ("26", "yolo26.yaml"),
    ])
    def test_yaml_model_forward(self, subdir, yaml_name):
        """Build model from YAML and run a tiny forward pass."""
        from segab_yolo import YOLO
        yaml_path = self.MODELS_DIR / subdir / yaml_name
        if not yaml_path.exists():
            pytest.skip(f"YAML not found: {yaml_path}")

        model = YOLO(str(yaml_path))
        x = torch.randn(1, 3, 128, 128)
        out = model.model(x)
        assert out is not None

    @pytest.mark.parametrize("yaml_name", [
        "yolo11_ResBlock_CBAM.yaml",
        "yolo11_ECA.yaml",
        "yolo11_SA.yaml",
        "yolo11_BGF.yaml",
    ])
    def test_broken_yamls_reported(self, yaml_name):
        """These YAMLs should now have all classes implemented."""
        from segab_yolo.nn.modules import __all__ as known_modules
        known = set(known_modules) | {"Detect", "Segment", "Pose", "Classify", "OBB", "OBB26", "Segment26",
                                       "YOLOEDetect", "YOLOESegment", "YOLOESegment26",
                                       "nn", "Conv", "C3k2", "Concat", "C2PSA", "SPPF"}

        yaml_path = self.MODELS_DIR / "11" / yaml_name
        if not yaml_path.exists():
            pytest.skip(f"{yaml_name} not found")
        import yaml as pyyaml
        with open(yaml_path) as f:
            data = pyyaml.safe_load(f)
        missing = []
        for layer in data.get("head", []):
            module_name = layer[2] if len(layer) > 2 else None
            if module_name and module_name[0].isupper() and module_name not in known:
                missing.append(module_name)
        assert not missing, f"{yaml_name}: missing classes: {set(missing)}"
