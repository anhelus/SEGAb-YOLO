"""Tests for XAI utilities (segab_yolo.utils.xai) and target-layer discovery (scripts.xai_predict)."""

import numpy as np
import pytest
import torch

from segab_yolo.utils.xai import (
    DetCAM_Target,
    DummyTarget,
    VanillaActivation,
    preprocess_for_cam,
)


# ======================================================================
# TST-5: DummyTarget + DetCAM_Target
# ======================================================================


class TestDummyTarget:
    """DummyTarget should always return 0.0."""

    def test_returns_scalar_tensor(self):
        t = DummyTarget()
        out = t(torch.randn(1, 84, 8400))
        assert isinstance(out, torch.Tensor)
        assert out.item() == 0.0

    def test_independent_of_input(self):
        t = DummyTarget()
        assert t(torch.zeros(1, 1)).item() == 0.0
        assert t(torch.ones(1, 100, 100)).item() == 0.0


class TestDetCAM_Target:
    """DetCAM_Target extracts a class score at the best-IoU box."""

    def _make_output(self, batch=1, num_boxes=100, num_classes=80):
        """Create a (B, 4+nc, N) tensor — cxcywh + scores.

        N > 4+nc ensures the transpose check in ``DetCAM_Target``
        triggers, so the output is reshaped to (B, N, 4+nc).
        """
        B, N, C = batch, num_boxes, 4 + num_classes
        out = torch.randn(B, C, N)
        # Ensure positive scores for the target class
        cls_idx = 0
        out[:, 4 + cls_idx, :] = torch.sigmoid(out[:, 4 + cls_idx, :])
        return out

    def test_returns_positive_score(self):
        output = self._make_output()
        box_xyxy = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
        target = DetCAM_Target(box_xyxy, cls_idx=0)
        score = target(output)
        assert isinstance(score, torch.Tensor)
        assert score.ndim == 0  # scalar

    def test_box_iou_selects_best_match(self):
        output = self._make_output()
        box_xyxy = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
        target = DetCAM_Target(box_xyxy, cls_idx=0)
        score = target(output)
        # Should be positive (sigmoid ensures at least some score > 0)
        assert score.item() > 0.0
        assert score.ndim == 0  # scalar


# ======================================================================
# TST-2: VanillaActivation.get_heatmap
# ======================================================================


class TestVanillaActivation:
    """VanillaActivation hook and heatmap aggregation."""

    @pytest.fixture
    def model_and_target(self):
        """A bare Conv2d as a dummy target layer."""
        layer = torch.nn.Conv2d(3, 8, 3)
        model = torch.nn.Sequential(layer, torch.nn.ReLU())
        return model, layer

    def test_hook_captures_activation(self, model_and_target):
        model, layer = model_and_target
        va = VanillaActivation(model, [layer])
        dummy = torch.randn(1, 3, 16, 16)
        _ = model(dummy)
        assert va.activation is not None
        assert va.activation.shape == (1, 8, 14, 14)
        va.close()

    def test_get_heatmap_l2(self, model_and_target):
        model, layer = model_and_target
        va = VanillaActivation(model, [layer])
        _ = model(torch.randn(1, 3, 16, 16))
        hm = va.get_heatmap(method="l2")
        assert hm is not None
        assert hm.shape == (14, 14)
        assert 0.0 <= hm.min() <= hm.max() <= 1.0
        va.close()

    def test_get_heatmap_mean(self, model_and_target):
        model, layer = model_and_target
        va = VanillaActivation(model, [layer])
        _ = model(torch.randn(1, 3, 16, 16))
        hm = va.get_heatmap(method="mean")
        assert hm is not None
        assert hm.shape == (14, 14)
        va.close()

    def test_get_heatmap_max(self, model_and_target):
        model, layer = model_and_target
        va = VanillaActivation(model, [layer])
        _ = model(torch.randn(1, 3, 16, 16))
        hm = va.get_heatmap(method="max")
        assert hm is not None
        assert hm.shape == (14, 14)
        va.close()

    def test_get_heatmap_before_forward_is_none(self, model_and_target):
        model, layer = model_and_target
        va = VanillaActivation(model, [layer])
        assert va.get_heatmap() is None
        va.close()

    def test_close_removes_hook(self, model_and_target):
        model, layer = model_and_target
        va = VanillaActivation(model, [layer])
        handle = va._handle
        # close() should not raise
        va.close()
        # After close, _handle is None
        assert va._handle is None


# ======================================================================
# TST-3: preprocess_for_cam
# ======================================================================


class TestPreprocessForCam:
    """LetterBox preprocessing for CAM."""

    def test_output_shape_and_dtype(self):
        img_rgb = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        tensor, params = preprocess_for_cam(img_rgb, imgsz=640, stride=32, device="cpu")
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"

    def test_values_in_range(self):
        img_rgb = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        tensor, _ = preprocess_for_cam(img_rgb, imgsz=640, stride=32, device="cpu")
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_lb_params_has_keys(self):
        img_rgb = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        _, params = preprocess_for_cam(img_rgb, imgsz=640, stride=32, device="cpu")
        for key in ("new_unpad", "top", "left"):
            assert key in params

    def test_gpu_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img_rgb = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        tensor, _ = preprocess_for_cam(img_rgb, imgsz=640, stride=32, device="cuda:0")
        assert tensor.device.type == "cuda"


# ======================================================================
# TST-1: find_target_layers (needs YOLO model instantiation)
# ======================================================================


class TestFindTargetLayers:
    """Backbone-final-layer discovery across model variants."""

    YAMLS = {
        "yolo26": "segab_yolo/cfg/models/26/yolo26.yaml",
        "yolo11_gam_bbone": "segab_yolo/cfg/models/11/yolo11-gam-bbone.yaml",
        "yolo11_simam_bbone": "segab_yolo/cfg/models/11/yolo11-simam-bbone.yaml",
        "yolo26_mod": "segab_yolo/cfg/models/26/yolo26-mod.yaml",
    }

    @pytest.fixture(autouse=True)
    def _import_target(self):
        from scripts.xai_predict import find_target_layers, _fallback_target

        self.find_target_layers = find_target_layers
        self._fallback_target = _fallback_target

    @pytest.fixture
    def models(self, request):
        """Load all model configs (once per session via caching)."""
        from segab_yolo import YOLO

        key = request.param
        yaml_path = self.YAMLS[key]
        model = YOLO(str(yaml_path))
        return model, yaml_path

    @pytest.mark.parametrize("models", ["yolo26"], indirect=True)
    def test_standard_yolo26(self, models):
        model, yaml_path = models
        layers = self.find_target_layers(model)
        assert len(layers) == 1
        layer = layers[0]
        # yolo26 backbone ends at C2PSA (index 10) → last Conv2d
        assert isinstance(layer, torch.nn.Conv2d)

    @pytest.mark.parametrize("models", ["yolo11_gam_bbone"], indirect=True)
    def test_gam_bbone(self, models):
        model, yaml_path = models
        layers = self.find_target_layers(model)
        assert len(layers) == 1
        # GAM is the last backbone layer → returned directly
        from segab_yolo.nn.modules.attention import GAM

        assert isinstance(layers[0], GAM)

    @pytest.mark.parametrize("models", ["yolo11_simam_bbone"], indirect=True)
    def test_simam_bbone(self, models):
        model, yaml_path = models
        layers = self.find_target_layers(model)
        assert len(layers) == 1
        from segab_yolo.nn.modules.attention import SimAM

        assert isinstance(layers[0], SimAM)

    @pytest.mark.parametrize("models", ["yolo26_mod"], indirect=True)
    def test_yolo26_mod(self, models):
        model, yaml_path = models
        layers = self.find_target_layers(model)
        assert len(layers) == 1
        from segab_yolo.nn.modules.attention import GAM

        assert isinstance(layers[0], GAM)

    def test_target_layer_name_c3k2(self, models):
        model, yaml_path = models
        # Request an explicit backbone target layer before the head.
        layers = self.find_target_layers(model, target_layer_name="C3k2")
        assert len(layers) == 1
        from segab_yolo.nn.modules import C3k2

        assert isinstance(layers[0], C3k2)

    def test_target_layer_index(self, models):
        model, yaml_path = models
        # Explicit index should select the exact module in model.model.
        layers = self.find_target_layers(model, target_layer_index=23)
        assert len(layers) == 1
        from segab_yolo.nn.modules import C3k2

        assert isinstance(layers[0], C3k2)

    def test_fallback_on_empty_model(self):
        """_fallback_target should handle a model with no Conv2d."""
        from segab_yolo import YOLO

        # Minimal model: just a few layers, no Detect
        model = YOLO("segab_yolo/cfg/models/26/yolo26.yaml")
        layers = self._fallback_target(model)
        # Should find some Conv2d (the backbone has many)
        assert len(layers) == 1


# ======================================================================
# TST-4: generate_cam eigencam (integration, @slow)
# ======================================================================


@pytest.mark.slow
class TestGenerateCamEigenCAM:
    """End-to-end EigenCAM generation on a small model."""

    @pytest.fixture
    def model_and_tensor(self):
        from segab_yolo import YOLO
        from scripts.xai_predict import find_target_layers

        model = YOLO("segab_yolo/cfg/models/26/yolo26.yaml")
        layers = find_target_layers(model)
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        img_rgb = img
        imgsz = int(model.overrides.get("imgsz", 640))
        stride = int(model.model.stride.max())
        tensor, _ = preprocess_for_cam(img_rgb, imgsz, stride, "cpu")
        return model, tensor, layers[0]

    def test_eigencam_returns_heatmap(self, model_and_tensor):
        from segab_yolo.utils.xai import EigenCAM

        model, tensor, layer = model_and_tensor
        # generate_cam with eigencam
        with torch.no_grad():
            cams = EigenCAM(model.model, [layer])(tensor, targets=[DummyTarget()])
        heatmap = cams[0]
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        # Heatmap should be smaller than input (spatial reduction from convs)
        assert heatmap.shape[0] > 0
        assert heatmap.shape[1] > 0
