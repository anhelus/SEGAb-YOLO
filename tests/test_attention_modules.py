"""Tests for GAM and SimAM attention modules."""

import pytest
import torch


@pytest.mark.parametrize("batch,channels,height,width", [
    (1, 64, 32, 32),
    (2, 128, 16, 16),
    (4, 32, 64, 64),
])
def test_simam_output_shape(batch, channels, height, width):
    from segab_yolo.nn.modules.attention import SimAM
    module = SimAM()
    x = torch.randn(batch, channels, height, width)
    out = module(x)
    assert out.shape == x.shape, f"SimAM: expected {x.shape}, got {out.shape}"


@pytest.mark.parametrize("batch,channels,height,width", [
    (1, 64, 32, 32),
    (2, 128, 16, 16),
    (4, 32, 64, 64),
])
def test_gam_output_shape(batch, channels, height, width):
    from segab_yolo.nn.modules.attention import GAM
    module = GAM(c1=channels)
    x = torch.randn(batch, channels, height, width)
    out = module(x)
    assert out.shape == x.shape, f"GAM: expected {x.shape}, got {out.shape}"


def test_simam_save_attention():
    from segab_yolo.nn.modules.attention import SimAM
    module = SimAM()
    module.save_attention = True
    x = torch.randn(1, 64, 32, 32)
    _ = module(x)
    assert module.last_attention is not None
    assert module.last_attention.shape == x.shape


def test_gam_save_attention():
    from segab_yolo.nn.modules.attention import GAM
    module = GAM(c1=64)
    module.save_attention = True
    x = torch.randn(1, 64, 32, 32)
    _ = module(x)
    assert module.last_attention is not None
    assert "channel" in module.last_attention
    assert "spatial" in module.last_attention


def test_simam_grad_flow():
    from segab_yolo.nn.modules.attention import SimAM
    module = SimAM()
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gam_grad_flow():
    from segab_yolo.nn.modules.attention import GAM
    module = GAM(c1=64)
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
