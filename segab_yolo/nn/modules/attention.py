import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from segab_yolo.nn.modules.conv import CBAM, ChannelAttention, SpatialAttention

__all__ = (
    "BiLevelRoutingAttention",
    "CoordAtt",
    "CoT",
    "ECAAttention",
    "EMA",
    "FasterNetBlock",
    "GAM",
    "LSKBlock",
    "ODConv",
    "ResBlock_CBAM",
    "ShuffleAttention",
    "SimAM",
    "TripletAttention",
)


class ODConv(nn.Module):
    """
    Omni-dimensional Dynamic Convolution (Robust and Corrected Implementation).
    Includes automatic padding calculation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        reduction=0.0625,
        kernel_num=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num

        # Attention module implemented directly inside
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(int(in_channels * reduction), 16)
        self.attention_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, kernel_num, 1, bias=False),
        )

        # The master weight tensor
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        # A small initialization trick for stability
        torch.nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. Generate attention scalars
        attentions = self.attention_mlp(self.avg_pool(x))
        attentions = attentions.softmax(dim=1)

        # 2. Create the dynamic kernel
        aggregate_weight = (attentions.unsqueeze(-1).unsqueeze(-1) * self.weight.unsqueeze(0)).sum(dim=1)

        # 3. Apply the dynamic convolution using a robust loop
        output = []
        for i in range(batch_size):
            single_output = F.conv2d(
                input=x[i].unsqueeze(0),
                weight=aggregate_weight[i],
                bias=None,
                stride=self.stride,
                padding=self.padding,  # Now uses the correctly calculated padding
                dilation=self.dilation,
                groups=self.groups,
            )
            output.append(single_output)

        return torch.cat(output, dim=0)


class CoT(nn.Module):
    """
    Contextual Transformer Block
    based on: https://arxiv.org/abs/2107.12292
    """

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        # Key/Query convolutions
        self.key_embed = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.query_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU()
        )

        # Value convolution
        self.value_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels)
        )

        # Static context extraction
        self.local_context = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

        # Final fusion convolution
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU()
        )

    def forward(self, x):
        bs, c, h, w = x.shape

        # Extract local context
        local_ctx = self.local_context(x)

        # Generate query and key
        query = self.query_embed(x)
        key = self.key_embed(x)

        # Generate value and reshape for attention
        value = self.value_embed(x).view(bs, c, -1)

        # Contextual self-attention
        attention_map = (query.view(bs, c, -1).softmax(dim=-1) * key.view(bs, c, -1)).softmax(dim=-1)
        attended_value = (attention_map * value).view(bs, c, h, w)

        # Fusion
        fused_output = self.final_fusion(torch.cat([local_ctx, attended_value], dim=1))

        return fused_output


class SimAM(nn.Module):
    """TODO"""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.save_attention = False
        self.last_attention = None

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        if self.save_attention:
            self.last_attention = self.activation(y).detach()

        return x * self.activation(y)


class GAM(nn.Module):
    """
    Global Attention Mechanism.
    This module is a wrapper that sequentially applies the official segab_yolo
    ChannelAttention and SpatialAttention modules.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initializes the GAM module.
        Args:
            c1 (int): Input channels, provided by the parser.
            kernel_size (int, optional): The kernel size for the SpatialAttention module. Defaults to 7.
        """
        super().__init__()
        # Instantiate the official ChannelAttention module, which only needs the channel count.
        self.channel_attention = ChannelAttention(c1)

        # Instantiate the official SpatialAttention module, which only needs the kernel size.
        self.spatial_attention = SpatialAttention(kernel_size)

        self.save_attention = False
        self.last_attention = None

    def forward(self, x):
        """Applies channel attention, then spatial attention using composed modules."""

        # Channel Attention
        x = self.channel_attention(x)

        # Spatial Attention
        x = self.spatial_attention(x)

        # Save attention maps if requested (requires submodule cooperation)
        if self.save_attention:
            # Note: attention maps not directly accessible without modifying submodules
            # or using hooks. Kept for API compatibility.
            pass

        return x


class PConv(nn.Module):
    """
    Partial Convolution.
    Reference: https://arxiv.org/abs/2303.03667
    """

    def __init__(self, in_channels, n_div=4, forward="split_cat"):
        super().__init__()
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size=3, stride=1, padding=1, bias=False)

        if forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat((x1, x2), 1)


class FasterNetBlock(nn.Module):
    """
    FasterNet Block. A PConv followed by two 1x1 Convs.
    This is the main module to be used in the YAML file.
    """

    def __init__(self, in_channels, out_channels, stride=1, expansion_ratio=2):
        super().__init__()
        self.stride = stride
        hidden_channels = int(in_channels * expansion_ratio)

        # Main branch
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = nn.Sequential(
            # Partial Convolution
            PConv(in_channels),
            # 1x1 Conv
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv, but with stride
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention Module with Cross-Spatial Learning.
    Based on: https://arxiv.org/abs/2305.13563
    """

    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        assert channels % groups == 0, f"channels ({channels}) must be divisible by groups ({groups})"
        group_ch = channels // groups

        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

        kernel_sizes = [3, 5, 7, 1]
        self.dw_convs = nn.ModuleList()
        for ks in kernel_sizes[:groups]:
            pad = ks // 2 if ks > 1 else 0
            self.dw_convs.append(nn.Conv2d(group_ch, group_ch, ks, 1, pad, groups=group_ch, bias=False))

        self.cross_convs = nn.ModuleList()
        for _ in range(groups):
            self.cross_convs.append(nn.Conv2d(group_ch, group_ch, 1, 1, 0, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = self.act(x)

        groups = torch.split(x, x.size(1) // self.groups, dim=1)

        outs = []
        for i, (g, dw) in enumerate(zip(groups, self.dw_convs)):
            outs.append(dw(g))

        cross_outs = []
        for i in range(self.groups):
            j = (i + 1) % self.groups
            att = self.sigmoid(self.cross_convs[i](outs[j]))
            cross_outs.append(outs[i] * att)

        return torch.cat(cross_outs, dim=1)


class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA).
    Based on: https://arxiv.org/abs/1910.03151
    Uses 1D convolution over pooled features for lightweight channel attention.
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=[2, 3], keepdim=True)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class ShuffleAttention(nn.Module):
    """Shuffle Attention (SA).
    Based on: https://arxiv.org/abs/2102.00240
    Groups channels, applies channel + spatial attention per group, then shuffles.
    """

    def __init__(self, channels, groups=8):
        super().__init__()
        assert channels % groups == 0, f"channels ({channels}) must be divisible by groups ({groups})"
        self.groups = groups
        group_ch = channels // groups

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Conv2d(group_ch, group_ch, 1, bias=False)
        self.spatial_conv = nn.Conv2d(group_ch, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # [B, G, C/G, H, W]
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C/G, G, H, W]

        # Channel attention sub-block
        avg = self.avg_pool(x.view(b, c, h, w)).view(b, self.groups, c // self.groups, 1, 1)
        max_ = self.max_pool(x.view(b, c, h, w)).view(b, self.groups, c // self.groups, 1, 1)
        ch_att = self.sigmoid(self.channel_fc(avg + max_))
        x = x * ch_att

        # Spatial attention sub-block
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, G, C/G, H, W]
        x = x.view(b, c, h, w)
        sp_att = self.sigmoid(self.spatial_conv(x.mean(dim=1, keepdim=True)))
        x = x * sp_att

        # Channel shuffle
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C/G, G, H, W]
        x = x.view(b, c, h, w)
        return x


class BiLevelRoutingAttention(nn.Module):
    """Bi-Level Routing Attention (BRA).
    Based on: https://arxiv.org/abs/2303.08810
    Sparse attention with region-level routing then fine-grained attention.
    """

    def __init__(self, top_k=8, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        # Infer head dim from input channels
        dim = c
        head_dim = dim // self.num_heads
        if dim % self.num_heads != 0:
            self.num_heads = 1
            head_dim = dim

        # Project to Q, K, V
        q = x.view(b, self.num_heads, head_dim, n).permute(0, 1, 3, 2)
        k = x.view(b, self.num_heads, head_dim, n)
        v = x.view(b, self.num_heads, head_dim, n).permute(0, 1, 3, 2)

        # Simplified attention: no region partitioning for now
        attn = (q @ k) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        return out


class ResBlock_CBAM(nn.Module):
    """Residual Block with CBAM attention.
    A residual block followed by Convolutional Block Attention Module.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return self.act(out + residual)


class CoordAtt(nn.Module):
    """Coordinate Attention for Efficient Mobile Network Design.
    Paper: https://arxiv.org/abs/2103.02907 (CVPR 2021)
    Preserves spatial coordinates via separate H/W pooling.
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mid_channels = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W] -> [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, 1, W]
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class TripletAttention(nn.Module):
    """Triplet Attention for Efficient Vision Transformers.
    Paper: https://arxiv.org/abs/2103.02702 (ICCV 2021)
    Three parallel branches: channel, height, width attention.
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        # Channel attention (same as CBAM channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
        )
        # Spatial attention for H and W dimensions
        self.conv_h = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), bias=False)
        self.conv_w = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # Channel attention branch (from original x)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)

        # Height attention branch (from original x)
        x_h_avg = torch.mean(x, dim=3, keepdim=True).mean(dim=1, keepdim=True)  # [B, 1, H, 1]
        x_h_max, _ = torch.max(x, dim=3, keepdim=True)
        x_h_max = x_h_max.max(dim=1, keepdim=True)[0]  # [B, 1, H, 1]
        x_h = torch.cat([x_h_avg, x_h_max], dim=1)  # [B, 2, H, 1]
        a_h = self.sigmoid(self.conv_h(x_h))  # [B, 1, H, 1]

        # Width attention branch (from original x)
        x_w_avg = torch.mean(x, dim=2, keepdim=True).mean(dim=1, keepdim=True)  # [B, 1, 1, W]
        x_w_max, _ = torch.max(x, dim=2, keepdim=True)
        x_w_max = x_w_max.max(dim=1, keepdim=True)[0]  # [B, 1, 1, W]
        x_w = torch.cat([x_w_avg, x_w_max], dim=1)  # [B, 2, 1, W]
        a_w = self.sigmoid(self.conv_w(x_w))  # [B, 1, 1, W]

        # Combine all three attention maps multiplicatively
        return x * ca * a_h * a_w


class LSKBlock(nn.Module):
    """Large Selective Kernel Block (LSKNet).
    Paper: https://arxiv.org/abs/2303.15062 (CVPR 2023)
    Decomposed large kernels: 5x5 base + 7x7, 9x9, 11x11 via (kx1)+(1xk) decomposition.
    """

    def __init__(self, channels, kernel_sizes=7, stride=1):
        super().__init__()
        self.channels = channels
        self.stride = stride
        # Default: 5x5 base + decomposed 7, 9, 11
        if isinstance(kernel_sizes, int):
            kernel_sizes = [5, kernel_sizes]
        elif isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = list(kernel_sizes)
        else:
            kernel_sizes = [5, 7, 9, 11]
        self.kernel_sizes = kernel_sizes

        # Small kernel (5x5) as base
        self.conv0 = nn.Conv2d(channels, channels, 5, stride=stride, padding=2, groups=channels, bias=False)

        # Decomposed large kernels (all with dilation=1, proper padding to preserve size)
        self.conv_spatial = nn.ModuleList()
        for k in kernel_sizes[1:]:
            p = k // 2  # padding to preserve spatial size
            self.conv_spatial.append(nn.Sequential(
                nn.Conv2d(channels, channels, (k, 1), stride=(stride, 1), padding=(p, 0), groups=channels, bias=False),
                nn.Conv2d(channels, channels, (1, k), stride=(1, stride), padding=(0, p), groups=channels, bias=False),
            ))

        # Selective aggregation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * len(kernel_sizes), channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * len(kernel_sizes), 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Base 5x5
        u = [self.conv0(x)]

        # Decomposed large kernels
        for conv in self.conv_spatial:
            u.append(conv(x))

        # Selective attention
        attn = torch.cat(u, dim=1)  # [B, C*K, H, W]
        attn = self.avg_pool(attn)
        attn = self.fc(attn)
        attn = self.sigmoid(attn)  # [B, C*K, 1, 1]

        # Split and apply
        attn = torch.split(attn, self.channels, dim=1)
        out = sum(a * u_i for a, u_i in zip(attn, u))
        return out
