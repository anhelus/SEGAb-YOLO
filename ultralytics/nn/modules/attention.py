import torch
import torch.nn as nn
from torch.nn import functional as F

from ultralytics.nn.modules.conv import ChannelAttention, SpatialAttention

__all__ = (
    'CoT',
    'ODConv',
    'SimAM',
    'GAM',
)


class ODConv(nn.Module):
    """
    Omni-dimensional Dynamic Convolution (Robust and Corrected Implementation).
    Includes automatic padding calculation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 dilation=1, groups=1, reduction=0.0625, kernel_num=4):
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
            nn.Conv2d(hidden_channels, kernel_num, 1, bias=False)
        )
        
        # The master weight tensor
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size))
        
        # A small initialization trick for stability
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

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
                padding=self.padding, # Now uses the correctly calculated padding
                dilation=self.dilation,
                groups=self.groups
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
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.query_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Value convolution
        self.value_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # Static context extraction
        self.local_context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # Final fusion convolution
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
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
    """ TODO
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
    
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class GAM(nn.Module):
    """
    Global Attention Mechanism.
    This module is a wrapper that sequentially applies the official Ultralytics
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

    def forward(self, x):
        """Applies channel attention, then spatial attention."""
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class PConv(nn.Module):
    """
    Partial Convolution.
    Reference: https://arxiv.org/abs/2303.03667
    """
    def __init__(self, in_channels, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size=3, stride=1, padding=1, bias=False)
        
        if forward == 'split_cat':
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
                nn.BatchNorm2d(out_channels)
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
