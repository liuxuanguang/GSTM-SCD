import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List
from utils.misc import initialize_weights
import os
from vmamba1 import VSSM, LayerNorm2d, VSSBlock, Permute

working_path = os.path.abspath('.')

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class STM(nn.Module):
    def __init__(self, encoder_dims, hidden_dim, channel_first, norm_layer, ssm_act_layer, mlp_act_layer):
        super(STM, self).__init__()
        hidden_dim = hidden_dim
        self.st_block = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims, out_channels=hidden_dim),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=hidden_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=16, ssm_ratio=2.0,
                     ssm_dt_rank="auto", ssm_act_layer=ssm_act_layer,
                     ssm_conv=3, ssm_conv_bias='false',
                     ssm_drop_rate=0, ssm_init="v0",
                     forward_type="v2", mlp_ratio=4.0,
                     mlp_act_layer=mlp_act_layer, mlp_drop_rate=0.0,
                     gmlp=False, use_checkpoint=False),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Smooth layer
        self.smooth_layer = ResBlock(hidden_dim, hidden_dim)
        
    def forward(self, x1, x2):

        B, C, H, W = x1.size()
        device = x1.device

        # Create an empty tensor of the correct shape (B, C, H, 2 * W)
        ct_tensor = torch.empty(B, C, H, 2*W).to(device)
        # Fill in odd columns with A and even columns with B
        ct_tensor[:, :, :, ::2] = x1  # odd columns
        ct_tensor[:, :, :, 1::2] = x2  # even columns
        
        xf = self.st_block(ct_tensor)        
        x1_f = xf[:, :, ::, ::2]
        x2_f = xf[:, :, ::, 1::2]
        
        x1_sm = self.smooth_layer(x1_f)
        x2_sm = self.smooth_layer(x2_f)

        return x1_sm, x2_sm


