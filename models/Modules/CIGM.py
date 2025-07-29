# -*- coding: utf-8 -*-
"""
@Create on ： 2023/9/9 20:50
@Author ： Zuoxibing
"""
import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CIGM(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.inchannels = inchannels
        self.channel_attention1 = ChannelAttention(self.inchannels, 8)
        self.channel_attention2 = ChannelAttention(self.inchannels, 8)
        self.spatial_attention = SpatialAttention()
        self.bn = nn.BatchNorm2d(self.inchannels)
        self.identity1 = nn.Identity()
        self.identity2 = nn.Identity()

    def forward(self, x1,x2,xc):
        att = self.spatial_attention(xc)
        x1 = self.channel_attention1(x1) * x1
        x2 = self.channel_attention1(x2) * x2
        out1 = self.identity1(x1 * att)
        out2 = self.identity2(x2 * att)
        outc = self.bn(xc * att)
        return out1, out2, outc