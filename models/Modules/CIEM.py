# -*- coding: utf-8 -*-
"""
@Create on ： 2023/9/9 20:50
@Author ： Zuoxibing
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        att_weight = self.activaton(y)

        return att_weight

class CIEM(nn.Module):
    def __init__(self, in_d, out_d, Lambda):
        super(CIEM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.Lambda = Lambda
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d*2, self.out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.bn1 = nn.BatchNorm2d(self.in_d)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(self.in_d)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(self.out_d)
        self.relu3 = nn.ReLU(inplace=True)
        self.SSFC1 = SimAM(e_lambda=self.Lambda)
        self.SSFC2 = SimAM(e_lambda=self.Lambda)

    def forward(self, input1, input2):
        diff = torch.abs(input1 - input2)
        att = self.SSFC1.forward(diff)
        diff = diff * att

        feature1 = self.relu1(input1 * att + input1)
        feature2 = self.relu2(input2 * att + input2)

        feature_cat = torch.cat([feature1, feature2], dim=1)
        feature_cat = self.conv_dr(feature_cat)
        att_cat = self.SSFC2.forward(feature_cat)
        feature_cat = att_cat * feature_cat

        feature_add = diff + feature_cat
        different =  self.relu3(feature_add)
        return different