import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from models.CSWin_Transformer import mit

args = {'hidden_size': 128 * 3,
        'mlp_dim': 256 * 3,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.}


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low // scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low // scale_ratio),
            nn.BatchNorm2d(in_channels_low // scale_ratio),
            nn.ReLU(inplace=True))
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


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


class SCanNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=416):
        super(SCanNet, self).__init__()
        feat_size = input_size // 4
        self.FCN = FCN(in_channels, pretrained=True)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=feat_size, in_chans=128 * 3, embed_dim=128 * 3)

        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1 = _DecoderBlock(128, 64, 128)
        self.Dec2 = _DecoderBlock(128, 64, 128)

        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD,
                           self.classifierCD)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):

        x = self.FCN.layer0(x)  # size:1/2
        x = self.FCN.maxpool(x)  # size:1/4
        x_low = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x_low)  # size:1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low

    def CD_forward(self, x1, x2):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x2], 1)
        xc = self.resCD(x)
        return x1, x2, xc

    def forward(self, x1, x2):
        x_size = x1.size()

        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)
        x1, x2, xc = self.CD_forward(x1, x2)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)

        x = torch.cat([x1, x2, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x2 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]

        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        change = self.classifierCD(xc)

        return F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'), F.upsample(
            change, x_size[2:], mode='bilinear').squeeze(1)


class SCanNet_MT(nn.Module):
    def __init__(self, in_channels=4, num_classes=13, input_size=512):
        super(SCanNet_MT, self).__init__()
        feat_size = input_size // 4
        self.FCN = FCN(in_channels, pretrained=True)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=feat_size, in_chans=128 * 3, embed_dim=128 * 3)
        self.softmax = nn.Softmax(dim=1)
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1 = _DecoderBlock(128, 64, 128)
        self.Dec2 = _DecoderBlock(128, 64, 128)
        self.Dec3 = _DecoderBlock(128, 64, 128)
        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierC = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.Dec1, self.Dec2, self.classifierA, self.classifierB, self.resCD, self.DecCD,
                           self.classifierCD)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):

        x = self.FCN.layer0(x)  # size:1/2
        x = self.FCN.maxpool(x)  # size:1/4
        x_low = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x_low)  # size:1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low

    def CD_forward(self, x1, x3):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x3], 1)
        xc = self.resCD(x)
        return x1, x3, xc

    def forward(self, x1, x2, x3):
        x_size = x1.size()
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)
        x3, x3_low = self.base_forward(x3)
        x1, x3, xc = self.CD_forward(x1, x3)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)
        x3 = self.Dec3(x3, x3_low)

        xc_low = torch.cat([x1_low, x3_low], 1)
        xc = self.DecCD(xc, xc_low)

        x = torch.cat([x1, x3, xc], 1)
        x = self.transformer(x)
        x1 = x[:, 0:128, :, :]
        x3 = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]

        out1 = self.classifierA(x1)
        out2 = self.classifierB(x2)
        out3 = self.classifierC(x3)
        change = self.classifierCD(xc)

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        change = torch.sigmoid(change)

        return F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'), F.upsample(
            out3, x_size[2:], mode='bilinear'), F.upsample(
            change, x_size[2:], mode='bilinear').squeeze(1)


class SCanNet_NMT(nn.Module):
    def __init__(self, in_channels=4, num_classes=7, input_size=512):
        super(SCanNet_NMT, self).__init__()
        feat_size = input_size // 4
        self.FCN = FCN(in_channels, pretrained=True)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.transformer = mit(img_size=feat_size, in_chans=128 * 3, embed_dim=128 * 3)
        self.softmax = nn.Softmax(dim=1)
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.Dec1 = _DecoderBlock(128, 64, 128)
        self.Dec2 = _DecoderBlock(128, 64, 128)
        self.Dec3 = _DecoderBlock(128, 64, 128)
        self.Dec4 = _DecoderBlock(128, 64, 128)
        self.Dec5 = _DecoderBlock(128, 64, 128)
        self.Dec6 = _DecoderBlock(128, 64, 128)

        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierC = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierD = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierE = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierF = nn.Conv2d(128, num_classes, kernel_size=1)

        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.Dec1, self.Dec2, self.Dec3, self.Dec4, self.Dec5, self.Dec6, self.classifierA,
                           self.classifierB, self.classifierC,
                           self.classifierD, self.classifierE, self.classifierF, self.resCD, self.DecCD,
                           self.classifierCD)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):

        x = self.FCN.layer0(x)  # size:1/2
        x = self.FCN.maxpool(x)  # size:1/4
        x_low = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x_low)  # size:1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        return x, x_low

    def CD_forward(self, x1, x2):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x2], 1)
        xc = self.resCD(x)
        return x1, x2, xc

    def forward(self, x1, x2, x3, x4, x5, x6, pair):
        x_size = x1.size()
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        x4 = x4.float()
        x5 = x5.float()
        x6 = x6.float()
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)
        x3, x3_low = self.base_forward(x3)
        x4, x4_low = self.base_forward(x4)
        x5, x5_low = self.base_forward(x5)
        x6, x6_low = self.base_forward(x6)

        x_high = [x1, x2, x3, x4, x5, x6]
        x_low = [x1_low, x2_low, x3_low, x4_low, x5_low, x6_low]
        x_high[pair[0]], x_high[pair[-1]], xc = self.CD_forward(x_high[pair[0]], x_high[pair[-1]])
        x_high[0] = self.Dec1(x_high[0], x_low[0])
        x_high[1] = self.Dec2(x_high[1], x_low[1])
        x_high[2] = self.Dec3(x_high[2], x_low[2])
        x_high[3] = self.Dec4(x_high[3], x_low[3])
        x_high[4] = self.Dec5(x_high[4], x_low[4])
        x_high[5] = self.Dec6(x_high[5], x_low[5])

        xc_low = torch.cat([x_low[pair[0]], x_low[pair[-1]]], 1)
        xc = self.DecCD(xc, xc_low)

        x = torch.cat([x_high[pair[0]], x_high[pair[-1]], xc], 1)
        x = self.transformer(x)
        x_high[pair[0]] = x[:, 0:128, :, :]
        x_high[pair[-1]] = x[:, 128:256, :, :]
        xc = x[:, 256:, :, :]

        out1 = self.classifierA(x_high[0])
        out2 = self.classifierB(x_high[1])
        out3 = self.classifierC(x_high[2])
        out4 = self.classifierD(x_high[3])
        out5 = self.classifierE(x_high[4])
        out6 = self.classifierF(x_high[5])
        change = self.classifierCD(xc)

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        out4 = self.softmax(out4)
        out5 = self.softmax(out5)
        out6 = self.softmax(out6)
        change = torch.sigmoid(change)
        # return [F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'),
        #         F.upsample(out3, x_size[2:], mode='bilinear'), F.upsample(out4, x_size[2:], mode='bilinear'),
        #         F.upsample(out5, x_size[2:], mode='bilinear'),
        #         F.upsample(out6, x_size[2:], mode='bilinear')], F.upsample(change, x_size[2:], mode='bilinear').squeeze(
        #     1)

        return (F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'),F.upsample(out3, x_size[2:], mode='bilinear'),
                F.upsample(out4, x_size[2:], mode='bilinear'),
                F.upsample(out5, x_size[2:], mode='bilinear'),F.upsample(out6, x_size[2:], mode='bilinear'), F.upsample(change, x_size[2:], mode='bilinear').squeeze(1))