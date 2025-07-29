import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
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


class SR(nn.Module):
    '''Spatial reasoning module'''

    # codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(SR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma * out

        return out


class CotSR(nn.Module):
    # codes derived from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)

        q2 = self.query_conv2(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)

        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2


class BiSRNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(BiSRNet, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.SR1 = SR(128)
        self.SR2 = SR(128)
        self.CotSR = CotSR(128)

        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        initialize_weights(self.resCD, self.classifierCD, self.classifier1, self.classifier2)

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

        x = self.FCN.layer0(x)  # size:1/4
        x = self.FCN.maxpool(x)  # size:1/4
        x = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x)  # size:1/8
        x = self.FCN.layer3(x)  # size:1/16
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)

        return x

    def CD_forward(self, x1, x2):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x2], 1)
        x = self.resCD(x)
        change = self.classifierCD(x)
        return change

    def forward(self, x1, x2):
        x_size = x1.size()
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        x1 = self.SR1(x1)
        x2 = self.SR2(x2)
        change = self.CD_forward(x1, x2)

        x1, x2 = self.CotSR(x1, x2)
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)

        return F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'), F.upsample(
            change, x_size[2:], mode='bilinear').squeeze(1)


class BiSRNet_MT(nn.Module):
    def __init__(self, in_channels=4, num_classes=13):
        super(BiSRNet_MT, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.SR1 = SR(128)
        self.SR2 = SR(128)
        self.SR3 = SR(128)
        self.CotSR = CotSR(128)
        self.softmax = nn.Softmax(dim=1)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        initialize_weights(self.resCD, self.classifierCD, self.classifier1, self.classifier2, self.classifier3)

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
        x = x.float()
        x = self.FCN.layer0(x)  # size:1/4
        x = self.FCN.maxpool(x)  # size:1/4
        x = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x)  # size:1/8
        x = self.FCN.layer3(x)  # size:1/16
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)

        return x

    def CD_forward(self, x1, x3):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x3], 1)
        x = self.resCD(x)
        change = self.classifierCD(x)
        return change

    def forward(self, x1, x2, x3):
        x_size = x1.size()
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        x3 = self.base_forward(x3)
        x1 = self.SR1(x1)
        x2 = self.SR2(x2)
        x3 = self.SR3(x3)
        change = self.CD_forward(x1, x3)

        x1, x3 = self.CotSR(x1, x3)
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        out3 = self.classifier3(x3)

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        change = torch.sigmoid(change)

        return F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'), F.upsample(
            out3, x_size[2:], mode='bilinear'), F.upsample(
            change, x_size[2:], mode='bilinear').squeeze(1)


class BiSRNet_NMT(nn.Module):
    def __init__(self, in_channels=4, num_classes=7):
        super(BiSRNet_NMT, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.SR1 = SR(128)
        self.SR2 = SR(128)
        self.SR3 = SR(128)
        self.SR4 = SR(128)
        self.SR5 = SR(128)
        self.SR6 = SR(128)
        self.CotSR = CotSR(128)
        self.softmax = nn.Softmax(dim=1)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier5 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier6 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        initialize_weights(self.resCD, self.classifierCD, self.classifier1, self.classifier2, self.classifier3,
                           self.classifier4, self.classifier5, self.classifier6)

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
        x = x.float()
        x = self.FCN.layer0(x)  # size:1/4
        x = self.FCN.maxpool(x)  # size:1/4
        x = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x)  # size:1/8
        x = self.FCN.layer3(x)  # size:1/16
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)

        return x

    def CD_forward(self, x1, x3):
        b, c, h, w = x1.size()
        x = torch.cat([x1, x3], 1)
        x = self.resCD(x)
        change = self.classifierCD(x)
        return change

    def forward(self, x1, x2, x3, x4, x5, x6):
        x_size = x1.size()
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        x3 = self.base_forward(x3)
        x4 = self.base_forward(x4)
        x5 = self.base_forward(x5)
        x6 = self.base_forward(x6)
        x1 = self.SR1(x1)
        x2 = self.SR2(x2)
        x3 = self.SR3(x3)
        x4 = self.SR4(x4)
        x5 = self.SR5(x5)
        x6 = self.SR6(x6)
        change = self.CD_forward(x1, x6)

        x1, x6 = self.CotSR(x1, x6)
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        out3 = self.classifier3(x3)
        out4 = self.classifier4(x4)
        out5 = self.classifier5(x5)
        out6 = self.classifier6(x6)

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        out4 = self.softmax(out4)
        out5 = self.softmax(out5)
        out6 = self.softmax(out6)
        change = torch.sigmoid(change)

        return (F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'),
                F.upsample(out3, x_size[2:], mode='bilinear'),
                F.upsample(out4, x_size[2:], mode='bilinear'), F.upsample(out5, x_size[2:], mode='bilinear'),
                F.upsample(out6, x_size[2:], mode='bilinear'),
                F.upsample(change, x_size[2:], mode='bilinear').squeeze(1))


class BiSRNet_NMT_random(nn.Module):
    def __init__(self, in_channels=4, num_classes=7):
        super(BiSRNet_NMT_random, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.SR1 = SR(128)
        self.SR2 = SR(128)
        self.SR3 = SR(128)
        self.SR4 = SR(128)
        self.SR5 = SR(128)
        self.SR6 = SR(128)
        self.CotSR = CotSR(128)
        self.softmax = nn.Softmax(dim=1)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier5 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier6 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        initialize_weights(self.resCD, self.classifierCD, self.classifier1, self.classifier2, self.classifier3,
                           self.classifier4, self.classifier5, self.classifier6)

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
        x = x.float()
        x = self.FCN.layer0(x)  # size:1/4
        x = self.FCN.maxpool(x)  # size:1/4
        x = self.FCN.layer1(x)  # size:1/4
        x = self.FCN.layer2(x)  # size:1/8
        x = self.FCN.layer3(x)  # size:1/16
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)

        return x

    def CD_forward(self, x, y):
        b, c, h, w = x.size()
        xy = torch.cat([x, y], 1)
        cd = self.resCD(xy)
        change = self.classifierCD(cd)
        return change

    def forward(self, x1, x2, x3, x4, x5, x6, pair):
        x_size = x1.size()
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        x3 = self.base_forward(x3)
        x4 = self.base_forward(x4)
        x5 = self.base_forward(x5)
        x6 = self.base_forward(x6)
        x1 = self.SR1(x1)
        x2 = self.SR2(x2)
        x3 = self.SR3(x3)
        x4 = self.SR4(x4)
        x5 = self.SR5(x5)
        x6 = self.SR6(x6)
        xs = [x1, x2, x3, x4, x5, x6]

        change = self.CD_forward(xs[pair[0]], xs[pair[-1]])

        xs[pair[0]], xs[pair[-1]] = self.CotSR(xs[pair[0]], xs[pair[-1]])

        out1 = self.classifier1(xs[0])
        out2 = self.classifier2(xs[1])
        out3 = self.classifier3(xs[2])
        out4 = self.classifier4(xs[3])
        out5 = self.classifier5(xs[4])
        out6 = self.classifier6(xs[5])

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        out4 = self.softmax(out4)
        out5 = self.softmax(out5)
        out6 = self.softmax(out6)
        change = torch.sigmoid(change)

        # return [F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'),
        #         F.upsample(out3, x_size[2:], mode='bilinear'),
        #         F.upsample(out4, x_size[2:], mode='bilinear'), F.upsample(out5, x_size[2:], mode='bilinear'),
        #         F.upsample(out6, x_size[2:], mode='bilinear')], F.upsample(change, x_size[2:], mode='bilinear').squeeze(1)

        return (F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear'),F.upsample(out3, x_size[2:], mode='bilinear'),
                F.upsample(out4, x_size[2:], mode='bilinear'), F.upsample(out5, x_size[2:], mode='bilinear'),F.upsample(out6, x_size[2:], mode='bilinear'),
                F.upsample(change, x_size[2:], mode='bilinear').squeeze(1))



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiSRNet().to(device)
    # model = BiGrootV_base(backbone='GrootV', pretrained=False, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)
    seg1, seg2, change = model(image1, image2)
    # print(seg1)
    from thop import profile

    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))