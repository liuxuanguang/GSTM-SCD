import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import initialize_weights


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(_EncoderBlock, self).__init__()
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = [
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        if self.downsample:
            x = self.maxpool(x)
        x = self.encode(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch1, in_ch1, kernel_size=2, stride=2)
        in_ch = in_ch1 + in_ch2
        self.decode = nn.Sequential(
            conv3x3(in_ch, in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            conv3x3(in_ch, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.decode(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        self.Enc0 = _EncoderBlock(in_channels, 16, downsample=False)
        self.Enc1 = _EncoderBlock(16, 32)
        self.Enc2 = _EncoderBlock(32, 64)
        self.Enc3 = _EncoderBlock(64, 128)
        self.Enc4 = _EncoderBlock(128, 128)

    def forward(self, x):
        enc0 = self.Enc0(x)
        enc1 = self.Enc1(enc0)
        enc2 = self.Enc2(enc1)
        enc3 = self.Enc3(enc2)
        enc4 = self.Enc4(enc3)

        encs = [enc0, enc1, enc2, enc3, enc4]
        return encs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.Dec3 = _DecoderBlock(128, 128, 64)
        self.Dec2 = _DecoderBlock(64, 64, 32)
        self.Dec1 = _DecoderBlock(32, 32, 16)
        self.Dec0 = _DecoderBlock(16, 16, 16)

    def forward(self, encs1, encs2):
        f1_enc0, f1_enc1, f1_enc2, f1_enc3, f1_enc4 = encs1
        f2_enc0, f2_enc1, f2_enc2, f2_enc3, f2_enc4 = encs2

        dec3 = self.Dec3(f1_enc4, torch.abs(f1_enc3 - f2_enc3))
        dec2 = self.Dec2(dec3, torch.abs(f1_enc2 - f2_enc2))
        dec1 = self.Dec1(dec2, torch.abs(f1_enc1 - f2_enc1))
        dec0 = self.Dec0(dec1, torch.abs(f1_enc0 - f2_enc0))
        return dec0


class FC_Siam_diff(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(FC_Siam_diff, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.classifier1 = nn.Conv2d(16, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(16, num_classes, kernel_size=1)

        initialize_weights(self.encoder, self.decoder, self.classifier1, self.classifier2)

    def forward(self, x1, x2):
        encs1 = self.encoder(x1)
        encs2 = self.encoder(x2)
        x = self.decoder(encs1, encs2)

        x1 = self.classifier1(x)
        x2 = self.classifier2(x)

        return x1, x2


class FC_Siam_diff_dynamic(nn.Module):
    def __init__(self, in_channels=4, num_classes=8):
        super(FC_Siam_diff_dynamic, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.classifier1 = nn.Conv2d(16, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(16, num_classes, kernel_size=1)

        initialize_weights(self.encoder, self.decoder, self.classifier1, self.classifier2)

    def forward(self, x1, x2, x3, x4, x5, x6, pair):
        xs = [x1.float(), x2.float(), x3.float(), x4.float(), x5.float(), x6.float()]
        encs1 = self.encoder(xs[pair[0]])
        encs2 = self.encoder(xs[pair[1]])
        x = self.decoder(encs1, encs2)

        xs[pair[0]] = self.classifier1(x)
        xs[pair[-1]] = self.classifier2(x)

        return [xs[pair[0]], xs[pair[-1]]]


