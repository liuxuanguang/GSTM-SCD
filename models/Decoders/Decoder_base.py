import torch
import torch.nn as nn
import torch.nn.functional as F

class Seg_Decoder(nn.Module):
    def __init__(self, channel_nums):
        super().__init__()

        self.channel_nums = channel_nums
        self.channel = 256
        # self.channel_nums = [96, 192, 384, 768]
        self.conv1 = self.Conv1(self.channel_nums[3] + self.channel_nums[2], self.channel)
        self.conv2 = self.Conv1(self.channel + self.channel_nums[1], self.channel)
        self.conv3 = self.Conv1(self.channel + self.channel_nums[0], self.channel)

    def forward(self, layers):
        layer0, layer1, layer2, layer3 = layers[0], layers[1], layers[2], layers[3]
        layer3 = F.interpolate(layer3, scale_factor=2, mode='bilinear', align_corners=True)

        layer2 = self.conv1(torch.cat((layer3, layer2), dim=1))
        layer2 = F.interpolate(layer2, scale_factor=2, mode='bilinear', align_corners=True)
        layer1 = self.conv2(torch.cat((layer2, layer1), dim=1))
        layer1 = F.interpolate(layer1, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv3(torch.cat((layer1, layer0), dim=1))

        return out

    def Conv1(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        return conv

class Seg_Decoder_ResNet(nn.Module):
    def __init__(self, channel_nums):
        super().__init__()

        self.channel_nums = channel_nums
        self.channel = 256
        # self.channel_nums = [96, 192, 384, 768]
        self.conv1 = self.Conv1(self.channel_nums[3] + self.channel_nums[2], self.channel)
        self.conv2 = self.Conv1(self.channel + self.channel_nums[1], self.channel)
        self.conv3 = self.Conv1(self.channel + self.channel_nums[0], self.channel)

    def forward(self, layers):
        layer0, layer1, layer2, layer3 = layers[0], layers[1], layers[2], layers[3]
        # layer3 = F.interpolate(layer3, scale_factor=2, mode='bilinear', align_corners=True)
        layer2 = self.conv1(torch.cat((layer3, layer2), dim=1))
        # layer2 = F.interpolate(layer2, scale_factor=2, mode='bilinear', align_corners=True)
        layer1 = self.conv2(torch.cat((layer2, layer1), dim=1))
        layer1 = F.interpolate(layer1, scale_factor=2, mode='bilinear', align_corners=True)
        layer1 = self.conv3(torch.cat((layer1, layer0), dim=1))


        return layer1

    def Conv1(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        return conv

class CD_Decoder(nn.Module):
    def __init__(self, channel_nums):
        super().__init__()

        self.channel_nums = channel_nums
        self.channel = 256
        # self.channel_nums = [96, 192, 384, 768]
        self.conv1 = self.Conv1(self.channel_nums[3] + self.channel_nums[2], self.channel)
        self.conv2 = self.Conv1(self.channel + self.channel_nums[1], self.channel)
        self.conv3 = self.Conv1(self.channel + self.channel_nums[0], self.channel)

    def forward(self, layers):
        layer0, layer1, layer2, layer3 = layers[0], layers[1], layers[2], layers[3]
        layer3 = F.interpolate(layer3, scale_factor=2, mode='bilinear', align_corners=True)

        layer2 = self.conv1(torch.cat((layer3, layer2), dim=1))
        layer2 = F.interpolate(layer2, scale_factor=2, mode='bilinear', align_corners=True)
        layer1 = self.conv2(torch.cat((layer2, layer1), dim=1))
        layer1 = F.interpolate(layer1, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv3(torch.cat((layer1, layer0), dim=1))

        return out

    def Conv1(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        return conv

class CD_Decoder_ResNet(nn.Module):
    def __init__(self, channel_nums):
        super().__init__()

        self.channel_nums = channel_nums
        self.channel = 256
        # self.channel_nums = [96, 192, 384, 768]
        self.conv1 = self.Conv1(self.channel_nums[3] + self.channel_nums[2], self.channel)
        self.conv2 = self.Conv1(self.channel + self.channel_nums[1], self.channel)
        self.conv3 = self.Conv1(self.channel + self.channel_nums[0], self.channel)

    def forward(self, layers):
        layer0, layer1, layer2, layer3 = layers[0], layers[1], layers[2], layers[3]
        # layer3 = F.interpolate(layer3, scale_factor=2, mode='bilinear', align_corners=True)
        layer2 = self.conv1(torch.cat((layer3, layer2), dim=1))
        # layer2 = F.interpolate(layer2, scale_factor=2, mode='bilinear', align_corners=True)
        layer1 = self.conv2(torch.cat((layer2, layer1), dim=1))
        layer1 = F.interpolate(layer1, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv3(torch.cat((layer1, layer0), dim=1))

        return out

    def Conv1(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        return conv