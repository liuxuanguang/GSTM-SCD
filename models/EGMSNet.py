from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder_base import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIGM import CIGM
from models.Modules.CIEM import CIEM
import torch
from torch import nn
import torch.nn.functional as F

def get_backbone(backbone, pretrained):
    if backbone == 'resnet18':
        backbone = resnet18(pretrained)
    elif backbone == 'resnet34':
        backbone = resnet34(pretrained)
    elif backbone == 'resnet50':
        backbone = resnet50(pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)
    return backbone

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3_dw(in_channel,out_channel,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=stride,padding=1,groups=in_channel,bias=True),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_dw(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_dw(planes, planes)
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

class EGMSNet(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(EGMSNet, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = get_backbone(backbone, pretrained)

        if backbone == "resnet18" or backbone == "resnet34":
            self.channel_nums = [64, 128, 256, 512]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50":
            self.Seg_Decoder = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = CD_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
        self.DIGM = CIGM(256)

        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True)
        )

        self.resCD = self._make_layer(ResBlock, 256, 256, self.M, stride=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
                                          nn.Conv2d(128, 1, kernel_size=1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature1_0, feature1_1, feature1_2, feature1_3 = feature1[0], feature1[1], feature1[2], feature1[3]
        feature2_0, feature2_1, feature2_2, feature2_3 = feature2[0], feature2[1], feature2[2], feature2[3]
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))

        #SegDecoder
        out_1 = self.Seg_Decoder(feature1)
        out_2 = self.Seg_Decoder(feature2)

        #CDDecoder
        out_change = self.CD_Decoder(feature_diff)


        #CIGM
        out_change = self.resCD(out_change)
        out_1, out_2, out_change = self.DIGM(out_1, out_2, out_change)

        change = F.interpolate(self.classifierCD(out_change), size=(h, w), mode='bilinear', align_corners=False)
        seg1 =  F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)

        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

class EGMSNet_MT(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(EGMSNet_MT, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = get_backbone(backbone, pretrained)
        self.softmax = nn.Softmax(dim=1)
        if backbone == "resnet18" or backbone == "resnet34":
            self.channel_nums = [64, 128, 256, 512]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50":
            self.Seg_Decoder = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = CD_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
        self.DIGM = CIGM(256)

        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True)
        )

        self.resCD = self._make_layer(ResBlock, 256, 256, self.M, stride=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
                                          nn.Conv2d(128, 1, kernel_size=1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        b, c, h, w = x1.shape
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature3 = self.backbone.forward(x3)
        feature1_0, feature1_1, feature1_2, feature1_3 = feature1[0], feature1[1], feature1[2], feature1[3]
        feature2_0, feature2_1, feature2_2, feature2_3 = feature2[0], feature2[1], feature2[2], feature2[3]
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature3[i]))

        #SegDecoder
        out_1 = self.Seg_Decoder(feature1)
        out_2 = self.Seg_Decoder(feature2)
        out_3 = self.Seg_Decoder(feature3)
        #CDDecoder
        out_change = self.CD_Decoder(feature_diff)
        #CIGM
        out_change = self.resCD(out_change)
        out_1, out_3, out_change = self.DIGM(out_1, out_3, out_change)

        change = F.interpolate(self.classifierCD(out_change), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg3 = F.interpolate(self.seg_conv(out_3), size=(h, w), mode='bilinear', align_corners=False)
        change = torch.sigmoid(change)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        seg3 = self.softmax(seg3)

        return seg1, seg2, seg3, change.squeeze(1)

# class EGMSNet_NMT(nn.Module):
#     def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
#         super(EGMSNet_NMT, self).__init__()
#         self.backbone_name = backbone
#         self.nclass = nclass
#         self.lightweight = lightweight
#         self.M = M
#         self.Lambda = Lambda
#         self.backbone = get_backbone(backbone, pretrained)
#         self.softmax = nn.Softmax(dim=1)
#         if backbone == "resnet18" or backbone == "resnet34":
#             self.channel_nums = [64, 128, 256, 512]
#         elif backbone == "resnet50":
#             self.channel_nums = [256, 512, 1024, 2048]
#
#         if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50":
#             self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
#             self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
#             self.Seg_Decoder3 = Seg_Decoder_ResNet(self.channel_nums)
#             self.Seg_Decoder4 = Seg_Decoder_ResNet(self.channel_nums)
#             self.Seg_Decoder5 = Seg_Decoder_ResNet(self.channel_nums)
#             self.Seg_Decoder6 = Seg_Decoder_ResNet(self.channel_nums)
#             self.CD_Decoder = CD_Decoder_ResNet(self.channel_nums)
#         else:
#             self.Seg_Decoder = Seg_Decoder(self.channel_nums)
#             self.CD_Decoder = CD_Decoder(self.channel_nums)
#
#         self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
#         self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
#         self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
#         self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
#         self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
#         self.DIGM = CIGM(256)
#
#         self.seg_conv = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.Dropout(0.2, False),
#             nn.Conv2d(128, self.nclass, 1, bias=True)
#         )
#
#         self.resCD = self._make_layer(ResBlock, 256, 256, self.M, stride=1)
#         self.classifierCD = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
#                                           nn.Conv2d(128, 1, kernel_size=1))
#
#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 conv1x1(inplanes, planes, stride),
#                 nn.BatchNorm2d(planes) )
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x1, x2, x3, x4, x5, x6):
#         b, c, h, w = x1.shape
#         x1 = x1.float()
#         x2 = x2.float()
#         x3 = x3.float()
#         x4 = x4.float()
#         x5 = x5.float()
#         x6 = x6.float()
#         feature1 = self.backbone.forward(x1)
#         feature2 = self.backbone.forward(x2)
#         feature3 = self.backbone.forward(x3)
#         feature4 = self.backbone.forward(x4)
#         feature5 = self.backbone.forward(x5)
#         feature6 = self.backbone.forward(x6)
#         feature1_0, feature1_1, feature1_2, feature1_3 = feature1[0], feature1[1], feature1[2], feature1[3]
#         feature2_0, feature2_1, feature2_2, feature2_3 = feature6[0], feature6[1], feature6[2], feature6[3]
#         feature_diff = []
#         for i in range(len(feature1)):
#             feature_diff.append(self.CFEM[i](feature1[i], feature6[i]))
#
#         #SegDecoder
#         out_1 = self.Seg_Decoder1(feature1)
#         out_2 = self.Seg_Decoder2(feature2)
#         out_3 = self.Seg_Decoder3(feature3)
#         out_4 = self.Seg_Decoder4(feature4)
#         out_5 = self.Seg_Decoder5(feature5)
#         out_6 = self.Seg_Decoder6(feature6)
#         #CDDecoder
#         out_change = self.CD_Decoder(feature_diff)
#         #CIGM
#         out_change = self.resCD(out_change)
#         out_1, out_6, out_change = self.DIGM(out_1, out_6, out_change)
#
#         change = F.interpolate(self.classifierCD(out_change), size=(h, w), mode='bilinear', align_corners=False)
#         seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
#         seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
#         seg3 = F.interpolate(self.seg_conv(out_3), size=(h, w), mode='bilinear', align_corners=False)
#         seg4 = F.interpolate(self.seg_conv(out_4), size=(h, w), mode='bilinear', align_corners=False)
#         seg5 = F.interpolate(self.seg_conv(out_5), size=(h, w), mode='bilinear', align_corners=False)
#         seg6 = F.interpolate(self.seg_conv(out_6), size=(h, w), mode='bilinear', align_corners=False)
#         change = torch.sigmoid(change)
#         seg1 = self.softmax(seg1)
#         seg2 = self.softmax(seg2)
#         seg3 = self.softmax(seg3)
#         seg4 = self.softmax(seg4)
#         seg5 = self.softmax(seg5)
#         seg6 = self.softmax(seg6)
#         return seg1, seg2, seg3, seg4, seg5, seg6, change.squeeze(1)

import random
import random
class EGMSNet_NMT(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(EGMSNet_NMT, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = get_backbone(backbone, pretrained)
        self.softmax = nn.Softmax(dim=1)
        if backbone == "resnet18" or backbone == "resnet34":
            self.channel_nums = [64, 128, 256, 512]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder3 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder4 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder5 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder6 = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = CD_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
        self.DIGM = CIGM(256)

        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True)
        )

        self.resCD = self._make_layer(ResBlock, 256, 256, self.M, stride=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
                                          nn.Conv2d(128, 1, kernel_size=1))

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

    def forward(self, x1, x2, x3, x4, x5, x6, pair):
        b, c, h, w = x1.shape
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        x4 = x4.float()
        x5 = x5.float()
        x6 = x6.float()
        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature3 = self.backbone.forward(x3)
        feature4 = self.backbone.forward(x4)
        feature5 = self.backbone.forward(x5)
        feature6 = self.backbone.forward(x6)
        feature1_0, feature1_1, feature1_2, feature1_3 = feature1[0], feature1[1], feature1[2], feature1[3]
        feature2_0, feature2_1, feature2_2, feature2_3 = feature6[0], feature6[1], feature6[2], feature6[3]

        # LXG:添加随机数
        feature = [feature1, feature2, feature3, feature4, feature5, feature6]
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature[pair[0]][i], feature[pair[-1]][i]))

        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        out_3 = self.Seg_Decoder3(feature3)
        out_4 = self.Seg_Decoder4(feature4)
        out_5 = self.Seg_Decoder5(feature5)
        out_6 = self.Seg_Decoder6(feature6)
        # CDDecoder
        out_change = self.CD_Decoder(feature_diff)

        out = [out_1, out_2, out_3, out_4, out_5, out_6]
        # CIGM
        # out_change = self.resCD(out_change)
        # out_1, out_6, out_change = self.DIGM(out[pair[0]], out[pair[-1]], out_change)

        change = F.interpolate(self.classifierCD(out_change), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg3 = F.interpolate(self.seg_conv(out_3), size=(h, w), mode='bilinear', align_corners=False)
        seg4 = F.interpolate(self.seg_conv(out_4), size=(h, w), mode='bilinear', align_corners=False)
        seg5 = F.interpolate(self.seg_conv(out_5), size=(h, w), mode='bilinear', align_corners=False)
        seg6 = F.interpolate(self.seg_conv(out_6), size=(h, w), mode='bilinear', align_corners=False)
        change = torch.sigmoid(change)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        seg3 = self.softmax(seg3)
        seg4 = self.softmax(seg4)
        seg5 = self.softmax(seg5)
        seg6 = self.softmax(seg6)
        return seg1, seg2, seg3, seg4, seg5, seg6, change.squeeze(1)
class REsnet34_encoder(nn.Module):
    def __init__(self):
        super(REsnet34_encoder, self).__init__()

        self.backbone = resnet34()

    def forward(self, x1, x2, x3):
        b, c, h, w = x1.shape
        feature_x1 = self.backbone.forward(x1)
        feature_x2 = self.backbone.forward(x2)
        feature_x3 = self.backbone.forward(x3)
        # 初始化feature1和feature2列表
        return feature_x1, feature_x2, feature_x3

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MTGrootV3D_SV3(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    model = REsnet34_encoder().to(device)
    # model = BiGrootV_base(backbone='GrootV', pretrained=False, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)
    image3 = torch.randn(1, 3, 512, 512).to(device)
    # seg1, seg2, seg3, change = model(image1, image2, image3)
    fs = model(image1, image2, image3)
    # print(seg1)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2, image3))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = EGMSNet(backbone='resnet34', pretrained=True, nclass=6, lightweight=True, M=6, Lambda=0.00005).to(device)
#     print(model)
#     image1 = torch.randn(1, 3, 512, 512).to(device)
#     image2 = torch.randn(1, 3, 512, 512).to(device)
#     from thop import profile
#     FLOPs, Params = profile(model, (image1, image2))
#     print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
#     with torch.no_grad():
#         tc1, tc2, cd = model.forward(image1, image2)
#     print(cd.shape, tc1.shape, tc2.shape)