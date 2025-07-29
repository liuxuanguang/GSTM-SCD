from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIGM import CIGM
from models.Modules.CIEM import  CIEM
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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EGMSNet(backbone='resnet34', pretrained=True, nclass=6, lightweight=True, M=6, Lambda=0.00005).to(device)
    print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
    with torch.no_grad():
        tc1, tc2, cd = model.forward(image1, image2)
    print(cd.shape, tc1.shape, tc2.shape)