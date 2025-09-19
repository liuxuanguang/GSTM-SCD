from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIGM import CIGM
from models.Modules.CIEM import CIEM
import torch
from torch import nn
import torch.nn.functional as F
from utils.misc import initialize_weights
from GrootV.classification.models.grootv import STM3DLayer, GOST_Mamba_bi

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
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

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

#TSSCS
class TSSCS_bi(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(TSSCS_bi, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = STM3DLayer(channels=128)
        self.smooth_layer_x = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_y = ResBlock(in_channels=128, out_channels=128, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()

        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_42[:, :, :, 0:W] = x
        ct_tensor_42[:, :, :, W:2*W] = y
        ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = f2[:, :, :, 0:W]
        yf_sm = f2[:, :, :, W:2*W]

        xf_sm = self.smooth_layer_x(xf_sm)
        yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm

# Proposed GSTM-SCD for bi-temporal SCD tasks
class GSTMSCD_Bitemporal_tiny(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda, first_channel, depths):
        super(GSTMSCD_Bitemporal_tiny, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.first_channel = first_channel
        self.depths = depths
        self.backbone = GOST_Mamba_bi(channels=self.first_channel, depths=self.depths)
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GOST-Mamba":
            self.channel_nums = [80, 160, 320, 640, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GOST-Mamba":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]
        self.MambaLayer = TSSCS_bi(self.channel_nums[3], False)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu后面不用
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        old_dict = self.backbone.state_dict
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.CD_Decoder,
                           self.seg_conv, self.classifierCD, self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4)
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

    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        return x_f, y_f

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = x1
        xy_in[:, :, :, w:2*w] = x2
        feature_xy = self.backbone.forward(xy_in)
        # Initialize the feature1 and feature2 lists
        feature1 = []
        feature2 = []
        # Traverse each matrix in feature_xy
        for matrix in feature_xy:
            W = matrix.size(3)
            left_part = matrix[:, :, :, :W // 2]
            right_part = matrix[:, :, :, W // 2:]
            feature1.append(left_part)
            feature2.append(right_part)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder1(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)


class GSTMSCD_Bitemporal_samll(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda, first_channel, depths):
        super(GSTMSCD_Bitemporal_samll, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.first_channel = first_channel
        self.depths = depths
        self.backbone = GOST_Mamba_bi(depths=self.depths, channels=self.first_channel)

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GOST-Mamba":
            self.channel_nums = [96, 192, 384, 768, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GOST-Mamba":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]
        self.MambaLayer = TSSCS_bi(self.channel_nums[3], False)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_small.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
                           self.seg_conv, self.classifierCD, self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4)
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

    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        return x_f, y_f

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = x1
        xy_in[:, :, :, w:2*w] = x2
        feature_xy = self.backbone.forward(xy_in)
        feature1 = []
        feature2 = []
        for matrix in feature_xy:
            W = matrix.size(3)
            left_part = matrix[:, :, :, :W // 2] 
            right_part = matrix[:, :, :, W // 2:] 
            feature1.append(left_part)
            feature2.append(right_part)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GSTMSCD_Bitemporal_samll(backbone='GOST-Mamba', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005, first_channel=96, depths=[2, 2, 13, 2]).to(device)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)
    seg1, seg2, change = model(image1, image2)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

