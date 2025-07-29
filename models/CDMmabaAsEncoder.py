# import nnunetv2.nets.SwinUMamba
from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, Seg_Decoder_CDMamba
# from models.Decoders.Decoder_base import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIGM import CIGM
from models.Modules.CIEM import CIEM
import torch
from torch import nn
import torch.nn.functional as F
from vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from utils.misc import initialize_weights
from models.testing_reorder import feature_resortV2, feature_resumption
from GrootV.classification.models.grootv import GrootVLayer, GrootV3DLayer, MTGrootV3DLayer
from GrootV.classification.models.grootv import GrootV, GrootV_3D

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

class ResBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResBlock1, self).__init__()
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


#使用3D树状扫描
class STM_MTGrootV3D_V2(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_MTGrootV3D_V2, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        # self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = MTGrootV3DLayer(channels=128)
        # self.smooth_layer_x = ResBlock1(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_y = ResBlock1(in_channels=128, out_channels=128, stride=1)
    def forward(self, x, y, z):
        B, C, H, W = x.size()

        ct_tensor_42 = torch.empty(B, C, H, 3 * W).cuda()
        ct_tensor_42[:, :, :, 0:W] = x
        ct_tensor_42[:, :, :, W:2*W] = y
        ct_tensor_42[:, :, :, 2*W:3*W] = z
        # ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = f2[:, :, :, 0:W]
        yf_sm = f2[:, :, :, W:2*W]
        zf_sm = f2[:, :, :, 2*W:3*W]
        # xf_sm = self.smooth_layer_x(xf_sm)
        # yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm, zf_sm

class STM_MTGrootV3D_V2_DynamicEarth(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_MTGrootV3D_V2_DynamicEarth, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        # self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = MTGrootV3DLayer(channels=640)
        # self.smooth_layer_x = ResBlock1(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_y = ResBlock1(in_channels=128, out_channels=128, stride=1)
    def forward(self, x1, x2, x3, x4, x5, x6):
        B, C, H, W = x1.size()

        ct_tensor_42 = torch.empty(B, C, H, 6 * W).cuda()
        ct_tensor_42[:, :, :, 0:W] = x1
        ct_tensor_42[:, :, :, W:2*W] = x2
        ct_tensor_42[:, :, :, 2*W:3*W] = x3
        ct_tensor_42[:, :, :, 3*W:4*W] = x4
        ct_tensor_42[:, :, :, 4*W:5*W] = x5
        ct_tensor_42[:, :, :, 5*W:6*W] = x6
        # ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        x1_sm = f2[:, :, :, 0:W]
        x2_sm = f2[:, :, :, W:2*W]
        x3_sm = f2[:, :, :, 2*W:3*W]
        x4_sm = f2[:, :, :, 3*W:4*W]
        x5_sm = f2[:, :, :, 4*W:5*W]
        x6_sm = f2[:, :, :, 5*W:6*W]
        # xf_sm = self.smooth_layer_x(xf_sm)
        # yf_sm = self.smooth_layer_x(yf_sm)

        return x1_sm, x2_sm, x3_sm, x4_sm, x5_sm, x6_sm

class STM_GrootV3D_V2(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV3D_V2, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        # self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = GrootV3DLayer(channels=640)
        # self.smooth_layer_x = ResBlock1(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_y = ResBlock1(in_channels=128, out_channels=128, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()

        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_42[:, :, :, 0:W] = x
        ct_tensor_42[:, :, :, W:2*W] = y
        # ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = f2[:, :, :, 0:W]
        yf_sm = f2[:, :, :, W:2*W]

        # xf_sm = self.smooth_layer_x(xf_sm)
        # yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm

class STM_GrootV3D_V2_small(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV3D_V2_small, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        # self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = GrootV3DLayer(channels=768)
        # self.smooth_layer_x = ResBlock1(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_y = ResBlock1(in_channels=128, out_channels=128, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()

        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_42[:, :, :, 0:W] = x
        ct_tensor_42[:, :, :, W:2*W] = y
        # ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = f2[:, :, :, 0:W]
        yf_sm = f2[:, :, :, W:2*W]

        # xf_sm = self.smooth_layer_x(xf_sm)
        # yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm
class MTGrootV3D_SV3(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(MTGrootV3D_SV3, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder3 = Seg_Decoder_ResNet(self.channel_nums)
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

        self.MambaLayer = STM_MTGrootV3D_V2(640, False)
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
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV3D_V1-STM_BiGrootV3D_V2-0114/resnet34/epoch66_Score38.41_mIOU72.90_Sek23.63_Fscd64.16_OA87.67.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # 防止权重不匹配
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                if key == 'patch_embed.conv1.weight':
                    # 检查当前模型的 conv1.weight 形状
                    current_conv1_weight = self.backbone.state_dict()[key]
                    # 创建一个新的权重，形状与当前模型一致
                    new_conv1_weight = torch.zeros_like(current_conv1_weight)
                    # 将预训练权重的前3通道复制到新权重的前3通道
                    new_conv1_weight[:, :3, :, :] = value
                    # 将新权重添加到 updated_weights
                    updated_weights[key] = new_conv1_weight
                else:
                    if key in self.backbone.state_dict():
                        updated_weights[key] = value
            else:
                if key in self.backbone.state_dict():
                    updated_weights[key] = value

        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.Seg_Decoder3, self.CD_Decoder,
                           self.seg_conv, self.classifierCD)

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

    def bi_mamba_forward(self, x, y, z):
        xf_sm1, yf_sm1, zf_sm1 = self.MambaLayer(x, y, z)
        yf_sm2, xf_sm2, zf_sm2 = self.MambaLayer(y, x, z)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        z_f = zf_sm1 + zf_sm2
        return x_f, y_f, z_f


    def forward(self, x1, x2, x3):
        b, c, h, w = x1.shape
        xy_in = torch.empty(b, c, h, 3 * w).cuda()
        xy_in[:, :, :, 0:w] = x1
        xy_in[:, :, :, w:2*w] = x2
        xy_in[:, :, :, 2*w:3*w] = x3
        feature_xy = self.backbone.forward(xy_in)
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        feature3 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//3
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            T3_part = matrix[:, :, :, 2*Ws:3*Ws]
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)
            feature3.append(T3_part)
        feature1_4, feature2_4, feature3_4 = self.bi_mamba_forward(feature1[-1], feature2[-1], feature3[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        feature3[-1] = feature3_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        out_3 = self.Seg_Decoder3(feature3)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature3[i], feature2[i]))
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg3 = F.interpolate(self.seg_conv(out_3), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        seg3 = self.softmax(seg3)
        change = torch.sigmoid(change)

        return seg1, seg2, seg3, change.squeeze(1)

from models.CDMamba.CDMamba import CDMamba
class CDMamba_WUSU(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(CDMamba_WUSU, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = CDMamba(spatial_dims=2, in_channels=4, out_channels=13, init_filters=16, norm=("GROUP", {"num_groups": 8}),
                    mode="AGLGF", conv_mode='orignal', local_query_model="orignal_dinner",
                    stage=4, mamba_act="silu", up_mode="SRCM", up_conv_mode='deepwise', blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                    resdiual=False, diff_abs="later")

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [16, 32, 64, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_CDMamba(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_CDMamba(self.channel_nums)
            self.Seg_Decoder3 = Seg_Decoder_CDMamba(self.channel_nums)
            self.CD_Decoder = Seg_Decoder_CDMamba(self.channel_nums)

        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]

        self.MambaLayer = STM_MTGrootV3D_V2(128, False)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(64, self.nclass, 1, bias=True),
            # relu后面不用
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.Seg_Decoder3, self.CD_Decoder,
                           self.seg_conv, self.classifierCD)

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

    def bi_mamba_forward(self, x, y, z):
        xf_sm1, yf_sm1, zf_sm1 = self.MambaLayer(x, y, z)
        yf_sm2, xf_sm2, zf_sm2 = self.MambaLayer(y, x, z)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        z_f = zf_sm1 + zf_sm2
        return x_f, y_f, z_f


    def forward(self, x1, x2, x3):
        b, c, h, w = x1.shape
        # 初始化feature1和feature2列表
        feature1 = self.backbone(x1)
        feature2 = self.backbone(x2)
        feature3 = self.backbone(x3)
        # 遍历A中的每个矩阵

        feature1_4, feature2_4, feature3_4 = self.bi_mamba_forward(feature1[-1], feature2[-1], feature3[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        feature3[-1] = feature3_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        out_3 = self.Seg_Decoder3(feature3)
        # CDDecoder

        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))

        xc = self.CD_Decoder(feature_diff)
        change13 = self.classifierCD(xc)
        change13 = F.interpolate(change13, size=(h, w), mode='bilinear', align_corners=False)

        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg3 = F.interpolate(self.seg_conv(out_3), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        seg3 = self.softmax(seg3)
        change13 = torch.sigmoid(change13)

        return seg1, seg2, seg3, change13.squeeze(1)





# LXG:baseline
import random
# 最好的模型
class MTGrootV3D_SV3_DynamicEarth_random(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(MTGrootV3D_SV3_DynamicEarth_random, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder3 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder4 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder5 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder6 = Seg_Decoder_ResNet(self.channel_nums)
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

        self.MambaLayer = STM_MTGrootV3D_V2_DynamicEarth(640, False)
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
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV3D_V1-STM_BiGrootV3D_V2-0114/resnet34/epoch66_Score38.41_mIOU72.90_Sek23.63_Fscd64.16_OA87.67.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # 防止权重不匹配
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                if key == 'patch_embed.conv1.weight':
                    # 检查当前模型的 conv1.weight 形状
                    current_conv1_weight = self.backbone.state_dict()[key]
                    # 创建一个新的权重，形状与当前模型一致
                    new_conv1_weight = torch.zeros_like(current_conv1_weight)
                    # 将预训练权重的前3通道复制到新权重的前3通道
                    new_conv1_weight[:, :3, :, :] = value
                    # 将新权重添加到 updated_weights
                    updated_weights[key] = new_conv1_weight
                else:
                    if key in self.backbone.state_dict():
                        updated_weights[key] = value
            else:
                if key in self.backbone.state_dict():
                    updated_weights[key] = value

        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2,self.Seg_Decoder3,self.Seg_Decoder4,self.Seg_Decoder5,self.Seg_Decoder6, self.CD_Decoder,
                           self.seg_conv, self.classifierCD)

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

    def bi_mamba_forward(self, x, y, z, d, s, b):
        xf_sm1, yf_sm1, zf_sm1, df_sm1, sf_sm1, bf_sm1 = self.MambaLayer(x, y, z, d, s, b)
        bf_sm2, sf_sm2, df_sm2, zf_sm2, yf_sm2, xf_sm2 = self.MambaLayer(b, s, d, z, y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        z_f = zf_sm1 + zf_sm2
        d_f = df_sm1 + df_sm2
        s_f = sf_sm1 + sf_sm2
        b_f = bf_sm1 + bf_sm2
        return x_f, y_f, z_f, d_f, s_f, b_f


    def forward(self, x1, x2, x3, x4, x5, x6):
        b, c, h, w = x1.shape
        xy_in = torch.empty(b, c, h, 6 * w).cuda()
        xy_in[:, :, :, 0:w] = x1
        xy_in[:, :, :, w:2*w] = x2
        xy_in[:, :, :, 2*w:3*w] = x3
        xy_in[:, :, :, 3*w:4*w] = x4
        xy_in[:, :, :, 4*w:5*w] = x5
        xy_in[:, :, :, 5*w:6*w] = x6
        feature_xy = self.backbone.forward(xy_in)
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        feature3 = []
        feature4 = []
        feature5 = []
        feature6 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//6
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            T3_part = matrix[:, :, :, 2*Ws:3*Ws]
            T4_part = matrix[:, :, :, 3*Ws:4*Ws]  # 左半部分
            T5_part = matrix[:, :, :, 4*Ws:5*Ws]  # 右半部分
            T6_part = matrix[:, :, :, 5*Ws:6*Ws]
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)
            feature3.append(T3_part)
            feature4.append(T4_part)
            feature5.append(T5_part)
            feature6.append(T6_part)
        feature1_4, feature2_4, feature3_4, feature4_4, feature5_4, feature6_4 = self.bi_mamba_forward(feature1[-1], feature2[-1], feature3[-1],feature4[-1], feature5[-1], feature6[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        feature3[-1] = feature3_4
        feature4[-1] = feature4_4
        feature5[-1] = feature5_4
        feature6[-1] = feature6_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        out_3 = self.Seg_Decoder3(feature3)
        out_4 = self.Seg_Decoder4(feature4)
        out_5 = self.Seg_Decoder5(feature5)
        out_6 = self.Seg_Decoder6(feature6)
        # CDDecoder
        # LXG:添加随机数
        feature = [feature1, feature2, feature3, feature4, feature5, feature6]
        pair = sorted(random.sample(range(6), 2))
        # pair = [1,2]
        print(pair)  # 输出可能是 [0, 1], [0, 2] 或 [1, 2]

        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature[pair[0]][i], feature[pair[-1]][i]))
        xc = self.CD_Decoder(feature_diff)

        # feature_diff = []
        # for i in range(len(feature1)):
        #     feature_diff.append(self.CFEM[i](feature1[i], feature6[i]))
        # xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg3 = F.interpolate(self.seg_conv(out_3), size=(h, w), mode='bilinear', align_corners=False)
        seg4 = F.interpolate(self.seg_conv(out_4), size=(h, w), mode='bilinear', align_corners=False)
        seg5 = F.interpolate(self.seg_conv(out_5), size=(h, w), mode='bilinear', align_corners=False)
        seg6 = F.interpolate(self.seg_conv(out_6), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        seg3 = self.softmax(seg3)
        seg4 = self.softmax(seg4)
        seg5 = self.softmax(seg5)
        seg6 = self.softmax(seg6)
        change = torch.sigmoid(change)

        return [seg1, seg2, seg3, seg4, seg5, seg6], change.squeeze(1), pair



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MTGrootV3D_SV3(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # model = ST_VSSM_Siam().to(device)
    model = CDMamba_WUSU(backbone='GrootV', pretrained=False, nclass=13, lightweight=True, M=6, Lambda=0.00005).to(device)
    print(model)
    image1 = torch.randn(1, 4, 512, 512).to(device)
    image2 = torch.randn(1, 4, 512, 512).to(device)
    image3 = torch.randn(1, 4, 512, 512).to(device)
    # seg1, seg2, seg3, change = model(image1, image2, image3)
    fs = model(image1, image2, image3)
    # print(seg1)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2, image3))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

