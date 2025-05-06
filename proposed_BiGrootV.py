# import nnunetv2.nets.SwinUMamba
from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
# from models.Decoders.Decoder_base import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIGM import CIGM
from models.Modules.CIEM import CIEM
import torch
from torch import nn
import torch.nn.functional as F
from vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from utils.misc import initialize_weights
from models.testing_reorder import feature_resortV2, feature_resumption
from GrootV.classification.models.grootv import GrootVLayer, GrootV3DLayer
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


#-----------------------------------------------------------------------------------------------

# 对应22.86值的时空建模
class STM_GrootV(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        self.conv1 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=320)
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=320)
        self.GrootV_S1 = GrootVLayer(channels=320)
        self.smooth_layer_x = ResBlock1(in_channels=640, out_channels=640, stride=1)
        self.smooth_layer_y = ResBlock1(in_channels=640, out_channels=640, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()
        ct_tensor_41 = torch.empty(B, C, 2 * H, W).cuda()
        ct_tensor_41[:, :, ::2, :] = x
        ct_tensor_41[:, :, 1::2, :] = y
        ct_tensor_41 = self.conv1(ct_tensor_41)
        if not self.channel_first:
            ct_tensor_41 = ct_tensor_41.permute(0, 2, 3, 1)
        f1 = self.GrootV_S1(ct_tensor_41)
        f1 = f1.permute(0, 3, 1, 2)

        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_42[:, :, :, ::2] = x
        ct_tensor_42[:, :, :, 1::2] = y
        ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = torch.concat([f1[:, :, ::2, :], f2[:, :, :, ::2]], dim=1)
        yf_sm = torch.concat([f1[:, :, 1::2, :], f2[:, :, :, 1::2]], dim=1)

        xf_sm = self.smooth_layer_x(xf_sm)
        yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm

class STM_GrootV2(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV2, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        self.conv1 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=320)
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=320)
        self.GrootV_S1 = GrootVLayer(channels=320)

        # self.fuse_layer_x = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=640*2, out_channels=640),
        #                                   nn.BatchNorm2d(640), nn.ReLU())
        # self.fuse_layer_y = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=640*2, out_channels=640),
        #                                   nn.BatchNorm2d(640), nn.ReLU())
        # 权重保存在New_SECOND_dataset-BiGrootV_V5-STM_GrootV2.1-softmax-1204是去除该操作
        self.smooth_layer_x = ResBlock1(in_channels=640, out_channels=640, stride=1)
        self.smooth_layer_y = ResBlock1(in_channels=640, out_channels=640, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()
        ct_tensor_41 = torch.empty(B, C, 2 * H, W).cuda()
        # ct_tensor_41[:, :, ::2, :] = x
        # ct_tensor_41[:, :, 1::2, :] = y
        ct_tensor_41[:, :, 0: H, :] = x
        ct_tensor_41[:, :, H: 2*H, :] = y

        ct_tensor_41 = self.conv1(ct_tensor_41)
        if not self.channel_first:
            ct_tensor_41 = ct_tensor_41.permute(0, 2, 3, 1)
        f1 = self.GrootV_S1(ct_tensor_41)
        f1 = f1.permute(0, 3, 1, 2)

        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        # ct_tensor_42[:, :, :, ::2] = x
        # ct_tensor_42[:, :, :, 1::2] = y
        ct_tensor_42[:, :, :, 0: W] = x
        ct_tensor_42[:, :, :, W:2*W] = y
        ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)


        xf_sm = torch.concat([f1[:, :, 0: H, :], f2[:, :, :, 0: W]], dim=1)
        yf_sm = torch.concat([f1[:, :, H: 2*H, :], f2[:, :, :, W: 2*W]], dim=1)

        xf_sm = self.smooth_layer_x(xf_sm)
        yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm

class STM_GrootV23(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV23, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        self.conv1 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=320)
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=320)
        self.conv3 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=320)
        self.GrootV_S1 = GrootVLayer(channels=320)
        self.fuse_layer_x = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=320*3, out_channels=640),
                                          nn.BatchNorm2d(640), nn.ReLU())
        self.fuse_layer_y = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=320*3, out_channels=640),
                                          nn.BatchNorm2d(640), nn.ReLU())
        # 权重保存在New_SECOND_dataset-BiGrootV_V5-STM_GrootV2.1-softmax-1204是去除该操作
        self.smooth_layer_x = ResBlock1(in_channels=640, out_channels=640, stride=1)
        self.smooth_layer_y = ResBlock1(in_channels=640, out_channels=640, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()
        ct_tensor_41 = torch.empty(B, C, 2 * H, W).cuda()
        ct_tensor_41[:, :, 0: H, :] = x
        ct_tensor_41[:, :, H: 2*H, :] = y
        ct_tensor_41 = self.conv1(ct_tensor_41)
        if not self.channel_first:
            ct_tensor_41 = ct_tensor_41.permute(0, 2, 3, 1)
        f1 = self.GrootV_S1(ct_tensor_41)
        f1 = f1.permute(0, 3, 1, 2)

        ct_tensor_42 = torch.empty(B, C, 2 * H, W).cuda()
        ct_tensor_42[:, :, ::2, :] = x
        ct_tensor_42[:, :, 1::2, :] = y
        ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        ct_tensor_43 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_43[:, :, :, ::2] = x
        ct_tensor_43[:, :, :, 1::2] = y
        ct_tensor_43 = self.conv3(ct_tensor_43)
        if not self.channel_first:
            ct_tensor_43 = ct_tensor_43.permute(0, 2, 3, 1)
        f3 = self.GrootV_S1(ct_tensor_43)
        f3 = f3.permute(0, 3, 1, 2)

        xf_sm = torch.concat([f1[:, :, 0: H, :], f2[:, :, ::2, :], f3[:, :, :, ::2]], dim=1)
        yf_sm = torch.concat([f1[:, :, H: 2*H, :], f2[:, :, 1::2, :], f3[:, :, :, 1::2]], dim=1)
        xf_sm = self.fuse_layer_x(xf_sm)
        yf_sm = self.fuse_layer_y(yf_sm)
        xf_sm = self.smooth_layer_x(xf_sm)
        yf_sm = self.smooth_layer_y(yf_sm)
        return xf_sm, yf_sm

# 改变扫描顺序
class STM(nn.Module):
    def __init__(self,encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer):
        super(STM, self).__init__()
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=640),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=640, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=1, ssm_ratio=2.0,
                     ssm_dt_rank="auto", ssm_act_layer=ssm_act_layer,
                     ssm_conv=3, ssm_conv_bias='false',
                     ssm_drop_rate=0, ssm_init="v0",
                     forward_type="v2", mlp_ratio=4.0,
                     mlp_act_layer=mlp_act_layer, mlp_drop_rate=0.0,
                     gmlp=False, use_checkpoint=False),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=640),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=640, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=1, ssm_ratio=2.0,
                     ssm_dt_rank="auto", ssm_act_layer=ssm_act_layer,
                     ssm_conv=3, ssm_conv_bias='false',
                     ssm_drop_rate=0, ssm_init="v0",
                     forward_type="v2", mlp_ratio=4.0,
                     mlp_act_layer=mlp_act_layer, mlp_drop_rate=0.0,
                     gmlp=False, use_checkpoint=False),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=640),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=640, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=1, ssm_ratio=2.0,
                     ssm_dt_rank="auto", ssm_act_layer=ssm_act_layer,
                     ssm_conv=3, ssm_conv_bias='false',
                     ssm_drop_rate=0, ssm_init="v0",
                     forward_type="v2", mlp_ratio=4.0,
                     mlp_act_layer=mlp_act_layer, mlp_drop_rate=0.0,
                     gmlp=False, use_checkpoint=False),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),)


        self.fuse_layer_x = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=640*2, out_channels=640),
                                          nn.BatchNorm2d(640), nn.ReLU())
        self.fuse_layer_y = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=640*2, out_channels=640),
                                          nn.BatchNorm2d(640), nn.ReLU())
        # self.fuse_layer_cf = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
        #                                   nn.BatchNorm2d(128), nn.ReLU())

        # Smooth layer
        self.smooth_layer_x = ResBlock1(in_channels=640, out_channels=640, stride=1)
        self.smooth_layer_y = ResBlock1(in_channels=640, out_channels=640, stride=1)
        # self.smooth_layer_cf = ResBlock1(in_channels=128, out_channels=128, stride=1)




    def forward(self, feat1_4, feat2_4):
        B, C, H, W = feat1_4.size()

        # Create an empty tensor of the correct shape (B, C, 2 * H, W)
        ct_tensor_41 = torch.empty(B, C, 2 * H, W).cuda()
        ct_tensor_41[:, :, 0:H, :] = feat1_4
        ct_tensor_41[:, :, H:2*H, :] = feat2_4
        p41 = self.st_block_42(ct_tensor_41)

        # Create an empty tensor of the correct shape (B, C, H, 2 * W)
        # ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        # # Fill in odd columns with A and even columns with B
        # ct_tensor_42[:, :, :, ::2] = feat1_4  # Odd columns
        # ct_tensor_42[:, :, :, 1::2] = feat2_4  # Even columns
        # p42 = self.st_block_42(ct_tensor_42)
        ct_tensor_43 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_43[:, :, :, 0:W] = feat1_4
        ct_tensor_43[:, :, :, W:2*W] = feat2_4
        p43 = self.st_block_43(ct_tensor_43)
        # xf_sm = torch.concat([p41[:, :, 0:H, :], p42[:, :, :, ::2], p43[:, :, :, 0:W]], dim=1)
        # yf_sm = torch.concat([p41[:, :, H:2*H, :], p42[:, :, :, 1::2], p43[:, :, :, W:2*W]], dim=1)
        xf_sm = torch.concat([p41[:, :, 0:H, :], p43[:, :, :, 0:W]], dim=1)
        yf_sm = torch.concat([p41[:, :, H:2*H, :], p43[:, :, :, W:2*W]], dim=1)
        xf_sm = self.fuse_layer_x(xf_sm.to(device='cuda'))
        xf_sm = self.smooth_layer_x(xf_sm)
        yf_sm = self.fuse_layer_y(yf_sm.to(device='cuda'))
        yf_sm = self.smooth_layer_x(yf_sm)


        return xf_sm, yf_sm



from GrootV.classification.models.grootv import GrootV, GrootV_3D


# Baseline
class BiGrootV_base(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV_base, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV()

        if backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet18" or backbone == "resnet34":
            self.channel_nums = [64, 128, 256, 512]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
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


        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(0.5, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu后面不用
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Dropout(0.5, False),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/GrootV-tiny-1111/resnet34/epoch10_Score35.17_mIOU71.18_Sek19.74_Fscd59.24_OA86.35.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)

        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder, self.seg_conv, self.classifierCD)

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

    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
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


# 精度为22.82，对应权重文件夹：GrootV-STM-tiny-Softmax-1113
class BiGrootV(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV, self).__init__()
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
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.MambaLayer = STM_GrootV(640, False)
        # self.DecCD = _DecoderBlock(128, 256, 128, scale_ratio=2)

        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu后面不用
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/GrootV-STM-tiny-Softmax-1113/resnet34/epoch59_Score37.78_mIOU72.60_Sek22.86_Fscd62.56_OA87.11.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder, self.MambaLayer,
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

    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        return x_f, y_f

    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(abs(feature1[i]-feature2[i]))

        xc = self.CD_Decoder(feature_diff)

        change = self.classifierCD(xc)
        # out_1, out_2, out_change = self.DIGM(out_1, out_2, out_change)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

class STM_GrootV_V2(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV_V2, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        self.conv1 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = GrootVLayer(channels=128)
        self.smooth_layer_x = ResBlock1(in_channels=256, out_channels=256, stride=1)
        self.smooth_layer_y = ResBlock1(in_channels=256, out_channels=256, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()
        ct_tensor_41 = torch.empty(B, C, 2 * H, W).cuda()
        ct_tensor_41[:, :, ::2, :] = x
        ct_tensor_41[:, :, 1::2, :] = y
        ct_tensor_41 = self.conv1(ct_tensor_41)
        if not self.channel_first:
            ct_tensor_41 = ct_tensor_41.permute(0, 2, 3, 1)
        f1 = self.GrootV_S1(ct_tensor_41)
        f1 = f1.permute(0, 3, 1, 2)

        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_42[:, :, :, ::2] = x
        ct_tensor_42[:, :, :, 1::2] = y
        ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = torch.concat([f1[:, :, 0:H, :], f2[:, :, :, 0:W]], dim=1)
        yf_sm = torch.concat([f1[:, :, H:2*H, :], f2[:, :, :, W:2*W]], dim=1)

        xf_sm = self.smooth_layer_x(xf_sm)
        yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm


#基于最好的改进，这里将时空建模模块的错误已经进行改正
class BiGrootV_V2(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV_V2, self).__init__()
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
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.MambaLayer = STM_GrootV(640, False)
        # self.DecCD = _DecoderBlock(128, 256, 128, scale_ratio=2)
        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]
        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu后面不用
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2, False),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV_V4-softmax-1202/resnet34/epoch80_Score38.11_mIOU73.08_Sek23.12_Fscd63.32_OA87.80.pth')
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
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = False
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder, self.MambaLayer,
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

    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))

        xc = self.CD_Decoder(feature_diff)

        change = self.classifierCD(xc)
        # out_1, out_2, out_change = self.DIGM(out_1, out_2, out_change)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

# V3去除了EGMSNet中变化特征增强模块，只使用简单的做差方法
class BiGrootV_V3(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV_V3, self).__init__()
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
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.MambaLayer = STM_GrootV(640, False)
        # self.DecCD = _DecoderBlock(128, 256, 128, scale_ratio=2)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu后面不用
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2, False),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV_V4-softmax-1202/resnet34/epoch80_Score38.11_mIOU73.08_Sek23.12_Fscd63.32_OA87.80.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中

                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = False
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder, self.MambaLayer,
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

    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        return x_f, y_f

    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(feature1[i]-feature2[i])

        xc = self.CD_Decoder(feature_diff)

        change = self.classifierCD(xc)
        # out_1, out_2, out_change = self.DIGM(out_1, out_2, out_change)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)


#使用串联方式获取变化特征
class BiGrootV_V5(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV_V5, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
            self.cd_channel_nums = [80*2, 160*2, 320*2, 640*2]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = CD_Decoder_ResNet(self.cd_channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        #对应New_SECOND_dataset-BiGrootV_V5-Retrain-1203的STM方法
        # self.MambaLayer = STM_GrootV(640, False)
        # 改为最初的STM方法，保存在New_SECOND_dataset-BiGrootV_V5-STM-Retrain-1203，最佳精度epoch83_Score38.45_mIOU73.17_Sek23.57_Fscd63.91_OA87.95
        # self.MambaLayer = STM_GrootV(640, False)
        # 基于STM_GrootV，使用最初STM的两种扫描方式
        self.MambaLayer = STM_GrootV23(640, False)
        # self.DecCD = _DecoderBlock(128, 256, 128, scale_ratio=2)

        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu后面不用
        )
        # self.softmax = nn.Softmax(dim=1)
        # self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2, False),
        #                                   nn.Conv2d(64, 1, kernel_size=1))
        # landsat不用Softmax和classifierCD的dropout
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/Landsat_SCD/BiGrootV_V5-nosoftmax_GrootV2.3-1209/resnet34/epoch97_Score69.89_mIOU89.30_Sek61.57_Fscd89.65_OA96.37.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        # after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = False
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder, self.MambaLayer,
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

    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        return x_f, y_f

    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(torch.cat([feature1[i], feature2[i]],dim=1))

        xc = self.CD_Decoder(feature_diff)

        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        # 训练landsat时不用
        # seg1 = self.softmax(seg1)
        # seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

class BiGrootV_V5Base(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV_V5Base, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
            self.cd_channel_nums = [80*2, 160*2, 320*2, 640*2]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = CD_Decoder_ResNet(self.cd_channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True)
        )
        #landsat不用，second使用softmax
        # self.softmax = nn.Softmax(dim=1)
        # 12月9号之前的landsat用dropout
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2, False),
                                          nn.Conv2d(64, 1, kernel_size=1))
        self.softmax = nn.Softmax(dim=1)
        # self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
        #                                   nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV_V4-softmax-1202/resnet34/epoch80_Score38.11_mIOU73.08_Sek23.12_Fscd63.32_OA87.80.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/课题研究/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
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
    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(torch.cat([feature1[i], feature2[i]],dim=1))
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        # out_1, out_2, out_change = self.DIGM(out_1, out_2, out_change)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)
# 使用新数据集训练，去除时空建模，消融实验2
class BiGrootV_V4(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV_V4, self).__init__()
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
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        # self.DecCD = _DecoderBlock(128, 256, 128, scale_ratio=2)
        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True),
            # relu后面不用
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.2, False),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/GrootV-STM-tiny-Softmax-1113/resnet34/epoch59_Score37.78_mIOU72.60_Sek22.86_Fscd62.56_OA87.11.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/文档/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
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
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder, self.seg_conv, self.classifierCD)

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

    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(feature1[i]-feature2[i])
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)


#使用3D树状扫描
class STM_GrootV3D_V2(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV3D_V2, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=768, out_channels=128)
        self.GrootV_S1 = GrootV3DLayer(channels=128)
        self.smooth_layer_x = ResBlock1(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_y = ResBlock1(in_channels=128, out_channels=128, stride=1)
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
class BiGrootV3D_V1(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV3D_V1, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.MambaLayer = STM_GrootV3D_V2(640, False)
        # self.DecCD = _DecoderBlock(128, 256, 128, scale_ratio=2)
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
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV_V4-softmax-1202/resnet34/epoch80_Score38.11_mIOU73.08_Sek23.12_Fscd63.32_OA87.80.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = False
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder, self.MambaLayer,
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

    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        # yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        # x_f = xf_sm1 + xf_sm2
        # y_f = yf_sm1 + yf_sm2
        return xf_sm1, yf_sm1

    def CD_forward(self, xy):
        b, c, h, w = xy.size()
        xc = self.resCD(xy)
        return xc

    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(feature1[i]-feature2[i])
        xc = self.CD_Decoder(feature_diff)

        xc = self.CD_Decoder(feature_diff)

        change = self.classifierCD(xc)
        # out_1, out_2, out_change = self.DIGM(out_1, out_2, out_change)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

# 直接在宽度串联，每一层级都进行时空关系建模,sek为23.63
class BiGrootV3D_V2(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV3D_V2, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        # self.backbone = GrootV(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)


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
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV_V4-softmax-1202/resnet34/epoch80_Score38.11_mIOU73.08_Sek23.12_Fscd63.32_OA87.80.pth')
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


    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = x1
        xy_in[:, :, :, w:2*w] = x2
        feature_xy = self.backbone.forward(xy_in)
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
            feature1.append(left_part)
            feature2.append(right_part)
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(feature1[i]-feature2[i])
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

# 仅增加最后一层特征的时空建模，Sek为23.76
class BiGrootV3D_SV2(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV3D_SV2, self).__init__()
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
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.MambaLayer = STM_GrootV3D_V2(640, False)
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
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV3D_V2-0116/resnet34/epoch63_Score38.21_mIOU72.76_Sek23.40_Fscd63.85_OA87.41.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = False
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
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
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
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
            feature_diff.append(feature1[i]-feature2[i])
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=1)
    return weight
# 增加最后一层特征的时空建模和变化解码强化模块，Sek为23.92
class BiGrootV3D_SV3(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV3D_SV3, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
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
        self.MambaLayer = STM_GrootV3D_V2(640, False)
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
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/Landsat_SCD/New_SECOND_dataset-BiGrootV3D_SV3-0216/resnet34/epoch72_Score67.01_mIOU88.14_Sek57.96_Fscd88.18_OA95.90.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
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
        # initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
        #                    self.seg_conv, self.classifierCD)
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
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
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
        # feature_diff = []
        # for i in range(len(feature1)):
        #     feature_diff.append(feature1[i]-feature2[i])
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)


class BiGrootV3D_SV3_small(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV3D_SV3_small, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 13, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [96, 192, 384, 768, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
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
        self.MambaLayer = STM_GrootV3D_V2(768, False)
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
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/Landsat_SCD/New_SECOND_dataset-BiGrootV3D_SV3-0216/resnet34/epoch72_Score67.01_mIOU88.14_Sek57.96_Fscd88.18_OA95.90.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
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
        # initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
        #                    self.seg_conv, self.classifierCD)
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
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
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
        # feature_diff = []
        # for i in range(len(feature1)):
        #     feature_diff.append(feature1[i]-feature2[i])
        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = self.softmax(seg1)
        seg2 = self.softmax(seg2)
        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)
# class BiGrootV3D_SV3(nn.Module):
#     def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
#         super(BiGrootV3D_SV3, self).__init__()
#         self.backbone_name = backbone
#         self.nclass = nclass
#         self.lightweight = lightweight
#         self.M = M
#         self.Lambda = Lambda
#         self.backbone = GrootV(depths=[2, 2, 9, 2])
#
#         if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
#             self.channel_nums = [80, 160, 320, 640, 640]
#         elif backbone == "resnet50":
#             self.channel_nums = [256, 512, 1024, 2048]
#
#         if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
#             self.Seg_Decoder1 = Seg_Decoder_ResNet(self.channel_nums)
#             self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
#             self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
#         else:
#             self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
#             self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
#             self.CD_Decoder = CD_Decoder(self.channel_nums)
#         self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
#         self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
#         self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
#         self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
#         self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
#         self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]
#         self.MambaLayer = STM_GrootV3D_V2(640, False)
#         self.seg_conv = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.Dropout(0.2, False),
#             nn.Conv2d(128, self.nclass, 1, bias=True),
#             # relu后面不用
#         )
#         self.softmax = nn.Softmax(dim=1)
#         self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
#                                           nn.Conv2d(64, 1, kernel_size=1))
#         updated_weights = {}
#         pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/Landsat_SCD/New_SECOND_dataset-BiGrootV3D_SV3-0216/resnet34/epoch72_Score67.01_mIOU88.14_Sek57.96_Fscd88.18_OA95.90.pth')
#         # 1. 过滤出预训练权重中在模型字典中也存在的键
#         for key, value in pretrained_weights.items():
#             if key.startswith('backbone.'):
#                 new_key = key.replace('backbone.', '')
#                 # 检查新的键是否存在于模型的 state_dict 中
#                 if new_key in self.backbone.state_dict():
#                     updated_weights[new_key] = value
#         # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
#         # new_dict = pretrained_weights['model']
#         # print(pretrained_weights)
#         # for key, value in new_dict.items():
#         #     if key.startswith(('patch_embed.', 'levels.')):
#         #         new_key = key
#         #         if new_key in self.backbone.state_dict():
#         #             updated_weights[new_key] = value
#         self.backbone.load_state_dict(updated_weights, strict=True)
#         after_weight = self.backbone.state_dict()
#         print('Successfully loaded pre-training weights!')
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#         initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
#                            self.seg_conv, self.classifierCD, self.MambaLayer)
#
#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 conv1x1(inplanes, planes, stride),
#                 nn.BatchNorm2d(planes))
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def bi_mamba_forward(self, x, y):
#         xf_sm1, yf_sm1 = self.MambaLayer(x, y)
#         yf_sm2, xf_sm2 = self.MambaLayer(y, x)
#         x_f = xf_sm1 + xf_sm2
#         y_f = yf_sm1 + yf_sm2
#         return x_f, y_f
#
#
#     def forward(self, x1, x2):
#         b, c, h, w = x1.shape
#         xy_in = torch.empty(b, c, h, 2 * w).cuda()
#         xy_in[:, :, :, 0:w] = x1
#         xy_in[:, :, :, w:2*w] = x2
#         feature_xy = self.backbone.forward(xy_in)
#         # 初始化feature1和feature2列表
#         feature1 = []
#         feature2 = []
#         # 遍历A中的每个矩阵
#         for matrix in feature_xy:
#             # 获取矩阵的W维度大小
#             W = matrix.size(3)
#             # 在W维度上左右分为两部分
#             left_part = matrix[:, :, :, :W // 2]  # 左半部分
#             right_part = matrix[:, :, :, W // 2:]  # 右半部分
#             # 将左右两部分分别存储到feature1和feature2列表中
#             feature1.append(left_part)
#             feature2.append(right_part)
#         feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
#         feature1[-1] = feature1_4
#         feature2[-1] = feature2_4
#         # SegDecoder
#         out_1 = self.Seg_Decoder1(feature1)
#         out_2 = self.Seg_Decoder2(feature2)
#         # CDDecoder
#         feature_diff = []
#         for i in range(len(feature1)):
#             feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))
#         xc = self.CD_Decoder(feature_diff)
#         change = self.classifierCD(xc)
#         change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)
#         seg1 = F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
#         seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)
#         seg1 = self.softmax(seg1)
#         seg2 = self.softmax(seg2)
#         change = torch.sigmoid(change)
#
#         return seg1, seg2, change.squeeze(1)


# LXG:精度F1最高为87.59
class BiGrootV3D_SV3_BCD(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV3D_SV3_BCD, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]
        self.MambaLayer = STM_GrootV3D_V2(640, False)
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
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/LEVIR-CD+/LEVIR-CD+_BiGrootV3D_SV3_BCD-0304/resnet34/epoch63_Recall89.54_Precision85.14_OA98.31_F187.29_IoU77.44._KC86.38.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = False
        initialize_weights(self.CD_Decoder, self.classifierCD, self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4)
        # initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
        #                    self.seg_conv, self.classifierCD)
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
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
            feature1.append(left_part)
            feature2.append(right_part)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4

        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))

        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)

        change = torch.sigmoid(change)

        return change.squeeze(1)


class BiGrootV_SV3_BCD(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(BiGrootV_SV3_BCD, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]
        self.MambaLayer = STM_GrootV3D_V2(640, False)
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
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/LEVIR-CD+/LEVIR-CD+_BiGrootV3D_SV3_BCD-0304/resnet34/epoch63_Recall89.54_Precision85.14_OA98.31_F187.29_IoU77.44._KC86.38.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD, self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4)
        # initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
        #                    self.seg_conv, self.classifierCD)
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
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
            feature1.append(left_part)
            feature2.append(right_part)
        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4

        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))

        xc = self.CD_Decoder(feature_diff)
        change = self.classifierCD(xc)
        change = F.interpolate(change, size=(h, w), mode='bilinear', align_corners=False)

        change = torch.sigmoid(change)

        return change.squeeze(1)


from models.Mamba_backbone import Mamba_encoder
class SB_Mmaba_SCD(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(SB_Mmaba_SCD, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = Mamba_encoder()

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 80, 160, 320, 640, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
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
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]

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
        # updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # self.backbone.load_state_dict(updated_weights, strict=True)
        # after_weight = self.backbone.state_dict()
        # print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
                           self.seg_conv, self.classifierCD, self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3)
        # initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
        #                    self.seg_conv, self.classifierCD)
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


    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = x1
        xy_in[:, :, :, w:2*w] = x2
        feature_xy = self.backbone.forward(xy_in)
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
            feature1.append(left_part)
            feature2.append(right_part)
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

class SSM_SCD_Tiny(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(SSM_SCD_Tiny, self).__init__()
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
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)

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

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = x1
        xy_in[:, :, :, w:2*w] = x2
        feature_xy = self.backbone.forward(xy_in)
        # 初始化feature1和feature2列表
        feature1 = []
        feature2 = []
        # 遍历A中的每个矩阵
        for matrix in feature_xy:
            # 获取矩阵的W维度大小
            W = matrix.size(3)
            # 在W维度上左右分为两部分
            left_part = matrix[:, :, :, :W // 2]  # 左半部分
            right_part = matrix[:, :, :, W // 2:]  # 右半部分
            # 将左右两部分分别存储到feature1和feature2列表中
            feature1.append(left_part)
            feature2.append(right_part)
        # SegDecoder
        out_1 = self.Seg_Decoder1(feature1)
        out_2 = self.Seg_Decoder2(feature2)
        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(feature1[i]-feature2[i])
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
    model = BiGrootV3D_SV3_small(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # model = BiGrootV_base(backbone='GrootV', pretrained=False, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)
    seg1, seg2, change = model(image1, image2)
    # print(seg1)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

