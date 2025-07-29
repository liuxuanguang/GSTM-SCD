# import nnunetv2.nets.SwinUMamba
from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, Seg_Decoder_changeformer
# from models.Decoders.Decoder_base import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIGM import CIGM
from models.Modules.CIEM import CIEM
import torch
from torch import nn
import torch.nn.functional as F
from vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from utils.misc import initialize_weights
from GrootV.classification.models.grootv import GrootV, GrootV_3D
from models.testing_reorder import feature_resortV2, feature_resumption
from GrootV.classification.models.grootv import GrootVLayer, GrootV3DLayer
from models.Changeformer.ChangeFormer import EncoderTransformer, EncoderTransformer_v3, ChangeFormerV6_encoder
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

class STM_GrootV3D_V2(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV3D_V2, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=512, out_channels=128)
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
class Bichangeformer_SV3(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(Bichangeformer_SV3, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = ChangeFormerV6_encoder()

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [64, 128, 320, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_changeformer(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_changeformer(self.channel_nums)
            self.CD_Decoder = Seg_Decoder_changeformer(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
        self.MambaLayer = STM_GrootV3D_V2(512, False)
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
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt')
        new_dict = pretrained_weights['model_G_state_dict']
        print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('Tenc_x2.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        self.backbone.load_state_dict(new_dict, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
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


class Bichangeformer_SV3_Siam(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(Bichangeformer_SV3_Siam, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = ChangeFormerV6_encoder()

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [64, 128, 320, 128]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.Seg_Decoder1 = Seg_Decoder_changeformer(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder_changeformer(self.channel_nums)
            self.CD_Decoder = Seg_Decoder_changeformer(self.channel_nums)
        else:
            self.Seg_Decoder1 = Seg_Decoder(self.channel_nums)
            self.Seg_Decoder2 = Seg_Decoder(self.channel_nums)
            self.CD_Decoder = CD_Decoder(self.channel_nums)
        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
        self.MambaLayer = STM_GrootV3D_V2(512, False)
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
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt')
        new_dict = pretrained_weights['model_G_state_dict']
        print(pretrained_weights)
        self.backbone.load_state_dict(new_dict, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded pre-training weights!')
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.Seg_Decoder1, self.Seg_Decoder2, self.CD_Decoder,
                           self.seg_conv, self.classifierCD, self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3)
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
        feature1 = self.backbone(x1)
        feature2 = self.backbone(x2)
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
    model = Bichangeformer_SV3_Siam(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # model = BiGrootV_base(backbone='GrootV', pretrained=False, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)
    seg1, seg2, change = model(image1, image2)
    print(seg1)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))