import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet18stem': 'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    'resnet50stem': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101stem': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}


def conv3x3(in_planes, outplanes, stride=1):
    # 带padding的3*3卷积
    return nn.Conv2d(in_planes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, layers, out_stride=8, use_stem=False, stem_channels=64, in_channels=3):
        self.inplanes = 64
        super(Resnet, self).__init__()
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        stride_list, dilation_list = outstride_to_strides_and_dilations[out_stride]

        self.use_stem = use_stem
        if use_stem:
            self.stem = nn.Sequential(
                conv3x3(in_channels, stem_channels // 2, stride=2),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=False),

                conv3x3(stem_channels // 2, stem_channels // 2),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=False),

                conv3x3(stem_channels // 2, stem_channels),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=False)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks=layers[0], stride=stride_list[0], dilation=dilation_list[0])
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=stride_list[1], dilation=dilation_list[1])
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=stride_list[2], dilation=dilation_list[2])
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=stride_list[3], dilation=dilation_list[3])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, contract_dilation=True):
        downsample = None
        dilations = [dilation] * blocks

        if contract_dilation and dilation > 1: dilations[0] = dilation // 2
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilations[0], downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilations[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_stem:
            x = self.stem(x)
        else:
            x = self.relu(self.bn1(self.conv1(x)))

        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        outs = [x, x1, x2, x3, x4]

        return tuple(outs)


def get_resnet18(pretrained=True):
    model = Resnet(BasicBlock, [2, 2, 2, 2], out_stride=32)
    if pretrained:
        checkpoint = model_zoo.load_url(model_urls['resnet18'])
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model


def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B, C, h, w)
    key_feats: (B, C, h, w)
    value_feats: (B, C, h, w)

    output: (B, C, h, w)
    """
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous() #(B, h*w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1) # (B, C, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous() # (B, h*w, C)

        sim_map = torch.matmul(query, key)
       
        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1) #(B, h*w, K)
        
        context = torch.matmul(sim_map, value) #(B, h*w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:]) #(B, C, h, w)

        context = self.out_project(context) #(B, C, h, w)
        return context
    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs-1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]

class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

class SFF(nn.Module):
    def __init__(self, in_channel):
        super(SFF, self).__init__()
        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel)
        self.catconv = conv_3x3(in_channel*2, in_channel)
        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels = in_channel,
            transform_channels = in_channel // 2,
            out_channels = in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1
        )
    
    def forward(self, x_small, x_big):
        img_size  =x_big.size(2), x_big.size(3)
        x_small = F.interpolate(x_small, img_size, mode="bilinear", align_corners=False)
        x = self.conv_small(x_small) + self.conv_big(x_big)
        new_x = self.attention(x, x, x_big)

        out = self.catconv(torch.cat([new_x, x_big], dim=1))
        return out

class SSFF(nn.Module):
    def __init__(self):
        super(SSFF, self).__init__()
        self.spatial = SpatialAttention()
    def forward(self, x_small, x_big):      # torch.Size([2, 128, 8, 8]) torch.Size([2, 128, 64, 64]) / 32 / 16
        img_shape = x_small.size(2), x_small.size(3)
        big_weight = self.spatial(x_big)
        big_weight = F.interpolate(big_weight, img_shape, mode="bilinear", align_corners=False)
        x_small = big_weight * x_small
        return x_small

class LightDecoder(nn.Module):
    def __init__(self, in_channel, num_class):
        super(LightDecoder, self).__init__()
        self.catconv = conv_3x3(in_channel*4, in_channel)
        self.decoder = nn.Conv2d(in_channel, num_class, 1)
    
    def forward(self, x1, x2, x3, x4):
        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        x3 = F.interpolate(x3, scale_factor=4, mode="bilinear")
        x4 = F.interpolate(x4, scale_factor=8, mode="bilinear")

        out = self.decoder(self.catconv(torch.cat([x1, x2, x3, x4], dim=1)))
        return out



# fca
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    # MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
    # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
    # planes * 4 -> channel, c2wh[planes] -> dct_h, c2wh[planes] -> dct_w
    # (64*4,56,56)
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        # print(channel, dct_h, dct_w)    # 64 56 56

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        # 返回x的多光谱向量
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        # 全连接层 + relu激活 + 全连接层 + sigmoid激活，返回一个通道注意力向量
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape       # torch.Size([2, 64, 64, 64])
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:      # dct_h=dct_w=56
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w)) # torch.Size([2, 64, 56, 56])
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)        # torch.Size([2, 64])

        y = self.fc(y).view(n, c, 1, 1)         # torch.Size([2, 64, 1, 1])
        # pytorch中的expand_as:扩张张量的尺寸至括号里张量的尺寸 torch.Size([2, 64, 64, 64]) 注意这里是逐元素相乘，不同于qkv的torch.matmul
        return x * y.expand_as(x)   # torch.Size([2, 64, 64, 64])

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    # MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        # 返回一组DCT滤波器
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):       # torch.Size([2, 64, 56, 56])
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        # DCT变换
        # x与DCT滤波器内积
        x = x * self.weight     # weight:torch.Size([64, 56, 56])  x:torch.Size([2, 64, 56, 56])
        # 消去x的2,3维
        result = torch.sum(x, dim=[2,3])        # result: torch.Size([2, 64])
        return result

    def build_filter(self, pos, freq, POS):     # 对应公式中i/j, h/w, H/W   一般是pos即i/j在变
                # self.build_filter(t_x, u_x, tile_size_x)  self.build_filter(t_y, v_y, tile_size_y)
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)        # 为什么是乘以根号2？
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
                # dct_h(height), dct_w(weight), mapper_x, mapper_y, channel(256,512,1024,2048)
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)     # (256,56,56)

        c_part = channel // len(mapper_x)       # c_part = 256/16 = 16

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 构建DCT滤波器，对应数学公式
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter


class DDLNet(nn.Module):
    def __init__(self, num_class=1, channel_list=[64, 128, 256, 512], transform_feat=128):
        super(DDLNet, self).__init__()

        self.backbone = get_resnet18()

        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.fca1 = MultiSpectralAttentionLayer(channel_list[0], c2wh[channel_list[0]], c2wh[channel_list[0]],  reduction=16, freq_sel_method = 'top16')
        self.fca2 = MultiSpectralAttentionLayer(channel_list[1], c2wh[channel_list[1]], c2wh[channel_list[1]],  reduction=16, freq_sel_method = 'top16')
        self.fca3 = MultiSpectralAttentionLayer(channel_list[2], c2wh[channel_list[2]], c2wh[channel_list[2]],  reduction=16, freq_sel_method = 'top16')
        self.fca4 = MultiSpectralAttentionLayer(channel_list[3], c2wh[channel_list[3]], c2wh[channel_list[3]],  reduction=16, freq_sel_method = 'top16')

        self.catconv1 = dsconv_3x3(channel_list[0] * 2, out_channel=128)
        self.catconv2 = dsconv_3x3(channel_list[1] * 2, out_channel=128)
        self.catconv3 = dsconv_3x3(channel_list[2] * 2, out_channel=128)
        self.catconv4 = dsconv_3x3(channel_list[3] * 2, out_channel=128)

        # self.sff1 = SFF(transform_feat)
        # self.sff2 = SFF(transform_feat)
        # self.sff3 = SFF(transform_feat)

        self.ssff1 = SSFF()
        self.ssff2 = SSFF()
        self.ssff3 = SSFF()

        self.lightdecoder = LightDecoder(transform_feat, num_class)

        self.catconv = conv_3x3(transform_feat*4, transform_feat)
    
    def forward(self, xa, xb):
        _, xA1, xA2, xA3, xA4 = self.backbone(xa) # [2, 64, 64, 64],[2, 128, 32, 32],[2, 256, 16, 16],[2, 512, 8, 8]
        _, xB1, xB2, xB3, xB4 = self.backbone(xb) # [2, 64, 64, 64],[2, 128, 32, 32],[2, 256, 16, 16],[2, 512, 8, 8]
        # print(x.shape, xA1.shape, xA2.shape, xA3.shape, xA4.shape)
        # ww
        # torch.Size([2, 64, 64, 64]) torch.Size([2, 128, 32, 32]) torch.Size([2, 256, 16, 16]) torch.Size([2, 512, 8, 8])

        x1 = self.fca1(xA1)     # torch.Size([2, 64, 64, 64])
        x2 = self.fca2(xA2)     # no change
        x3 = self.fca3(xA3)
        x4 = self.fca4(xA4)

        x11 = self.fca1(xB1)
        x22 = self.fca2(xB2)
        x33 = self.fca3(xB3)
        x44 = self.fca4(xB4)

        # 深度可分离卷积
        x111 = self.catconv1(torch.cat([x11 - x1, x1], dim=1))      # torch.Size([2, 128, 64, 64])
        x222 = self.catconv2(torch.cat([x22 - x2, x2], dim=1))      # torch.Size([2, 128, 32, 32])
        x333 = self.catconv3(torch.cat([x33 - x3, x3], dim=1))      # torch.Size([2, 128, 16, 16])
        x444 = self.catconv4(torch.cat([x44 - x4, x4], dim=1))      # torch.Size([2, 128, 8, 8])

        x1_new = self.ssff1(x444, x111)     # torch.Size([2, 128, 8, 8])
        x2_new = self.ssff2(x444, x222)     # torch.Size([2, 128, 8, 8])
        x3_new = self.ssff3(x444, x333)     # torch.Size([2, 128, 8, 8])

        x4_new = self.catconv(torch.cat([x444, x1_new, x2_new, x3_new], dim=1))     # torch.Size([2, 128, 8, 8])

        out = self.lightdecoder(x111, x222, x333, x4_new)               # torch.Size([2, 1, 64, 64])
        out = F.interpolate(out, scale_factor=4, mode="bilinear")       # torch.Size([2, 1, 256, 256])
        return [out]


if __name__ == "__main__":
    xa = torch.randn(2, 3, 256, 256)
    xb = torch.randn(2, 3, 256, 256)
    model = DDLNet()
    out = model(xa, xb)
    print(out.shape)
