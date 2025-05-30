import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from GrootV.classification.models.tree_scanning import Tree_SSM, Tree_SSM3D, MTTree_SSM3D
from fvcore.nn import flop_count
import copy


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class StemLayer(nn.Module):
    r""" Stem layer of GrootV
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=4,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x

# 为保持局部特征，在这里改进
class DownsampleLayer(nn.Module):
    r""" Downsample layer of GrootV
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.channels = channels
        # 根据 channels 的值动态创建卷积层
        # 当 channels == 320 时，stride 设置为 1,计算消耗太多
        # if channels > 80:
        #     self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=1, padding=1, bias=False)
        # else:
        #     self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x
#
# class DownsampleLayer(nn.Module):
#     r""" Downsample layer of GrootV
#     Args:
#         channels (int): number of input channels
#         norm_layer (str): normalization layer
#     """
#
#     def __init__(self, channels, norm_layer='LN'):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels,
#                               2 * channels,
#                               kernel_size=3,
#                               stride=1,
#                               padding=1,
#                               bias=False)
#         self.conv2 = nn.Conv2d(channels,
#                               2 * channels,
#                               kernel_size=3,
#                               stride=2,
#                               padding=1,
#                               bias=False)
#         self.norm = build_norm_layer(2 * channels, norm_layer,
#                                      'channels_first', 'channels_last')
#
#     def forward(self, x):
#         B, H, W, C = x.size()
#         if C == 64 or C == 256:
#             x = self.conv1(x.permute(0, 3, 1, 2))
#         else:
#             x = self.conv2(x.permute(0, 3, 1, 2))
#         x = self.norm(x)
#         return x
class MLPLayer(nn.Module):
    r""" MLP layer of GrootV
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 原始GrootV
class GrootVLayer(nn.Module):

    def __init__(self,
                 channels,
                 mlp_ratio=4.,
                 drop=0.,
                 norm_layer='LN',
                 drop_path=0.,
                 act_layer='GELU',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 ):
        super().__init__()
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.TreeSSM = Tree_SSM(
                d_model=channels, 
                d_state=1, 
                ssm_ratio=2,
                ssm_rank_ratio=2,
                dt_rank='auto',
                act_layer=nn.SiLU,
                # ==========================
                d_conv=3,
                conv_bias=False,
                # ==========================
                dropout=0.,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.TreeSSM(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.TreeSSM(self.norm1(x)))  # x(1, 128, 128, 80) -> B H W C
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.TreeSSM(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.TreeSSM(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
class GrootVBlock(nn.Module):

    def __init__(self,
                 channels,
                 depth,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.0,
                 drop_path=0.0,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            GrootVLayer(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                with_cp=with_cp,
        ) for i in range(depth)
        ])
        self.norm = build_norm_layer(channels, 'LN')
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x

class GrootV(nn.Module):

    def __init__(self,
                 channels=80,
                 depths=[2, 2, 9, 2],
                 # depths=[2, 2, 9],
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 post_norm=False,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2**(self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio

        print(f'using core type: tree_scanning_algorithm')
        print(f'using activation layer: {act_layer}')
        print(f'using main norm layer: {norm_layer}')
        print(f'using dpr: {drop_path_type}, {drop_path_rate}')

        in_chans = 4
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = GrootVBlock(
                channels=int(channels * 2**i),
                depth=depths[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                # 特征分辨率改这里，这是22.86精度的
                downsample=(i < self.num_levels-1),
                layer_scale=layer_scale,
                with_cp=with_cp,
            )
            self.levels.append(level)


        self.num_layers = len(depths)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = 'levels.{}.blocks.{}.'.format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios['levels.0.blocks.0.']
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios['levels.1.blocks.0.']
        lr_ratios["levels.0.norm"] = lr_ratios['levels.1.blocks.0.']
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios['levels.2.blocks.0.']
        lr_ratios["levels.1.norm"] = lr_ratios['levels.2.blocks.0.']
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios['levels.3.blocks.0.']
        lr_ratios["levels.2.norm"] = lr_ratios['levels.3.blocks.0.']
        return lr_ratios
        
    def forward(self, x):
        fs = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        fs.append(x.permute(0, 3, 1, 2))
        for level in self.levels:
            x = level(x)
            fs.append(x.permute(0, 3, 1, 2))
        return fs

# 修改后的适用于双时相影像的3D-GrootV
class GrootV3DLayer(nn.Module):

    def __init__(self,
                 channels,
                 mlp_ratio=4.,
                 drop=0.,
                 norm_layer='LN',
                 drop_path=0.,
                 act_layer='GELU',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 ):
        super().__init__()
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.TreeSSM = Tree_SSM3D(
                d_model=channels,
                d_state=1,
                ssm_ratio=2,
                ssm_rank_ratio=2,
                dt_rank='auto',
                act_layer=nn.SiLU,
                # ==========================
                d_conv=3,
                conv_bias=False,
                # ==========================
                dropout=0.,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.TreeSSM(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.TreeSSM(self.norm1(x)))  # x(1, 128, 128, 80) -> B H W C
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.TreeSSM(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.TreeSSM(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class GrootV3DBlock(nn.Module):

    def __init__(self,
                 channels,
                 depth,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.0,
                 drop_path=0.0,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            GrootV3DLayer(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                with_cp=with_cp,
        ) for i in range(depth)
        ])
        self.norm = build_norm_layer(channels, 'LN')
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x
class GrootV_3D(nn.Module):
# lxg: tiny参数为channels=80, depths=[2, 2, 9, 2]
# lxg: small参数为channels=96, depths=[2, 2, 13, 2]
    def __init__(self,
                 channels=80,
                 depths=[2, 2, 9, 2],
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 post_norm=False,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2 ** (self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio

        print(f'using core type: tree_scanning_algorithm')
        print(f'using activation layer: {act_layer}')
        print(f'using main norm layer: {norm_layer}')
        print(f'using dpr: {drop_path_type}, {drop_path_rate}')

        in_chans = 4
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = GrootV3DBlock(
                channels=int(channels * 2 ** i),
                depth=depths[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                # 特征分辨率改这里，这是22.86精度的
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                with_cp=with_cp,
            )
            self.levels.append(level)
        self.num_layers = len(depths)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = 'levels.{}.blocks.{}.'.format(layer_num, block_num)
                decay = 1.0 * (decay_ratio ** idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios['levels.0.blocks.0.']
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios['levels.1.blocks.0.']
        lr_ratios["levels.0.norm"] = lr_ratios['levels.1.blocks.0.']
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios['levels.2.blocks.0.']
        lr_ratios["levels.1.norm"] = lr_ratios['levels.2.blocks.0.']
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios['levels.3.blocks.0.']
        lr_ratios["levels.2.norm"] = lr_ratios['levels.3.blocks.0.']
        return lr_ratios

    def forward(self, x):
        fs = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        fs.append(x.permute(0, 3, 1, 2))
        # for level in self.levels:
        for index, level in enumerate(self.levels):
            if index==3:
                x_f = level(x)
                B, C, H, total_W = x.shape
                W = total_W // 2  # 计算 W 的值
                # 在 W 维度上切片，获取左半部分和右半部分
                left_half = x[..., :W]
                right_half = x[..., W:]
                # 交换左半部分和右半部分的位置
                x_b = torch.cat((right_half, left_half), dim=-1)
                x_b = level(x_b)
                x = x_f + x_b
            else:
                x = level(x)
            fs.append(x.permute(0, 3, 1, 2))
        return fs


# 修改后的适用于多时相影像的3D-GrootV
class MTGrootV3DLayer(nn.Module):

    def __init__(self,
                 channels,
                 mlp_ratio=4.,
                 drop=0.,
                 norm_layer='LN',
                 drop_path=0.,
                 act_layer='GELU',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 Tem=3
                 ):
        super().__init__()
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.TreeSSM = MTTree_SSM3D(
                d_model=channels,
                d_state=1,
                ssm_ratio=2,
                ssm_rank_ratio=2,
                dt_rank='auto',
                act_layer=nn.SiLU,
                # ==========================
                d_conv=3,
                conv_bias=False,
                # ==========================
                dropout=0.,
                Tem=3,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.TreeSSM(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.TreeSSM(self.norm1(x)))  # x(1, 128, 128, 80) -> B H W C
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.TreeSSM(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.TreeSSM(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
class GrootVBlock_3D(nn.Module):

    def __init__(self,
                 channels,
                 depth,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.0,
                 drop_path=0.0,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 with_cp=False,
                 ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            MTGrootV3DLayer(
                channels=channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                with_cp=with_cp,
        ) for i in range(depth)
        ])
        self.norm = build_norm_layer(channels, 'LN')
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x

class MTGrootV_3D(nn.Module):

    def __init__(self,
                 channels=80,
                 depths=[2, 2, 9, 2],
                 # depths=[2, 2, 9],
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 post_norm=False,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2 ** (self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio

        print(f'using core type: tree_scanning_algorithm')
        print(f'using activation layer: {act_layer}')
        print(f'using main norm layer: {norm_layer}')
        print(f'using dpr: {drop_path_type}, {drop_path_rate}')

        in_chans = 3
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = GrootVBlock_3D(
                channels=int(channels * 2 ** i),
                depth=depths[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                # 特征分辨率改这里，这是22.86精度的
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                with_cp=with_cp,
            )
            self.levels.append(level)

        self.num_layers = len(depths)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = 'levels.{}.blocks.{}.'.format(layer_num, block_num)
                decay = 1.0 * (decay_ratio ** idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios['levels.0.blocks.0.']
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios['levels.1.blocks.0.']
        lr_ratios["levels.0.norm"] = lr_ratios['levels.1.blocks.0.']
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios['levels.2.blocks.0.']
        lr_ratios["levels.1.norm"] = lr_ratios['levels.2.blocks.0.']
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios['levels.3.blocks.0.']
        lr_ratios["levels.2.norm"] = lr_ratios['levels.3.blocks.0.']
        return lr_ratios

    def forward(self, x):
        fs = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        fs.append(x.permute(0, 3, 1, 2))
        for level in self.levels:
            x = level(x)
            fs.append(x.permute(0, 3, 1, 2))
        return fs

# model = GrootV()
# fx = torch.randn(1, 3, 512, 512)
# output = model(fx)
# print(output)
