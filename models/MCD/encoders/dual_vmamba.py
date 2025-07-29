import torch
import torch.nn as nn
import sys
sys.path.append('../..')
sys.path.append('..')
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from ..net_utils import FeatureFusionModule as FFM
# from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from models.MCD.engine.logger import get_logger
from models.MCD.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
from deepspeed.profiling.flops_profiler import FlopsProfiler

logger = get_logger()

# on the SECOND
# class RGBXTransformer(nn.Module):
#     def __init__(self,
#                  num_classes=1000,
#                  norm_layer=nn.LayerNorm,
#                  depths=[2, 2, 27, 2],  # [2,2,27,2] for vmamba small
#                  dims=96,
#                  pretrained=None,
#                  mlp_ratio=4.0,
#                  downsample_version='v1',
#                  ape=False,
#                  img_size=[480, 640],
#                  patch_size=4,
#                  drop_path_rate=0.2,
#                  **kwargs):
#         super().__init__()
#
#         self.ape = ape
#
#         self.vssm = Backbone_VSSM(
#             pretrained=pretrained,
#             norm_layer=norm_layer,
#             num_classes=num_classes,
#             depths=depths,
#             dims=dims,
#             mlp_ratio=mlp_ratio,
#             downsample_version=downsample_version,
#             drop_path_rate=drop_path_rate,
#         )
#
#         self.cross_mamba = nn.ModuleList(
#             CrossMambaFusionBlock(
#                 hidden_dim=dims * (2 ** i),
#                 mlp_ratio=0.0,
#                 d_state=4,
#             ) for i in range(4)
#         )
#         self.channel_attn_mamba = nn.ModuleList(
#             ConcatMambaFusionBlock(
#                 hidden_dim=dims * (2 ** i),
#                 mlp_ratio=0.0,
#                 d_state=4,
#             ) for i in range(4)
#         )
#
#         # absolute position embedding
#         if self.ape:
#             self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
#             self.absolute_pos_embed = []
#             self.absolute_pos_embed_x = []
#             self.absolute_pos_embed_y = []
#             for i_layer in range(len(depths)):
#                 input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
#                                     self.patches_resolution[1] // (2 ** i_layer))
#                 dim = int(dims * (2 ** i_layer))
#                 absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
#                 trunc_normal_(absolute_pos_embed, std=.02)
#                 absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
#                 trunc_normal_(absolute_pos_embed_x, std=.02)
#                 absolute_pos_embed_y = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
#                 trunc_normal_(absolute_pos_embed_y, std=.02)
#                 self.absolute_pos_embed.append(absolute_pos_embed)
#                 self.absolute_pos_embed_x.append(absolute_pos_embed_x)
#                 self.absolute_pos_embed_y.append(absolute_pos_embed_y)
#
#     def forward_features(self, x_A, x_B):
#         """
#         x_A: B x C x H x W
#         """
#         B = x_A.shape[0]
#         outs_fused = []
#
#         outs_A = self.vssm(x_A)  # B x C x H x W
#         outs_B = self.vssm(x_B)  # B x C x H x W
#         outss_A = []
#         outss_B = []
#         for i in range(4):
#             if self.ape:
#                 # this has been discarded
#                 out_A = self.absolute_pos_embed[i].to(outs_A[i].device) + outs_A[i]
#                 out_B = self.absolute_pos_embed_x[i].to(outs_B[i].device) + outs_B[i]
#                 outss_A.append(out_A)
#                 outss_B.append(out_B)
#             else:
#                 out_A = outs_A[i]
#                 out_B = outs_B[i]
#                 outss_A.append(out_A)
#                 outss_B.append(out_B)
#             x_fuse = self.channel_attn_mamba[i](out_A.permute(0, 2, 3, 1).contiguous(),
#                                                 out_B.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
#             # x_fuse = (out_A - out_B)
#             outs_fused.append(x_fuse)
#         return outss_A, outss_B, outs_fused
#
#     def forward(self, x_A, x_B):
#         out = self.forward_features(x_A, x_B)
#         return out

# on the WUSU
# class RGBXTransformer(nn.Module):
#     def __init__(self,
#                  num_classes=1000,
#                  norm_layer=nn.LayerNorm,
#                  depths=[2,2,27,2], # [2,2,27,2] for vmamba small
#                  dims=96,
#                  pretrained=None,
#                  mlp_ratio=4.0,
#                  downsample_version='v1',
#                  ape=False,
#                  img_size=[480, 640],
#                  patch_size=4,
#                  drop_path_rate=0.2,
#                  **kwargs):
#         super().__init__()
#
#         self.ape = ape
#
#         self.vssm = Backbone_VSSM(
#             pretrained=pretrained,
#             norm_layer=norm_layer,
#             num_classes=num_classes,
#             depths=depths,
#             dims=dims,
#             mlp_ratio=mlp_ratio,
#             downsample_version=downsample_version,
#             drop_path_rate=drop_path_rate,
#         )
#
#         self.cross_mamba = nn.ModuleList(
#             CrossMambaFusionBlock(
#                 hidden_dim=dims * (2 ** i),
#                 mlp_ratio=0.0,
#                 d_state=4,
#             ) for i in range(4)
#         )
#         self.channel_attn_mamba = nn.ModuleList(
#             ConcatMambaFusionBlock(
#                 hidden_dim=dims * (2 ** i),
#                 mlp_ratio=0.0,
#                 d_state=4,
#             ) for i in range(4)
#         )
#
#         # absolute position embedding
#         if self.ape:
#             self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
#             self.absolute_pos_embed = []
#             self.absolute_pos_embed_x = []
#             self.absolute_pos_embed_y = []
#             for i_layer in range(len(depths)):
#                 input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
#                                       self.patches_resolution[1] // (2 ** i_layer))
#                 dim=int(dims * (2 ** i_layer))
#                 absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
#                 trunc_normal_(absolute_pos_embed, std=.02)
#                 absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
#                 trunc_normal_(absolute_pos_embed_x, std=.02)
#                 absolute_pos_embed_y = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
#                 trunc_normal_(absolute_pos_embed_y, std=.02)
#                 self.absolute_pos_embed.append(absolute_pos_embed)
#                 self.absolute_pos_embed_x.append(absolute_pos_embed_x)
#                 self.absolute_pos_embed_y.append(absolute_pos_embed_y)
#     def forward_features(self, x_A, x_B, x_C):
#         """
#         x_A: B x C x H x W
#         """
#         B = x_A.shape[0]
#         outs_fused = []
#
#         outs_A = self.vssm(x_A) # B x C x H x W
#         outs_B = self.vssm(x_B) # B x C x H x W
#         outs_C = self.vssm(x_C)
#         outss_A = []
#         outss_B = []
#         outss_C = []
#         for i in range(4):
#             if self.ape:
#                 # this has been discarded
#                 out_A = self.absolute_pos_embed[i].to(outs_A[i].device) + outs_A[i]
#                 out_B = self.absolute_pos_embed_x[i].to(outs_B[i].device) + outs_B[i]
#                 out_C = self.absolute_pos_embed_x[i].to(outs_C[i].device) + outs_C[i]
#                 outss_A.append(out_A)
#                 outss_B.append(out_B)
#                 outss_C.append(out_C)
#             else:
#                 out_A = outs_A[i]
#                 out_B = outs_B[i]
#                 out_C = outs_C[i]
#                 outss_A.append(out_A)
#                 outss_B.append(out_B)
#                 outss_C.append(out_C)
#             x_fuse = self.channel_attn_mamba[i](out_A.permute(0, 2, 3, 1).contiguous(), out_C.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
#             # x_fuse = (out_A - out_B)
#             outs_fused.append(x_fuse)
#         return outss_A, outss_B, outss_C, outs_fused
#
#     def forward(self, x_A, x_B, x_C):
#         out = self.forward_features(x_A, x_B, x_C)
#         return out

# on the DynamicEarthNet
class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 2, 27, 2],  # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed_a = []
            self.absolute_pos_embed_b = []
            self.absolute_pos_embed_c = []
            self.absolute_pos_embed_d = []
            self.absolute_pos_embed_e = []
            self.absolute_pos_embed_f = []
            for i_layer in range(len(depths)):
                input_resolution = (self.patches_resolution[0] // (2 ** i_layer),
                                    self.patches_resolution[1] // (2 ** i_layer))
                dim = int(dims * (2 ** i_layer))
                absolute_pos_embed_a = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_a, std=.02)
                absolute_pos_embed_b = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_b, std=.02)
                absolute_pos_embed_c = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_c, std=.02)
                absolute_pos_embed_d = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_d, std=.02)
                absolute_pos_embed_e = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_e, std=.02)
                absolute_pos_embed_f = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_f, std=.02)
                self.absolute_pos_embed_a.append(absolute_pos_embed_a)
                self.absolute_pos_embed_b.append(absolute_pos_embed_b)
                self.absolute_pos_embed_c.append(absolute_pos_embed_c)
                self.absolute_pos_embed_d.append(absolute_pos_embed_d)
                self.absolute_pos_embed_e.append(absolute_pos_embed_e)
                self.absolute_pos_embed_f.append(absolute_pos_embed_f)

    def forward_features(self, x_A, x_B, x_C, x_D, x_E, x_F, pair):
        """
        x_A: B x C x H x W
        """
        B = x_A.shape[0]
        outs_fused = []

        outs_A = self.vssm(x_A)  # B x C x H x W
        outs_B = self.vssm(x_B)  # B x C x H x W
        outs_C = self.vssm(x_C)
        outs_D = self.vssm(x_D)  # B x C x H x W
        outs_E = self.vssm(x_E)  # B x C x H x W
        outs_F = self.vssm(x_F)
        outs = [outs_A, outs_B, outs_C, outs_D, outs_E, outs_F]
        outss_A = []
        outss_B = []
        outss_C = []
        outss_D = []
        outss_E = []
        outss_F = []
        for i in range(4):
            if self.ape:
                out_A = self.absolute_pos_embed_a[i].to(outs[0][i].device) + outs[0][i]
                out_B = self.absolute_pos_embed_b[i].to(outs[1][i].device) + outs[1][i]
                out_C = self.absolute_pos_embed_c[i].to(outs[2][i].device) + outs[2][i]
                out_D = self.absolute_pos_embed_d[i].to(outs[3][i].device) + outs[3][i]
                out_E = self.absolute_pos_embed_e[i].to(outs[4][i].device) + outs[4][i]
                out_F = self.absolute_pos_embed_f[i].to(outs[5][i].device) + outs[5][i]
                out = [out_A, out_B, out_C, out_D, out_E, out_F]
                outss_A.append(out_A)
                outss_B.append(out_B)
                outss_C.append(out_C)
                outss_D.append(out_D)
                outss_E.append(out_E)
                outss_F.append(out_F)
            else:
                out_A = outs[0][i]
                out_B = outs[1][i]
                out_C = outs[2][i]
                out_D = outs[3][i]
                out_E = outs[4][i]
                out_F = outs[5][i]
                out = [out_A, out_B, out_C, out_D, out_E, out_F]
                outss_A.append(out_A)
                outss_B.append(out_B)
                outss_C.append(out_C)
                outss_D.append(out_D)
                outss_E.append(out_E)
                outss_F.append(out_F)
            x_fuse = self.channel_attn_mamba[i](out[pair[0]].permute(0, 2, 3, 1).contiguous(),
                                                out[pair[-1]].permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1,
                                                                                                        2).contiguous()
            # x_fuse = (out_A - out_B)
            outs_fused.append(x_fuse)
        return outss_A, outss_B, outss_C, outss_D, outss_E, outss_F, outs_fused

    def forward(self, x_A, x_B, x_C, x_D, x_E, x_F, pair):
        out = self.forward_features(x_A, x_B, x_C, x_D, x_E, x_F, pair)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/models/FEMCD/vssm1_tiny_0230s_ckpt_epoch_264.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))


class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))

if __name__ == "__main__":
    print("model small")
    model_small = vssm_small().cuda()
    prof = FlopsProfiler(model_small, None)

    dummy1 = torch.ones((1,3,256,256)).cuda()
    dummy2 = torch.ones((1,3,256,256)).cuda()
    prof.start_profile()
    dummy_out = model_small(dummy1, dummy2)
    prof.stop_profile()
    flops = prof.get_total_flops()
    print("Small GFlops: ", flops/(10**9))
    prof.end_profile()
