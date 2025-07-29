import torch
import torch.distributed as dist
import numpy as np
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from einops import rearrange

from tree_scan import _C

class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None

mst = _MST.apply

def norm2_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = (diff * diff).sum(dim=1)
    return weight

def norm1_distance(fm_ref, fm_tar):
    diff = fm_ref - fm_tar
    weight = torch.abs(diff).sum(dim=1)
    return weight

class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
    
    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2), 
                           col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def _build_feature_weight_cosine(self, fm, max_tree):
        batch,dim = fm.shape[0],fm.shape[1]
        weight_row = torch.cosine_similarity(fm[:, :, :-1, :].reshape(batch,dim,-1), fm[:, :, 1:, :].reshape(batch,dim,-1),dim=1)
        # import pdb;pdb.set_trace()
        weight_col = torch.cosine_similarity(fm[:, :, :, :-1].reshape(batch,dim,-1), fm[:, :, :, 1:].reshape(batch,dim,-1),dim=1)
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(weight)   # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(-weight)   # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in, max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree


class MinimumSpanning3DTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanning3DTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        # 修改后代码
        # 获取左半部分和右半部分的索引
        mid = width // 2
        left_raw_index = raw_index[:, :mid]
        right_raw_index = raw_index[:, mid:]

        # 获取左半部分的row_index和col_index
        left_row_index = torch.stack([left_raw_index[:-1, :], left_raw_index[1:, :]], 2)
        left_col_index = torch.stack([left_raw_index[:, :-1], left_raw_index[:, 1:]], 2)

        # 获取右半部分的row_index和col_index
        right_row_index = torch.stack([right_raw_index[:-1, :], right_raw_index[1:, :]], 2)
        right_col_index = torch.stack([right_raw_index[:, :-1], right_raw_index[:, 1:]], 2)

        # 获取左半部分和右半部分对应位置的索引
        # cross_row_index = torch.stack([left_raw_index[:-1, :], right_raw_index[1:, :]], 2)
        cross_col_index = torch.stack([left_raw_index[:, :], right_raw_index[:, :]], 2)

        # 合并所有索引
        index = torch.cat([
            left_row_index.reshape(1, -1, 2),
            left_col_index.reshape(1, -1, 2),
            right_row_index.reshape(1, -1, 2),
            right_col_index.reshape(1, -1, 2),
            cross_col_index.reshape(1, -1, 2)
        ], 1)

        index = index.expand(batch, -1, -1)

        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weihth_bi = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight_bi = weihth_bi.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def _build_feature_weight_cosine(self, fm, max_tree):
        batch, dim = fm.shape[0], fm.shape[1]
        half_width = fm.shape[3] // 2
        fm_left_half = fm[:, :, :, :half_width]
        fm_right_half = fm[:, :, :, half_width:]
        # 左边weight_row和weight_col
        left_weight_row = torch.cosine_similarity(fm_left_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_left_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        left_weight_col = torch.cosine_similarity(fm_left_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_left_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # 右边weight_row和weight_col
        right_weight_row = torch.cosine_similarity(fm_right_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_right_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        right_weight_col = torch.cosine_similarity(fm_right_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_right_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # 左右对应位置权重
        weight_bi = torch.cosine_similarity(fm_left_half.reshape(batch, dim, -1),
                                             fm_right_half.reshape(batch, dim, -1), dim=1)


        weight = torch.cat([left_weight_row, left_weight_col, right_weight_row, right_weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(
                    weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(
                    -weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in, max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree


class MinimumSpanningMT3DTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None, Tem=3):
        super(MinimumSpanningMT3DTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    @staticmethod
    def _build_matrix_index(fm, Tem=3):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        # 修改后代码
        # 获取左半部分和右半部分的索引
        mid = width // Tem
        T1_raw_index = raw_index[:, :mid]
        T2_raw_index = raw_index[:, mid:2*mid]
        T3_raw_index = raw_index[:, 2*mid:3*mid]
        # 获取T1的row_index和col_index
        T1_row_index = torch.stack([T1_raw_index[:-1, :], T1_raw_index[1:, :]], 2)
        T1_col_index = torch.stack([T1_raw_index[:, :-1], T1_raw_index[:, 1:]], 2)

        # 获取T2的row_index和col_index
        T2_row_index = torch.stack([T2_raw_index[:-1, :], T2_raw_index[1:, :]], 2)
        T2_col_index = torch.stack([T2_raw_index[:, :-1], T2_raw_index[:, 1:]], 2)

        # 获取T3的row_index和col_index
        T3_row_index = torch.stack([T3_raw_index[:-1, :], T3_raw_index[1:, :]], 2)
        T3_col_index = torch.stack([T3_raw_index[:, :-1], T3_raw_index[:, 1:]], 2)
        # 获取左半部分和右半部分对应位置的索引
        # cross_row_index = torch.stack([left_raw_index[:-1, :], right_raw_index[1:, :]], 2)
        cross_col_index = torch.stack([T1_raw_index[:, :], T2_raw_index[:, :]], 2)

        # 合并所有索引
        index = torch.cat([
            T1_row_index.reshape(1, -1, 2),
            T1_col_index.reshape(1, -1, 2),
            T2_row_index.reshape(1, -1, 2),
            T2_col_index.reshape(1, -1, 2),
            T3_row_index.reshape(1, -1, 2),
            T3_col_index.reshape(1, -1, 2),
            cross_col_index.reshape(1, -1, 2)
        ], 1)

        index = index.expand(batch, -1, -1)

        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = norm2_distance(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weihth_bi = norm2_distance(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight_bi = weihth_bi.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def _build_feature_weight_cosine(self, fm, max_tree):
        batch, dim = fm.shape[0], fm.shape[1]
        Tem = 3
        half_width = fm.shape[3] // Tem
        fm_T1_half = fm[:, :, :, :half_width]
        fm_T2_half = fm[:, :, :, half_width:2*half_width]
        fm_T3_half = fm[:, :, :, 2*half_width:3*half_width]
        # T1的weight_row和weight_col
        T1_weight_row = torch.cosine_similarity(fm_T1_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_T1_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        T1_weight_col = torch.cosine_similarity(fm_T1_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_T1_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # T2的weight_row和weight_col
        T2_weight_row = torch.cosine_similarity(fm_T2_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_T2_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        T2_weight_col = torch.cosine_similarity(fm_T2_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_T2_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # T3的weight_row和weight_col
        T3_weight_row = torch.cosine_similarity(fm_T3_half[:, :, :-1, :].reshape(batch, dim, -1),
                                             fm_T3_half[:, :, 1:, :].reshape(batch, dim, -1), dim=1)
        T3_weight_col = torch.cosine_similarity(fm_T3_half[:, :, :, :-1].reshape(batch, dim, -1),
                                             fm_T3_half[:, :, :, 1:].reshape(batch, dim, -1), dim=1)

        # 左右对应位置权重
        weight_bi = torch.cosine_similarity(fm_T1_half.reshape(batch, dim, -1),
                                             fm_T3_half.reshape(batch, dim, -1), dim=1)


        weight = torch.cat([T1_weight_row, T1_weight_col, T2_weight_row, T2_weight_col, T3_weight_row, T3_weight_col, weight_bi], dim=1)
        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(
                    weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(
                    -weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_cosine(guide_in, max_tree)
            else:
                weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree


import torch
import torch.nn as nn


class MinimumSpanningMTnDTree_revised(nn.Module):
    def __init__(self, distance_func, mapping_func=None, Tem=3):
        super(MinimumSpanningMTnDTree_revised, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
        self.Tem = Tem

    @staticmethod
    def _build_matrix_index(fm, Tem=3):
        batch, height, width = fm.shape[0], fm.shape[2], fm.shape[3]
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        Tem = 3
        # print('TEM:', Tem)
        # 动态分割时相
        phase_width = width // Tem
        phase_indices = [raw_index[:, i * phase_width:(i + 1) * phase_width] for i in range(Tem)]

        # 计算每个时相的行索引和列索引
        row_indices = []
        col_indices = []
        for phase_index in phase_indices:
            row_index = torch.stack([phase_index[:-1, :], phase_index[1:, :]], dim=2)
            col_index = torch.stack([phase_index[:, :-1], phase_index[:, 1:]], dim=2)
            row_indices.append(row_index.reshape(1, -1, 2))
            col_indices.append(col_index.reshape(1, -1, 2))

        # 计算跨时相的索引
        cross_indices = []
        for i in range(Tem - 1):
            cross_index = torch.stack([phase_indices[i][:, :], phase_indices[i + 1][:, :]], dim=2)
            cross_indices.append(cross_index.reshape(1, -1, 2))

        # 合并所有索引
        index = torch.cat(row_indices + col_indices + cross_indices, dim=1)
        index = index.expand(batch, -1, -1)

        return index

    def _build_feature_weight_ED(self, fm):
        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem
        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
        # 计算每个时相的行权重和列权重
        weights = []
        # 使用欧式距离计算每个时相的行权重和列权重
        pdist = nn.PairwiseDistance(p=2)
        for phase_feature in phase_features:
            weight_row = pdist(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1)
            )
            weight_col = pdist(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1)
            )
            weights.extend([weight_row, weight_col])
        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = pdist(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1)
            )
            weights.append(weight_cross)

        weight = torch.cat(weights, dim=1)
        return weight

    def _build_feature_weight_norm1_distance(self, fm):
        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem
        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
        # 计算每个时相的行权重和列权重
        weights = []
        # 使用欧式距离计算每个时相的行权重和列权重
        for phase_feature in phase_features:
            weight_row = norm1_distance(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1)
            )
            weight_col = norm1_distance(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1)
            )
            weights.extend([weight_row, weight_col])
        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = norm1_distance(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1)
            )
            weights.append(weight_cross)

        weight = torch.cat(weights, dim=1)
        return weight

    def _build_feature_weight_norm2_distance(self, fm):
        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem
        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
        # 计算每个时相的行权重和列权重
        weights = []
        # 使用欧式距离计算每个时相的行权重和列权重
        for phase_feature in phase_features:
            weight_row = norm2_distance(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1)
            )
            weight_col = norm2_distance(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1)
            )
            weights.extend([weight_row, weight_col])
        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = norm2_distance(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1)
            )
            weights.append(weight_cross)

        weight = torch.cat(weights, dim=1)
        return weight

    def _build_feature_weight_consine(self, fm, max_tree):
        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem
        # print(self.Tem)
        # print('使用余弦相似度计算特征距离')
        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
        # 计算每个时相的行权重和列权重
        weights = []
        # 使用余弦相似度计算每个时相的行权重和列权重
        for phase_feature in phase_features:
            weight_row = torch.cosine_similarity(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1),
                dim=1
            )
            weight_col = torch.cosine_similarity(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1),
                dim=1
            )
            weights.extend([weight_row, weight_col])

        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = torch.cosine_similarity(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1),
                dim=1
            )
            weights.append(weight_cross)
        # 合并权重
        weight = torch.cat(weights, dim=1)

        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(-weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight


        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem
        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
        # 计算每个时相的行权重和列权重
        weights = []
        # 使用余弦相似度计算每个时相的行权重和列权重
        for phase_feature in phase_features:
            weight_row = torch.cosine_similarity(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1),
                dim=1
            )
            weight_col = torch.cosine_similarity(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1),
                dim=1
            )
            weights.extend([weight_row, weight_col])

        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = torch.cosine_similarity(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1),
                dim=1
            )
            weights.append(weight_cross)
        # 合并权重
        weight = torch.cat(weights, dim=1)

        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(-weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in, 3)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_consine(guide_in, max_tree)
            elif self.distance_func == "norm2_distance":
                weight = self._build_feature_weight_norm2_distance(guide_in)
            elif self.distance_func == "ED":
                weight = self._build_feature_weight_ED(guide_in)
            else:
                weight = self._build_feature_weight_norm1_distance(guide_in)
                print('norm1_distance!')
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree



# 回复审稿人3意见，采用14方向邻接矩阵
import torch
import torch.nn as nn
class MinimumSpanningMTnDTree_AE(nn.Module):
    def __init__(self, distance_func, mapping_func=None, Tem=6):
        super(MinimumSpanningMTnDTree_AE, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
        self.Tem = Tem

    @staticmethod
    def _build_matrix_index(fm, Tem=6):
        batch, height, width = fm.shape[0], fm.shape[2], fm.shape[3]
        device = fm.device

        # 创建行和列索引
        row = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=device).unsqueeze(1)
        raw_index = row + col * width
        # print(Tem)
        # 动态分割时相
        phase_width = width // Tem
        if phase_width == 0:
            return torch.empty(0, 2, dtype=torch.int32, device=device)

        phase_indices = []
        for i in range(Tem):
            start = i * phase_width
            end = (i + 1) * phase_width
            phase = raw_index[:, start:end]
            if phase.numel() > 0:  # 确保非空
                phase_indices.append(phase)

        # 计算每个时相的行索引和列索引
        row_indices = []
        col_indices = []
        for phase_index in phase_indices:
            h, w = phase_index.shape

            # 行邻接（垂直方向）- 确保高度至少为2
            if h >= 2:
                row_index = torch.stack([phase_index[:-1, :], phase_index[1:, :]], dim=2)
                row_indices.append(row_index.reshape(1, -1, 2))

            # 列邻接（水平方向）- 确保宽度至少为2
            if w >= 2:
                col_index = torch.stack([phase_index[:, :-1], phase_index[:, 1:]], dim=2)
                col_indices.append(col_index.reshape(1, -1, 2))

        # 计算跨时相的索引（包括四个方向的邻接）
        cross_indices = []
        for i in range(Tem - 1):
            phase1 = phase_indices[i]
            phase2 = phase_indices[i + 1]

            h1, w1 = phase1.shape
            h2, w2 = phase2.shape

            # 获取公共区域尺寸
            min_h = min(h1, h2)
            min_w = min(w1, w2)

            if min_h == 0 or min_w == 0:
                continue

            positions = []

            # 1. 相同位置
            same_position = torch.stack([
                phase1[:min_h, :min_w].reshape(-1),
                phase2[:min_h, :min_w].reshape(-1)
            ], dim=1).unsqueeze(0)
            positions.append(same_position)

            # 2. 上邻居（上偏移1像素）
            if min_h > 1:
                up_position = torch.stack([
                    phase1[1:min_h, :min_w].reshape(-1),
                    phase2[:min_h - 1, :min_w].reshape(-1)
                ], dim=1).unsqueeze(0)
                positions.append(up_position)

            # 3. 下邻居（下偏移1像素）
            if min_h > 1:
                down_position = torch.stack([
                    phase1[:min_h - 1, :min_w].reshape(-1),
                    phase2[1:min_h, :min_w].reshape(-1)
                ], dim=1).unsqueeze(0)
                positions.append(down_position)

            # 4. 左邻居（左偏移1像素）
            if min_w > 1:
                left_position = torch.stack([
                    phase1[:min_h, 1:min_w].reshape(-1),
                    phase2[:min_h, :min_w - 1].reshape(-1)
                ], dim=1).unsqueeze(0)
                positions.append(left_position)

            # 5. 右邻居（右偏移1像素）
            if min_w > 1:
                right_position = torch.stack([
                    phase1[:min_h, :min_w - 1].reshape(-1),
                    phase2[:min_h, 1:min_w].reshape(-1)
                ], dim=1).unsqueeze(0)
                positions.append(right_position)

            if positions:
                cross_index = torch.cat(positions, dim=1)
                cross_indices.append(cross_index)

        # 合并所有索引
        all_indices = []
        if row_indices:
            all_indices.append(torch.cat(row_indices, dim=1))
        if col_indices:
            all_indices.append(torch.cat(col_indices, dim=1))
        if cross_indices:
            all_indices.append(torch.cat(cross_indices, dim=1))

        if not all_indices:
            return torch.empty(0, 2, dtype=torch.int32, device=device)

        index = torch.cat(all_indices, dim=1)
        return index.expand(batch, -1, -1)

    def _compute_edge_weight(self, feature_map, index1, index2):
        """计算两个位置之间的特征距离"""
        batch_size, channels = feature_map.shape[:2]

        # 展平特征图
        flat_features = feature_map.view(batch_size, channels, -1)

        # 获取特征向量
        features1 = flat_features[:, :, index1]
        features2 = flat_features[:, :, index2]

        if self.distance_func == "ED":
            # 欧氏距离
            return torch.norm(features1 - features2, p=2, dim=1)
        elif self.distance_func == "norm1_distance":
            # L1距离
            return torch.norm(features1 - features2, p=1, dim=1)
        elif self.distance_func == "norm2_distance":
            # L2距离
            return torch.norm(features1 - features2, p=2, dim=1)
        elif self.distance_func == "Cosine":
            # 余弦相似度
            return torch.cosine_similarity(features1, features2, dim=1)
        else:
            # 默认为L1距离
            return torch.norm(features1 - features2, p=1, dim=1)

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            # 构建索引矩阵
            index = self._build_matrix_index(guide_in, self.Tem)

            if index.numel() == 0:
                return None

            # 从索引中提取节点索引
            index1 = index[:, :, 0]  # 起始节点索引
            index2 = index[:, :, 1]  # 目标节点索引

            # 计算所有边的权重
            weight = self._compute_edge_weight(guide_in, index1, index2)

            # 如果需要映射函数
            if self.mapping_func is not None and self.distance_func == "Cosine":
                if max_tree:
                    weight = self.mapping_func(weight)
                else:
                    weight = self.mapping_func(-weight)

            # 计算最小生成树
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])

        return tree


# class MinimumSpanningMTnDTree_AE(nn.Module):
#     def __init__(self, distance_func, mapping_func=None, Tem=3):
#         super(MinimumSpanningMTnDTree_AE, self).__init__()
#         self.distance_func = distance_func
#         self.mapping_func = mapping_func
#         self.Tem = Tem
#     @staticmethod
#     def _build_matrix_index(fm, Tem=3):
#         batch, height, width = fm.shape[0], fm.shape[2], fm.shape[3]
#         row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
#         col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
#         raw_index = row + col * width
#         # print(Tem)
#         # 动态分割时相
#         phase_width = width // Tem
#         phase_indices = [raw_index[:, i * phase_width:(i + 1) * phase_width] for i in range(Tem)]
#
#         # 计算每个时相的行索引和列索引
#         row_indices = []
#         col_indices = []
#         for phase_index in phase_indices:
#             # 行方向邻接（垂直方向）
#             row_index = torch.stack([phase_index[:-1, :], phase_index[1:, :]], dim=2)
#             # 列方向邻接（水平方向）
#             col_index = torch.stack([phase_index[:, :-1], phase_index[:, 1:]], dim=2)
#             row_indices.append(row_index.reshape(1, -1, 2))
#             col_indices.append(col_index.reshape(1, -1, 2))
#
#         # 计算跨时相的索引 - 扩展为包括上下左右四个邻居
#         cross_indices = []
#         for i in range(Tem - 1):
#             current_phase = phase_indices[i]
#             next_phase = phase_indices[i + 1]
#
#             # 相同位置
#             same_position = torch.stack([current_phase, next_phase], dim=2)
#
#             # 上邻居 (row-1, col)
#             up_mask = torch.zeros_like(current_phase, dtype=torch.bool)
#             up_mask[1:, :] = True  # 第一行没有上邻居
#             up_current = current_phase[1:, :]
#             up_next = next_phase[:-1, :]
#             up_position = torch.stack([up_current, up_next], dim=2)
#
#             # 下邻居 (row+1, col)
#             down_mask = torch.zeros_like(current_phase, dtype=torch.bool)
#             down_mask[:-1, :] = True  # 最后一行没有下邻居
#             down_current = current_phase[:-1, :]
#             down_next = next_phase[1:, :]
#             down_position = torch.stack([down_current, down_next], dim=2)
#
#             # 左邻居 (row, col-1)
#             left_mask = torch.zeros_like(current_phase, dtype=torch.bool)
#             left_mask[:, 1:] = True  # 第一列没有左邻居
#             left_current = current_phase[:, 1:]
#             left_next = next_phase[:, :-1]
#             left_position = torch.stack([left_current, left_next], dim=2)
#
#             # 右邻居 (row, col+1)
#             right_mask = torch.zeros_like(current_phase, dtype=torch.bool)
#             right_mask[:, :-1] = True  # 最后一列没有右邻居
#             right_current = current_phase[:, :-1]
#             right_next = next_phase[:, 1:]
#             right_position = torch.stack([right_current, right_next], dim=2)
#
#             # 合并所有位置
#             all_positions = torch.cat([
#                 same_position.reshape(-1, 2),
#                 up_position.reshape(-1, 2),
#                 down_position.reshape(-1, 2),
#                 left_position.reshape(-1, 2),
#                 right_position.reshape(-1, 2)
#             ], dim=0)
#
#             cross_indices.append(all_positions.unsqueeze(0))
#
#         # 合并所有索引
#         row_tensor = torch.cat(row_indices, dim=1)
#         col_tensor = torch.cat(col_indices, dim=1)
#         cross_tensor = torch.cat(cross_indices, dim=1)
#
#         index = torch.cat([row_tensor, col_tensor, cross_tensor], dim=1)
#         index = index.expand(batch, -1, -1)
#
#         return index
#
#     def _build_feature_weight_ED(self, fm):
#         batch, dim = fm.shape[0], fm.shape[1]
#         height, width = fm.shape[2], fm.shape[3]
#         phase_width = width // self.Tem
#         # 分割特征图
#         phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
#         # 计算每个时相的行权重和列权重
#         weights = []
#         # 使用欧式距离计算每个时相的行权重和列权重
#         pdist = nn.PairwiseDistance(p=2)
#         for phase_feature in phase_features:
#             weight_row = pdist(
#                 phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
#                 phase_feature[:, :, 1:, :].reshape(batch, dim, -1)
#             )
#             weight_col = pdist(
#                 phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
#                 phase_feature[:, :, :, 1:].reshape(batch, dim, -1)
#             )
#             weights.extend([weight_row, weight_col])
#         # 计算跨时相权重
#         for i in range(self.Tem - 1):
#             weight_cross = pdist(
#                 phase_features[i].reshape(batch, dim, -1),
#                 phase_features[i + 1].reshape(batch, dim, -1)
#             )
#             weights.append(weight_cross)
#
#         weight = torch.cat(weights, dim=1)
#         return weight
#
#     def _build_feature_weight_norm1_distance(self, fm):
#         batch, dim = fm.shape[0], fm.shape[1]
#         height, width = fm.shape[2], fm.shape[3]
#         phase_width = width // self.Tem
#         # 分割特征图
#         phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
#         # 计算每个时相的行权重和列权重
#         weights = []
#         # 使用欧式距离计算每个时相的行权重和列权重
#         for phase_feature in phase_features:
#             weight_row = norm1_distance(
#                 phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
#                 phase_feature[:, :, 1:, :].reshape(batch, dim, -1)
#             )
#             weight_col = norm1_distance(
#                 phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
#                 phase_feature[:, :, :, 1:].reshape(batch, dim, -1)
#             )
#             weights.extend([weight_row, weight_col])
#         # 计算跨时相权重
#         for i in range(self.Tem - 1):
#             weight_cross = norm1_distance(
#                 phase_features[i].reshape(batch, dim, -1),
#                 phase_features[i + 1].reshape(batch, dim, -1)
#             )
#             weights.append(weight_cross)
#
#         weight = torch.cat(weights, dim=1)
#         return weight
#
#     def _build_feature_weight_norm2_distance(self, fm):
#         batch, dim = fm.shape[0], fm.shape[1]
#         height, width = fm.shape[2], fm.shape[3]
#         phase_width = width // self.Tem
#         # 分割特征图
#         phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
#         # 计算每个时相的行权重和列权重
#         weights = []
#         # 使用欧式距离计算每个时相的行权重和列权重
#         for phase_feature in phase_features:
#             weight_row = norm2_distance(
#                 phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
#                 phase_feature[:, :, 1:, :].reshape(batch, dim, -1)
#             )
#             weight_col = norm2_distance(
#                 phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
#                 phase_feature[:, :, :, 1:].reshape(batch, dim, -1)
#             )
#             weights.extend([weight_row, weight_col])
#         # 计算跨时相权重
#         for i in range(self.Tem - 1):
#             weight_cross = norm2_distance(
#                 phase_features[i].reshape(batch, dim, -1),
#                 phase_features[i + 1].reshape(batch, dim, -1)
#             )
#             weights.append(weight_cross)
#
#         weight = torch.cat(weights, dim=1)
#         return weight
#
#     def _build_feature_weight_consine(self, fm, max_tree):
#         batch, dim = fm.shape[0], fm.shape[1]
#         height, width = fm.shape[2], fm.shape[3]
#         phase_width = width // self.Tem
#         # print(self.Tem)
#         # print('使用余弦相似度计算特征距离')
#         # 分割特征图
#         phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
#         # 计算每个时相的行权重和列权重
#         weights = []
#         # 使用余弦相似度计算每个时相的行权重和列权重
#         for phase_feature in phase_features:
#             weight_row = torch.cosine_similarity(
#                 phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
#                 phase_feature[:, :, 1:, :].reshape(batch, dim, -1),
#                 dim=1
#             )
#             weight_col = torch.cosine_similarity(
#                 phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
#                 phase_feature[:, :, :, 1:].reshape(batch, dim, -1),
#                 dim=1
#             )
#             weights.extend([weight_row, weight_col])
#
#         # 计算跨时相权重
#         for i in range(self.Tem - 1):
#             weight_cross = torch.cosine_similarity(
#                 phase_features[i].reshape(batch, dim, -1),
#                 phase_features[i + 1].reshape(batch, dim, -1),
#                 dim=1
#             )
#             weights.append(weight_cross)
#         # 合并权重
#         weight = torch.cat(weights, dim=1)
#
#         if self.mapping_func is not None:
#             if max_tree:
#                 weight = self.mapping_func(weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
#             else:
#                 weight = self.mapping_func(-weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
#         return weight
#
#
#         batch, dim = fm.shape[0], fm.shape[1]
#         height, width = fm.shape[2], fm.shape[3]
#         phase_width = width // self.Tem
#         # 分割特征图
#         phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
#         # 计算每个时相的行权重和列权重
#         weights = []
#         # 使用余弦相似度计算每个时相的行权重和列权重
#         for phase_feature in phase_features:
#             weight_row = torch.cosine_similarity(
#                 phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
#                 phase_feature[:, :, 1:, :].reshape(batch, dim, -1),
#                 dim=1
#             )
#             weight_col = torch.cosine_similarity(
#                 phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
#                 phase_feature[:, :, :, 1:].reshape(batch, dim, -1),
#                 dim=1
#             )
#             weights.extend([weight_row, weight_col])
#
#         # 计算跨时相权重
#         for i in range(self.Tem - 1):
#             weight_cross = torch.cosine_similarity(
#                 phase_features[i].reshape(batch, dim, -1),
#                 phase_features[i + 1].reshape(batch, dim, -1),
#                 dim=1
#             )
#             weights.append(weight_cross)
#         # 合并权重
#         weight = torch.cat(weights, dim=1)
#
#         if self.mapping_func is not None:
#             if max_tree:
#                 weight = self.mapping_func(weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
#             else:
#                 weight = self.mapping_func(-weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
#         return weight
#
#
#     def forward(self, guide_in, max_tree=False):
#         with torch.no_grad():
#             index = self._build_matrix_index(guide_in, 3)
#             if self.distance_func == "Cosine":
#                 weight = self._build_feature_weight_consine(guide_in, max_tree)
#             elif self.distance_func == "norm2_distance":
#                 weight = self._build_feature_weight_norm2_distance(guide_in)
#             elif self.distance_func == "ED":
#                 weight = self._build_feature_weight_ED(guide_in)
#             else:
#                 weight = self._build_feature_weight_norm1_distance(guide_in)
#                 print('norm1_distance!')
#             tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
#             # tree = mst(index, weight, guide_in.shape[2])
#         return tree
class MinimumSpanningMTnDTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None, Tem=6):
        super(MinimumSpanningMTnDTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
        self.Tem = Tem

    @staticmethod
    def _build_matrix_index(fm, Tem):
        batch, height, width = fm.shape[0], fm.shape[2], fm.shape[3]
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width

        # 动态分割时相
        phase_width = width // Tem
        phase_indices = [raw_index[:, i * phase_width:(i + 1) * phase_width] for i in range(Tem)]

        # 计算每个时相的行索引和列索引
        row_indices = []
        col_indices = []
        for phase_index in phase_indices:
            row_index = torch.stack([phase_index[:-1, :], phase_index[1:, :]], dim=2)
            col_index = torch.stack([phase_index[:, :-1], phase_index[:, 1:]], dim=2)
            row_indices.append(row_index.reshape(1, -1, 2))
            col_indices.append(col_index.reshape(1, -1, 2))

        # 计算跨时相的索引
        cross_indices = []
        for i in range(Tem - 1):
            cross_index = torch.stack([phase_indices[i][:, :], phase_indices[i + 1][:, :]], dim=2)
            cross_indices.append(cross_index.reshape(1, -1, 2))

        # 合并所有索引
        index = torch.cat(row_indices + col_indices + cross_indices, dim=1)
        index = index.expand(batch, -1, -1)

        return index

    def _build_feature_weight_ED(self, fm):
        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem
        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
        # 计算每个时相的行权重和列权重
        weights = []
        # 使用欧式距离计算每个时相的行权重和列权重
        pdist = nn.PairwiseDistance(p=2)
        for phase_feature in phase_features:
            weight_row = pdist(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1)
            )
            weight_col = pdist(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1)
            )
            weights.extend([weight_row, weight_col])
        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = pdist(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1)
            )
            weights.append(weight_cross)

        weight = torch.cat(weights, dim=1)
        return weight
    def _build_feature_weight_consine(self, fm, max_tree):
        batch, dim = fm.shape[0], fm.shape[1]
        height, width = fm.shape[2], fm.shape[3]
        phase_width = width // self.Tem
        # 分割特征图
        phase_features = [fm[:, :, :, i * phase_width:(i + 1) * phase_width] for i in range(self.Tem)]
        # 计算每个时相的行权重和列权重
        weights = []
        # 使用余弦相似度计算每个时相的行权重和列权重
        for phase_feature in phase_features:
            weight_row = torch.cosine_similarity(
                phase_feature[:, :, :-1, :].reshape(batch, dim, -1),
                phase_feature[:, :, 1:, :].reshape(batch, dim, -1),
                dim=1
            )
            weight_col = torch.cosine_similarity(
                phase_feature[:, :, :, :-1].reshape(batch, dim, -1),
                phase_feature[:, :, :, 1:].reshape(batch, dim, -1),
                dim=1
            )
            weights.extend([weight_row, weight_col])

        # 计算跨时相权重
        for i in range(self.Tem - 1):
            weight_cross = torch.cosine_similarity(
                phase_features[i].reshape(batch, dim, -1),
                phase_features[i + 1].reshape(batch, dim, -1),
                dim=1
            )
            weights.append(weight_cross)
        # 合并权重
        weight = torch.cat(weights, dim=1)

        if self.mapping_func is not None:
            if max_tree:
                weight = self.mapping_func(weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
            else:
                weight = self.mapping_func(-weight)  # cosine similarity needs "-weight" for min tree, "weight" for max tree
        return weight

    def forward(self, guide_in, max_tree=False):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in, 3)
            if self.distance_func == "Cosine":
                weight = self._build_feature_weight_consine(guide_in, max_tree)
            else:
                weight = self._build_feature_weight_ED(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
            # tree = mst(index, weight, guide_in.shape[2])
        return tree

    #     if self.mapping_func is not None:
    #         weight = self.mapping_func(weight)
    #     return weight
    #
    # def forward(self, guide_in, max_tree=False):
    #     with torch.no_grad():
    #         index = self._build_matrix_index(guide_in, self.Tem)
    #         weight = self._build_feature_weight(guide_in)
    #         tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
    #     return tree