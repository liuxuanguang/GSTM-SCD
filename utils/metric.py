import math
import os.path
from scipy.stats import hmean
import numpy as np

import itertools
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils.colormap import heatmap, annotate_heatmap

def cal_kappa(hist):
    if hist.sum() == 0:
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class IOUandSek:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):

        mask = (label_pred >= 0) & (label_pred < self.num_classes)

        # 检查 mask 后的长度
        true_len = len(label_true[mask])
        pred_len = len(label_pred[mask])
        label_l = label_true[mask].astype(int)
        label_p = label_pred[mask]
        # 检查是否相等且非零
        assert true_len == pred_len and true_len > 0, "Label lengths after masking do not match or are zero"

        # 检查最大索引值
        max_index = max(self.num_classes * label_true[mask].astype(int) + label_pred[mask])
        max_index_tensor = self.num_classes * label_true[mask].astype(int) + label_pred[mask]
        # print(f"Max index: {max_index}, Expected max: {self.num_classes ** 2 - 1}")

        # 确保 num_classes 是你期望的值
        # print(f"Number of classes: {self.num_classes}")

        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            # print((lp.flatten()).shape, (lt.flatten()).shape)
            # print((lt.flatten()).shape)
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def reset(self):

        """重置统计数据"""

        self.hist = np.zeros((self.num_classes, self.num_classes))

    def color_map_WUSU(self, path):
        ax = plt.plot()
        # y = ['Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']
        # x = ['Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']
        y = ['nochange', 'Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']
        x = ['nochange', 'Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_WUSU.png')
        plt.savefig(save_path, transparent=True, dpi=800)
    def color_map_SECOND(self, path):
        ax = plt.plot()
        # y = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        # x = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        y = ['Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        x = ['Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        # confusion = np.array(self.hist, dtype=int)
        # confusion[0][0] = 0
        confusion = np.array(self.hist[1:, 1:], dtype=int)
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_SECOND.png')
        plt.savefig(save_path, transparent=True, dpi=800)


    def color_map_FZSCD(self, path):
        ax = plt.plot()
        y = ['No change', 'Bare', 'Building', 'Vegetation', 'Water', 'Road', 'others']
        x = ['No change', 'Bare', 'Building', 'Vegetation', 'Water', 'Road', 'others']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_SECOND.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_DynamicEarth(self, path):
        ax = plt.plot()
        # y = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        # x = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        y = ['nochange', 'bg', 'impervious surface', 'agriculture', 'forest & other vegetation', 'wetlands', 'soil', 'water', 'snow & ice']
        x = ['nochange', 'bg', 'impervious surface', 'agriculture', 'forest & other vegetation', 'wetlands', 'soil', 'water', 'snow & ice']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_DynamicEarth.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_Landsat_SCD(self, path):
        ax = plt.plot()
        y = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
        x = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_LandSat_SCD.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_HRSCD(self):
        ax = plt.plot()
        y = ['No change', 'Artificial surfaces', 'Agricultural area', 'Forests', 'Wetlands', 'Water']
        x = ['No change', 'Artificial surfaces', 'Agricultural area', 'Forests', 'Wetlands', 'Water']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", size=6, threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = '/disk527/Datadisk/b527_cfz/SenseEarth2020-ChangeDetection/utils/method_1.png'
        plt.savefig(save_path, transparent=True, dpi=800)

    def evaluate(self):
        hist = self.hist
        TN, FP, FN, TP = hist[0][0], hist[1][0], hist[0][1], hist[1][1]
        pr = TP / (TP + FP)    # precision
        re = TP / (TP + FN)    # recall
        F1 = 2*pr*re / (pr + re)
        return F1
        
    def evaluate_SECOND(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)

        hist = self.hist.copy()
        OA = (np.diag(hist).sum()) / (hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        if change_pred_sum == 0:
            SC_Precision = 0
        SC_Recall = SC_TP / change_label_sum
        if change_label_sum == 0:
            SC_Recall = 0

        Fscd = stats.hmean([SC_Precision, SC_Recall])

        return change_ratio, score, miou, sek, Fscd, OA, SC_Precision, SC_Recall

    def evaluate_classification(self):
        """
        计算土地覆盖分类的评价指标

        返回:
            miou (float): 平均交并比 (mIoU)
            oa (float): 总体精度 (Overall Accuracy)
            f1 (float): 宏平均F1分数
            precision (float): 宏平均精确率
            recall (float): 宏平均召回率
        """
        # 计算标准分类指标
        hist = self.hist.copy()  # 使用完整的混淆矩阵

        # 1. 总体精度 (OA)
        oa = np.diag(hist).sum() / hist.sum()

        # 2. 计算每个类别的IoU
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        miou = np.nanmean(iou)  # 平均交并比 (mIoU)

        # 3. 计算每个类别的精确率和召回率
        precision_per_class = np.diag(hist) / (hist.sum(0) + 1e-10)
        recall_per_class = np.diag(hist) / (hist.sum(1) + 1e-10)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-10)

        # 4. 计算宏平均指标
        precision = np.nanmean(precision_per_class)  # 宏平均精确率
        recall = np.nanmean(recall_per_class)  # 宏平均召回率
        f1 = np.nanmean(f1_per_class)  # 宏平均F1分数

        return miou, oa, f1, precision, recall

    def evaluate_WUSU(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)

        hist = self.hist.copy()
        OA = (np.diag(hist).sum()) / (hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        if change_pred_sum == 0:
            SC_Precision = 0
        SC_Recall = SC_TP / change_label_sum
        if change_label_sum == 0:
            SC_Recall = 0

        Fscd = stats.hmean([SC_Precision, SC_Recall])

        return miou, Fscd, OA

    def evaluate_BCD1(self):
        """
        计算双时相建筑物变化检测的精度评价指标。
        返回: Recall, Precision, OA, F1, IoU, KC
        """
        # 确保混淆矩阵是2x2的
        if self.hist.shape != (2, 2):
            # 如果不是2x2，尝试提取前2x2部分
            if self.hist.shape[0] >= 2 and self.hist.shape[1] >= 2:
                hist = self.hist[:2, :2]
            else:
                # 无法计算，返回0
                return 0, 0, 0, 0, 0, 0
        else:
            hist = self.hist

        # 直接使用混淆矩阵的值
        TN = hist[0][0]
        FP = hist[0][1]
        FN = hist[1][0]
        TP = hist[1][1]

        # 计算召回率 (Recall)
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 计算精确度 (Precision)
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # 计算总体精度 (OA)
        OA = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

        # 计算 F1 分数
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

        # 计算 IoU (Intersection over Union)
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        # 计算 Kappa 系数 (KC)
        total = TP + FP + FN + TN
        p_o = OA  # 观察一致性
        p_e = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total ** 2) if total > 0 else 0
        KC = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0

        return Recall, Precision, OA, F1, IoU, KC


    def evaluate_BCD(self):
        """
        计算双时相建筑物变化检测的精度评价指标。
        返回: Recall, Precision, OA, F1, IoU, KC
        """
        # 构建混淆矩阵
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]  # TN (True Negative)
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]  # FP (False Positive)
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]  # FN (False Negative)
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()  # TP (True Positive)

        # 计算召回率 (Recall)
        TP = confusion_matrix[1][1]
        FN = confusion_matrix[1][0]
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 计算精确度 (Precision)
        FP = confusion_matrix[0][1]
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # 计算总体精度 (OA)
        OA = (TP + confusion_matrix[0][0]) / confusion_matrix.sum()

        # 计算 F1 分数
        F1 = hmean([Precision, Recall]) if (Precision > 0 and Recall > 0) else 0

        # 计算 IoU (Intersection over Union)
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        # 计算 Kappa 系数 (KC)
        total = confusion_matrix.sum()
        p_o = OA  # 观察一致性
        p_e = ((TP + FP) * (TP + FN) + (FN + confusion_matrix[0][0]) * (FP + confusion_matrix[0][0])) / (
                    total ** 2)  # 随机一致性
        KC = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0

        return Recall, Precision, OA, F1, IoU, KC


    def evaluate_inference(self):
        ''' BCD '''
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)
        TN, FP, FN, TP = confusion_matrix[0][0], confusion_matrix[1][0], confusion_matrix[0][1], confusion_matrix[1][1]
        pr = TP / (TP + FP)    # precision
        re = TP / (TP + FN)    # recall
        F1 = 2*pr*re / (pr + re)

        ''' SCD '''
        hist = self.hist.copy()
        oa = (np.diag(hist).sum())/(hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        if change_pred_sum == 0:
            SC_Precision = 0
        SC_Recall = SC_TP / change_label_sum
        if change_label_sum == 0:
            SC_Recall = 0
        Fscd = stats.hmean([SC_Precision, SC_Recall])
        # acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        # return change_ratio, score, miou, sek, Fscd, oa, iou[1], F1, kappa, pr, re

        return change_ratio, oa, miou, sek, Fscd, score, SC_Precision, SC_Recall

    def miou(self):
        confusion_matrix = self.hist[1:, 1:]
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))
        return iou, np.mean(iou)
