import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import cv2
import random
import re
from datetime import datetime, timedelta
import tifffile
from skimage import io


def extract_location_id(filename):
    """从文件名中提取位置ID（例如1311_3077_13）"""
    pattern = r'(\d+_\d+_\d+)_\d{4}-\d{2}-\d{2}\.tif'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def find_matching_files(base_dir, location_id):
    """在目录中查找匹配位置ID的所有文件"""
    matching_files = []
    for filename in os.listdir(base_dir):
        if location_id in filename and filename.endswith('.tif'):
            matching_files.append(filename)
    return sorted(matching_files)  # 按文件名排序确保时间顺序


def read_image(image_path):
    """读取单波段灰度图、RGB影像或四波段遥感影像"""
    try:
        # 使用tifffile读取图像
        img = tifffile.imread(image_path)

        # 处理不同维度的图像
        if img.ndim == 2:  # 单波段灰度图
            img = np.expand_dims(img, axis=0)  # 增加通道维度
        elif img.ndim == 3:  # 多波段图像
            # 将通道维度放在最前面 (channels, height, width)
            img = np.transpose(img, (2, 0, 1))

        return img.astype('float32')

    except Exception as e:
        print(f"读取图像 {image_path} 时出错: {str(e)}")
        raise


def rand_rot90_flip_MCD(img_A, img_B, img_C, img_D, img_E, img_F, label_A, label_B, label_C, label_D, label_E, label_F,
                        label_cd):
    """对4波段影像和灰度标签进行同步数据增强"""
    angle = random.choice([0, 90, 180, 270])
    flip_type = random.choice(['none', 'horizontal', 'vertical'])

    def apply_transform(data, is_label=False):
        if is_label:
            data = np.expand_dims(data, axis=0)

        # 旋转操作
        if angle == 90:
            data = np.rot90(data, k=1, axes=(1, 2))
        elif angle == 180:
            data = np.rot90(data, k=2, axes=(1, 2))
        elif angle == 270:
            data = np.rot90(data, k=3, axes=(1, 2))

        # 翻转操作
        if flip_type == 'horizontal':
            data = np.flip(data, axis=2)
        elif flip_type == 'vertical':
            data = np.flip(data, axis=1)

        if is_label:
            data = np.squeeze(data, axis=0)

        # 确保数组在内存中是连续的
        return np.ascontiguousarray(data)

    img_A = apply_transform(img_A)
    img_B = apply_transform(img_B)
    img_C = apply_transform(img_C)
    img_D = apply_transform(img_D)
    img_E = apply_transform(img_E)
    img_F = apply_transform(img_F)

    label_A = apply_transform(label_A, is_label=True)
    label_B = apply_transform(label_B, is_label=True)
    label_C = apply_transform(label_C, is_label=True)
    label_D = apply_transform(label_D, is_label=True)
    label_E = apply_transform(label_E, is_label=True)
    label_F = apply_transform(label_F, is_label=True)
    label_cd = apply_transform(label_cd, is_label=True)

    return img_A, img_B, img_C, img_D, img_E, img_F, label_A, label_B, label_C, label_D, label_E, label_F, label_cd


# 常量定义保持不变
num_classes = 8
ST_COLORMAP = [[197, 0, 255], [255, 219, 88], [128, 128, 0], [0, 0, 128], [38, 115, 0], [165, 42, 42],
               [176, 226, 255]]
ST_CLASSES = ['background', 'impervious surface', 'agriculture', 'forest & other vegetation', 'wetlands', 'soil',
              'water', 'snow & ice']

MEAN_A = np.array([184.530226, 245.369933, 287.284581, 636.725934])
STD_A = np.array([457.925692, 575.479819, 716.003299, 1233.677155])

MEAN_B = np.array([182.348677, 257.678226, 295.770668, 707.539389])
STD_B = np.array([447.063670, 604.456624, 744.189421, 1341.465072])

MEAN_C = np.array([179.188286, 245.856369, 282.507839, 659.767521])
STD_C = np.array([471.656230, 596.523909, 732.180482, 1273.457777])

MEAN_D = np.array([177.473588, 238.493912, 277.326044, 616.301305])
STD_D = np.array([429.244153, 549.205164, 679.699752, 1197.202361])

MEAN_E = np.array([183.249813, 254.955249, 294.087111, 697.785509])
STD_E = np.array([458.778723, 591.724693, 730.240061, 1325.141909])

MEAN_F = np.array([176.202393, 241.355481, 277.609658, 657.072685])
STD_F = np.array([438.080904, 561.792241, 699.111058, 1262.025594])

root = '/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/随机时相测试/'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def normalize_image(im, time='A'):
    """
    对影像进行归一化。
    Args:
        im (np.ndarray): 输入影像，形状为 (4, H, W)。
        time (str): 影像时间（'A', 'B', 'C'），用于选择均值和标准差。
    Returns:
        np.ndarray: 归一化后的影像。
    """
    assert time in ['A', 'B', 'C', 'D', 'E', 'F']

    if time == 'A':
        mean = MEAN_A[:, np.newaxis, np.newaxis]
        std = STD_A[:, np.newaxis, np.newaxis]
    elif time == 'B':
        mean = MEAN_B[:, np.newaxis, np.newaxis]
        std = STD_B[:, np.newaxis, np.newaxis]
    elif time == 'C':
        mean = MEAN_C[:, np.newaxis, np.newaxis]
        std = STD_C[:, np.newaxis, np.newaxis]
    elif time == 'D':
        mean = MEAN_D[:, np.newaxis, np.newaxis]
        std = STD_D[:, np.newaxis, np.newaxis]
    elif time == 'E':
        mean = MEAN_E[:, np.newaxis, np.newaxis]
        std = STD_E[:, np.newaxis, np.newaxis]
    else:
        mean = MEAN_F[:, np.newaxis, np.newaxis]
        std = STD_F[:, np.newaxis, np.newaxis]

    # 归一化
    im = (im - mean) / std
    return im


def read_RSimages(mode):
    """读取六个时相的影像和标签"""
    # 基础目录
    base_dir = os.path.join(root, mode)

    # 影像和标签目录
    img_dirs = [
        os.path.join(base_dir, 'im1'),
        os.path.join(base_dir, 'im2'),
        os.path.join(base_dir, 'im3'),
        os.path.join(base_dir, 'im4'),
        os.path.join(base_dir, 'im5'),
        os.path.join(base_dir, 'im6')
    ]

    label_dirs = [
        os.path.join(base_dir, 'label1'),
        os.path.join(base_dir, 'label2'),
        os.path.join(base_dir, 'label3'),
        os.path.join(base_dir, 'label4'),
        os.path.join(base_dir, 'label5'),
        os.path.join(base_dir, 'label6')
    ]

    # 获取所有位置ID（从第一个时相目录）
    location_ids = set()
    for filename in os.listdir(img_dirs[0]):
        if filename.endswith('.tif'):
            location_id = extract_location_id(filename)
            if location_id:
                location_ids.add(location_id)

    # 为每个位置ID收集六个时相的文件路径
    all_img_paths = []
    all_label_paths = []
    all_labels_cd = []

    for location_id in location_ids:
        img_paths = []
        label_paths = []

        # 为每个时相收集匹配的文件
        for i in range(6):
            img_dir = img_dirs[i]
            label_dir = label_dirs[i]

            # 查找匹配的文件
            img_files = find_matching_files(img_dir, location_id)
            label_files = find_matching_files(label_dir, location_id)

            # 确保每个时相只有一个匹配文件
            if len(img_files) == 1 and len(label_files) == 1:
                img_paths.append(os.path.join(img_dir, img_files[0]))
                label_paths.append(os.path.join(label_dir, label_files[0]))
            else:
                print(
                    f"警告: 位置 {location_id} 在时相 {i + 1} 有 {len(img_files)} 个影像文件和 {len(label_files)} 个标签文件")
                break

        # 如果收集到六个时相的文件，添加到结果列表
        if len(img_paths) == 6:
            all_img_paths.append(img_paths)
            all_label_paths.append(label_paths)

            # 读取第一个和最后一个时相的标签来计算变化
            label_first = io.imread(label_paths[0])
            label_last = io.imread(label_paths[-1])

            # 确保标签是二维数组
            if label_first.ndim == 3:
                label_first = label_first[:, :, 0]  # 取第一个通道
            if label_last.ndim == 3:
                label_last = label_last[:, :, 0]  # 取第一个通道

            label_cd = label_last - label_first
            label_cd[label_cd != 0] = 1
            all_labels_cd.append(label_cd)

    print(f"{len(all_img_paths)} {mode} 位置加载完成，每个位置有6个时相")

    # 重组数据结构：每个时相一个列表
    imgs_list_A, imgs_list_B, imgs_list_C, imgs_list_D, imgs_list_E, imgs_list_F = [], [], [], [], [], []
    labels_A, labels_B, labels_C, labels_D, labels_E, labels_F = [], [], [], [], [], []

    for img_paths, label_paths in zip(all_img_paths, all_label_paths):
        imgs_list_A.append(img_paths[0])
        imgs_list_B.append(img_paths[1])
        imgs_list_C.append(img_paths[2])
        imgs_list_D.append(img_paths[3])
        imgs_list_E.append(img_paths[4])
        imgs_list_F.append(img_paths[5])

        # 读取并处理标签
        def process_label(path):
            label = io.imread(path)
            if label.ndim == 3:
                # 如果是RGB标签，转换为类别索引
                return Color2Index(label)
            return label

        labels_A.append(process_label(label_paths[0]))
        labels_B.append(process_label(label_paths[1]))
        labels_C.append(process_label(label_paths[2]))
        labels_D.append(process_label(label_paths[3]))
        labels_E.append(process_label(label_paths[4]))
        labels_F.append(process_label(label_paths[5]))

    return (imgs_list_A, imgs_list_B, imgs_list_C, imgs_list_D, imgs_list_E, imgs_list_F,
            labels_A, labels_B, labels_C, labels_D, labels_E, labels_F, all_labels_cd)


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        (self.imgs_list_A, self.imgs_list_B, self.imgs_list_C,
         self.imgs_list_D, self.imgs_list_E, self.imgs_list_F,
         self.labels_A, self.labels_B, self.labels_C,
         self.labels_D, self.labels_E, self.labels_F,
         self.labels_cd) = read_RSimages(mode)

    def get_mask_name(self, idx):
        return os.path.basename(self.imgs_list_A[idx])

    def __getitem__(self, idx):
        img_A = read_image(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')

        img_B = read_image(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')

        img_C = read_image(self.imgs_list_C[idx])
        img_C = normalize_image(img_C, 'C')

        img_D = read_image(self.imgs_list_D[idx])
        img_D = normalize_image(img_D, 'D')

        img_E = read_image(self.imgs_list_E[idx])
        img_E = normalize_image(img_E, 'E')

        img_F = read_image(self.imgs_list_F[idx])
        img_F = normalize_image(img_F, 'F')

        # 获取图像名称
        img_name = os.path.basename(self.imgs_list_A[idx])

        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        label_C = self.labels_C[idx]
        label_D = self.labels_D[idx]
        label_E = self.labels_E[idx]
        label_F = self.labels_F[idx]
        label_cd = self.labels_cd[idx]

        if self.random_flip:
            (img_A, img_B, img_C, img_D, img_E, img_F,
             label_A, label_B, label_C, label_D, label_E, label_F,
             label_cd) = rand_rot90_flip_MCD(
                img_A, img_B, img_C, img_D, img_E, img_F,
                label_A, label_B, label_C, label_D, label_E, label_F,
                label_cd)

        return (img_A, img_B, img_C, img_D, img_E, img_F,
                label_A, label_B, label_C, label_D, label_E, label_F,
                label_cd, img_name)

    def __len__(self):
        return len(self.imgs_list_A)