import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import datasets.transform_MT as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
# from osgeo import gdal_array
from osgeo import gdal
import cv2
from osgeo import gdal
import numpy as np
import random
from PIL import Image
def read_image(image_path):
    """
    高效读取单波段灰度图、RGB影像或四波段遥感影像（RGB+NIR）。

    参数:
    image_path (str): 影像文件的路径。

    返回:
    img_data (numpy.ndarray): 影像数据。
    bands (list): 波段名称列表。
    """
    # 注册所有驱动
    gdal.AllRegister()
    # 打开影像文件
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if not dataset:
        raise IOError("无法打开文件")
    # 获取影像的基本信息
    nBandCount = dataset.RasterCount  # 波段数
    nRows = dataset.RasterYSize  # 影像的高度（像元数目），行数
    nCols = dataset.RasterXSize  # 影像的宽度（像元数目），列数
    # 读取所有波段的数据
    img_data = dataset.ReadAsArray().astype('float32')  # 读取所有波段数据并转换为float32类型
    # 根据波段数量确定波段名称
    if nBandCount == 1:  # 灰度图
        bands = ['Gray']
    elif nBandCount == 3:  # RGB影像
        bands = ['Red', 'Green', 'Blue']
    elif nBandCount == 4:  # 四波段遥感影像（RGB+NIR）
        bands = ['Red', 'Green', 'Blue', 'NIR']
    else:
        raise ValueError("Unsupported number of bands. Only single band (grayscale), RGB, or RGB+NIR are supported.")
    # 关闭数据集
    dataset = None
    return img_data


import random
import torch
import torchvision.transforms.functional as F

def rand_rot90_flip_MCD(img_A, img_B, img_C, label_A, label_B, label_C, label_cd):
    """
    对4波段影像和灰度标签进行同步数据增强，包括随机旋转和翻转。
    Args:
        img_A (np.ndarray): 影像A，形状为 (4, H, W)。
        img_B (np.ndarray): 影像B，形状为 (4, H, W)。
        img_C (np.ndarray): 影像C，形状为 (4, H, W)。
        label_A (np.ndarray): 标签A，形状为 (H, W)。
        label_B (np.ndarray): 标签B，形状为 (H, W)。
        label_C (np.ndarray): 标签C，形状为 (H, W)。
        label_cd (np.ndarray): 变化检测标签，形状为 (H, W)。
    Returns:
        img_A, img_B, img_C, label_A, label_B, label_C, label_cd: 增强后的影像和标签。
    """
    # 随机旋转角度（0°, 90°, 180°, 270°）
    angle = random.choice([0, 90, 180, 270])

    # 随机翻转（水平或垂直）
    flip_type = random.choice(['none', 'horizontal', 'vertical'])

    # 对影像和标签进行相同的旋转和翻转操作
    def apply_transform(data, is_label=False):
        # 将 NumPy 数组转换为 PyTorch 张量
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # 如果是标签，增加一个通道维度
        if is_label:
            data = data.unsqueeze(0)  # 形状从 (H, W) 变为 (1, H, W)

        # 旋转操作
        if angle != 0:
            data = F.rotate(data, angle)

        # 翻转操作
        if flip_type == 'horizontal':
            data = F.hflip(data)
        elif flip_type == 'vertical':
            data = F.vflip(data)

        # 如果是标签，去掉通道维度
        if is_label:
            data = data.squeeze(0)  # 形状从 (1, H, W) 变为 (H, W)

        return data

    # 对影像A、B、C进行增强
    img_A = apply_transform(img_A)
    img_B = apply_transform(img_B)
    img_C = apply_transform(img_C)

    # 对标签A、B、C和变化检测标签进行增强
    label_A = apply_transform(label_A, is_label=True)
    label_B = apply_transform(label_B, is_label=True)
    label_C = apply_transform(label_C, is_label=True)
    label_cd = apply_transform(label_cd, is_label=True)

    return img_A, img_B, img_C, label_A, label_B, label_C, label_cd

num_classes = 13
ST_COLORMAP = [[255, 255, 255],[0, 0, 0],[255, 211, 127],[255, 0, 0],[255, 115, 127],[255,255,255], [38, 115, 0],[38, 115, 0],[170, 255, 0],[0, 197, 255],[0, 92, 230],[0, 255, 197],[197, 0, 255]]
ST_CLASSES = ['nochange', 'Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']

MEAN_A = np.array([89.147606, 75.60075, 75.206795, 76.10553])
STD_A = np.array([39.584747, 35.20891, 35.645096, 37.21572])
MEAN_B = np.array([69.99051, 57.549652, 59.85536, 62.881676])
STD_B = np.array([39.687336, 35.002705, 33.52335, 32.46179])
MEAN_C = np.array([56.94247, 51.62498, 55.48882, 55.334343])
STD_C = np.array([39.50115, 34.213947, 34.078976, 33.744305])
# root = '/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/debug_data'
root = '/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


# def normalize_image(im, time='A'):
#     assert time in ['A', 'B', 'C']
#     if time == 'A':
#         im = (im - MEAN_A) / STD_A
#     if time == 'B':
#         im = (im - MEAN_B) / STD_B
#     else:
#         im = (im - MEAN_C) / STD_C
#     return im
def normalize_image(im, time='A'):
    """
    对影像进行归一化。
    Args:
        im (np.ndarray): 输入影像，形状为 (4, H, W)。
        time (str): 影像时间（'A', 'B', 'C'），用于选择均值和标准差。
    Returns:
        np.ndarray: 归一化后的影像。
    """
    assert time in ['A', 'B', 'C']

    if time == 'A':
        mean = MEAN_A[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_A[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
    elif time == 'B':
        mean = MEAN_B[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_B[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
    else:
        mean = MEAN_C[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_C[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)

    # 归一化
    im = (im - mean) / std
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


def read_RSimages(mode, rescale=False):
    # assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    img_C_dir = os.path.join(root, mode, 'im3')
    # NOT Use rgb labels:
    # label_A_dir = os.path.join(root, mode, 'labelA')
    # label_B_dir = os.path.join(root, mode, 'labelB')
    # To use rgb labels:
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    label_C_dir = os.path.join(root, mode, 'label3')
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, imgs_list_C, labels_A, labels_B, labels_C, labels_cd = [], [], [], [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:] == '.tif'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            img_C_path = os.path.join(img_C_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            label_C_path = os.path.join(label_C_dir, it)
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            imgs_list_C.append(img_C_path)
            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)
            label_C = io.imread(label_C_path)
            label_cd = label_C - label_A
            label_cd[label_cd != 0] = 1
            # for rgb labels:
            # label_A = Color2Index(label_A)
            # label_B = Color2Index(label_B)
            # label_C = Color2Index(label_C)
            labels_A.append(label_A)
            labels_B.append(label_B)
            labels_C.append(label_C)
            labels_cd.append(label_cd)
        count += 1
        if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))

    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_A, imgs_list_B, imgs_list_C, labels_A, labels_B, labels_C, labels_cd


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.imgs_list_C, self.labels_A, self.labels_B, self.labels_C, self.labels_cd = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):

        img_A = read_image(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = read_image(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        img_C = read_image(self.imgs_list_C[idx])
        img_C = normalize_image(img_C, 'C')
        # 获取图像名称
        img_name_A = os.path.split(self.imgs_list_A[idx])[-1]

        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        label_C = self.labels_C[idx]
        label_cd = self.labels_cd[idx]
        if self.random_flip:
            img_A, img_B, img_C, label_A, label_B, label_C, label_cd = rand_rot90_flip_MCD(img_A, img_B, img_C, label_A, label_B, label_C, label_cd)
        # return F.to_tensor(img_A), F.to_tensor(img_B), F.to_tensor(img_C), torch.from_numpy(label_A), torch.from_numpy(label_B), torch.from_numpy(label_C), torch.from_numpy(label_cd)
        return img_A, img_B, img_C, label_A, label_B, label_C, label_cd, img_name_A

    def __len__(self):
        return len(self.imgs_list_A)


class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'im1')
        imgB_dir = os.path.join(test_dir, 'im2')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:] == '.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len