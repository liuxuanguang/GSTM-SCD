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
import re
from datetime import datetime, timedelta


def increase_month(date_str, months_to_add):
    # 将字符串转换为datetime对象
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # 计算新的月份和年份
    new_month = date_obj.month + months_to_add
    new_year = date_obj.year + (new_month - 1) // 12
    new_month = (new_month - 1) % 12 + 1

    # 创建新的datetime对象
    new_date_obj = date_obj.replace(year=new_year, month=new_month)

    # 将新的datetime对象转换回字符串
    new_date_str = new_date_obj.strftime("%Y-%m-%d")

    return new_date_str


def process_string(input_str, months_to_add):
    # 使用正则表达式提取日期部分
    match = re.search(r"\d{4}-\d{2}-\d{2}", input_str)
    if match:
        date_str = match.group(0)
        new_date_str = increase_month(date_str, months_to_add)

        # 替换原字符串中的日期部分
        new_str = input_str.replace(date_str, new_date_str)
        return new_str
    else:
        return "日期未找到"





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

def rand_rot90_flip_MCD(img_A, img_B, img_C, img_D, img_E, img_F, label_A, label_B, label_C, label_D, label_E, label_F, label_cd):
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
    img_D = apply_transform(img_D)
    img_E = apply_transform(img_E)
    img_F = apply_transform(img_F)
    # 对标签A、B、C和变化检测标签进行增强
    label_A = apply_transform(label_A, is_label=True)
    label_B = apply_transform(label_B, is_label=True)
    label_C = apply_transform(label_C, is_label=True)
    label_D = apply_transform(label_D, is_label=True)
    label_E = apply_transform(label_E, is_label=True)
    label_F = apply_transform(label_F, is_label=True)
    label_cd = apply_transform(label_cd, is_label=True)

    return img_A, img_B, img_C, img_D, img_E, img_F, label_A, label_B, label_C, label_D, label_E, label_F, label_cd

num_classes = 8
ST_COLORMAP = [[197, 0, 255], [255, 219, 88], [128, 128, 0], [0, 0, 128], [38, 115, 0], [165, 42, 42], [176, 226, 255]]
ST_CLASSES = ['background', 'impervious surface', 'agriculture', 'forest & other vegetation', 'wetlands', 'soil', 'water', 'snow & ice']

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
root = '/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/'
# root = '/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/随机时相测试/'
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
    assert time in ['A', 'B', 'C', 'D', 'E', 'F']

    if time == 'A':
        mean = MEAN_A[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_A[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
    elif time == 'B':
        mean = MEAN_B[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_B[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
    elif time == 'C':
        mean = MEAN_C[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_C[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
    elif time == 'D':
        mean = MEAN_D[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_D[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
    elif time == 'E':
        mean = MEAN_E[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_E[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
    else:
        mean = MEAN_F[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)
        std = STD_F[:, np.newaxis, np.newaxis]  # 将 (4,) 扩展为 (4, 1, 1)

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
    img_D_dir = os.path.join(root, mode, 'im4')
    img_E_dir = os.path.join(root, mode, 'im5')
    img_F_dir = os.path.join(root, mode, 'im6')
    # NOT Use rgb labels:
    # label_A_dir = os.path.join(root, mode, 'labelA')
    # label_B_dir = os.path.join(root, mode, 'labelB')
    # To use rgb labels:
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    label_C_dir = os.path.join(root, mode, 'label3')
    label_D_dir = os.path.join(root, mode, 'label4')
    label_E_dir = os.path.join(root, mode, 'label5')
    label_F_dir = os.path.join(root, mode, 'label6')
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, imgs_list_C, imgs_list_D, imgs_list_E, imgs_list_F, labels_A, labels_B, labels_C, labels_D, labels_E, labels_F, labels_cd = [], [], [], [], [], [], [],[], [], [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:] == '.tif'):
            img_A_path = os.path.join(img_A_dir, it)
            new_strB = process_string(it, 4)
            img_B_path = os.path.join(img_B_dir, new_strB)
            new_strC = process_string(it, 8)
            img_C_path = os.path.join(img_C_dir, new_strC)
            new_strD = process_string(it, 12)
            img_D_path = os.path.join(img_D_dir, new_strD)
            new_strE = process_string(it, 16)
            img_E_path = os.path.join(img_E_dir, new_strE)
            new_strF = process_string(it, 20)
            img_F_path = os.path.join(img_F_dir, new_strF)

            label_A_path = os.path.join(label_A_dir, it)
            new_label_B = process_string(it, 4)
            label_B_path = os.path.join(label_B_dir, new_label_B)
            new_label_C = process_string(it, 8)
            label_C_path = os.path.join(label_C_dir, new_label_C)
            new_label_D = process_string(it, 12)
            label_D_path = os.path.join(label_D_dir, new_label_D)
            new_label_E = process_string(it, 16)
            label_E_path = os.path.join(label_E_dir, new_label_E)
            new_label_F = process_string(it, 20)
            label_F_path = os.path.join(label_F_dir, new_label_F)

            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            imgs_list_C.append(img_C_path)
            imgs_list_D.append(img_D_path)
            imgs_list_E.append(img_E_path)
            imgs_list_F.append(img_F_path)
            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)
            label_C = io.imread(label_C_path)
            label_D = io.imread(label_D_path)
            label_E = io.imread(label_E_path)
            label_F = io.imread(label_F_path)
            label_cd = label_F - label_A
            label_cd[label_cd != 0] = 1
            # for rgb labels:
            # label_A = Color2Index(label_A)
            # label_B = Color2Index(label_B)
            # label_C = Color2Index(label_C)
            labels_A.append(label_A)
            labels_B.append(label_B)
            labels_C.append(label_C)
            labels_D.append(label_D)
            labels_E.append(label_E)
            labels_F.append(label_F)
            labels_cd.append(label_cd)
        count += 1
        if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))

    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_A, imgs_list_B, imgs_list_C, imgs_list_D, imgs_list_E, imgs_list_F, labels_A, labels_B, labels_C, labels_D, labels_E, labels_F, labels_cd


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.imgs_list_C, self.imgs_list_D, self.imgs_list_E, self.imgs_list_F, self.labels_A, self.labels_B, self.labels_C, self.labels_D, self.labels_E, self.labels_F, self.labels_cd = read_RSimages(mode)

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
        img_D = read_image(self.imgs_list_D[idx])
        img_D = normalize_image(img_D, 'D')
        img_E = read_image(self.imgs_list_E[idx])
        img_E = normalize_image(img_E, 'E')
        img_F = read_image(self.imgs_list_F[idx])
        img_F = normalize_image(img_F, 'F')
        # 获取图像名称
        img_name_A = os.path.split(self.imgs_list_A[idx])[-1]

        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        label_C = self.labels_C[idx]
        label_D = self.labels_D[idx]
        label_E = self.labels_E[idx]
        label_F = self.labels_F[idx]
        label_cd = self.labels_cd[idx]
        if self.random_flip:
            img_A, img_B, img_C, img_D, img_E, img_F, label_A, label_B, label_C, label_D, label_E, label_F, label_cd = rand_rot90_flip_MCD(img_A, img_B, img_C, img_D, img_E, img_F, label_A, label_B, label_C, label_D, label_E, label_F, label_cd)
        # return F.to_tensor(img_A), F.to_tensor(img_B), F.to_tensor(img_C), torch.from_numpy(label_A), torch.from_numpy(label_B), torch.from_numpy(label_C), torch.from_numpy(label_cd)
        return img_A, img_B, img_C, img_D, img_E, img_F, label_A, label_B, label_C, label_D, label_E, label_F, label_cd, img_name_A

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
#
# import os
# import numpy as np
# import torch
# from skimage import io
# from torch.utils import data
# import matplotlib.pyplot as plt
# from osgeo import gdal
# import random
# from PIL import Image
# from torchvision.transforms import functional as F
#
# # 定义常量
# num_classes = 8
# ST_COLORMAP = [[255, 255, 255], [197, 0, 255], [255, 219, 88], [128, 128, 0], [0, 0, 128], [38, 115, 0], [165, 42, 42], [176, 226, 255]]
# ST_CLASSES = ['nochange', 'impervious surface', 'agriculture', 'forest & other vegetation', 'wetlands', 'soil', 'water', 'snow & ice']
#
# MEAN_A = np.array([89.147606, 75.60075, 75.206795, 76.10553])
# STD_A = np.array([39.584747, 35.20891, 35.645096, 37.21572])
# MEAN_B = np.array([69.99051, 57.549652, 59.85536, 62.881676])
# STD_B = np.array([39.687336, 35.002705, 33.52335, 32.46179])
# MEAN_C = np.array([56.94247, 51.62498, 55.48882, 55.334343])
# STD_C = np.array([39.50115, 34.213947, 34.078976, 33.744305])
#
# root = '/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_process/train/DynamicEarth512/'
#
# colormap2label = np.zeros(256 ** 3)
# for i, cm in enumerate(ST_COLORMAP):
#     colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
#
# def read_image(image_path):
#     """
#     高效读取单波段灰度图、RGB影像或四波段遥感影像（RGB+NIR）。
#     参数:
#     image_path (str): 影像文件的路径。
#     返回:
#     img_data (numpy.ndarray): 影像数据。
#     bands (list): 波段名称列表。
#     """
#     gdal.AllRegister()
#     dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
#     if not dataset:
#         raise IOError("无法打开文件")
#     nBandCount = dataset.RasterCount
#     img_data = dataset.ReadAsArray().astype('float32')
#     dataset = None
#     return img_data
#
# def rand_rot90_flip_MCD(*args):
#     """
#     对多个影像和标签进行同步数据增强，包括随机旋转和翻转。
#     Args:
#         *args: 多个影像和标签，顺序为 img1, img2, ..., imgN, label1, label2, ..., labelN, label_cd。
#     Returns:
#         增强后的影像和标签。
#     """
#     angle = random.choice([0, 90, 180, 270])
#     flip_type = random.choice(['none', 'horizontal', 'vertical'])
#
#     def apply_transform(data, is_label=False):
#         if isinstance(data, np.ndarray):
#             data = torch.from_numpy(data)
#         if is_label:
#             data = data.unsqueeze(0)
#         if angle != 0:
#             data = F.rotate(data, angle)
#         if flip_type == 'horizontal':
#             data = F.hflip(data)
#         elif flip_type == 'vertical':
#             data = F.vflip(data)
#         if is_label:
#             data = data.squeeze(0)
#         return data
#
#     results = []
#     for i, arg in enumerate(args):
#         is_label = i >= len(args) // 2  # 后半部分是标签
#         results.append(apply_transform(arg, is_label))
#     return results
#
# def normalize_image(im, time='A'):
#     """
#     对影像进行归一化。
#     Args:
#         im (np.ndarray): 输入影像，形状为 (4, H, W)。
#         time (str): 影像时间（'A', 'B', 'C'），用于选择均值和标准差。
#     Returns:
#         np.ndarray: 归一化后的影像。
#     """
#     assert time in ['A', 'B', 'C']
#     mean = MEAN_A[:, np.newaxis, np.newaxis] if time == 'A' else MEAN_B[:, np.newaxis, np.newaxis] if time == 'B' else MEAN_C[:, np.newaxis, np.newaxis]
#     std = STD_A[:, np.newaxis, np.newaxis] if time == 'A' else STD_B[:, np.newaxis, np.newaxis] if time == 'B' else STD_C[:, np.newaxis, np.newaxis]
#     im = (im - mean) / std
#     return im
#
# def read_RSimages(mode, num_images=6):
#     """
#     读取多个影像和标签。
#     Args:
#         mode (str): 数据集模式（'train', 'val', 'test'）。
#         num_images (int): 影像数量（默认为 6）。
#     Returns:
#         imgs_list (list): 影像路径列表。
#         labels_list (list): 标签路径列表。
#         labels_cd (list): 变化检测标签列表。
#     """
#     imgs_list = []
#     labels_list = []
#     labels_cd = []
#
#     for i in range(1, num_images + 1):
#         img_dir = os.path.join(root, mode, f'im{i}')
#         label_dir = os.path.join(root, mode, f'label{i}')
#         data_list = os.listdir(img_dir)
#
#         for it in data_list:
#             if it.endswith('.tif'):
#                 img_path = os.path.join(img_dir, it)
#                 label_path = os.path.join(label_dir, it)
#                 imgs_list.append(img_path)
#                 label = io.imread(label_path)
#                 labels_list.append(label)
#                 if i == 1:
#                     label_cd = np.zeros_like(label)
#                 else:
#                     label_cd = label - labels_list[0]
#                     label_cd[label_cd != 0] = 1
#                 labels_cd.append(label_cd)
#
#     print(f'{len(imgs_list)} {mode} images loaded.')
#     return imgs_list, labels_list, labels_cd
#
# class Data(data.Dataset):
#     def __init__(self, mode, random_flip=False, num_images=6):
#         self.random_flip = random_flip
#         self.num_images = num_images
#         self.imgs_list, self.labels_list, self.labels_cd = read_RSimages(mode, num_images)
#
#     def __getitem__(self, idx):
#         imgs = [read_image(self.imgs_list[idx + i * len(self.imgs_list) // self.num_images]) for i in range(self.num_images)]
#         labels = [self.labels_list[idx + i * len(self.labels_list) // self.num_images] for i in range(self.num_images)]
#         label_cd = self.labels_cd[idx]
#
#         if self.random_flip:
#             imgs_labels = rand_rot90_flip_MCD(*imgs, *labels, label_cd)
#             imgs = imgs_labels[:self.num_images]
#             labels = imgs_labels[self.num_images:-1]
#             label_cd = imgs_labels[-1]
#
#         return imgs, labels, label_cd
#
#     def __len__(self):
#         return len(self.imgs_list) // self.num_images