from datasets.augmentation import augmentation_compose
import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChangeDetection_SECOND(Dataset):
    #SECOND
    CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']
    # FZSCD
    # CLASSES = ['未变化区域', '水体', '地面', '植被', '建筑物']
    # WUSU
    # CLASSES = ['road','low-building', 'high-building', 'Arable', 'woodland', 'grassland', 'river', 'lake', 'structure', 'excavation', 'bare']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_SECOND, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))
        # 高斯平滑
        gaussian1 = cv2.GaussianBlur(img1, (3, 3), 0)
        gaussian2 = cv2.GaussianBlur(img2, (3, 3), 0)
        # 再通过拉普拉斯算子做边缘检测
        ed1 = cv2.Laplacian(gaussian1, -1, ksize=3)
        ed2 = cv2.Laplacian(gaussian2, -1, ksize=3)

        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))
        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0 ] = 1


        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'ed1': ed1, 'ed2': ed2, 'mask1': mask1, 'mask2': mask2,
                                        'gt_mask': mask_bin})
            img1, img2, ed1, ed2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['ed1'], sample['ed2'], sample['mask1'], \
                                                    sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)



class ChangeDetection_FZSCD(Dataset):

    # FZSCD
    CLASSES = ['未变化区域', '裸地', '建筑物', '植被', '水体', '道路', '其他']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_FZSCD, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]
        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id).replace('T1m','T2')))
        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id).replace('T1m','T1_label')))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id).replace('T1m','T2_label')))
        mask1[mask1 == 15] = 0
        mask2[mask2 == 15] = 0
        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                        'gt_mask': mask_bin})
            img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                                                    sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)

