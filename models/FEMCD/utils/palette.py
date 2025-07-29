import numpy as np
import os
import shutil
from PIL import Image
from tqdm import tqdm


def color_map():
    cmap = np.zeros((7, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 255])
    cmap[2] = np.array([128, 128, 128])
    cmap[3] = np.array([0, 128, 0])
    cmap[4] = np.array([0, 255, 0])
    cmap[5] = np.array([128, 0, 0])
    cmap[6] = np.array([255, 0, 0])

    return cmap

def color_map_WUSU13():
    cmap = np.zeros((13, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 0])
    cmap[2] = np.array([255, 211, 127])
    cmap[3] = np.array([255, 0, 0])
    cmap[4] = np.array([255, 115, 127])
    cmap[5] = np.array([255, 255, 255])
    cmap[6] = np.array([38, 115, 0])
    cmap[7] = np.array([38, 115, 0])
    cmap[8] = np.array([170, 255, 0])
    cmap[9] = np.array([0, 197, 255])
    cmap[10] = np.array([0, 92, 230])
    cmap[11] = np.array([0, 255, 197])
    cmap[12] = np.array([197, 0, 255])

    return cmap

def color_map_WUSU12():
    cmap = np.zeros((12, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 0])
    cmap[2] = np.array([255, 211, 127])
    cmap[3] = np.array([255, 0, 0])
    cmap[4] = np.array([255, 115, 127])
    cmap[5] = np.array([38, 115, 0])
    cmap[6] = np.array([38, 115, 0])
    cmap[7] = np.array([170, 255, 0])
    cmap[8] = np.array([0, 197, 255])
    cmap[9] = np.array([0, 92, 230])
    cmap[10] = np.array([0, 255, 197])
    cmap[11] = np.array([197, 0, 255])

    return cmap


def color_map_DynamicEarth():
    cmap = np.zeros((8, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([130, 130, 130])
    cmap[2] = np.array([240, 236, 64])
    cmap[3] = np.array([33, 189, 0])
    cmap[4] = np.array([19, 0, 230])
    cmap[5] = np.array([166, 88, 0])
    cmap[6] = np.array([25, 198, 255])
    cmap[7] = np.array([128, 215, 255])
    return cmap



def color_map_HRSCD():
    cmap = np.zeros((6, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 255])
    cmap[2] = np.array([128, 128, 128])
    cmap[3] = np.array([0, 128, 0])
    cmap[4] = np.array([0, 255, 0])
    cmap[5] = np.array([128, 0, 0])
    # cmap[6] = np.array([255, 0, 0])

    return cmap


def color_map_Landsat_SCD():
    cmap = np.zeros((5, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 155, 0])
    cmap[2] = np.array([255, 165, 0])
    cmap[3] = np.array([230, 30, 100])
    cmap[4] = np.array([0, 170, 240])

    return cmap
