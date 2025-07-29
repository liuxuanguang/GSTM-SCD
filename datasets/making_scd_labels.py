import os
from PIL import Image
import numpy as np
from tqdm import tqdm  # 用于显示进度条

# 定义文件夹路径
folder_2018 = '/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label1'
folder_2020 = '/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label3'
output_folder1 = '/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/scd-label1'
output_folder2 = '/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/scd-label3'
# 确保输出文件夹存在
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
# 获取2018年和2020年文件夹中的文件列表
files_2018 = sorted([f for f in os.listdir(folder_2018) if f.endswith('.tif')])
files_2020 = sorted([f for f in os.listdir(folder_2020) if f.endswith('.tif')])

# 确保文件数量一致
assert len(files_2018) == len(files_2020), "2018年和2020年的文件数量不一致！"

# 遍历每一对文件
for file_2018, file_2020 in tqdm(zip(files_2018, files_2020), total=len(files_2018)):
    # 读取2018年和2020年的标签图像
    path_2018 = os.path.join(folder_2018, file_2018)
    path_2020 = os.path.join(folder_2020, file_2020)

    # 使用PIL读取图像
    img_2018 = Image.open(path_2018)
    img_2020 = Image.open(path_2020)

    # 将图像转换为numpy数组
    label_2018 = np.array(img_2018)
    label_2020 = np.array(img_2020)

    # 获取变化区域
    change_mask = (label_2018 != label_2020)

    # 将2018年和2020年标签图像中除变化区域以外的地方掩膜为0
    masked_label_2018 = np.where(change_mask, label_2018, 0)
    masked_label_2020 = np.where(change_mask, label_2020, 0)

    # 将numpy数组转换回PIL图像
    masked_img_2018 = Image.fromarray(masked_label_2018)
    masked_img_2020 = Image.fromarray(masked_label_2020)

    # 复制原图像的调色板（如果有）
    if 'P' in img_2018.mode:  # 如果原图像是调色板模式
        masked_img_2018.putpalette(img_2018.getpalette())
    if 'P' in img_2020.mode:  # 如果原图像是调色板模式
        masked_img_2020.putpalette(img_2020.getpalette())

    # 保存掩膜后的2018年标签图像
    output_path_2018 = os.path.join(output_folder1, file_2018)
    masked_img_2018.save(output_path_2018)

    # 保存掩膜后的2020年标签图像
    output_path_2020 = os.path.join(output_folder2, file_2020)
    masked_img_2020.save(output_path_2020)

print("批量处理完成！")