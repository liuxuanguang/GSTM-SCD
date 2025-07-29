import os
import numpy as np
from osgeo import gdal
from glob import glob
from tqdm import tqdm
import argparse
import re
import time
# 在这里设置您的路径
PRED_CLASS_DIR_T1 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/WUSU_SCDNet/resnet34/pred1_semantic/"
PRED_CLASS_DIR_T2 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/WUSU_SCDNet/resnet34/pred2_semantic/"
PRED_CLASS_DIR_T3 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/WUSU_SCDNet/resnet34/pred3_semantic/"

LABEL_DIR_T1 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label1/"
LABEL_DIR_T2 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label2/"
LABEL_DIR_T3 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label3/"

PRED_CHANGE_DIR_LONG = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/WUSU_SCDNet/resnet34/pred_change/"




def read_tif_as_gray(image_path):
    """读取tif图像并转为单通道灰度图"""
    ds = gdal.Open(image_path)
    if ds is None:
        raise ValueError(f"无法读取文件: {image_path}")

    # 读取所有通道数据
    img_arr = np.array([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)])

    # 如果是三通道RGB，转换为灰度图（取平均值）
    if img_arr.shape[0] == 3:
        img_arr = img_arr.mean(axis=0).astype(np.int16)
    elif img_arr.shape[0] == 1:
        img_arr = img_arr[0].astype(np.int16)
    else:
        raise ValueError(f"不支持的通道数: {img_arr.shape[0]}")

    ds = None  # 释放资源
    return img_arr


def calculate_iou(conf_matrix, valid_classes=None):
    """通过混淆矩阵计算IoU，可指定有效类别"""
    # 计算每个类别的交并比
    ious = []
    n_classes = conf_matrix.shape[0]

    if valid_classes is None:
        valid_classes = range(n_classes)
    else:
        valid_classes = set(valid_classes)

    for i in range(n_classes):
        if i not in valid_classes:
            ious.append(float('nan'))
            continue

        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        denominator = tp + fp + fn
        if denominator == 0:
            ious.append(float('nan'))  # 忽略未出现的类别
        else:
            ious.append(tp / denominator)

    # 计算平均交并比（忽略NaN值）
    miou = np.nanmean(ious)
    return ious, miou


def extract_suffix_from_filename(filename):
    """从文件名中提取区域和时间标识（如 'region1_2020'）"""
    # 匹配常见的文件名模式，如：xxx_region_date_xxx.tif
    match = re.search(r'([A-Za-z0-9]+_[0-9]{4})', os.path.basename(filename))
    if match:
        return match.group(1)
    else:
        # 如果没有特定模式，则返回文件名前缀
        return os.path.splitext(os.path.basename(filename))[0][:10]


def match_files_by_suffix(files1, files2):
    """根据提取的后缀匹配两组文件"""
    matched_files1 = []
    matched_files2 = []

    # 提取后缀映射
    suffix_map1 = {extract_suffix_from_filename(f): f for f in files1}
    suffix_map2 = {extract_suffix_from_filename(f): f for f in files2}

    # 寻找共同后缀
    common_suffixes = set(suffix_map1.keys()) & set(suffix_map2.keys())

    for suffix in common_suffixes:
        matched_files1.append(suffix_map1[suffix])
        matched_files2.append(suffix_map2[suffix])

    print(f"成功匹配 {len(matched_files1)} 对文件")

    if not matched_files1:
        raise ValueError("未找到匹配的文件对！请确保文件名包含相同的区域和时间标识")

    return matched_files1, matched_files2


def generate_change_labels(label_dir1, label_dir2):
    """根据双时相土地覆盖分类标签生成变化检测标签"""
    print("\n正在生成变化检测标签...")

    # 获取标签文件并匹配
    label_files1 = sorted(glob(os.path.join(label_dir1, '*.tif')))
    label_files2 = sorted(glob(os.path.join(label_dir2, '*.tif')))

    label_files1, label_files2 = match_files_by_suffix(label_files1, label_files2)

    # 检查文件数量
    if len(label_files1) != len(label_files2):
        raise ValueError(f"{os.path.basename(label_dir1)}和{os.path.basename(label_dir2)}文件夹中的文件数量不匹配")

    # 生成变化标签字典（用于后续计算变化mIoU）
    change_labels = {}

    for label_path1, label_path2 in tqdm(zip(label_files1, label_files2), total=len(label_files1), desc="生成变化标签"):
        # 读取土地覆盖分类标签
        label_img1 = read_tif_as_gray(label_path1)
        label_img2 = read_tif_as_gray(label_path2)

        # 验证尺寸一致性
        if label_img1.shape != label_img2.shape:
            raise ValueError(f"尺寸不匹配: {label_path1} vs {label_path2}")

        # 生成变化标签：像素值一致为未变化(0)，不一致为变化(1)
        change_label = (label_img1 != label_img2).astype(np.uint8)

        # 存储生成的标签
        file_id = extract_suffix_from_filename(label_path1)
        change_labels[file_id] = change_label

    print(f"从{os.path.basename(label_dir1)}到{os.path.basename(label_dir2)}生成了 {len(change_labels)} 个变化检测标签")
    return change_labels


def evaluate_classification(pred_dir, label_dir, time_point, ignore_class=0):
    """评估土地覆盖分类结果，忽略指定类别"""
    print("\n" + "=" * 70)
    print(f"时相 {time_point} 土地覆盖分类精度评估".center(70))
    print("=" * 70)

    # 获取预测和标签文件
    pred_files = sorted(glob(os.path.join(pred_dir, '*.tif')))
    label_files = sorted(glob(os.path.join(label_dir, '*.tif')))

    if not pred_files or not label_files:
        raise ValueError("未找到.tif文件！请检查路径是否正确")

    print(f"找到 {len(pred_files)} 个预测文件和 {len(label_files)} 个标签文件")

    # 匹配文件
    pred_files, label_files = match_files_by_suffix(pred_files, label_files)

    # 自动确定类别数量
    max_class = 0
    print("正在确定类别数量...")
    for file in tqdm(label_files, desc="分析标签"):
        img = read_tif_as_gray(file)
        max_class = max(max_class, np.max(img))

    n_classes = int(max_class) + 1
    print(f"检测到 {n_classes} 个类别 (忽略类别 {ignore_class})")
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)

    # 处理每对图像
    print(f"\n计算时相 {time_point} 土地覆盖分类混淆矩阵 (忽略类别 {ignore_class})...")
    for pred_path, label_path in tqdm(zip(pred_files, label_files), total=len(pred_files), desc=f"评估T{time_point}"):
        pred_img = read_tif_as_gray(pred_path)
        label_img = read_tif_as_gray(label_path)

        # 验证尺寸一致性
        if pred_img.shape != label_img.shape:
            raise ValueError(f"尺寸不匹配: {pred_path} vs {label_path}")

        # 创建有效像素掩码 (忽略类别0)
        valid_mask = (label_img != ignore_class)

        # 展平图像并应用掩码
        pred_flat = pred_img[valid_mask].ravel()
        label_flat = label_img[valid_mask].ravel()

        # 更新混淆矩阵
        for p, l in zip(pred_flat, label_flat):
            if l < n_classes:  # 忽略超出范围的标签
                conf_matrix[int(l), int(p)] += 1

    # 确定有效类别（所有类别中排除被忽略的类别）
    valid_classes = [i for i in range(n_classes) if i != ignore_class]

    # 计算mIoU，只考虑有效类别
    class_ious, miou = calculate_iou(conf_matrix, valid_classes=valid_classes)

    # 打印结果
    print("\n" + "=" * 50)
    print(f"时相 {time_point} 土地覆盖分类评估结果 (基于 {len(pred_files)} 对图像)")
    print(f"总类别数: {n_classes} | 有效类别数: {len(valid_classes)} (忽略类别 {ignore_class})")
    print("有效类别IoU值:")
    for i in valid_classes:
        iou = class_ious[i]
        if not np.isnan(iou):
            print(f"  类别 {i}: {iou:.4f}")

    print(f"\n时相 {time_point} 平均交并比 (mIoU，仅有效类别): {miou:.4f}")
    print(f"混淆矩阵摘要 (前{min(10, n_classes)}x{min(10, n_classes)}):")
    print(conf_matrix[:min(10, n_classes), :min(10, n_classes)])  # 避免打印过大矩阵
    print("=" * 50)

    return miou, conf_matrix, valid_classes


def evaluate_change_detection(pred_dir, change_labels):
    """评估变化检测结果"""
    print("\n" + "=" * 70)
    print(f"长期变化检测精度评估".center(70))
    print("=" * 70)

    # 获取变化预测文件
    pred_files = sorted(glob(os.path.join(pred_dir, '*.tif')))

    if not pred_files:
        raise ValueError("未找到变化预测.tif文件！")

    print(f"找到 {len(pred_files)} 个变化预测文件")

    # 初始化混淆矩阵（二分类：未变化=0，变化=1）
    conf_matrix = np.zeros((2, 2), dtype=np.uint64)

    # 匹配预测文件对应的标签
    matched_count = 0
    print(f"\n计算变化检测混淆矩阵...")
    for pred_path in tqdm(pred_files, desc="处理变化预测"):
        # 提取文件标识符
        file_id = extract_suffix_from_filename(pred_path)

        # 检查是否有对应的变化标签
        if file_id in change_labels:
            # 读取预测图像
            pred_img = read_tif_as_gray(pred_path)

            # 二值化预测（非0即1）
            pred_img = (pred_img > 0).astype(np.uint8)

            # 获取对应的变化标签
            true_img = change_labels[file_id]

            # 验证尺寸一致性
            if pred_img.shape != true_img.shape:
                print(f"尺寸不匹配: 预测 {pred_img.shape} vs 标签 {true_img.shape}, 文件: {file_id}")
                continue

            # 展平图像
            pred_flat = pred_img.ravel()
            true_flat = true_img.ravel()

            # 更新混淆矩阵
            for p, t in zip(pred_flat, true_flat):
                conf_matrix[int(t), int(p)] += 1

            matched_count += 1

    print(f"成功匹配 {matched_count}/{len(pred_files)} 对变化预测和标签")

    if matched_count == 0:
        raise ValueError(f"未能匹配任何变化预测文件和标签！")

    # 计算变化检测指标
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"\n变化检测混淆矩阵:")
    print(f"TN (真负): {tn} | FP (假正): {fp}")
    print(f"FN (假负): {fn} | TP (真正): {tp}")

    # 计算各项指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    overall_accuracy = (tp + tn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0

    # 打印结果
    print(f"\n变化检测性能指标:")
    print(f"精度 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    print(f"变化类别IoU: {iou:.4f}")
    print(f"总体准确率: {overall_accuracy:.4f}")
    print("=" * 50)

    return iou, precision, recall, f1_score


def save_results(results, save_dir):
    """保存评估结果到文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    result_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")

    with open(result_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("三时相土地覆盖分类与长期变化检测评估结果\n".center(70))
        f.write("=" * 70 + "\n\n")

        # 保存评估信息
        f.write(f"评估时间: {time.ctime()}\n")
        f.write(f"评估图像数量: {results['img_count']}\n")
        f.write("\n")

        # 保存土地覆盖分类结果
        f.write("\n" + "=" * 70 + "\n")
        f.write("土地覆盖分类评估结果\n".center(70) + "\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"时相1预测文件夹: {PRED_CLASS_DIR_T1}\n")
        f.write(f"时相1标签文件夹: {LABEL_DIR_T1}\n")
        f.write(f"时相1 mIoU: {results['class_miou_t1']:.4f}\n")
        f.write(f"时相1有效类别数: {len(results['valid_classes_t1'])}\n")

        f.write("\n")

        f.write(f"时相2预测文件夹: {PRED_CLASS_DIR_T2}\n")
        f.write(f"时相2标签文件夹: {LABEL_DIR_T2}\n")
        f.write(f"时相2 mIoU: {results['class_miou_t2']:.4f}\n")
        f.write(f"时相2有效类别数: {len(results['valid_classes_t2'])}\n")

        f.write("\n")

        f.write(f"时相3预测文件夹: {PRED_CLASS_DIR_T3}\n")
        f.write(f"时相3标签文件夹: {LABEL_DIR_T3}\n")
        f.write(f"时相3 mIoU: {results['class_miou_t3']:.4f}\n")
        f.write(f"时相3有效类别数: {len(results['valid_classes_t3'])}\n")

        f.write("\n")

        f.write(f"平均土地覆盖分类 mIoU: {results['avg_class_miou']:.4f}\n\n")

        # 保存长期变化检测结果
        f.write("\n" + "=" * 70 + "\n")
        f.write("长期变化检测评估结果 (时相1->时相3)\n".center(70) + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"预测文件夹: {PRED_CHANGE_DIR_LONG}\n\n")

        f.write(f"变化检测IoU: {results['long_change_iou']:.4f}\n")
        f.write(f"精度 (Precision): {results['long_change_precision']:.4f}\n")
        f.write(f"召回率 (Recall): {results['long_change_recall']:.4f}\n")
        f.write(f"F1分数: {results['long_change_f1']:.4f}\n\n")

        # 详细结果
        f.write("\n" + "=" * 70 + "\n")
        f.write("详细统计结果\n".center(70) + "\n")
        f.write("=" * 70 + "\n")
        f.write(f"时相1有效类别: {sorted(results['valid_classes_t1'])}\n")
        f.write(f"时相2有效类别: {sorted(results['valid_classes_t2'])}\n")
        f.write(f"时相3有效类别: {sorted(results['valid_classes_t3'])}\n")

    print(f"\n评估结果已保存至: {result_file}")
    return result_file


def main():
    """主评估函数"""
    start_time = time.time()

    # 设置GDAL不抛出警告
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    try:
        print("=" * 80)
        print("三时相土地覆盖分类与长期变化检测评估".center(80))
        print("=" * 80)
        print(f"开始时间: {time.ctime()}")
        print("忽略土地覆盖分类中的类别: 0 (背景)")
        print("=" * 80)

        # 1. 评估三个时相的土地覆盖分类结果，忽略类别0
        class_miou_t1, conf_matrix_t1, valid_classes_t1 = evaluate_classification(
            PRED_CLASS_DIR_T1, LABEL_DIR_T1, 1, ignore_class=0
        )

        class_miou_t2, conf_matrix_t2, valid_classes_t2 = evaluate_classification(
            PRED_CLASS_DIR_T2, LABEL_DIR_T2, 2, ignore_class=0
        )

        class_miou_t3, conf_matrix_t3, valid_classes_t3 = evaluate_classification(
            PRED_CLASS_DIR_T3, LABEL_DIR_T3, 3, ignore_class=0
        )

        # 计算平均mIoU
        avg_class_miou = np.mean([class_miou_t1, class_miou_t2, class_miou_t3])

        # 2. 生成长期变化标签 (时相1到时相3)
        print("\n" + "=" * 80)
        print("生成长期变化标签 (时相1到时相3)".center(80))
        print("=" * 80)
        long_change_labels = generate_change_labels(LABEL_DIR_T1, LABEL_DIR_T3)

        # 3. 评估长期变化检测结果
        long_change_iou, long_precision, long_recall, long_f1 = evaluate_change_detection(
            PRED_CHANGE_DIR_LONG, long_change_labels
        )

        # 4. 汇总结果
        img_count = len(long_change_labels)  # 所有时相应有相同数量的图像

        results = {
            # 土地覆盖分类
            'class_miou_t1': class_miou_t1,
            'class_miou_t2': class_miou_t2,
            'class_miou_t3': class_miou_t3,
            'avg_class_miou': avg_class_miou,
            'valid_classes_t1': valid_classes_t1,
            'valid_classes_t2': valid_classes_t2,
            'valid_classes_t3': valid_classes_t3,
            'img_count': img_count,

            # 长期变化检测
            'long_change_iou': long_change_iou,
            'long_change_precision': long_precision,
            'long_change_recall': long_recall,
            'long_change_f1': long_f1,
        }

        # 5. 保存结果
        save_results_dir = os.path.dirname(PRED_CLASS_DIR_T1)
        result_file = save_results(results, save_results_dir)

        end_time = time.time()
        elapsed = end_time - start_time

        print("\n" + "=" * 80)
        print("评估完成!".center(80))
        print("=" * 80)
        print(f"平均土地覆盖分类 mIoU (忽略背景): {avg_class_miou:.4f}")
        print(f"长期变化检测 IoU: {long_change_iou:.4f} | F1分数: {long_f1:.4f}")
        print(f"处理图像数量: {img_count}")
        print(f"总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
        print(f"详细结果已保存至: {result_file}")
        print("=" * 80)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

        if "TIFF" in str(e) and "read" in str(e):
            print("提示：考虑使用PIL库代替GDAL，请安装: pip install Pillow")


if __name__ == "__main__":
    main()