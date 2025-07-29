import os
import numpy as np
import cv2
import time
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import rasterio

# SECOND
# PRED_DIR = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/SECOND/FZ-SCD-Sek22.4/resnet34/pred_change/"
# GT_DIR = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/SECOND_total_test/test/label1/"
# GT_DIR = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND/New_SECOND/test667/label1/"
# GT_DIR = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/SECOND_512size/test/label1"

# Landsat
PRED_DIR = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/Landsat_SCD/proposed/resnet34/pred_change/"
GT_DIR = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/Landsat-SCD/test/labelA/"


def fast_read_image(path):
    """高效读取图像数据"""
    # 使用OpenCV读取普通图像格式
    if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        return img

    # 使用rasterio读取地理空间图像格式
    elif path.lower().endswith(('.tif', '.tiff')):
        with rasterio.open(path) as src:
            return src.read(1)

    else:
        raise ValueError(f"不支持的图像格式: {os.path.splitext(path)[1]}")


def process_file(args):
    """处理单个文件对并计算指标"""
    pred_path, gt_path = args
    try:
        # 读取预测图像（0-未变化, 255-变化）
        pred = fast_read_image(pred_path)

        # 二值化预测图像：变化区域设为1，未变化设为0
        pred_binary = (pred == 255).astype(np.uint8)

        # 读取真实标签（0-未变化, >0-变化）
        gt = fast_read_image(gt_path)

        # 二值化真实标签：所有大于0的值设为1（变化）
        gt_binary = (gt > 0).astype(np.uint8)

        # 计算混淆矩阵元素
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))

        # 计算变化区域的IoU
        iou_change = tp / (tp + fp + fn + 1e-10)

        # 计算未变化区域的IoU
        iou_nochange = tn / (tn + fp + fn + 1e-10)

        # 计算整体mIoU
        miou = (iou_change + iou_nochange) / 2

        # 计算F1分数
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        return {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'iou_change': iou_change, 'iou_nochange': iou_nochange,
            'miou': miou, 'f1': f1
        }

    except Exception as e:
        print(f"处理文件出错 {pred_path}: {str(e)}")
        return None


def evaluate_change_detection(pred_dir, gt_dir):
    """
    高效评估二值变化检测结果

    参数:
        pred_dir: 预测结果目录（0-未变化, 255-变化）
        gt_dir: 真实标签目录（0-未变化, >0-变化）

    返回:
        包含所有评估指标的字典
    """
    start_time = time.time()

    # 获取所有预测文件
    pred_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        pred_files.extend([f for f in os.listdir(pred_dir) if f.lower().endswith(ext)])

    if not pred_files:
        raise ValueError(f"在目录 {pred_dir} 中找不到任何图像文件")

    # 准备文件对 (预测路径, 真实标签路径)
    file_pairs = []
    for f in pred_files:
        pred_path = os.path.join(pred_dir, f)
        gt_path = os.path.join(gt_dir, f)

        if not os.path.exists(gt_path):
            # 尝试不同的文件扩展名
            base_name = os.path.splitext(f)[0]
            found = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                alt_gt_path = os.path.join(gt_dir, base_name + ext)
                if os.path.exists(alt_gt_path):
                    gt_path = alt_gt_path
                    found = True
                    break

            if not found:
                print(f"警告: 找不到与 {f} 匹配的真实标签文件，跳过")
                continue

        file_pairs.append((pred_path, gt_path))

    print(f"找到 {len(file_pairs)} 对文件进行评估")

    # 使用多进程并行处理
    results = []
    with Pool(processes=min(cpu_count(), 8)) as pool:  # 限制最多8个进程
        with tqdm(total=len(file_pairs), desc="处理图像") as pbar:
            for result in pool.imap_unordered(process_file, file_pairs):
                if result is not None:
                    results.append(result)
                pbar.update(1)

    if not results:
        raise RuntimeError("所有文件处理失败，无有效结果")

    # 汇总所有结果
    total_tp = total_fp = total_fn = total_tn = 0
    total_miou = total_f1 = 0

    for res in results:
        total_tp += res['tp']
        total_fp += res['fp']
        total_fn += res['fn']
        total_tn += res['tn']
        total_miou += res['miou']
        total_f1 += res['f1']

    num_samples = len(results)

    # 计算整体统计量
    precision = total_tp / (total_tp + total_fp + 1e-10)
    recall = total_tp / (total_tp + total_fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    # 变化区域的IoU
    iou_change = total_tp / (total_tp + total_fp + total_fn + 1e-10)

    # 未变化区域的IoU
    iou_nochange = total_tn / (total_tn + total_fp + total_fn + 1e-10)

    # 整体mIoU
    miou = (iou_change + iou_nochange) / 2

    # 执行时间
    elapsed = time.time() - start_time

    return {
        'samples': num_samples,
        'time_elapsed': elapsed,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'tn': total_tn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou_change': iou_change,
        'iou_nochange': iou_nochange,
        'miou': miou,
        'f1_mean': total_f1 / num_samples,
        'miou_mean': total_miou / num_samples
    }


def print_results(results):
    """打印评估结果"""
    print("\n评估结果:")
    print(f"处理图像数量: {results['samples']}")
    print(f"总耗时: {results['time_elapsed']:.2f} 秒")
    print(f"平均每张图像耗时: {results['time_elapsed'] / results['samples'] * 1000:.2f} 毫秒")
    print("\n混淆矩阵:")
    print(f"真正例 (TP): {results['tp']}")
    print(f"假正例 (FP): {results['fp']}")
    print(f"假负例 (FN): {results['fn']}")
    print(f"真负例 (TN): {results['tn']}")
    print("\n指标:")
    print(f"精确率 (Precision): {results['precision']:.6f}")
    print(f"召回率 (Recall): {results['recall']:.6f}")
    print(f"F1分数 (F1-Score): {results['f1_score']:.6f}")
    print(f"变化区域IoU: {results['iou_change']:.6f}")
    print(f"未变化区域IoU: {results['iou_nochange']:.6f}")
    print(f"整体mIoU: {results['miou']:.6f}")
    # print(f"平均每张图像的F1分数: {results['f1_mean']:.6f}")
    # print(f"平均每张图像的mIoU: {results['miou_mean']:.6f}")


if __name__ == "__main__":
    print("开始二值变化检测评估")
    print(f"预测目录: {PRED_DIR}")
    print(f"真实标签目录: {GT_DIR}")

    try:
        results = evaluate_change_detection(PRED_DIR, GT_DIR)
        print_results(results)
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")