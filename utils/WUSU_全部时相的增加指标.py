import os
import numpy as np
import multiprocessing
import tqdm
import time
import re
import rasterio
from rasterio.windows import Window
import itertools
import traceback  # 添加traceback模块以处理异常

# 在这里设置您的路径
PRED_CLASS_DIR_T1 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/TSSCD-全部时相/resnet34/pred_LC/pred_semantic_time1/"
PRED_CLASS_DIR_T2 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/TSSCD-全部时相/resnet34/pred_LC/pred_semantic_time2/"
PRED_CLASS_DIR_T3 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/TSSCD-全部时相/resnet34/pred_LC/pred_semantic_time3/"
LABEL_DIR_T1 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label1/"
LABEL_DIR_T2 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label2/"
LABEL_DIR_T3 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label3/"

# 双时相变化检测的基础目录
PRED_CHANGE_DIR_BASE = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/TSSCD-全部时相/resnet34/"

# 定义所有时相的路径
TIME_POINTS = 3
PRED_CLASS_DIRS = {
    1: PRED_CLASS_DIR_T1,
    2: PRED_CLASS_DIR_T2,
    3: PRED_CLASS_DIR_T3
}

LABEL_DIRS = {
    1: LABEL_DIR_T1,
    2: LABEL_DIR_T2,
    3: LABEL_DIR_T3
}

# 双时相组合
TIME_PAIRS = list(itertools.combinations(range(1, TIME_POINTS + 1), 2))


def extract_match_key(filename):
    """
    从文件名中提取匹配键
    格式: 区域代码 + 分块代码
    示例:
        '2697_3715_13_2018-06-01_0.png' -> ('2697_3715_13', '0')
        '2697_3715_13_2018-02-01_0.tif' -> ('2697_3715_13', '0')
    """
    # 移除文件扩展名
    base_name = os.path.splitext(filename)[0]

    # 分割文件名各部分
    parts = base_name.split('_')

    # 区域代码是前三部分(如果有至少四部分的话)
    region_code = '_'.join(parts[:2]) if len(parts) >= 2 else '_'.join(parts[:-1])

    # 分块代码是最后一部分
    tile_code = parts[-1] if parts else ""

    return (region_code, tile_code)


def match_files_by_region_and_tile(pred_dir, label_dir):
    """
    根据区域代码和分块代码匹配预测文件和标签文件
    """
    try:
        # 获取所有文件
        pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.tif', '.png'))]
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.tif', '.png'))]

        # 创建匹配键到文件路径的映射
        pred_dict = {}
        for f in pred_files:
            key = extract_match_key(f)
            if key not in pred_dict:
                pred_dict[key] = []
            pred_dict[key].append(os.path.join(pred_dir, f))

        label_dict = {}
        for f in label_files:
            key = extract_match_key(f)
            if key not in label_dict:
                label_dict[key] = []
            label_dict[key].append(os.path.join(label_dir, f))

        # 找出共同匹配键
        common_keys = set(pred_dict.keys()) & set(label_dict.keys())

        if not common_keys:
            # 输出一些调试信息
            print("预测文件示例:")
            for k in list(pred_dict.keys())[:5]:
                print(f"  匹配键: {k} -> 文件: {pred_dict[k]}")
            print("\n标签文件示例:")
            for k in list(label_dict.keys())[:5]:
                print(f"  匹配键: {k} -> 文件: {label_dict[k]}")

            raise ValueError("未找到匹配的文件对! 请检查文件名格式")

        # 创建匹配的文件对
        matched_preds = []
        matched_labels = []

        for key in common_keys:
            # 每个匹配键可能有多个文件，我们取第一个
            matched_preds.append(pred_dict[key][0])
            matched_labels.append(label_dict[key][0])

        print(f"成功匹配 {len(matched_preds)} 对文件")
        return matched_preds, matched_labels
    except Exception as e:
        print(f"匹配文件时出错: {str(e)}")
        traceback.print_exc()
        return [], []


def read_tif_as_gray(image_path):
    """使用rasterio高效读取tif或png图像并转为单通道灰度图"""
    try:
        with rasterio.open(image_path) as src:
            # 如果是多波段图像，取平均值
            if src.count > 1:
                img = np.mean(src.read(), axis=0).astype(np.int16)
            else:
                img = src.read(1).astype(np.int16)
        return img
    except Exception as e:
        print(f"读取文件 {image_path} 时出错: {str(e)}")
        return None


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


def generate_change_labels(label_dir1, label_dir2):
    """根据双时相土地覆盖分类标签生成变化检测标签"""
    print(f"\n正在生成从{os.path.basename(label_dir1)}到{os.path.basename(label_dir2)}的变化检测标签...")
    try:
        # 获取标签文件并匹配
        label_files1 = [f for f in os.listdir(label_dir1) if f.lower().endswith(('.tif', '.png'))]
        label_files2 = [f for f in os.listdir(label_dir2) if f.lower().endswith(('.tif', '.png'))]

        # 创建匹配键到文件路径的映射
        label_dict1 = {}
        for f in label_files1:
            key = extract_match_key(f)
            if key not in label_dict1:
                label_dict1[key] = []
            label_dict1[key].append(os.path.join(label_dir1, f))

        label_dict2 = {}
        for f in label_files2:
            key = extract_match_key(f)
            if key not in label_dict2:
                label_dict2[key] = []
            label_dict2[key].append(os.path.join(label_dir2, f))

        # 找出共同匹配极简版结果/动态EarthNet/1/键
        common_keys = set(label_dict1.keys()) & set(label_dict2.keys())

        if not common_keys:
            # 输出调试信息
            print(f"{os.path.basename(label_dir1)}文件示例:")
            for k in list(label_dict1.keys())[:5]:
                print(f"  匹配键: {k} -> 文件: {label_dict1[k][0]}")
            print(f"\n{os.path.basename(label_dir2)}文件示例:")
            for k in list(label_dict2.keys())[:5]:
                print(f"  匹配键: {k} -> 文件: {label_dict2[k][0]}")

            raise ValueError(f"在 {label_dir1} 和 {label_dir2} 中没有找到共同的文件匹配键！")

        print(f"找到 {len(common_keys)} 对匹配文件")

        # 生成变化标签字典
        change_labels = {}

        for key in tqdm.tqdm(common_keys, desc="生成变化标签"):
            try:
                # 只取每个匹配键的第一个文件
                label_path1 = label_dict1[key][0]
                label_path2 = label_dict2[key][0]

                # 读取土地覆盖分类标签
                label_img1 = read_tif_as_gray(label_path1)
                label_img2 = read_tif_as_gray(label_path2)

                if label_img1 is None or label_img2 is None:
                    continue

                # 处理图像尺寸不一致的情况
                if label_img1.shape != label_img2.shape:
                    min_height = min(label_img1.shape[0], label_img2.shape[0])
                    min_width = min(label_img1.shape[1], label_img2.shape[1])
                    label_img1 = label_img1[:min_height, :min_width]
                    label_img2 = label_img2[:min_height, :min_width]

                # 生成变化标签：像素值一致为未变化(0)，不一致为变化(1)
                change_label = (label_img1 != label_img2).astype(np.uint8)

                # 获取原始文件名作为键（不带路径）
                filename = os.path.basename(label_path1)
                change_labels[filename] = change_label
            except Exception as e:
                print(f"处理键 {key} 时出错: {str(e)}")
                traceback.print_exc()

        print(
            f"从{os.path.basename(label_dir1)}到{os.path.basename(label_dir2)}生成了 {len(change_labels)} 个变化检测标签")
        return change_labels
    except Exception as e:
        print(f"生成变化标签时出错: {str(e)}")
        traceback.print_exc()
        return {}


def process_image_pair(args):
    """处理单个图像对并更新混淆矩阵"""
    try:
        pred_path, label_path, ignore_class, n_classes = args
        # 读取图像
        pred_img = read_tif_as_gray(pred_path)
        label_img = read_tif_as_gray(label_path)

        if pred_img is None or label_img is None:
            return np.zeros((n_classes, n_classes), dtype=np.uint64)

        # 处理尺寸不一致问题
        if pred_img.shape != label_img.shape:
            min_height = min(pred_img.shape[0], label_img.shape[0])
            min_width = min(pred_img.shape[1], label_img.shape[1])
            pred_img = pred_img[:min_height, :min_width]
            label_img = label_img[:min_height, :min_width]

        # 创建有效像素掩码 (忽略类别0)
        valid_mask = (label_img != ignore_class)

        # 提取有效像素
        pred_flat = pred_img[valid_mask].ravel()
        label_flat = label_img[valid_mask].ravel()

        # 创建局部混淆矩阵
        conf_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)

        # 使用向量化操作更新混淆矩阵
        # 只处理有效的标签（在0到n_classes-1之间）
        valid_indices = (label_flat >= 0) & (label_flat < n_classes)
        valid_labels = label_flat[valid_indices]
        valid_preds = pred_flat[valid_indices]

        if len(valid_labels) > 0:
            # 使用bincount进行高效统计
            indices = valid_labels * n_classes + valid_preds
            counts = np.bincount(indices, minlength=n_classes * n_classes)

            # 确保数据类型匹配
            counts = counts.astype(np.uint64)
            conf_matrix = counts.reshape((n_classes, n_classes))

        return conf_matrix
    except Exception as e:
        print(f"处理图像对时出错: {str(e)}")
        return np.zeros((n_classes, n_classes), dtype=np.uint64)


def evaluate_classification(pred_dir, label_dir, time_point, ignore_class=0):
    """评估土地覆盖分类结果，忽略指定类别"""
    print("\n" + "=" * 70)
    print(f"时相 {time_point} 土地覆盖分类精度评估".center(70))
    print("=" * 70)

    try:
        # 使用新的匹配规则匹配文件
        pred_files, label_files = match_files_by_region_and_tile(pred_dir, label_dir)

        if not pred_files:
            print(f"警告: 未找到匹配的文件对! 请检查文件和路径")
            return 0.0, None, []

        print(f"处理 {len(pred_files)} 对文件")

        # 自动确定类别数量
        max_class = 0
        print("正在确定类别数量...")
        for file_path in tqdm.tqdm(label_files, desc="分析标签"):
            img = read_tif_as_gray(file_path)
            if img is not None:
                max_class = max(max_class, np.max(img))

        n_classes = max(1, int(max_class) + 1)
        print(f"检测到 {n_classes} 个类别 (忽略类别 {ignore_class})")

        # 使用多进程并行处理图像对
        print(f"\n并行计算时相 {time_point} 土地覆盖分类混淆矩阵...")

        # 计算合适的进程数
        num_processes = min(multiprocessing.cpu_count(), 8, len(pred_files))
        pool = multiprocessing.Pool(processes=num_processes) if num_processes > 1 else None

        # 准备参数
        args_list = [(pred_files[i], label_files[i], ignore_class, n_classes)
                     for i in range(len(pred_files))]

        # 处理结果
        conf_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)

        if pool:
            results = pool.imap_unordered(process_image_pair, args_list)
            for result in tqdm.tqdm(results, total=len(args_list), desc=f"评估T{time_point}"):
                conf_matrix += result
            pool.close()
            pool.join()
        else:
            for args in tqdm.tqdm(args_list, desc=f"评估T{time_point}"):
                result = process_image_pair(args)
                conf_matrix += result

        # 确定有效类别（所有类别中排除被忽略的类别）
        valid_classes = [i for i in range(n_classes) if i != ignore_class]

        # 计算mIoU，只考虑有效类别
        class_ious, miou = calculate_iou(conf_matrix, valid_classes=valid_classes)

        # 打印结果
        print("\n" + "=" * 50)
        print(f"时相 {time_point} 土地覆盖分类评估结果 (基于 {len(pred_files)} 对图像)")
        print(f"总类别数: {n_classes} | 有效类别数: {len(valid_classes)} (忽略类别 {ignore_class})")
        print(f"\n时相 {time_point} 平均交并比 (mIoU，仅有效类别): {miou:.4f}")
        print("=" * 50)

        return miou, conf_matrix, valid_classes
    except Exception as e:
        print(f"评估土地覆盖分类时出错: {str(e)}")
        traceback.print_exc()
        return 0.0, None, []


def process_change_pair(args):
    """处理单个变化检测图像对并更新混淆矩阵"""
    try:
        pred_path, true_img = args
        if true_img is None:
            return np.zeros((2, 2), dtype=np.uint64)

        # 读取预测图像
        pred_img = read_tif_as_gray(pred_path)
        if pred_img is None:
            return np.zeros((2, 2), dtype=np.uint64)

        # 处理图像尺寸不一致
        if pred_img.shape != true_img.shape:
            min_height = min(pred_img.shape[0], true_img.shape[0])
            min_width = min(pred_img.shape[1], true_img.shape[1])
            pred_img = pred_img[:min_height, :min_width]
            true_img = true_img[:min_height, :min_width]

        # 二值化预测（非0即1）
        pred_img = (pred_img > 0).astype(np.uint8)

        # 创建局部混淆矩阵
        conf_matrix = np.zeros((2, 2), dtype=np.uint64)

        # 使用向量化操作更新混淆矩阵
        t = true_img.ravel()
        p = pred_img.ravel()

        # 使用bincount进行高效统计
        indices = t * 2 + p
        counts = np.bincount(indices, minlength=4)

        # 确保数据类型匹配
        counts = counts.astype(np.uint64)
        conf_matrix = counts.reshape((2, 2))

        return conf_matrix
    except Exception as e:
        print(f"处理变化检测对时出错: {str(e)}")
        return np.zeros((2, 2), dtype=np.uint64)


def evaluate_change_detection(pred_dir, change_labels, t1, t2):
    """评估变化检测结果"""
    print("\n" + "=" * 70)
    print(f"双时相变化检测精度评估 (T{t1} -> T{t2})".center(70))
    print("=" * 70)

    print(f"预测目录: {pred_dir}")
    try:
        # 获取变化预测文件
        pred_files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
                      if f.lower().endswith(('.tif', '.png'))]

        if not pred_files:
            print(f"警告: 未在 {pred_dir} 中找到变化预测文件！")
            return None

        print(f"找到 {len(pred_files)} 个变化预测文件")

        # 匹配预测文件对应的标签
        matched_files = []
        matched_labels = []

        for pred_path in pred_files:
            file_name = os.path.basename(pred_path)
            match_key = extract_match_key(file_name)

            # 查找匹配的变化标签
            matched = False
            for label_file in change_labels.keys():
                label_match_key = extract_match_key(label_file)
                if match_key == label_match_key:
                    matched_files.append(pred_path)
                    matched_labels.append(change_labels[label_file])
                    matched = True
                    break

        print(f"成功匹配 {len(matched_files)}/{len(pred_files)} 对变化预测和标签")

        if not matched_files:
            # 输出一些调试信息
            print("预测文件示例:")
            for f in pred_files[:5]:
                key = extract_match_key(os.path.basename(f))
                print(f"  匹配键: {key} -> 文件: {f}")

            print("\n变化标签示例:")
            for f in list(change_labels.keys())[:5]:
                key = extract_match_key(f)
                print(f"  匹配键: {key} -> 文件: {f}")

            print("警告: 未能匹配任何变化预测文件和标签！")
            return None

        # 使用多进程并行处理
        print(f"\n并行计算变化检测混淆矩阵...")

        # 计算合适的进程数
        num_processes = min(multiprocessing.cpu_count(), 8, len(matched_files))
        pool = multiprocessing.Pool(processes=num_processes) if num_processes > 1 else None

        # 准备参数
        args_list = [(matched_files[i], matched_labels[i]) for i in range(len(matched_files))]

        # 使用imap_unordered并行处理
        conf_matrix = np.zeros((2, 2), dtype=np.uint64)
        if pool:
            for result in tqdm.tqdm(pool.imap_unordered(process_change_pair, args_list),
                                    total=len(args_list), desc=f"处理T{t1}-T{t2}变化预测"):
                conf_matrix += result
            pool.close()
            pool.join()
        else:
            for result in tqdm.tqdm(args_list, desc=f"处理T{t1}-T{t2}变化预测"):
                conf_matrix += process_change_pair(result)

        # 计算变化检测指标
        try:
            tn, fp, fn, tp = conf_matrix.ravel()
        except ValueError:
            print("错误: 无法计算混淆矩阵")
            return None

        print(f"\n变化检测混淆矩阵:")
        print(f"TN (真负): {tn} | FP (假正): {fp}")
        print(f"FN (假负): {fn} | TP (真正): {tp}")

        # 计算各项指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 计算每个类别的IoU (变化和未变化)
        iou_change = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        iou_nochange = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0

        # 计算平均IoU (mIoU)
        miou = (iou_change + iou_nochange) / 2

        overall_accuracy = (tp + tn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0

        # 打印结果
        print(f"\n变化检测性能指标:")
        print(f"精度 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1_score:.4f}")
        print(f"变化类别IoU: {iou_change:.4f}")
        print(f"未变化类别IoU: {iou_nochange:.4f}")
        print(f"平均IoU (mIoU): {miou:.4f}")
        print(f"总体准确率: {overall_accuracy:.4f}")
        print("=" * 50)

        return {
            'iou_change': iou_change,
            'iou_nochange': iou_nochange,
            'miou': miou,
            'precision': precision,
            'recall': recall,
            'f1': f1_score,
            'oa': overall_accuracy,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    except Exception as e:
        print(f"评估变化检测时出错: {str(e)}")
        traceback.print_exc()
        return None


def save_results(results, save_dir):
    """保存评估结果到文件"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(save_dir, exist_ok=True)
        result_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")

        with open(result_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("六时相土地覆盖分类与变化检测评估结果\n".center(70))
            f.write("=" * 70 + "\n\n")

            # 保存评估信息
            f.write(f"评估时间: {time.ctime()}\n")
            if 'img_count' in results:
                f.write(f"评估图像数量: {results['img_count']}\n")
            f.write(f"双时相组合数量: {len(TIME_PAIRS)}\n")
            f.write("\n")

            # 保存土地覆盖分类结果
            f.write("\n" + "=" * 70 + "\n")
            f.write("土地覆盖分类评估结果\n".center(70) + "\n")
            f.write("=" * 70 + "\n\n")

            # 输出每个时相的结果
            for t in range(1, TIME_POINTS + 1):
                f.write(f"[时相 {t}] 预测文件夹: {PRED_CLASS_DIRS[t]}\n")
                f.write(f"[时相 {t}] 标签文件夹: {LABEL_DIRS[t]}\n")
                f.write(f"时相 {t} mIoU: {results.get(f'class_miou_t{t}', 0):.4f}\n")
                if f'valid_classes_t{t}' in results and results[f'valid_classes_t{t}']:
                    f.write(f"时相 {t} 有效类别数: {len(results[f'valid_classes_t{t}'])}\n\n")
                else:
                    f.write("\n")

            if 'avg_class_miou' in results:
                f.write(f"平均土地覆盖分类 mIoU: {results['avg_class_miou']:.4f}\n\n")

            # 保存变化检测结果
            f.write("\n" + "=" * 70 + "\n")
            f.write("双时相变化检测评估结果\n".center(70) + "\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"双时相变化检测基础目录: {PRED_CHANGE_DIR_BASE}\n\n")

            # 输出所有双时相组合的变化检测结果
            if 'change_metrics' in results:
                for pair in TIME_PAIRS:
                    t1, t2 = pair
                    pair_key = f"pair_{t1}{t2}"
                    metrics = results['change_metrics'].get(pair_key, {})

                    f.write(f"双时相 T{t1}->T{t2} 评估结果:\n")
                    if metrics:
                        f.write(f"  变化IoU: {metrics.get('iou_change', 0):.4f}\n")
                        f.write(f"  未变化IoU: {metrics.get('iou_nochange', 0):.4f}\n")
                        f.write(f"  mIoU: {metrics.get('miou', 0):.4f}\n")
                        f.write(f"  精度 (Precision): {metrics.get('precision', 0):.4f}\n")
                        f.write(f"  召回率 (Recall): {metrics.get('recall', 0):.4f}\n")
                        f.write(f"  F1分数: {metrics.get('f1', 0):.4f}\n")
                        f.write(f"  总体准确率: {metrics.get('oa', 0):.4f}\n")
                        f.write(f"  预测目录: {metrics.get('pred_dir', 'N/A')}\n\n")
                    else:
                        f.write("  评估失败\n\n")

                # 计算并输出平均指标
                ious = [m.get('miou', 0) for m in results['change_metrics'].values() if m]
                precisions = [m.get('precision', 0) for m in results['change_metrics'].values() if m]
                recalls = [m.get('recall', 0) for m in results['change_metrics'].values() if m]
                f1s = [m.get('f1', 0) for m in results['change_metrics'].values() if m]

                if ious:
                    avg_iou = np.mean(ious)
                    avg_precision = np.mean(precisions)
                    avg_recall = np.mean(recalls)
                    avg_f1 = np.mean(f1s)

                    f.write("\n变化检测平均指标:\n")
                    f.write(f"  平均mIoU: {avg_iou:.4f}\n")
                    f.write(f"  平均精度 (Precision): {avg_precision:.4f}\n")
                    f.write(f"  平均召回率 (Recall): {avg_recall:.4f}\n")
                    f.write(f"  平均F1分数: {avg_f1:.4f}\n\n")

            # 详细结果
            f.write("\n" + "=" * 70 + "\n")
            f.write("详细统计结果\n".center(70) + "\n")
            f.write("=" * 70 + "\n")
            for t in range(1, TIME_POINTS + 1):
                if f'valid_classes_t{t}' in results and results[f'valid_classes_t{t}']:
                    f.write(f"时相 {t} 有效类别: {sorted(results[f'valid_classes_t{t}'])}\n")

        print(f"\n评估结果已保存至: {result_file}")
        return result_file
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
        traceback.print_exc()
        return ""


def main():
    """主评估函数"""
    start_time = time.time()

    print("=" * 80)
    print("六时相土地覆盖分类与变化检测评估".center(80))
    print("=" * 80)
    print(f"开始时间: {time.ctime()}")
    print(f"时相数量: {TIME_POINTS}")
    print(f"双时相组合: {TIME_PAIRS}")
    print("忽略土地覆盖分类中的类别: 0 (背景)")
    print(f"使用 {multiprocessing.cpu_count()} 个CPU核心进行并行计算")
    print("文件匹配规则: 区域代码 + 分块代码")
    print("=" * 80)

    # 1. 评估六个时相的土地覆盖分类结果，忽略类别0
    class_results = {}
    for t in range(1, TIME_POINTS + 1):
        print(f"\n评估时相 {t} 的土地覆盖分类...")
        miou, conf_matrix, valid_classes = evaluate_classification(
            PRED_CLASS_DIRS[t], LABEL_DIRS[t], t, ignore_class=0
        )
        class_results[f'class_miou_t{t}'] = miou
        class_results[f'conf_matrix_t{t}'] = conf_matrix
        class_results[f'valid_classes_t{t}'] = valid_classes

    # 计算平均mIoU
    miou_values = [class_results.get(f'class_miou_t{t}', 0) for t in range(1, TIME_POINTS + 1)]
    avg_class_miou = np.mean(miou_values) if any(miou_values) else 0.0

    # 2. 评估所有双时相的变化检测
    change_results = {}
    img_count = 0

    for t1, t2 in TIME_PAIRS:
        print(f"\n{'=' * 80}")
        print(f"评估双时相变化检测 (时相{t1} -> 时相{t2})".center(80))
        print(f"{'=' * 80}\n")

        # 生成真实变化标签
        change_labels = generate_change_labels(LABEL_DIRS[t1], LABEL_DIRS[t2])
        img_count = max(img_count, len(change_labels))

        # 构建预测结果目录
        pred_change_dir = os.path.join(PRED_CHANGE_DIR_BASE, f"pair_{t1}{t2}", "change_map")

        # 检查预测目录是否存在
        if not os.path.exists(pred_change_dir):
            print(f"警告: 预测目录不存在 - {pred_change_dir}")
            continue

        # 评估变化检测结果
        metrics = evaluate_change_detection(pred_change_dir, change_labels, t1, t2)

        if metrics:
            # 添加预测目录信息
            metrics['pred_dir'] = pred_change_dir

            # 记录结果
            pair_key = f"pair_{t1}{t2}"
            change_results[pair_key] = metrics
        else:
            print(f"双时相 T{t1}->T{t2} 评估失败")

    # 3. 汇总结果
    results = {
        # 土地覆盖分类
        'avg_class_miou': avg_class_miou,
        'img_count': img_count,

        # 变化检测
        'change_metrics': change_results,
    }

    # 添加每个时相的分类结果
    for t in range(1, TIME_POINTS + 1):
        results[f'class_miou_t{t}'] = class_results.get(f'class_miou_t{t}', 0.0)
        results[f'valid_classes_t{t}'] = class_results.get(f'valid_classes_t{t}', [])

    # 4. 保存结果
    save_results_dir = os.path.dirname(PRED_CLASS_DIRS[1])
    result_file = save_results(results, save_results_dir)

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n" + "=" * 80)
    print("评估完成!".center(80))
    print("=" * 80)

    if img_count > 0:
        print(f"评估图像数量: {img_count}")
        print(f"平均土地覆盖分类 mIoU (忽略背景): {avg_class_miou:.4f}")
        print(f"评估了 {len(change_results)} 组双时相变化检测")

        # 计算平均变化检测指标
        if change_results:
            avg_miou = np.mean([m['miou'] for m in change_results.values()])
            avg_f1 = np.mean([m['f1'] for m in change_results.values()])
            print(f"平均变化检测 mIoU: {avg_miou:.4f} | F1分数: {avg_f1:.4f}")

    print(f"总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
    if result_file:
        print(f"详细结果已保存至: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    main()
# import os
# import numpy as np
# import multiprocessing
# import tqdm
# import time
# import re
# import rasterio
# from rasterio.windows import Window
# import itertools
#
# # 在这里设置您的路径
# PRED_CLASS_DIR_T1 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/baseline-全部时相/resnet34/semantic_predictions/time_1/"
# PRED_CLASS_DIR_T2 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/baseline-全部时相/resnet34/semantic_predictions/time_2/"
# PRED_CLASS_DIR_T3 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/baseline-全部时相/resnet34/semantic_predictions/time_3/"
# PRED_CLASS_DIR_T4 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/baseline-全部时相/resnet34/semantic_predictions/time_4/"
# PRED_CLASS_DIR_T5 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/baseline-全部时相/resnet34/semantic_predictions/time_5/"
# PRED_CLASS_DIR_T6 = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/baseline-全部时相/resnet34/semantic_predictions/time_6/"
#
# LABEL_DIR_T1 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label1/"
# LABEL_DIR_T2 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label2/"
# LABEL_DIR_T3 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label3/"
# LABEL_DIR_T4 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label4/"
# LABEL_DIR_T5 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label5/"
# LABEL_DIR_T6 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label6/"
#
# # 双时相变化检测的基础目录
# PRED_CHANGE_DIR_BASE = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/baseline-全部时相/resnet34/"
#
# # 定义所有时相的路径
# TIME_POINTS = 6
# PRED_CLASS_DIRS = {
#     1: PRED_CLASS_DIR_T1,
#     2: PRED_CLASS_DIR_T2,
#     3: PRED_CLASS_DIR_T3,
#     4: PRED_CLASS_DIR_T4,
#     5: PRED_CLASS_DIR_T5,
#     6: PRED_CLASS_DIR_T6
# }
#
# LABEL_DIRS = {
#     1: LABEL_DIR_T1,
#     2: LABEL_DIR_T2,
#     3: LABEL_DIR_T3,
#     4: LABEL_DIR_T4,
#     5: LABEL_DIR_T5,
#     6: LABEL_DIR_T6
# }
#
# # 双时相组合
# TIME_PAIRS = list(itertools.combinations(range(1, TIME_POINTS + 1), 2))
#
#
# def extract_match_key(filename):
#     """
#     从文件名中提取匹配键
#     格式: 区域代码 + 分块代码
#     示例:
#         '2697_3715_13_2018-06-01_0.png' -> ('2697_3715_13', '0')
#         '2697_3715_13_2018-02-01_0.tif' -> ('2697_3715_13', '0')
#     """
#     # 移除文件扩展名
#     base_name = os.path.splitext(filename)[0]
#
#     # 分割文件名各部分
#     parts = base_name.split('_')
#
#     # 区域代码是前三部分(如果有至少四部分的话)
#     region_code = '_'.join(parts[:3]) if len(parts) >= 4 else '_'.join(parts[:-1])
#
#     # 分块代码是最后一部分
#     tile_code = parts[-1] if parts else ""
#
#     return (region_code, tile_code)
#
#
# def match_files_by_region_and_tile(pred_dir, label_dir):
#     """
#     根据区域代码和分块代码匹配预测文件和标签文件
#     """
#     # 获取所有文件
#     pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.tif', '.png'))]
#     label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.tif', '.png'))]
#
#     # 创建匹配键到文件路径的映射
#     pred_dict = {}
#     for f in pred_files:
#         key = extract_match_key(f)
#         if key not in pred_dict:
#             pred_dict[key] = []
#         pred_dict[key].append(os.path.join(pred_dir, f))
#
#     label_dict = {}
#     for f in label_files:
#         key = extract_match_key(f)
#         if key not in label_dict:
#             label_dict[key] = []
#         label_dict[key].append(os.path.join(label_dir, f))
#
#     # 找出共同匹配键
#     common_keys = set(pred_dict.keys()) & set(label_dict.keys())
#
#     if not common_keys:
#         # 输出一些调试信息
#         print("预测文件示例:")
#         for k in list(pred_dict.keys())[:5]:
#             print(f"  匹配键: {k} -> 文件: {pred_dict[k]}")
#         print("\n标签文件示例:")
#         for k in list(label_dict.keys())[:5]:
#             print(f"  匹配键: {k} -> 文件: {label_dict[k]}")
#
#         raise ValueError("未找到匹配的文件对! 请检查文件名格式")
#
#     # 创建匹配的文件对
#     matched_preds = []
#     matched_labels = []
#
#     for key in common_keys:
#         # 每个匹配键可能有多个文件，我们取第一个
#         matched_preds.append(pred_dict[key][0])
#         matched_labels.append(label_dict[key][0])
#
#     print(f"成功匹配 {len(matched_preds)} 对文件")
#     return matched_preds, matched_labels
#
#
# def read_tif_as_gray(image_path):
#     """使用rasterio高效读取tif或png图像并转为单通道灰度图"""
#     with rasterio.open(image_path) as src:
#         # 如果是多波段图像，取平均值
#         if src.count > 1:
#             img = np.mean(src.read(), axis=0).astype(np.int16)
#         else:
#             img = src.read(1).astype(np.int16)
#     return img
#
#
# def calculate_iou(conf_matrix, valid_classes=None):
#     """通过混淆矩阵计算IoU，可指定有效类别"""
#     # 计算每个类别的交并比
#     ious = []
#     n_classes = conf_matrix.shape[0]
#
#     if valid_classes is None:
#         valid_classes = range(n_classes)
#     else:
#         valid_classes = set(valid_classes)
#
#     for i in range(n_classes):
#         if i not in valid_classes:
#             ious.append(float('nan'))
#             continue
#
#         tp = conf_matrix[i, i]
#         fp = np.sum(conf_matrix[:, i]) - tp
#         fn = np.sum(conf_matrix[i, :]) - tp
#
#         denominator = tp + fp + fn
#         if denominator == 0:
#             ious.append(float('nan'))  # 忽略未出现的类别
#         else:
#             ious.append(tp / denominator)
#
#     # 计算平均交并比（忽略NaN值）
#     miou = np.nanmean(ious)
#     return ious, miou
#
#
# def generate_change_labels(label_dir1, label_dir2):
#     """根据双时相土地覆盖分类标签生成变化检测标签"""
#     print(f"\n正在生成从{os.path.basename(label_dir1)}到{os.path.basename(label_dir2)}的变化检测标签...")
#
#     # 获取标签文件并匹配
#     label_files1 = [f for f in os.listdir(label_dir1) if f.lower().endswith(('.tif', '.png'))]
#     label_files2 = [f for f in os.listdir(label_dir2) if f.lower().endswith(('.tif', '.png'))]
#
#     # 创建匹配键到文件路径的映射
#     label_dict1 = {}
#     for f in label_files1:
#         key = extract_match_key(f)
#         if key not in label_dict1:
#             label_dict1[key] = []
#         label_dict1[key].append(os.path.join(label_dir1, f))
#
#     label_dict2 = {}
#     for f in label_files2:
#         key = extract_match_key(f)
#         if key not in label_dict2:
#             label_dict2[key] = []
#         label_dict2[key].append(os.path.join(label_dir2, f))
#
#     # 找出共同匹配键
#     common_keys = set(label_dict1.keys()) & set(label_dict2.keys())
#
#     if not common_keys:
#         raise ValueError(f"在 {label_dir1} 和 {label_dir2} 中没有找到共同的文件匹配键！")
#
#     print(f"找到 {len(common_keys)} 对匹配文件")
#
#     # 生成变化标签字典
#     change_labels = {}
#
#     for key in tqdm.tqdm(common_keys, desc="生成变化标签"):
#         # 只取每个匹配键的第一个文件
#         label_path1 = label_dict1[key][0]
#         label_path2 = label_dict2[key][0]
#
#         # 读取土地覆盖分类标签
#         label_img1 = read_tif_as_gray(label_path1)
#         label_img2 = read_tif_as_gray(label_path2)
#
#         # 生成变化标签：像素值一致为未变化(0)，不一致为变化(1)
#         change_label = (label_img1 != label_img2).astype(np.uint8)
#
#         # 获取原始文件名作为键（不带路径）
#         filename = os.path.basename(label_path1)
#         change_labels[filename] = change_label
#
#     print(f"从{os.path.basename(label_dir1)}到{os.path.basename(label_dir2)}生成了 {len(change_labels)} 个变化检测标签")
#     return change_labels
#
#
# def process_image_pair(args):
#     """处理单个图像对并更新混淆矩阵"""
#     pred_path, label_path, ignore_class, n_classes = args
#     # 读取图像
#     pred_img = read_tif_as_gray(pred_path)
#     label_img = read_tif_as_gray(label_path)
#
#     # 创建有效像素掩码 (忽略类别0)
#     valid_mask = (label_img != ignore_class)
#
#     # 提取有效像素
#     pred_flat = pred_img[valid_mask].ravel()
#     label_flat = label_img[valid_mask].ravel()
#
#     # 创建局部混淆矩阵
#     conf_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)
#
#     # 使用向量化操作更新混淆矩阵
#     # 只处理有效的标签（在0到n_classes-1之间）
#     valid_indices = (label_flat >= 0) & (label_flat < n_classes)
#     valid_labels = label_flat[valid_indices]
#     valid_preds = pred_flat[valid_indices]
#
#     if len(valid_labels) > 0:
#         # 使用bincount进行高效统计
#         indices = valid_labels * n_classes + valid_preds
#         counts = np.bincount(indices, minlength=n_classes * n_classes)
#
#         # 确保数据类型匹配
#         counts = counts.astype(np.uint64)
#         conf_matrix = counts.reshape((n_classes, n_classes))
#
#     return conf_matrix
#
#
# def evaluate_classification(pred_dir, label_dir, time_point, ignore_class=0):
#     """评估土地覆盖分类结果，忽略指定类别"""
#     print("\n" + "=" * 70)
#     print(f"时相 {time_point} 土地覆盖分类精度评估".center(70))
#     print("=" * 70)
#
#     # 使用新的匹配规则匹配文件
#     pred_files, label_files = match_files_by_region_and_tile(pred_dir, label_dir)
#
#     if not pred_files:
#         raise ValueError("未找到匹配的文件对! 请检查文件和路径")
#
#     print(f"处理 {len(pred_files)} 对文件")
#
#     # 自动确定类别数量
#     max_class = 0
#     print("正在确定类别数量...")
#     for file_path in tqdm.tqdm(label_files, desc="分析标签"):
#         img = read_tif_as_gray(file_path)
#         max_class = max(max_class, np.max(img))
#
#     n_classes = int(max_class) + 1
#     print(f"检测到 {n_classes} 个类别 (忽略类别 {ignore_class})")
#
#     # 使用多进程并行处理图像对
#     print(f"\n并行计算时相 {time_point} 土地覆盖分类混淆矩阵...")
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
#
#     # 准备参数
#     args_list = [(pred_files[i], label_files[i], ignore_class, n_classes)
#                  for i in range(len(pred_files))]
#
#     # 使用imap_unordered并行处理
#     conf_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)
#     for result in tqdm.tqdm(pool.imap_unordered(process_image_pair, args_list),
#                             total=len(args_list), desc=f"评估T{time_point}"):
#         conf_matrix += result
#
#     pool.close()
#     pool.join()
#
#     # 确定有效类别（所有类别中排除被忽略的类别）
#     valid_classes = [i for i in range(n_classes) if i != ignore_class]
#
#     # 计算mIoU，只考虑有效类别
#     class_ious, miou = calculate_iou(conf_matrix, valid_classes=valid_classes)
#
#     # 打印结果
#     print("\n" + "=" * 50)
#     print(f"时相 {time_point} 土地覆盖分类评估结果 (基于 {len(pred_files)} 对图像)")
#     print(f"总类别数: {n_classes} | 有效类别数: {len(valid_classes)} (忽略类别 {ignore_class})")
#     print(f"\n时相 {time_point} 平均交并比 (mIoU，仅有效类别): {miou:.4f}")
#     print("=" * 50)
#
#     return miou, conf_matrix, valid_classes
#
#
# def process_change_pair(args):
#     """处理单个变化检测图像对并更新混淆矩阵"""
#     pred_path, true_img = args
#     # 读取预测图像
#     pred_img = read_tif_as_gray(pred_path)
#
#     # 二值化预测（非0即1）
#     pred_img = (pred_img > 0).astype(np.uint8)
#
#     # 创建局部混淆矩阵
#     conf_matrix = np.zeros((2, 2), dtype=np.uint64)
#
#     # 使用向量化操作更新混淆矩阵
#     t = true_img.ravel()
#     p = pred_img.ravel()
#
#     # 使用bincount进行高效统计
#     indices = t * 2 + p
#     counts = np.bincount(indices, minlength=4)
#
#     # 确保数据类型匹配
#     counts = counts.astype(np.uint64)
#     conf_matrix = counts.reshape((2, 2))
#
#     return conf_matrix
#
#
# def evaluate_change_detection(pred_dir, change_labels, t1, t2):
#     """评估变化检测结果"""
#     print("\n" + "=" * 70)
#     print(f"双时相变化检测精度评估 (T{t1} -> T{t2})".center(70))
#     print("=" * 70)
#
#     print(f"预测目录: {pred_dir}")
#
#     # 获取变化预测文件
#     pred_files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
#                   if f.lower().endswith(('.tif', '.png'))]
#
#     if not pred_files:
#         raise ValueError("未找到变化预测文件！")
#
#     print(f"找到 {len(pred_files)} 个变化预测文件")
#
#     # 匹配预测文件对应的标签
#     matched_files = []
#     matched_labels = []
#
#     for pred_path in pred_files:
#         file_name = os.path.basename(pred_path)
#         match_key = extract_match_key(file_name)
#
#         # 查找匹配的变化标签
#         for label_file in change_labels.keys():
#             label_match_key = extract_match_key(label_file)
#             if match_key == label_match_key:
#                 matched_files.append(pred_path)
#                 matched_labels.append(change_labels[label_file])
#                 break
#
#     print(f"成功匹配 {len(matched_files)}/{len(pred_files)} 对变化预测和标签")
#
#     if not matched_files:
#         # 输出一些调试信息
#         print("预测文件示例:")
#         for f in pred_files[:5]:
#             key = extract_match_key(os.path.basename(f))
#             print(f"  匹配键: {key} -> 文件: {f}")
#
#         print("\n变化标签示例:")
#         for f in list(change_labels.keys())[:5]:
#             key = extract_match_key(f)
#             print(f"  匹配键: {key} -> 文件: {f}")
#
#         raise ValueError("未能匹配任何变化预测文件和标签！")
#
#     # 使用多进程并行处理
#     print(f"\n并行计算变化检测混淆矩阵...")
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
#
#     # 准备参数
#     args_list = [(matched_files[i], matched_labels[i]) for i in range(len(matched_files))]
#
#     # 使用imap_unordered并行处理
#     conf_matrix = np.zeros((2, 2), dtype=np.uint64)
#     for result in tqdm.tqdm(pool.imap_unordered(process_change_pair, args_list),
#                             total=len(args_list), desc=f"处理T{t1}-T{t2}变化预测"):
#         conf_matrix += result
#
#     pool.close()
#     pool.join()
#
#     # 计算变化检测指标
#     tn, fp, fn, tp = conf_matrix.ravel()
#     print(f"\n变化检测混淆矩阵:")
#     print(f"TN (真负): {tn} | FP (假正): {fp}")
#     print(f"FN (假负): {fn} | TP (真正): {tp}")
#
#     # 计算各项指标
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#     f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#     iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
#     overall_accuracy = (tp + tn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
#
#     # 打印结果
#     print(f"\n变化检测性能指标:")
#     print(f"精度 (Precision): {precision:.4f}")
#     print(f"召回率 (Recall): {recall:.4f}")
#     print(f"F1分数: {f1_score:.4f}")
#     print(f"变化类别IoU: {iou:.4f}")
#     print(f"总体准确率: {overall_accuracy:.4f}")
#     print("=" * 50)
#
#     return {
#         'iou': iou,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1_score,
#         'oa': overall_accuracy,
#         'tn': tn,
#         'fp': fp,
#         'fn': fn,
#         'tp': tp
#     }
#
#
# def save_results(results, save_dir):
#     """保存评估结果到文件"""
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     os.makedirs(save_dir, exist_ok=True)
#     result_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")
#
#     with open(result_file, "w") as f:
#         f.write("=" * 70 + "\n")
#         f.write("六时相土地覆盖分类与变化检测评估结果\n".center(70))
#         f.write("=" * 70 + "\n\n")
#
#         # 保存评估信息
#         f.write(f"评估时间: {time.ctime()}\n")
#         f.write(f"评估图像数量: {results['img_count']}\n")
#         f.write(f"双时相组合数量: {len(TIME_PAIRS)}\n")
#         f.write("\n")
#
#         # 保存土地覆盖分类结果
#         f.write("\n" + "=" * 70 + "\n")
#         f.write("土地覆盖分类评估结果\n".center(70) + "\n")
#         f.write("=" * 70 + "\n\n")
#
#         # 输出每个时相的结果
#         for t in range(1, TIME_POINTS + 1):
#             f.write(f"[时相 {t}] 预测文件夹: {PRED_CLASS_DIRS[t]}\n")
#             f.write(f"[时相 {t}] 标签文件夹: {LABEL_DIRS[t]}\n")
#             f.write(f"时相 {t} mIoU: {results[f'class_miou_t{t}']:.4f}\n")
#             f.write(f"时相 {t} 有效类别数: {len(results[f'valid_classes_t{t}'])}\n\n")
#
#         f.write(f"平均土地覆盖分类 mIoU: {results['avg_class_miou']:.4f}\n\n")
#
#         # 保存变化检测结果
#         f.write("\n" + "=" * 70 + "\n")
#         f.write("双时相变化检测评估结果\n".center(70) + "\n")
#         f.write("=" * 70 + "\n\n")
#         f.write(f"双时相变化检测基础目录: {PRED_CHANGE_DIR_BASE}\n\n")
#
#         # 输出所有双时相组合的变化检测结果
#         for pair in TIME_PAIRS:
#             t1, t2 = pair
#             pair_key = f"pair_{t1}{t2}"
#             metrics = results['change_metrics'][pair_key]
#
#             f.write(f"双时相 T{t1}->T{t2} 评估结果:\n")
#             f.write(f"  变化IoU: {metrics['iou']:.4f}\n")
#             f.write(f"  精度 (Precision): {metrics['precision']:.4f}\n")
#             f.write(f"  召回率 (Recall): {metrics['recall']:.4f}\n")
#             f.write(f"  F1分数: {metrics['f1']:.4f}\n")
#             f.write(f"  总体准确率: {metrics['oa']:.4f}\n")
#             f.write(f"  TN/FP/FN/TP: {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}\n")
#             f.write(f"  预测目录: {metrics['pred_dir']}\n\n")
#
#         # 计算并输出平均指标
#         avg_iou = np.mean([results['change_metrics'][f"pair_{t1}{t2}"]['iou'] for (t1, t2) in TIME_PAIRS])
#         avg_precision = np.mean([results['change_metrics'][f"pair_{t1}{t2}"]['precision'] for (t1, t2) in TIME_PAIRS])
#         avg_recall = np.mean([results['change_metrics'][f"pair_{t1}{t2}"]['recall'] for (t1, t2) in TIME_PAIRS])
#         avg_f1 = np.mean([results['change_metrics'][f"pair_{t1}{t2}"]['f1'] for (t1, t2) in TIME_PAIRS])
#
#         f.write("\n变化检测平均指标:\n")
#         f.write(f"  平均IoU: {avg_iou:.4f}\n")
#         f.write(f"  平均精度 (Precision): {avg_precision:.4f}\n")
#         f.write(f"  平均召回率 (Recall): {avg_recall:.4f}\n")
#         f.write(f"  平均F1分数: {avg_f1:.4f}\n\n")
#
#         # 详细结果
#         f.write("\n" + "=" * 70 + "\n")
#         f.write("详细统计结果\n".center(70) + "\n")
#         f.write("=" * 70 + "\n")
#         for t in range(1, TIME_POINTS + 1):
#             f.write(f"时相 {t} 有效类别: {sorted(results[f'valid_classes_t{t}'])}\n")
#
#     print(f"\n评估结果已保存至: {result_file}")
#     return result_file
#
#
# def main():
#     """主评估函数"""
#     start_time = time.time()
#
#     try:
#         print("=" * 80)
#         print("六时相土地覆盖分类与变化检测评估".center(80))
#         print("=" * 80)
#         print(f"开始时间: {time.ctime()}")
#         print(f"时相数量: {TIME_POINTS}")
#         print(f"双时相组合: {TIME_PAIRS}")
#         print("忽略土地覆盖分类中的类别: 0 (背景)")
#         print(f"使用 {multiprocessing.cpu_count()} 个CPU核心进行并行计算")
#         print("文件匹配规则: 区域代码 + 分块代码")
#         print("=" * 80)
#
#         # 1. 评估六个时相的土地覆盖分类结果，忽略类别0
#         class_results = {}
#         for t in range(1, TIME_POINTS + 1):
#             print(f"\n评估时相 {t} 的土地覆盖分类...")
#             miou, conf_matrix, valid_classes = evaluate_classification(
#                 PRED_CLASS_DIRS[t], LABEL_DIRS[t], t, ignore_class=0
#             )
#             class_results[f'class_miou_t{t}'] = miou
#             class_results[f'conf_matrix_t{t}'] = conf_matrix
#             class_results[f'valid_classes_t{t}'] = valid_classes
#
#         # 计算平均mIoU
#         miou_values = [class_results[f'class_miou_t{t}'] for t in range(1, TIME_POINTS + 1)]
#         avg_class_miou = np.mean(miou_values)
#
#         # 2. 评估所有双时相的变化检测
#         change_results = {}
#
#         for t1, t2 in TIME_PAIRS:
#             print(f"\n{'=' * 80}")
#             print(f"评估双时相变化检测 (时相{t1} -> 时相{t2})".center(80))
#             print(f"{'=' * 80}\n")
#
#             # 生成真实变化标签
#             change_labels = generate_change_labels(LABEL_DIRS[t1], LABEL_DIRS[t2])
#
#             # 构建预测结果目录
#             pred_change_dir = os.path.join(PRED_CHANGE_DIR_BASE, f"pair_{t1}{t2}", "change_map")
#
#             # 检查预测目录是否存在
#             if not os.path.exists(pred_change_dir):
#                 print(f"警告: 预测目录不存在 - {pred_change_dir}")
#                 continue
#
#             # 评估变化检测结果
#             metrics = evaluate_change_detection(pred_change_dir, change_labels, t1, t2)
#
#             # 添加预测目录信息
#             metrics['pred_dir'] = pred_change_dir
#
#             # 记录结果
#             pair_key = f"pair_{t1}{t2}"
#             change_results[pair_key] = metrics
#
#         if not change_results:
#             raise ValueError("未评估任何双时相变化检测结果！请检查预测目录")
#
#         # 3. 汇总结果
#         img_count = len(change_labels)  # 所有时相应有相同数量的图像
#
#         results = {
#             # 土地覆盖分类
#             'avg_class_miou': avg_class_miou,
#             'img_count': img_count,
#
#             # 变化检测
#             'change_metrics': change_results,
#         }
#
#         # 添加每个时相的分类结果
#         for t in range(1, TIME_POINTS + 1):
#             results[f'class_miou_t{t}'] = class_results[f'class_miou_t{t}']
#             results[f'valid_classes_t{t}'] = class_results[f'valid_classes_t{t}']
#
#         # 4. 保存结果
#         save_results_dir = os.path.dirname(PRED_CLASS_DIRS[1])
#         result_file = save_results(results, save_results_dir)
#
#         end_time = time.time()
#         elapsed = end_time - start_time
#
#         print("\n" + "=" * 80)
#         print("评估完成!".center(80))
#         print("=" * 80)
#         print(f"平均土地覆盖分类 mIoU (忽略背景): {avg_class_miou:.4f}")
#         print(f"评估了 {len(change_results)} 组双时相变化检测")
#
#         # 计算平均变化检测指标
#         avg_change_iou = np.mean([m['iou'] for m in change_results.values()])
#         avg_change_f1 = np.mean([m['f1'] for m in change_results.values()])
#
#         print(f"平均变化检测 IoU: {avg_change_iou:.4f} | F1分数: {avg_change_f1:.4f}")
#         print(f"处理图像数量: {img_count}")
#         print(f"总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
#         print(f"详细结果已保存至: {result_file}")
#         print("=" * 80)
#
#     except Exception as e:
#         print(f"\n错误: {str(e)}")
#         import traceback
#         print(traceback.format_exc())
#
#
# if __name__ == "__main__":
#     # 设置多进程启动方法
#     multiprocessing.set_start_method('spawn', force=True)
#     main()