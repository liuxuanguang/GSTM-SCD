import os
import numpy as np
import multiprocessing
import tqdm
import time
import re
import rasterio
import itertools
import traceback


# 双时相变化检测的基础目录
PRED_CHANGE_DIR_BASE = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/DynamicEarth/FC_Siam_diff-全部时相/resnet34/"
# 标签数据路径 - 为每个时相指定
LABEL_DIR_T1 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label1/"
LABEL_DIR_T2 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label2/"
LABEL_DIR_T3 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label3/"
LABEL_DIR_T4 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label4/"
LABEL_DIR_T5 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label5/"
LABEL_DIR_T6 = "/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/val/label6/"
# 时相数量
TIME_POINTS = 6

# 定义所有时相的路径
LABEL_DIRS = {
    1: LABEL_DIR_T1,
    2: LABEL_DIR_T2,
    3: LABEL_DIR_T3,
    4: LABEL_DIR_T4,
    5: LABEL_DIR_T5,
    6: LABEL_DIR_T6
}

# 双时相组合 - 所有可能的配对
TIME_PAIRS = list(itertools.combinations(range(1, TIME_POINTS + 1), 2))


def extract_match_key(filename):
    """
    从文件名中提取匹配键（区域代码 + 分块代码）
    示例:
        '2697_3715_13_2018-06-01_0.png' -> ('2697_3715_13', '0')
    """
    # 移除文件扩展名
    base_name = os.path.splitext(filename)[0]

    # 分割文件名各部分
    parts = base_name.split('_')

    # 区域代码是前三部分(如果有至少四部分)
    if len(parts) >= 4:
        region_code = '_'.join(parts[:3])
        tile_code = parts[-1]
    else:
        region_code = '_'.join(parts[:-1])
        tile_code = parts[-1] if parts else ""

    return (region_code, tile_code)


def match_files_by_region_and_tile(dir1, dir2):
    """
    根据区域代码和分块代码匹配两个目录中的文件
    """
    # 获取所有文件
    files1 = [f for f in os.listdir(dir1) if f.lower().endswith(('.tif', '.png'))]
    files2 = [f for f in os.listdir(dir2) if f.lower().endswith(('.tif', '.png'))]

    # 创建匹配键到文件路径的映射
    dict1 = {}
    for f in files1:
        key = extract_match_key(f)
        if key not in dict1:
            dict1[key] = []
        dict1[key].append(os.path.join(dir1, f))

    dict2 = {}
    for f in files2:
        key = extract_match_key(f)
        if key not in dict2:
            dict2[key] = []
        dict2[key].append(os.path.join(dir2, f))

    # 找出共同匹配键
    common_keys = set(dict1.keys()) & set(dict2.keys())

    if not common_keys:
        # 输出调试信息
        print(f"{os.path.basename(dir1)}文件示例:")
        for k in list(dict1.keys())[:5]:
            print(f"  匹配键: {k} -> 文件: {dict1[k][0]}")
        print(f"\n{os.path.basename(dir2)}文件示例:")
        for k in list(dict2.keys())[:5]:
            print(f"  匹配键: {k} -> 文件: {dict2[k][0]}")

        raise ValueError(f"未找到匹配的文件对! 请检查 {dir1} 和 {dir2} 中的文件名格式")

    print(f"成功匹配 {len(common_keys)} 对文件")
    return common_keys, dict1, dict2


def read_tif_as_gray(image_path):
    """使用rasterio读取图像并转为单通道灰度图"""
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


def generate_change_labels(label_dir1, label_dir2):
    """根据双时相土地覆盖分类标签生成变化检测标签"""
    print(f"\n生成变化检测标签 ({os.path.basename(label_dir1)} -> {os.path.basename(label_dir2)})")

    # 获取匹配的文件对
    common_keys, label_dict1, label_dict2 = match_files_by_region_and_tile(label_dir1, label_dir2)

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

            if label_img1 is None or label_img2 is None or label_img1.shape != label_img2.shape:
                continue
            # 生成变化标签：像素值一致为未变化(0)，不一致为变化(1)
            change_label = (label_img1 != label_img2).astype(np.uint8)

            # 获取原始文件名作为键
            filename = os.path.basename(label_path1)
            change_labels[filename] = change_label
        except Exception as e:
            print(f"处理键 {key} 时出错: {str(e)}")

    print(f"生成 {len(change_labels)} 个变化检测标签")
    return change_labels

def process_change_pair(args):
    """处理单个变化检测图像对并计算混淆矩阵"""
    pred_path, true_img = args
    try:
        # 读取预测图像
        pred_img = read_tif_as_gray(pred_path)
        if pred_img is None:
            return np.zeros((2, 2), dtype=np.uint64)

        # 二值化预测（非0即1）
        pred_img = (pred_img > 0).astype(np.uint8)

        # 处理图像尺寸不一致
        if pred_img.shape != true_img.shape:
            min_height = min(pred_img.shape[0], true_img.shape[0])
            min_width = min(pred_img.shape[1], true_img.shape[1])
            pred_img = pred_img[:min_height, :min_width]
            true_img = true_img[:min_height, :min_width]

        # 创建局部混淆矩阵
        conf_matrix = np.zeros((2, 2), dtype=np.uint64)

        # 展平数组
        t = true_img.ravel()
        p = pred_img.ravel()

        # 使用bincount高效计算混淆矩阵
        indices = t * 2 + p
        counts = np.bincount(indices, minlength=4)
        counts = counts.astype(np.uint64)
        conf_matrix = counts.reshape((2, 2))

        return conf_matrix
    except Exception as e:
        print(f"处理文件 {pred_path} 时出错: {str(e)}")
        return np.zeros((2, 2), dtype=np.uint64)


def evaluate_change_detection(pred_dir, change_labels, t1, t2):
    """评估变化检测结果"""
    print("\n" + "=" * 70)
    print(f"双时相变化检测精度评估 (T{t1} -> T{t2})".center(70))
    print("=" * 70)

    # 检查预测目录
    if not os.path.exists(pred_dir):
        print(f"错误: 预测目录不存在 {pred_dir}")
        return None

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
    unmatched_files = []

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
        if not matched:
            unmatched_files.append(pred_path)
    print(f"成功匹配 {len(matched_files)}/{len(pred_files)} 对变化预测和标签")

    # 显示未匹配文件警告
    if len(unmatched_files) > 0:
        print(f"警告: {len(unmatched_files)} 个预测文件未能匹配到标签文件")
        for f in unmatched_files[:5]:
            print(f"  未匹配文件: {os.path.basename(f)}")
    if not matched_files:
        print("错误: 未能匹配任何预测文件和标签！")
        return None

    # 使用多进程并行处理
    print(f"\n并行计算变化检测混淆矩阵...")

    # 准备参数
    args_list = [(matched_files[i], matched_labels[i]) for i in range(len(matched_files))]

    # 计算合适的进程数
    num_processes = min(multiprocessing.cpu_count(), 8, len(args_list))

    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
        results = []
        with tqdm.tqdm(total=len(args_list), desc=f"处理T{t1}-T{t2}变化预测") as pbar:
            for result in pool.imap_unordered(process_change_pair, args_list):
                results.append(result)
                pbar.update(1)

        pool.close()
        pool.join()
    else:
        results = []
        for args in tqdm.tqdm(args_list, desc=f"处理T{t1}-T{t2}变化预测"):
            results.append(process_change_pair(args))

    # 合并结果
    conf_matrix = sum(results) if results else np.zeros((2, 2), dtype=np.uint64)

    # 计算混淆矩阵各项
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

    # 计算每个类别的IoU
    iou_change = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    iou_nochange = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0

    # 计算平均IoU (mIoU)
    miou = (iou_change + iou_nochange) / 2

    # 计算总体准确率
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
        'tp': tp,
        'num_files': len(matched_files),
        'pred_dir': pred_dir
    }


def save_change_results(results, save_dir):
    """保存变化检测评估结果到文件"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    result_file = os.path.join(save_dir, f"change_detection_results_{timestamp}.txt")

    with open(result_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("双时相变化检测评估结果\n".center(70))
        f.write("=" * 70 + "\n\n")

        # 基本信息
        f.write(f"评估时间: {time.ctime()}\n")
        f.write(f"双时相组合数量: {len(results['change_metrics'])}\n")
        f.write(f"双时相变化检测基础目录: {results['pred_change_dir_base']}\n\n")

        # 统计成功和失败评估
        valid_pairs = [pair for pair, metrics in results['change_metrics'].items() if metrics is not None]
        failed_pairs = [pair for pair, metrics in results['change_metrics'].items() if metrics is None]

        f.write(f"成功评估组合数: {len(valid_pairs)}")
        f.write(f" | 失败评估组合数: {len(failed_pairs)}\n\n")

        # 成功评估的详细结果
        if valid_pairs:
            f.write("\n" + "=" * 70 + "\n")
            f.write("成功评估组合的详细结果\n".center(70))
            f.write("=" * 70 + "\n\n")

            for pair_key in valid_pairs:
                metrics = results['change_metrics'][pair_key]
                # 安全解析pair_key
                if '_' in pair_key:
                    parts = pair_key.split('_')
                    if len(parts) >= 2:
                        t1, t2 = parts[-2], parts[-1]
                    else:
                        t1, t2 = "N/A", "N/A"
                else:
                    t1, t2 = "N/A", "N/A"

                f.write(f"双时相 T{t1}->T{t2} 评估结果:\n")
                f.write(f"  预测目录: {metrics['pred_dir']}\n")
                f.write(f"  评估文件数量: {metrics['num_files']}\n")
                f.write(f"  变化类别IoU: {metrics['iou_change']:.4f}\n")
                f.write(f"  未变化类别IoU: {metrics['iou_nochange']:.4f}\n")
                f.write(f"  平均IoU (mIoU): {metrics['miou']:.4f}\n")
                f.write(f"  精度 (Precision): {metrics['precision']:.4f}\n")
                f.write(f"  召回率 (Recall): {metrics['recall']:.4f}\n")
                f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                f.write(f"  总体准确率: {metrics['oa']:.4f}\n")
                f.write(f"  TN/FP/FN/TP: {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}\n\n")

        # 失败评估的详细情况
        if failed_pairs:
            f.write("\n" + "=" * 70 + "\n")
            f.write("失败评估组合列表\n".center(70))
            f.write("=" * 70 + "\n\n")

            for pair_key in failed_pairs:
                # 安全解析pair_key
                if '_' in pair_key:
                    parts = pair_key.split('_')
                    if len(parts) >= 2:
                        t1, t2 = parts[-2], parts[-1]
                    else:
                        t1, t2 = "N/A", "N/A"
                else:
                    t1, t2 = "N/A", "N/A"

                f.write(f"组合 T{t1}->T{t2} 评估失败\n")

        # 整体统计结果
        if valid_pairs:
            # 计算平均值
            miou_values = [results['change_metrics'][pair]['miou'] for pair in valid_pairs]
            iou_change_values = [results['change_metrics'][pair]['iou_change'] for pair in valid_pairs]
            iou_nochange_values = [results['change_metrics'][pair]['iou_nochange'] for pair in valid_pairs]
            precision_values = [results['change_metrics'][pair]['precision'] for pair in valid_pairs]
            recall_values = [results['change_metrics'][pair]['recall'] for pair in valid_pairs]
            f1_values = [results['change_metrics'][pair]['f1'] for pair in valid_pairs]

            avg_miou = np.mean(miou_values)
            avg_iou_change = np.mean(iou_change_values)
            avg_iou_nochange = np.mean(iou_nochange_values)
            avg_precision = np.mean(precision_values)
            avg_recall = np.mean(recall_values)
            avg_f1 = np.mean(f1_values)

            # 写入统计结果
            f.write("\n" + "=" * 70 + "\n")
            f.write("总体评估结果\n".center(70))
            f.write("=" * 70 + "\n\n")

            f.write(f"平均变化类别IoU: {avg_iou_change:.4f}\n")
            f.write(f"平均未变化类别IoU: {avg_iou_nochange:.4f}\n")
            f.write(f"平均mIoU: {avg_miou:.4f}\n")
            f.write(f"平均精度 (Precision): {avg_precision:.4f}\n")
            f.write(f"平均召回率 (Recall): {avg_recall:.4f}\n")
            f.write(f"平均F1分数: {avg_f1:.4f}\n\n")

            f.write(f"所有组合平均变化检测性能:\n")
            f.write(f"  mIoU: {avg_miou:.4f} | F1: {avg_f1:.4f}\n")

        f.write("\n评估完成时间: " + time.ctime() + "\n")

    print(f"\n变化检测评估结果已保存至: {result_file}")
    return result_file


def main():
    """变化检测评估主函数"""
    start_time = time.time()

    results = {
        'change_metrics': {},  # 存储所有组合的变化检测评估结果
        'pred_change_dir_base': PRED_CHANGE_DIR_BASE
    }

    try:
        print("=" * 80)
        print("双时相变化检测评估".center(80))
        print("=" * 80)
        print(f"开始时间: {time.ctime()}")
        print(f"时相数量: {TIME_POINTS}")
        print(f"双时相组合数: {len(TIME_PAIRS)}")
        print(f"可用CPU核心数: {multiprocessing.cpu_count()}")
        print("=" * 80)

        # 评估所有双时相组合的变化检测
        for t1, t2 in TIME_PAIRS:
            # 创建pair_key用于存储结果
            pair_key = f"pair_{t1}_{t2}"
            print(f"\n{'=' * 80}")
            print(f"评估双时相变化检测 (时相{t1} -> 时相{t2})".center(80))
            print(f"{'=' * 80}")

            try:
                # 生成真实变化标签
                print("\n步骤1: 生成真实变化标签...")
                change_labels = generate_change_labels(LABEL_DIRS[t1], LABEL_DIRS[t2])

                if not change_labels:
                    print("错误: 未能生成变化标签，跳过此组合")
                    results['change_metrics'][pair_key] = None
                    continue

                # 构建预测结果目录 - 这里保持无下划线的格式
                pred_change_dir = os.path.join(PRED_CHANGE_DIR_BASE, f"pair_{t1}{t2}", "change_map")
                print(f"步骤2: 预测目录 - {pred_change_dir}")

                # 检查预测目录是否存在
                if not os.path.exists(pred_change_dir):
                    print(f"错误: 预测目录不存在")
                    results['change_metrics'][pair_key] = None
                    continue

                # 评估变化检测结果
                print("步骤3: 开始评估变化检测...")
                metrics = evaluate_change_detection(pred_change_dir, change_labels, t1, t2)

                if metrics is None:
                    print("错误: 未能完成评估")
                    results['change_metrics'][pair_key] = None
                else:
                    results['change_metrics'][pair_key] = metrics
                    print(f"成功完成评估: 平均mIoU = {metrics['miou']:.4f}")
            except Exception as e:
                print(f"评估失败: {str(e)}")
                results['change_metrics'][pair_key] = None
                traceback.print_exc()

        # 保存结果
        print("\n所有组合评估完成，正在生成最终报告...")
        save_results_dir = os.path.join(PRED_CHANGE_DIR_BASE, "evaluation_results")
        os.makedirs(save_results_dir, exist_ok=True)
        result_file = save_change_results(results, save_results_dir)

        # 统计结果
        valid_results = [metrics for metrics in results['change_metrics'].values() if metrics is not None]
        num_valid = len(valid_results)
        num_total = len(results['change_metrics'])

        end_time = time.time()
        elapsed = end_time - start_time

        # 计算平均性能指标
        if valid_results:
            avg_miou = np.mean([m['miou'] for m in valid_results])
            avg_f1 = np.mean([m['f1'] for m in valid_results])

        print("\n" + "=" * 80)
        print("评估完成!".center(80))
        print("=" * 80)
        print(f"评估组合总数: {num_total}")
        print(f"成功评估组合数: {num_valid}")
        print(f"失败评估组合数: {num_total - num_valid}")
        if valid_results:
            print(f"平均mIoU: {avg_miou:.4f}")
            print(f"平均F1分数: {avg_f1:.4f}")
        print(f"总耗时: {elapsed / 60:.2f} 分钟")
        print(f"详细结果已保存至: {result_file}")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("评估因错误终止".center(80))
        print("=" * 80)
        print(f"错误: {str(e)}")
        traceback.print_exc()
        print("=" * 80)


if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    main()