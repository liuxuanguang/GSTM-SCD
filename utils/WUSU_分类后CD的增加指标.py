import os
import numpy as np
import tqdm
import time
import itertools
from PIL import Image
from utils.metric import IOUandSek  # 确保从您的metric模块导入IOUandSek


def extract_match_key(filename):
    """从文件名中提取匹配键（区域代码 + 分块代码）"""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    region_code = '_'.join(parts[:3]) if len(parts) >= 4 else '_'.join(parts[:-1])
    tile_code = parts[-1] if (len(parts) >= 4 or parts) else ""
    return (region_code, tile_code)


def match_files_by_region_and_tile(dir1, dir2):
    """根据区域代码和分块代码匹配两个目录中的文件"""
    get_files = lambda d: [f for f in os.listdir(d) if f.lower().endswith(('.tif', '.png'))]

    files1 = get_files(dir1)
    files2 = get_files(dir2)

    # 创建映射字典
    dict1 = {}
    for f in files1:
        key = extract_match_key(f)
        dict1.setdefault(key, []).append(os.path.join(dir1, f))

    dict2 = {}
    for f in files2:
        key = extract_match_key(f)
        dict2.setdefault(key, []).append(os.path.join(dir2, f))

    # 找出共同键
    common_keys = set(dict1.keys()) & set(dict2.keys())

    if not common_keys:
        # 简洁的错误信息
        sample1 = [f"{k}->{dict1[k][0]}" for k in list(dict1.keys())[:2]] if dict1 else []
        sample2 = [f"{k}->{dict2[k][0]}" for k in list(dict2.keys())[:2]] if dict2 else []
        print(f"{os.path.basename(dir1)}示例: {sample1}")
        print(f"{os.path.basename(dir2)}示例: {sample2}")
        raise ValueError(f"未找到匹配文件对! 检查 {dir1} 和 {dir2}")

    print(f"成功匹配 {len(common_keys)} 对文件")
    return common_keys, dict1, dict2


def calculate_bcd_metrics(tp, fp, fn, tn):
    """计算二值变化检测指标"""
    # 计算指标
    total = tp + fp + fn + tn

    # 精度 (Precision)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 召回率 (Recall)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 总体准确率 (OA)
    oa = (tp + tn) / total if total > 0 else 0

    # F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 变化类IoU
    iou_change = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # 未变化类IoU
    iou_nochange = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0

    # 平均IoU (mIoU)
    miou = (iou_change + iou_nochange) / 2

    return {
        'oa': oa,
        'miou': miou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'iou_change': iou_change
    }


def evaluate_all_pairs(pred_dirs, label_dirs, output_dir, num_classes=13):
    """评估所有双时相组合的二值变化检测和语义变化检测精度"""
    start_time = time.time()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 双时相组合 (6个时相对应C(6,2)=15个组合)
    time_pairs = list(itertools.combinations(range(1, 4), 2))

    # 存储所有组合的结果
    all_results = {}

    # 处理每个双时相组合
    for t1, t2 in time_pairs:
        print(f"\n处理时相组合 T{t1} - T{t2}")
        print(f"预测目录1 (T{t1}): {pred_dirs[t1]}")
        print(f"预测目录2 (T{t2}): {pred_dirs[t2]}")
        print(f"标签目录1 (T{t1}): {label_dirs[t1]}")
        print(f"标签目录2 (T{t2}): {label_dirs[t2]}")

        try:
            # 匹配文件
            common_keys_t1, pred_dict_t1, label_dict_t1 = match_files_by_region_and_tile(pred_dirs[t1], label_dirs[t1])
            common_keys_t2, pred_dict_t2, label_dict_t2 = match_files_by_region_and_tile(pred_dirs[t2], label_dirs[t2])

            # 获取共同键 (两个时相都有数据的区域)
            common_keys = common_keys_t1 & common_keys_t2
            if not common_keys:
                print(f"警告: 时相组合 T{t1}-T{t2} 没有共同匹配键，跳过评估")
                continue

            print(f"找到 {len(common_keys)} 个有效图像对")

            # 初始化全局混淆矩阵元素 (用于二值变化检测)
            global_tp, global_fp, global_fn, global_tn = 0, 0, 0, 0

            # 初始化语义变化检测指标计算器
            scd_metric = IOUandSek(num_classes=num_classes)

            # 处理每个匹配的图像对
            for key in tqdm.tqdm(common_keys, desc=f"处理图像对 T{t1}-T{t2}"):
                # 获取文件路径
                pred_t1_path = pred_dict_t1[key][0]
                pred_t2_path = pred_dict_t2[key][0]
                label_t1_path = label_dict_t1[key][0]
                label_t2_path = label_dict_t2[key][0]

                # 读取图像
                pred_t1 = np.array(Image.open(pred_t1_path))
                pred_t2 = np.array(Image.open(pred_t2_path))
                label_t1 = np.array(Image.open(label_t1_path))
                label_t2 = np.array(Image.open(label_t2_path))

                # 统一图像尺寸
                min_height = min(pred_t1.shape[0], pred_t2.shape[0], label_t1.shape[0], label_t2.shape[0])
                min_width = min(pred_t1.shape[1], pred_t2.shape[1], label_t1.shape[1], label_t2.shape[1])

                pred_t1 = pred_t1[:min_height, :min_width]
                pred_t2 = pred_t2[:min_height, :min_width]
                label_t1 = label_t1[:min_height, :min_width]
                label_t2 = label_t2[:min_height, :min_width]
                # 生成变化检测结果（比较两个时相的分类结果）
                change_pred = (pred_t1 != pred_t2).astype(np.uint8)
                # 生成SCD变化检测结果
                pred_t1[change_pred == 0] = 0
                pred_t2[change_pred == 0] = 0
                # 生成真实变化标签（比较两个时相的标签）
                change_gt = (label_t1 != label_t2).astype(np.uint8)

                # 计算当前图像的混淆矩阵元素 (二值变化检测)
                TP = np.sum((change_pred == 1) & (change_gt == 1))
                FP = np.sum((change_pred == 1) & (change_gt == 0))
                FN = np.sum((change_pred == 0) & (change_gt == 1))
                TN = np.sum((change_pred == 0) & (change_gt == 0))

                # 累加到全局混淆矩阵
                global_tp += TP
                global_fp += FP
                global_fn += FN
                global_tn += TN

                # 为语义变化检测准备数据
                # 创建掩膜后的标签（只保留变化区域）
                label_t1_masked = label_t1.copy()
                label_t1_masked[change_gt == 0] = 0
                label_t2_masked = label_t2.copy()
                label_t2_masked[change_gt == 0] = 0

                # 添加到语义变化检测指标计算器
                scd_metric.add_batch(pred_t1, label_t1_masked)
                scd_metric.add_batch(pred_t2, label_t2_masked)

            # 如果所有图像都没有有效像素，跳过该组合
            total = global_tp + global_fp + global_fn + global_tn
            if total == 0:
                print(f"警告: 时相组合 T{t1}-T{t2} 所有图像都没有有效像素，跳过")
                continue

            # 计算整个组合的二值变化检测指标
            bcd_metrics = calculate_bcd_metrics(global_tp, global_fp, global_fn, global_tn)

            # 计算整个组合的语义变化检测指标
            scd_results = scd_metric.evaluate_inference()
            change_ratio, oa, miou, sek, fscd, score, precision_scd, recall_scd = scd_results

            # 存储结果
            all_results[(t1, t2)] = {
                'bcd': bcd_metrics,
                'scd': {
                    'change_ratio': change_ratio,
                    'oa': oa,
                    'miou': miou,
                    'sek': sek,
                    'fscd': fscd,
                    'score': score,
                    'precision': precision_scd,
                    'recall': recall_scd
                }
            }

            print(f"\n组合 T{t1}-T{t2} 结果:")
            print("二值变化检测:")
            print(f"  总体准确率 (OA): {bcd_metrics['oa']:.4f}")
            print(f"  平均交并比 (mIoU): {bcd_metrics['miou']:.4f}")
            print(f"  F1分数: {bcd_metrics['f1']:.4f}")
            print(f"  精度: {bcd_metrics['precision']:.4f}")
            print(f"  召回率: {bcd_metrics['recall']:.4f}")
            print(f"  变化类IoU: {bcd_metrics['iou_change']:.4f}")

            print("\n语义变化检测:")
            print(f"  变化比例: {change_ratio:.4f}")
            print(f"  总体准确率 (OA): {oa:.4f}")
            print(f"  平均交并比 (mIoU): {miou:.4f}")
            print(f"  空间精度 (Sek): {sek:.4f}")
            print(f"  Fscd分数: {fscd:.4f}")
            print(f"  语义得分: {score:.4f}")
            print(f"  精度: {precision_scd:.4f}")
            print(f"  召回率: {recall_scd:.4f}")

        except Exception as e:
            print(f"处理组合 T{t1}-T{t2} 时出错: {str(e)}")
            continue

    # 如果没有任何组合成功评估
    if not all_results:
        print("警告: 没有任何时相组合成功评估，检查文件路径和匹配情况")
        return None, None

    # 计算平均指标
    avg_bcd_metrics = {
        'oa': np.mean([r['bcd']['oa'] for r in all_results.values()]),
        'miou': np.mean([r['bcd']['miou'] for r in all_results.values()]),
        'f1': np.mean([r['bcd']['f1'] for r in all_results.values()]),
        'precision': np.mean([r['bcd']['precision'] for r in all_results.values()]),
        'recall': np.mean([r['bcd']['recall'] for r in all_results.values()]),
        'iou_change': np.mean([r['bcd']['iou_change'] for r in all_results.values()]),
    }

    avg_scd_metrics = {
        'change_ratio': np.mean([r['scd']['change_ratio'] for r in all_results.values()]),
        'oa': np.mean([r['scd']['oa'] for r in all_results.values()]),
        'miou': np.mean([r['scd']['miou'] for r in all_results.values()]),
        'sek': np.mean([r['scd']['sek'] for r in all_results.values()]),
        'fscd': np.mean([r['scd']['fscd'] for r in all_results.values()]),
        'score': np.mean([r['scd']['score'] for r in all_results.values()]),
        'precision': np.mean([r['scd']['precision'] for r in all_results.values()]),
        'recall': np.mean([r['scd']['recall'] for r in all_results.values()]),
    }

    # 保存结果到文件
    result_file = os.path.join(output_dir, "all_pairs_results.txt")
    with open(result_file, "w") as f:
        f.write("变化检测评估结果 (所有双时相组合)\n")
        f.write("=" * 80 + "\n")
        f.write(f"评估时间: {time.ctime()}\n")
        f.write(f"时相组合数量: {len(all_results)}\n")
        f.write(f"类别数量: {num_classes}\n")
        f.write("\n")

        # 详细结果
        f.write("各组合详细结果:\n")
        for (t1, t2), metrics in all_results.items():
            f.write(f"组合 T{t1}-T{t2}:\n")

            f.write("二值变化检测:\n")
            f.write(f"  总体准确率 (OA): {metrics['bcd']['oa']:.4f}\n")
            f.write(f"  平均交并比 (mIoU): {metrics['bcd']['miou']:.4f}\n")
            f.write(f"  F1分数: {metrics['bcd']['f1']:.4f}\n")
            f.write(f"  精度: {metrics['bcd']['precision']:.4f}\n")
            f.write(f"  召回率: {metrics['bcd']['recall']:.4f}\n")
            f.write(f"  变化类IoU: {metrics['bcd']['iou_change']:.4f}\n")

            f.write("\n语义变化检测:\n")
            f.write(f"  变化比例: {metrics['scd']['change_ratio']:.4f}\n")
            f.write(f"  总体准确率 (OA): {metrics['scd']['oa']:.4f}\n")
            f.write(f"  平均交并比 (mIoU): {metrics['scd']['miou']:.4f}\n")
            f.write(f"  空间精度 (Sek): {metrics['scd']['sek']:.4f}\n")
            f.write(f"  Fscd分数: {metrics['scd']['fscd']:.4f}\n")
            f.write(f"  语义得分: {metrics['scd']['score']:.4f}\n")
            f.write(f"  精度: {metrics['scd']['precision']:.4f}\n")
            f.write(f"  召回率: {metrics['scd']['recall']:.4f}\n")
            f.write("-" * 80 + "\n")

        # 平均结果
        f.write("\n平均结果 (所有双时相组合):\n")
        f.write("二值变化检测:\n")
        f.write(f"  平均总体准确率: {avg_bcd_metrics['oa']:.4f}\n")
        f.write(f"  平均交并比 (mIoU): {avg_bcd_metrics['miou']:.4f}\n")
        f.write(f"  平均F1分数: {avg_bcd_metrics['f1']:.4f}\n")
        f.write(f"  平均精度: {avg_bcd_metrics['precision']:.4f}\n")
        f.write(f"  平均召回率: {avg_bcd_metrics['recall']:.4f}\n")
        f.write(f"  平均变化类IoU: {avg_bcd_metrics['iou_change']:.4f}\n")

        f.write("\n语义变化检测:\n")
        f.write(f"  平均变化比例: {avg_scd_metrics['change_ratio']:.4f}\n")
        f.write(f"  平均总体准确率: {avg_scd_metrics['oa']:.4f}\n")
        f.write(f"  平均交并比 (mIoU): {avg_scd_metrics['miou']:.4f}\n")
        f.write(f"  平均空间精度 (Sek): {avg_scd_metrics['sek']:.4f}\n")
        f.write(f"  平均Fscd分数: {avg_scd_metrics['fscd']:.4f}\n")
        f.write(f"  平均语义得分: {avg_scd_metrics['score']:.4f}\n")
        f.write(f"  平均精度: {avg_scd_metrics['precision']:.4f}\n")
        f.write(f"  平均召回率: {avg_scd_metrics['recall']:.4f}\n")

    # 计算总耗时
    elapsed = time.time() - start_time
    print(f"\n评估完成! 总耗时: {elapsed:.2f}秒")
    print(f"结果已保存至: {result_file}")

    return all_results, {'bcd': avg_bcd_metrics, 'scd': avg_scd_metrics}


if __name__ == "__main__":
    # 设置路径 - 时相1-6预测目录
    PRED_DIRS = {
        1: "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/TSSCD-全部时相/resnet34/pred_LC/pred_semantic_time1/",
        2: "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/TSSCD-全部时相/resnet34/pred_LC/pred_semantic_time2/",
        3: "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/TSSCD-全部时相/resnet34/pred_LC/pred_semantic_time3/"
    }

    # 设置路径 - 时相1-6标签目录
    LABEL_DIRS = {
        1: "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label1/",
        2: "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label2/",
        3: "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/wusu512_process/NewWUSU/val/label3/"
    }

    # 设置输出目录
    OUTPUT_DIR = "/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/pred_results/WUSU/TSSCD-全部时相/resnet34/pred_SCD/"

    # 设置类别数量 (根据您的数据集)
    NUM_CLASSES = 13  # 包括背景类别

    # 运行评估
    results, avg_results = evaluate_all_pairs(PRED_DIRS, LABEL_DIRS, OUTPUT_DIR, num_classes=NUM_CLASSES)

    # 打印平均结果
    if avg_results:
        print("\n所有双时相组合的平均结果:")
        print("二值变化检测:")
        print(f"  平均总体准确率: {avg_results['bcd']['oa']:.4f}")
        print(f"  平均交并比 (mIoU): {avg_results['bcd']['miou']:.4f}")
        print(f"  平均F1分数: {avg_results['bcd']['f1']:.4f}")
        print(f"  平均精度: {avg_results['bcd']['precision']:.4f}")
        print(f"  平均召回率: {avg_results['bcd']['recall']:.4f}")
        print(f"  平均变化类IoU: {avg_results['bcd']['iou_change']:.4f}")

        print("\n语义变化检测:")
        print(f"  平均变化比例: {avg_results['scd']['change_ratio']:.4f}")
        print(f"  平均总体准确率: {avg_results['scd']['oa']:.4f}")
        print(f"  平均交并比 (mIoU): {avg_results['scd']['miou']:.4f}")
        print(f"  平均空间精度 (Sek): {avg_results['scd']['sek']:.4f}")
        print(f"  平均Fscd分数: {avg_results['scd']['fscd']:.4f}")
        print(f"  平均语义得分: {avg_results['scd']['score']:.4f}")
        print(f"  平均精度: {avg_results['scd']['precision']:.4f}")
        print(f"  平均召回率: {avg_results['scd']['recall']:.4f}")
    else:
        print("\n未能计算平均结果，请检查输入数据")