import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import PIL.Image as Image
import datasets.MultiSiamese_RS_ST_TL as RS
# from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth as Net
# from models.HRSCD_str4 import HRSCD_str4_NMT as Net
# from models.FC_Siam_diff import FC_Siam_diff_dynamic as Net
# from models.EGMSNet import EGMSNet_NMT as Net
# from models.FC_Siam_conv import FC_Siam_conv_Dynamic as Net
# from models.SCanNet import SCanNet_NMT_infer as Net
from models.sitsscd.multiutae import MultiUTAE as Net
# from models.UNet_3D import UNet3D as Net
# from models.HRSCD_str3 import HRSCD_str3_dymamic as Net
# from models.BiSRNet import BiSRNet_NMT_random as Net
# from models.SCanNet import SCanNet_NMT as Net
# from models.SSCDl import SSCDl_NMT as Net
from utils.palette import color_map_DynamicEarth, color_map_WUSU12
from utils.metric import IOUandSek
from tqdm import tqdm
from torch.utils.data import DataLoader
from thop import profile
import time
import argparse
import collections
import time


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default="WUSU")
        parser.add_argument("--Net_name", type=str, default="UTAE-全部时相")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                            help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str,
                            default=r"/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_process/train/DynamicEarth512/test")
        parser.add_argument("--load_from", type=str,
                            default=r"/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/WUSU/UTAE_epoch95_mIOU72.94_Fscd68.25_OA88.70.pth")
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--pretrained", type=bool, default=True,
                            help='initialize the backbone with pretrained parameters')
        parser.add_argument("--tta", dest="tta", action="store_true",
                            help='test_time_augmentation')
        parser.add_argument("--M", type=int, default=6)
        parser.add_argument("--Lambda", type=float, default=0.00005)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args

def inference_scd(args):
    begin_time = time.time()
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)

    # 为当前时相对创建一个目录
    pair_dir = os.path.join(pred_dir, 'pred_LC')
    os.makedirs(pair_dir, exist_ok=True)

    # 创建保存路径
    pred_save_path1 = os.path.join(pair_dir, 'pred_semantic_time1')
    pred_save_path2 = os.path.join(pair_dir, 'pred_semantic_time2')
    pred_save_path3 = os.path.join(pair_dir, 'pred_semantic_time3')

    os.makedirs(pred_save_path1, exist_ok=True)
    os.makedirs(pred_save_path2, exist_ok=True)
    os.makedirs(pred_save_path3, exist_ok=True)

    testset = RS.Data(mode="val", random_flip=False)
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                            pin_memory=True, num_workers=8, drop_last=False)
    # MultiUTAE
    model = Net(input_dim=4, num_classes=len(RS.ST_CLASSES), in_features=512, T=3)
    # UNet3D
    # model = Net(in_channel=4, n_classes=len(RS.ST_CLASSES), timesteps=3, dropout=0.2)
    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)
    model = model.cuda()
    model.eval()
    metric = IOUandSek(num_classes=len(RS.ST_CLASSES) + 1)
    cmap = color_map_WUSU12()

    tbar = tqdm(testloader)
    with torch.no_grad():
        for img1, img2, img3, label1, label2, label3, label_bn, id in tbar:
            # 确保输入数据为float32
            img1, img2, img3 = img1.float().cuda(), img2.float().cuda(), img3.float().cuda()
            # 创建输入张量
            input = [img1, img2, img3]
            #  MultiUTAE:dim=1,UNet3D:dim=2
            input = torch.stack(input, dim=1)
            batch_size = input.size(0)
            # 用于MultiUTAE
            # 创建位置信息并确保为float32
            time_positions = torch.arange(1, 4).float().cuda()  # 添加.float()
            batch_positions = time_positions[None, :, None, None].expand(batch_size, -1, -1, -1)
            batch = {"data": input, "positions": batch_positions}
            # 前向传播
            outs = model(batch)
            out = outs["logits"]
            # 初始化存储列表
            time_series_predictions = []
            # 循环提取每个时相的数据
            for time_idx in range(3):
                # 提取当前时相的预测结果
                time_pred = out[:, time_idx, :, :, :]  # 索引操作
                # assert time_pred.shape == (2, 8, 512, 512)
                # 添加到结果列表
                time_series_predictions.append(time_pred)
            out1 = torch.argmax(time_series_predictions[0], dim=1).cpu().numpy()
            out2 = torch.argmax(time_series_predictions[1], dim=1).cpu().numpy()
            out3 = torch.argmax(time_series_predictions[2], dim=1).cpu().numpy()
            # LCs = [out1.squeeze(0), out2.squeeze(0), out3.squeeze(0)]
            sample_predictions = []
            for i in range(batch_size):
                sample_predictions.append([
                    out1[i],  # [H, W]
                    out2[i],
                    out3[i]
                ])
            metric.add_batch(out1, label1.numpy())
            metric.add_batch(out2, label2.numpy())
            metric.add_batch(out3, label3.numpy())
            ####################################用于MultiUTAE########################################

            ####################################用于UNet3D########################################
            # out = model(input)
            # # 初始化存储列表
            # time_series_predictions = []
            # # 循环提取每个时相的数据
            # for time_idx in range(3):
            #     # 提取当前时相的预测结果
            #     time_pred = out[:, :, time_idx, :, :]  # 索引操作
            #     # assert time_pred.shape == (2, 8, 512, 512)
            #     # 添加到结果列表
            #     time_series_predictions.append(time_pred)
            # out1 = torch.argmax(time_series_predictions[0], dim=1).cpu().numpy()
            # out2 = torch.argmax(time_series_predictions[1], dim=1).cpu().numpy()
            # out3 = torch.argmax(time_series_predictions[2], dim=1).cpu().numpy()
            # sample_predictions = []
            # for i in range(batch_size):
            #     sample_predictions.append([
            #         out1[i],  # [H, W]
            #         out2[i],
            #         out3[i]
            #     ])
            # metric.add_batch(out1, label1.numpy())
            # metric.add_batch(out2, label2.numpy())
            # metric.add_batch(out3, label3.numpy())
            # 保存预测结果
            for idx in range(batch_size):
                lc1, lc2, lc3 = sample_predictions[idx]
                # 保存语义变化图1
                mask1 = Image.fromarray(lc1.astype(np.uint8)).convert('P')
                mask1.putpalette(cmap)
                mask1.save(os.path.join(pred_save_path1, id[idx]))

                mask2 = Image.fromarray(lc2.astype(np.uint8)).convert('P')
                mask2.putpalette(cmap)
                mask2.save(os.path.join(pred_save_path2, id[idx]))

                mask3 = Image.fromarray(lc3.astype(np.uint8)).convert('P')
                mask3.putpalette(cmap)
                mask3.save(os.path.join(pred_save_path3, id[idx]))

        # 计算指标
        change_ratio, OA, mIoU, Sek, Fscd, Score, Precision_scd, Recall_scd = metric.evaluate_inference()
        print('预测完成')
        time_use = time.time() - begin_time

    # 保存当前时相组合的指标
    metric_file = os.path.join(pair_dir, 'metrics.txt')
    with open(metric_file, 'w', encoding='utf-8') as f:
        f.write("##################### Pair Metrics #####################\n")
        f.write("Inference time (s): " + str(round(time_use, 2)) + '\n')
        f.write("Change ratio (%): " + str(round(change_ratio * 100, 2)) + '\n')
        f.write("OA (%): " + str(round(OA * 100, 2)) + '\n')
        f.write("mIoU (%): " + str(round(mIoU * 100, 2)) + '\n')
        f.write("Sek (%): " + str(round(Sek * 100, 2)) + '\n')
        f.write("Fscd (%): " + str(round(Fscd * 100, 2)) + '\n')
        f.write("Score (%): " + str(round(Score * 100, 2)) + '\n')
        f.write("Precision_scd (%): " + str(round(Precision_scd * 100, 2)) + '\n')
        f.write("Recall_scd (%): " + str(round(Recall_scd * 100, 2)) + '\n')

    # 返回指标用于后续计算平均值
    return {
        'time_use': time_use,
        'change_ratio': change_ratio,
        'OA': OA,
        'mIoU': mIoU,
        'Sek': Sek,
        'Fscd': Fscd,
        'Score': Score,
        'Precision_scd': Precision_scd,
        'Recall_scd': Recall_scd
    }


def calculate_global_metrics(all_metrics, pred_dir):
    """计算所有时相组合的平均指标"""
    if not all_metrics:
        print("没有指标可用于计算全局平均值")
        return

    # 初始化指标求和
    metrics_sum = collections.defaultdict(float)

    # 计算指标总和
    for metrics in all_metrics:
        for key in metrics:
            if key != 'pair' and key != 'time_use':  # 排除特殊键
                metrics_sum[key] += metrics[key]

    # 计算平均值
    num_pairs = len(all_metrics)
    avg_metrics = {key: metrics_sum[key] / num_pairs for key in metrics_sum}

    # 添加平均推理时间
    total_time = sum(metrics['time_use'] for metrics in all_metrics)
    avg_time = total_time / num_pairs if num_pairs > 0 else 0

    # 保存全局指标
    global_metric_file = os.path.join(pred_dir, 'global_metrics.txt')
    with open(global_metric_file, 'w', encoding='utf-8') as f:
        f.write("##################### Global Metrics #####################\n")
        f.write(f"Total Time Pairs: {num_pairs}\n")
        f.write("Total Inference time (s): " + str(round(total_time, 2)) + '\n')
        f.write("Average Inference time per pair (s): " + str(round(avg_time, 2)) + '\n')

        for key, value in avg_metrics.items():
            if key in ['change_ratio', 'OA', 'mIoU', 'Sek', 'Fscd', 'Score', 'Precision_scd', 'Recall_scd']:
                percent_value = round(value * 100, 2)
                f.write(f"{key} (%): {percent_value}\n")

        f.write("\nDetailed Pairs Metrics:\n")
        for metrics in all_metrics:
            pair_str = f"{metrics['pair'][0]}-{metrics['pair'][1]}"
            f.write(f"\nPair {pair_str}:\n")
            for key in ['change_ratio', 'OA', 'mIoU', 'Sek', 'Fscd', 'Score', 'Precision_scd', 'Recall_scd']:
                percent_value = round(metrics[key] * 100, 2)
                f.write(f"  {key}: {percent_value}%\n")

    print(f"全局指标已保存到: {global_metric_file}")


if __name__ == "__main__":
    args = Options().parse()
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)
    os.makedirs(pred_dir, exist_ok=True)

    # 收集所有指标
    all_metrics = []
    metrics = inference_scd(args)
    all_metrics.append(metrics)

