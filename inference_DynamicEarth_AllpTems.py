import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import PIL.Image as Image
# import datasets.MultiSiamese_RS_ST_TL_DynamocEarth as RS
import datasets.MultiSiamese_RS_ST_TL_DynamocEarth_random as RS
from models.GSTMSCD_MTSCD import GSTMSCD_Dynamic as Net
# from models.proposed_MTGrootV3D import GSTMSCD_baseline as Net
# from models.HRSCD_str4 import HRSCD_str4_NMT as Net
# from models.EGMSNet import EGMSNet_NMT as Net
# from models.SCanNet import SCanNet_NMT_infer as Net
# from models.HRSCD_str3 import HRSCD_str3_dymamic as Net
# from models.BiSRNet import BiSRNet_NMT_random as Net
# from models.SCanNet import SCanNet_NMT as Net
# from models.SSCDl import SSCDl_NMT as Net
# from models.FEMCD.FEMCD import FEMCD_net_dynamic as Net
# from models.FEMCD.FEMCD import FEMCD_net_dynamic as Net
from utils.palette import color_map_DynamicEarth
from utils.metric import IOUandSek
from tqdm import tqdm
from torch.utils.data import DataLoader
from thop import profile
import time
import argparse
import copy
import collections
import time


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default="DynamicEarth")
        parser.add_argument("--Net_name", type=str, default="GSTMSCD")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                            help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str,
                            default=r"/DynamicEarthNet")
        parser.add_argument("--load_from", type=str,
                            default=r"best_model.pth")
        parser.add_argument("--test_batch_size", type=int, default=4)
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


def inference_ss(args, pair):
    begin_time = time.time()
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)
    semantic_dir = os.path.join(pred_dir, 'semantic_predictions')
    os.makedirs(semantic_dir, exist_ok=True)
    for i in range(1, 7):
        time_dir = os.path.join(semantic_dir, f'time_{i}')
        os.makedirs(time_dir, exist_ok=True)

    testset = RS.Data(mode="val", random_flip=False)
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                            pin_memory=True, num_workers=8, drop_last=False)
    model = Net(args.backbone, args.pretrained, len(RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
    # model = Net(4, num_classes=len(RS.ST_CLASSES))
    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)

    model = model.cuda()
    model.eval()
    
    if args.test_batch_size > 0:
        for vi, data in enumerate(testloader):
            if vi == 0:
                img1, img2, img3, img4, img5, img6, label1, label2, label3, label4, label5, label6, label_bn, id = data
                img1, img2, img3, img4, img5, img6 = img1.cuda().float(), img2.cuda().float(), img3.cuda().float(), img4.cuda().float(), img5.cuda().float(), img6.cuda().float()
                FLOPs, Params = profile(model, (img1, img2, img3, img4, img5, img6, pair))
                print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
                break

    cmap = color_map_DynamicEarth()
    tbar = tqdm(testloader)
    with torch.no_grad():
        for img1, img2, img3, img4, img5, img6, label1, label2, label3, label4, label5, label6, label_bn, id in tbar:
            img1, img2, img3, img4, img5, img6 = img1.cuda().float(), img2.cuda().float(), img3.cuda().float(), img4.cuda().float(), img5.cuda().float(), img6.cuda().float()
            # img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
            out1, out2, out3, out4, out5, out6, out_bn = model(img1, img2, img3, img4, img5, img6, pair)

            outputs = [torch.argmax(out, dim=1).cpu().numpy() for out in [out1, out2, out3, out4, out5, out6]]
            for time_idx in range(6):
                time_dir = os.path.join(semantic_dir, f'time_{time_idx + 1}')
                for i in range(outputs[0].shape[0]):
                    mask = Image.fromarray(outputs[time_idx][i].astype(np.uint8)).convert('P')
                    mask.putpalette(cmap)
                    mask.save(os.path.join(time_dir, id[i]))
def inference_scd(args, pair):
    begin_time = time.time()
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)

    pair_dir = os.path.join(pred_dir, f'pair_{pair[0]+1}{pair[1]+1}')
    os.makedirs(pair_dir, exist_ok=True)

    pred_save_path1 = os.path.join(pair_dir, 'pred_semantic_time1')
    pred_save_path2 = os.path.join(pair_dir, 'pred_semantic_time2')
    pred_save_pathcd = os.path.join(pair_dir, 'change_map')
    os.makedirs(pred_save_path1, exist_ok=True)
    os.makedirs(pred_save_path2, exist_ok=True)
    os.makedirs(pred_save_pathcd, exist_ok=True)

    testset = RS.Data(mode="val", random_flip=False)
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                            pin_memory=True, num_workers=8, drop_last=False)
    model = Net(args.backbone, args.pretrained, len(RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
    # model = Net(4, num_classes=len(RS.ST_CLASSES))
    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)

    model = model.cuda()
    model.eval()

    metric = IOUandSek(num_classes=len(RS.ST_CLASSES) + 1)
    cmap = color_map_DynamicEarth()

    tbar = tqdm(testloader)
    with torch.no_grad():
        for img1, img2, img3, img4, img5, img6, label1, label2, label3, label4, label5, label6, label_bn, id in tbar:
            img1, img2, img3, img4, img5, img6 = img1.cuda().float(), img2.cuda().float(), img3.cuda().float(), img4.cuda().float(), img5.cuda().float(), img6.cuda().float()
            # img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
            out1, out2, out3, out4, out5, out6, out_bn = model(img1, img2, img3, img4, img5, img6, pair)

            seg_predictions = [torch.argmax(out, dim=1).cpu().numpy() for out in [out1, out2, out3, out4, out5, out6]]
            change_prediction = (out_bn > 0.5).cpu().numpy().astype(np.uint8)

            time1_pred = seg_predictions[pair[0]]
            time2_pred = seg_predictions[pair[1]]

            time1_pred_masked = time1_pred.copy()
            time2_pred_masked = time2_pred.copy()
            time1_pred_masked[change_prediction == 0] = 0
            time2_pred_masked[change_prediction == 0] = 0
            
            labels_np = {
                'time1': label1.numpy() if pair[0] == 0 else
                label2.numpy() if pair[0] == 1 else
                label3.numpy() if pair[0] == 2 else
                label4.numpy() if pair[0] == 3 else
                label5.numpy() if pair[0] == 4 else
                label6.numpy(),
                'time2': label1.numpy() if pair[1] == 0 else
                label2.numpy() if pair[1] == 1 else
                label3.numpy() if pair[1] == 2 else
                label4.numpy() if pair[1] == 3 else
                label5.numpy() if pair[1] == 4 else
                label6.numpy(),
                'change': label_bn.numpy()
            }

            time1_gt_masked = labels_np['time1'].copy()
            time2_gt_masked = labels_np['time2'].copy()
            time1_gt_masked[labels_np['change'] == 0] = 0
            time2_gt_masked[labels_np['change'] == 0] = 0

            metric.add_batch(time1_pred_masked, time1_gt_masked)
            metric.add_batch(time2_pred_masked, time2_gt_masked)

            for idx in range(time1_pred.shape[0]):
                mask1 = Image.fromarray(time1_pred_masked[idx].astype(np.uint8)).convert('P')
                mask1.putpalette(cmap)
                mask1.save(os.path.join(pred_save_path1, id[idx]))

                mask2 = Image.fromarray(time2_pred_masked[idx].astype(np.uint8)).convert('P')
                mask2.putpalette(cmap)
                mask2.save(os.path.join(pred_save_path2, id[idx]))
                
                change_img = Image.fromarray(change_prediction[idx] * 255)
                change_img.save(os.path.join(pred_save_pathcd, id[idx]))

        change_ratio, OA, mIoU, Sek, Fscd, Score, Precision_scd, Recall_scd = metric.evaluate_inference()
        print(f'时相T{pair[0]+1}-T{pair[1]+1}预测完成')
        time_use = time.time() - begin_time

    metric_file = os.path.join(pair_dir, 'metrics.txt')
    with open(metric_file, 'w', encoding='utf-8') as f:
        f.write("##################### Pair Metrics #####################\n")
        f.write(f"Time Pair: {pair[0]} -> {pair[1]}\n")
        f.write("Inference time (s): " + str(round(time_use, 2)) + '\n')
        f.write("Change ratio (%): " + str(round(change_ratio * 100, 2)) + '\n')
        f.write("OA (%): " + str(round(OA * 100, 2)) + '\n')
        f.write("mIoU (%): " + str(round(mIoU * 100, 2)) + '\n')
        f.write("Sek (%): " + str(round(Sek * 100, 2)) + '\n')
        f.write("Fscd (%): " + str(round(Fscd * 100, 2)) + '\n')
        f.write("Score (%): " + str(round(Score * 100, 2)) + '\n')
        f.write("Precision_scd (%): " + str(round(Precision_scd * 100, 2)) + '\n')
        f.write("Recall_scd (%): " + str(round(Recall_scd * 100, 2)) + '\n')

    return {
        'pair': (pair[0], pair[1]),
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
    """Calculate the average metrics for all phase combinations"""
    if not all_metrics:
        print("没有指标可用于计算全局平均值")
        return

    metrics_sum = collections.defaultdict(float)

    for metrics in all_metrics:
        for key in metrics:
            if key != 'pair' and key != 'time_use':  # 排除特殊键
                metrics_sum[key] += metrics[key]

    num_pairs = len(all_metrics)
    avg_metrics = {key: metrics_sum[key] / num_pairs for key in metrics_sum}
    
    total_time = sum(metrics['time_use'] for metrics in all_metrics)
    avg_time = total_time / num_pairs if num_pairs > 0 else 0

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

    all_metrics = []

    # Process all phases
    for i in range(6):
        for j in range(6):
            if i == 0 and j == 0:
                # 第一个时相，语义分割
                pair = [i, j]
                inference_ss(args, pair)
            elif i != j and i < j:
                # 时相变化检测
                pair = [i, j]
                print(f"\n{'=' * 50}")
                print(f"处理时相组合: T{i} -> T{j}")
                print(f"{'=' * 50}\n")

                metrics = inference_scd(args, pair)
                all_metrics.append(metrics)
            else:
                continue

    if all_metrics:
        calculate_global_metrics(all_metrics, pred_dir)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torch
# import numpy as np
# import PIL.Image as Image
# import datasets.MultiSiamese_RS_ST_TL_DynamocEarth as RS
# # from models.proposed_BiGrootV import BiGrootV_V2 as Net
# # from models.proposed_V1 import BiGrootV as Net
# # from models.proposed_BiGrootV import BiGrootV_V5Base as Net
# # from models.proposed_BiGrootV import BiGrootV_V5 as Net
# # from models.EGMSNet import EGMSNet_NMT as Net
# # from models.proposed_BiGrootV import BiGrootV3D_V2 as Net
# # from models.SSCDl import SSCDl_NMT as Net
# # from models.BiSRNet import BiSRNet_NMT as Net
# # from models.proposed_MTGrootV3D import MTGrootV3D_SV4_DynamicEarth as Net
# # from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth_random_base as Net
# # from models.SCanNet import SCanNet_NMT as Net
# # from models.TED import TED_NMT as Net
# from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth as Net
# from utils.palette import color_map_DynamicEarth
# from utils.metric import IOUandSek
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from thop import profile
# import time
# import argparse
# import copy
#
# class Options:
#     def __init__(self):
#         parser = argparse.ArgumentParser('Semantic Change Detection')
#         parser.add_argument("--data_name", type=str, default="DynamicEarth")
#         parser.add_argument("--Net_name", type=str, default="GSTMSCD-全部时相测试")
#         parser.add_argument("--lightweight", dest="lightweight", action="store_true",
#                            help='lightweight head for fewer parameters and faster speed')
#         parser.add_argument("--backbone", type=str, default="resnet34")
#         parser.add_argument("--data_root", type=str, default=r"/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_4band/train/DynamicEarth512/test")   #/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_process/train/DynamicEarth512/test
#         parser.add_argument("--load_from", type=str,
#                             default=r"/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/DynamicEarth/Proposed_epoch94_Score43.75_mIOU76.74_Sek29.61_Fscd69.46_OA88.69.pth")
#         parser.add_argument("--test_batch_size", type=int, default=4)
#         parser.add_argument("--pretrained", type=bool, default=True,
#                            help='initialize the backbone with pretrained parameters')
#         parser.add_argument("--tta", dest="tta", action="store_true",
#                            help='test_time_augmentation')
#         parser.add_argument("--M", type=int, default=6)
#         parser.add_argument("--Lambda", type=float, default=0.00005)
#         self.parser = parser
#
#     def parse(self):
#         args = self.parser.parse_args()
#         print(args)
#         return args
#
# def inference_ss(args, pair):
#     begin_time = time.time()
#     working_path = os.path.dirname(os.path.abspath(__file__))
#     pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)
#     pred_save_path1_semantic = os.path.join(pred_dir, 'pred1_semantic')
#     pred_save_path2_semantic = os.path.join(pred_dir, 'pred2_semantic')
#     pred_save_path3_semantic = os.path.join(pred_dir, 'pred3_semantic')
#     pred_save_path4_semantic = os.path.join(pred_dir, 'pred4_semantic')
#     pred_save_path5_semantic = os.path.join(pred_dir, 'pred5_semantic')
#     pred_save_path6_semantic = os.path.join(pred_dir, 'pred6_semantic')
#
#     if not os.path.exists(pred_save_path1_semantic): os.makedirs(pred_save_path1_semantic)
#     if not os.path.exists(pred_save_path2_semantic): os.makedirs(pred_save_path2_semantic)
#     if not os.path.exists(pred_save_path3_semantic): os.makedirs(pred_save_path3_semantic)
#     if not os.path.exists(pred_save_path4_semantic): os.makedirs(pred_save_path4_semantic)
#     if not os.path.exists(pred_save_path5_semantic): os.makedirs(pred_save_path5_semantic)
#     if not os.path.exists(pred_save_path6_semantic): os.makedirs(pred_save_path6_semantic)
#
#     testset = RS.Data(mode="val", pair=pair, random_flip=False)
#     testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
#                                 pin_memory=True, num_workers=0, drop_last=False)
#     model = Net(args.backbone, args.pretrained, len(RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
#     # model = Net(in_channels=4, num_classes=7)
#     if args.load_from:
#         model.load_state_dict(torch.load(args.load_from), strict=True)
#
#     model = model.cuda()
#     model.eval()
#
#     # calculate Pamrams and FLOPs
#     for vi, data in enumerate(testloader):
#         if vi == 0:
#             img1, img2, img3, img4, img5, img6, label1, label2, label3, label4, label5, label6, label_bn, id = data
#             img1, img2, img3, img4, img5, img6 = img1.cuda().float(), img2.cuda().float(), img3.cuda().float(), img4.cuda().float(), img5.cuda().float(), img6.cuda().float()
#             break
#     FLOPs, Params = profile(model, (img1, img2, img3, img4, img5, img6, pair))
#     print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
#     tbar = tqdm(testloader)
#     metric = IOUandSek(num_classes=len(RS.ST_CLASSES)+1)
#     with torch.no_grad():
#         for img1, img2, img3, img4, img5, img6, label1, label2, label3, label4, label5, label6, label_bn, id in tbar:
#             img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
#             out1, out2, out3, out4, out5, out6, out_bn = model(img1, img2, img3, img4, img5, img6, pair)
#             pred1_seg = torch.argmax(out1, dim=1).cpu().numpy()
#             pred2_seg = torch.argmax(out2, dim=1).cpu().numpy()
#             pred3_seg = torch.argmax(out3, dim=1).cpu().numpy()
#             pred4_seg = torch.argmax(out4, dim=1).cpu().numpy()
#             pred5_seg = torch.argmax(out5, dim=1).cpu().numpy()
#             pred6_seg = torch.argmax(out6, dim=1).cpu().numpy()
#
#             cmap = color_map_DynamicEarth()
#
#             for i in range(out1.shape[0]):
#                 mask1_seg = Image.fromarray(pred1_seg[i].astype(np.uint8)).convert('P')
#                 mask1_seg.putpalette(cmap)
#                 mask1_seg.save(os.path.join(pred_save_path1_semantic, id[i]))
#                 mask2_seg = Image.fromarray(pred2_seg[i].astype(np.uint8)).convert('P')
#                 mask2_seg.putpalette(cmap)
#                 mask2_seg.save(os.path.join(pred_save_path2_semantic, id[i]))
#                 mask3_seg = Image.fromarray(pred3_seg[i].astype(np.uint8)).convert('P')
#                 mask3_seg.putpalette(cmap)
#                 mask3_seg.save(os.path.join(pred_save_path3_semantic, id[i]))
#                 mask4_seg = Image.fromarray(pred4_seg[i].astype(np.uint8)).convert('P')
#                 mask4_seg.putpalette(cmap)
#                 mask4_seg.save(os.path.join(pred_save_path4_semantic, id[i]))
#                 mask5_seg = Image.fromarray(pred5_seg[i].astype(np.uint8)).convert('P')
#                 mask5_seg.putpalette(cmap)
#                 mask5_seg.save(os.path.join(pred_save_path5_semantic, id[i]))
#                 mask6_seg = Image.fromarray(pred6_seg[i].astype(np.uint8)).convert('P')
#                 mask6_seg.putpalette(cmap)
#                 mask6_seg.save(os.path.join(pred_save_path6_semantic, id[i]))
#
# def inference_scd(args, pair):
#     begin_time = time.time()
#     working_path = os.path.dirname(os.path.abspath(__file__))
#     pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)
#     pred_save_path1 = os.path.join(pred_dir, f'pred1_{pair[0]}{pair[1]}')
#     pred_save_path2 = os.path.join(pred_dir, f'pred2_{pair[0]}{pair[1]}')
#     pred_save_path3 = os.path.join(pred_dir, f'pred3_{pair[0]}{pair[1]}')
#     pred_save_path4 = os.path.join(pred_dir, f'pred4_{pair[0]}{pair[1]}')
#     pred_save_path5 = os.path.join(pred_dir, f'pred5_{pair[0]}{pair[1]}')
#     pred_save_path6 = os.path.join(pred_dir, f'pred6_{pair[0]}{pair[1]}')
#     pred_save_path1_rgb = os.path.join(pred_dir, f'pred1_rgb_{pair[0]}{pair[1]}')
#     pred_save_path2_rgb = os.path.join(pred_dir, f'pred2_rgb_{pair[0]}{pair[1]}')
#     pred_save_path3_rgb = os.path.join(pred_dir, f'pred3_rgb_{pair[0]}{pair[1]}')
#     pred_save_path4_rgb = os.path.join(pred_dir, f'pred4_rgb_{pair[0]}{pair[1]}')
#     pred_save_path5_rgb = os.path.join(pred_dir, f'pred5_rgb_{pair[0]}{pair[1]}')
#     pred_save_path6_rgb = os.path.join(pred_dir, f'pred6_rgb_{pair[0]}{pair[1]}')
#     pred_save_pathcd = os.path.join(pred_dir, f'pred_change_{pair[0]}{pair[1]}')
#
#     if not os.path.exists(pred_save_path1): os.makedirs(pred_save_path1)
#     if not os.path.exists(pred_save_path2): os.makedirs(pred_save_path2)
#     if not os.path.exists(pred_save_path3): os.makedirs(pred_save_path3)
#     if not os.path.exists(pred_save_path4): os.makedirs(pred_save_path4)
#     if not os.path.exists(pred_save_path5): os.makedirs(pred_save_path5)
#     if not os.path.exists(pred_save_path6): os.makedirs(pred_save_path6)
#     if not os.path.exists(pred_save_path1_rgb): os.makedirs(pred_save_path1_rgb)
#     if not os.path.exists(pred_save_path2_rgb): os.makedirs(pred_save_path2_rgb)
#     if not os.path.exists(pred_save_path3_rgb): os.makedirs(pred_save_path3_rgb)
#     if not os.path.exists(pred_save_path4_rgb): os.makedirs(pred_save_path4_rgb)
#     if not os.path.exists(pred_save_path5_rgb): os.makedirs(pred_save_path5_rgb)
#     if not os.path.exists(pred_save_path6_rgb): os.makedirs(pred_save_path6_rgb)
#     if not os.path.exists(pred_save_pathcd): os.makedirs(pred_save_pathcd)
#     testset = RS.Data(mode="val", pair=pair, random_flip=False)
#     testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
#                                 pin_memory=True, num_workers=0, drop_last=False)
#     model = Net(args.backbone, args.pretrained, len(RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
#     # model = Net(in_channels=4, num_classes=7)
#     if args.load_from:
#         model.load_state_dict(torch.load(args.load_from), strict=True)
#
#     model = model.cuda()
#     model.eval()
#     tbar = tqdm(testloader)
#     metric = IOUandSek(num_classes=len(RS.ST_CLASSES)+1)
#     with torch.no_grad():
#         for img1, img2, img3, img4, img5, img6, label1, label2, label3, label4, label5, label6, label_bn, id in tbar:
#             img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
#             out1, out2, out3, out4, out5, out6, out_bn = model(img1, img2, img3, img4, img5, img6, pair)
#             pred1_seg = torch.argmax(out1, dim=1).cpu().numpy()
#             pred2_seg = torch.argmax(out2, dim=1).cpu().numpy()
#             pred3_seg = torch.argmax(out3, dim=1).cpu().numpy()
#             pred4_seg = torch.argmax(out4, dim=1).cpu().numpy()
#             pred5_seg = torch.argmax(out5, dim=1).cpu().numpy()
#             pred6_seg = torch.argmax(out6, dim=1).cpu().numpy()
#             out_bn = (out_bn > 0.5).cpu().numpy().astype(np.uint8)
#
#             out1 = copy.deepcopy(pred1_seg)
#             out2 = copy.deepcopy(pred2_seg)
#             out3 = copy.deepcopy(pred3_seg)
#             out4 = copy.deepcopy(pred4_seg)
#             out5 = copy.deepcopy(pred5_seg)
#             out6 = copy.deepcopy(pred6_seg)
#
#             out1[out_bn == 0] = 0
#             out2[out_bn == 0] = 0
#             out3[out_bn == 0] = 0
#             out4[out_bn == 0] = 0
#             out5[out_bn == 0] = 0
#             out6[out_bn == 0] = 0
#
#             label1[label_bn == 0] = 0
#             label2[label_bn == 0] = 0
#             label3[label_bn == 0] = 0
#             label4[label_bn == 0] = 0
#             label5[label_bn == 0] = 0
#             label6[label_bn == 0] = 0
#
#             outs = [out1, out2, out3, out4, out5, out6]
#             labels = [label1, label2, label3, label4, label5, label6]
#             cmap = color_map_DynamicEarth()
#
#             for i in range(out1.shape[0]):
#                 mask1 = Image.fromarray(out1[i].astype(np.uint8))
#                 mask1.save(os.path.join(pred_save_path1, id[i]))
#                 mask1_rgb = mask1.convert('P')
#                 mask1_rgb.putpalette(cmap)
#                 mask1_rgb.save(os.path.join(pred_save_path1_rgb, id[i]))
#                 mask2 = Image.fromarray(out2[i].astype(np.uint8))
#                 mask2.save(os.path.join(pred_save_path2, id[i]))
#                 mask2_rgb = mask2.convert('P')
#                 mask2_rgb.putpalette(cmap)
#                 mask2_rgb.save(os.path.join(pred_save_path2_rgb, id[i]))
#                 mask3 = Image.fromarray(out3[i].astype(np.uint8))
#                 mask3.save(os.path.join(pred_save_path3, id[i]))
#                 mask3_rgb = mask3.convert('P')
#                 mask3_rgb.putpalette(cmap)
#                 mask3_rgb.save(os.path.join(pred_save_path3_rgb, id[i]))
#                 mask4 = Image.fromarray(out4[i].astype(np.uint8))
#                 mask4.save(os.path.join(pred_save_path4, id[i]))
#                 mask4_rgb = mask4.convert('P')
#                 mask4_rgb.putpalette(cmap)
#                 mask4_rgb.save(os.path.join(pred_save_path4_rgb, id[i]))
#                 mask5 = Image.fromarray(out5[i].astype(np.uint8))
#                 mask5.save(os.path.join(pred_save_path5, id[i]))
#                 mask5_rgb = mask5.convert('P')
#                 mask5_rgb.putpalette(cmap)
#                 mask5_rgb.save(os.path.join(pred_save_path5_rgb, id[i]))
#                 mask6 = Image.fromarray(out6[i].astype(np.uint8))
#                 mask6.save(os.path.join(pred_save_path6, id[i]))
#                 mask6_rgb = mask6.convert('P')
#                 mask6_rgb.putpalette(cmap)
#                 mask6_rgb.save(os.path.join(pred_save_path6_rgb, id[i]))
#                 mask_bn = Image.fromarray(out_bn[i]*255)
#                 mask_bn.save(os.path.join(pred_save_pathcd, id[i]))
#                 print(id[i])
#             metric.add_batch(outs[pair[0]], labels[pair[0]].numpy())
#             metric.add_batch(outs[pair[-1]], labels[pair[-1]].numpy())
#
#         metric.color_map_DynamicEarth(pred_dir)      #需根据数据集调整函数
#
#         change_ratio, OA, mIoU, Sek, Fscd, Score, Precision_scd, Recall_scd = metric.evaluate_inference()
#
#         # print('==>change_ratio', change_ratio)
#         # print('==>oa', OA)
#         # print('==>miou', mIoU)
#         # print('==>sek', Sek)
#         # print('==>Fscd', Fscd)
#         # print('==>score', Score)
#         # print('==>SC_Precision', Precision_scd)
#         # print('==>SC_Recall', Recall_scd)
#         print(f'时相T{pair[0]}{pair[1]}预测完成')
#         time_use = time.time() - begin_time
#
#     metric_file = os.path.join(pred_dir, f'metric.txt_{pair[0]}{pair[1]}')
#     f = open(metric_file, 'w', encoding='utf-8')
#     f.write("Data：" + str(args.data_name) + '\n')
#     f.write("model：" + str(args.Net_name) + '\n')
#     f.write("##################### metric #####################"+'\n')
#     f.write("infer time (s) ：" + str(round(time_use, 2)) + '\n')
#     f.write('\n')
#     f.write("change_ratio (%) ：" + str(round(change_ratio * 100, 2)) + '\n')
#     f.write("OA (%) ：" + str(round(OA * 100, 2)) + '\n')
#     f.write("mIoU (%) ：" + str(round(mIoU * 100, 2)) + '\n')
#     f.write("Sek (%) ：" + str(round(Sek * 100, 2)) + '\n')
#     f.write("Fscd (%) ：" + str(round(Fscd * 100, 2)) + '\n')
#     f.write("Score (%) ：" + str(round(Score * 100, 2)) + '\n')
#     f.write("Precision_scd (%) ：" + str(round(Precision_scd * 100, 2)) + '\n')
#     f.write("Recall_scd (%) ：" + str(round(Recall_scd * 100, 2)) + '\n')
#
#     f.close()
#
#
# if __name__ == "__main__":
#     args = Options().parse()
#     for i in range(6):
#         for j in range(6):
#             if i == 0 and j == 0:
#                 pair = [i, j]
#                 inference_ss(args, pair)
#             elif i != j and i < j:
#                 pair = [i, j]
#                 inference_scd(args, pair)
#             else:
#                 continue
