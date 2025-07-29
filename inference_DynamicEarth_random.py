import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import PIL.Image as Image

import datasets.MultiSiamese_RS_ST_TL_DynamocEarth as RS
# from models.proposed_BiGrootV import BiGrootV_V2 as Net
# from models.proposed_V1 import BiGrootV as Net
# from models.proposed_BiGrootV import BiGrootV_V5Base as Net
# from models.proposed_BiGrootV import BiGrootV_V5 as Net
# from models.EGMSNet import EGMSNet_NMT as Net
# from models.proposed_BiGrootV import BiGrootV3D_V2 as Net
# from models.SSCDl import SSCDl_NMT as Net
# from models.BiSRNet import BiSRNet_NMT as Net
# from models.proposed_MTGrootV3D import MTGrootV3D_SV4_DynamicEarth as Net
from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth_random_AE2 as Net
# from models.SCanNet import SCanNet_NMT as Net
# from models.TED import TED_NMT as Net
from utils.palette import color_map_DynamicEarth
from utils.metric import IOUandSek
from tqdm import tqdm
from torch.utils.data import DataLoader
from thop import profile
import time
import argparse
import copy

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default="DynamicEarth")
        parser.add_argument("--Net_name", type=str, default="FC-Siam-diff")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                           help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default=r"/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_process/train/DynamicEarth512/test")
        parser.add_argument("--load_from", type=str,
                            default=r"/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/DynamicEarth/MTGrootV3D_SV3_DynamicEarth_random_AE2_0417/resnet34/epoch25_Score53.25_mIOU86.78_Sek38.88_Fscd64.18_OA76.44.pth")
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

def inference(args):
    begin_time = time.time()
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)
    pred_save_path1 = os.path.join(pred_dir, 'pred1')
    pred_save_path2 = os.path.join(pred_dir, 'pred2')
    pred_save_path3 = os.path.join(pred_dir, 'pred3')
    pred_save_path4 = os.path.join(pred_dir, 'pred4')
    pred_save_path5 = os.path.join(pred_dir, 'pred5')
    pred_save_path6 = os.path.join(pred_dir, 'pred6')
    pred_save_path1_rgb = os.path.join(pred_dir, 'pred1_rgb')
    pred_save_path2_rgb = os.path.join(pred_dir, 'pred2_rgb')
    pred_save_path3_rgb = os.path.join(pred_dir, 'pred3_rgb')
    pred_save_path4_rgb = os.path.join(pred_dir, 'pred4_rgb')
    pred_save_path5_rgb = os.path.join(pred_dir, 'pred5_rgb')
    pred_save_path6_rgb = os.path.join(pred_dir, 'pred6_rgb')
    pred_save_path1_semantic = os.path.join(pred_dir, 'pred1_semantic')
    pred_save_path2_semantic = os.path.join(pred_dir, 'pred2_semantic')
    pred_save_path3_semantic = os.path.join(pred_dir, 'pred3_semantic')
    pred_save_path4_semantic = os.path.join(pred_dir, 'pred4_semantic')
    pred_save_path5_semantic = os.path.join(pred_dir, 'pred5_semantic')
    pred_save_path6_semantic = os.path.join(pred_dir, 'pred6_semantic')
    pred_save_pathcd = os.path.join(pred_dir, 'pred_change')

    if not os.path.exists(pred_save_path1): os.makedirs(pred_save_path1)
    if not os.path.exists(pred_save_path2): os.makedirs(pred_save_path2)
    if not os.path.exists(pred_save_path3): os.makedirs(pred_save_path3)
    if not os.path.exists(pred_save_path4): os.makedirs(pred_save_path4)
    if not os.path.exists(pred_save_path5): os.makedirs(pred_save_path5)
    if not os.path.exists(pred_save_path6): os.makedirs(pred_save_path6)
    if not os.path.exists(pred_save_path1_rgb): os.makedirs(pred_save_path1_rgb)
    if not os.path.exists(pred_save_path2_rgb): os.makedirs(pred_save_path2_rgb)
    if not os.path.exists(pred_save_path3_rgb): os.makedirs(pred_save_path3_rgb)
    if not os.path.exists(pred_save_path4_rgb): os.makedirs(pred_save_path4_rgb)
    if not os.path.exists(pred_save_path5_rgb): os.makedirs(pred_save_path5_rgb)
    if not os.path.exists(pred_save_path6_rgb): os.makedirs(pred_save_path6_rgb)
    if not os.path.exists(pred_save_path1_semantic): os.makedirs(pred_save_path1_semantic)
    if not os.path.exists(pred_save_path2_semantic): os.makedirs(pred_save_path2_semantic)
    if not os.path.exists(pred_save_path3_semantic): os.makedirs(pred_save_path3_semantic)
    if not os.path.exists(pred_save_path4_semantic): os.makedirs(pred_save_path4_semantic)
    if not os.path.exists(pred_save_path5_semantic): os.makedirs(pred_save_path5_semantic)
    if not os.path.exists(pred_save_path6_semantic): os.makedirs(pred_save_path6_semantic)
    if not os.path.exists(pred_save_pathcd): os.makedirs(pred_save_pathcd)

    testset = RS.Data(mode="val", random_flip=False)
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                pin_memory=True, num_workers=0, drop_last=False)
    model = Net(args.backbone, args.pretrained, len(RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
    # model = Net(in_channels=4, num_classes=7)
    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)

    model = model.cuda()
    model.eval()

    tbar = tqdm(testloader)
    metric = IOUandSek(num_classes=len(RS.ST_CLASSES))
    with torch.no_grad():
        for img1, img2, img3, img4, img5, img6, label1, label2, label3, label4, label5, label6, label_bn, id in tbar:
            img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
            out, out_bn, pair = model(img1, img2, img3, img4, img5, img6)
            label = [label1, label2, label3, label4, label5, label6]
            mask_bn = label[pair[0]] - label[pair[-1]]
            mask_bn[mask_bn != 0] = 1

            pred = [torch.argmax(out[0], dim=1).cpu().numpy(), torch.argmax(out[1], dim=1).cpu().numpy(),
                    torch.argmax(out[2], dim=1).cpu().numpy(), torch.argmax(out[3], dim=1).cpu().numpy(),
                    torch.argmax(out[4], dim=1).cpu().numpy(), torch.argmax(out[5], dim=1).cpu().numpy()]

            out_bn = (out_bn > 0.5).cpu().numpy().astype(np.uint8)

            out1 = copy.deepcopy(pred[0])
            out2 = copy.deepcopy(pred[1])
            out3 = copy.deepcopy(pred[2])
            out4 = copy.deepcopy(pred[3])
            out5 = copy.deepcopy(pred[4])
            out6 = copy.deepcopy(pred[5])
            #
            pred1_seg = pred[0]
            pred2_seg = pred[1]
            pred3_seg = pred[2]
            pred4_seg = pred[3]
            pred5_seg = pred[4]
            pred6_seg = pred[5]

            out1[out_bn == 0] = 0
            out2[out_bn == 0] = 0
            out3[out_bn == 0] = 0
            out4[out_bn == 0] = 0
            out5[out_bn == 0] = 0
            out6[out_bn == 0] = 0
            # out_ls = [out1, out2, out3, out4, out5, out6]

            label1[mask_bn == 0] = 0
            label2[mask_bn == 0] = 0
            label3[mask_bn == 0] = 0
            label4[mask_bn == 0] = 0
            label5[mask_bn == 0] = 0
            label6[mask_bn == 0] = 0
            # label_ls = [label1, label2, label3, label4, label5, label6]

            cmap = color_map_DynamicEarth()

            for i in range(out1.shape[0]):
                mask1 = Image.fromarray(out1[i].astype(np.uint8))
                mask1.save(os.path.join(pred_save_path1, id[i]))
                mask1_rgb = mask1.convert('P')
                mask1_rgb.putpalette(cmap)
                mask1_rgb.save(os.path.join(pred_save_path1_rgb, id[i]))
                mask1_seg = Image.fromarray(pred1_seg[i].astype(np.uint8)).convert('P')
                mask1_seg.putpalette(cmap)
                mask1_seg.save(os.path.join(pred_save_path1_semantic, id[i]))

                mask2 = Image.fromarray(out2[i].astype(np.uint8))
                mask2.save(os.path.join(pred_save_path2, id[i]))
                mask2_rgb = mask2.convert('P')
                mask2_rgb.putpalette(cmap)
                mask2_rgb.save(os.path.join(pred_save_path2_rgb, id[i]))
                mask2_seg = Image.fromarray(pred2_seg[i].astype(np.uint8)).convert('P')
                mask2_seg.putpalette(cmap)
                mask2_seg.save(os.path.join(pred_save_path2_semantic, id[i]))


                mask3 = Image.fromarray(out3[i].astype(np.uint8))
                mask3.save(os.path.join(pred_save_path3, id[i]))
                mask3_rgb = mask3.convert('P')
                mask3_rgb.putpalette(cmap)
                mask3_rgb.save(os.path.join(pred_save_path3_rgb, id[i]))
                mask3_seg = Image.fromarray(pred3_seg[i].astype(np.uint8)).convert('P')
                mask3_seg.putpalette(cmap)
                mask3_seg.save(os.path.join(pred_save_path3_semantic, id[i]))

                mask4 = Image.fromarray(out4[i].astype(np.uint8))
                mask4.save(os.path.join(pred_save_path4, id[i]))
                mask4_rgb = mask4.convert('P')
                mask4_rgb.putpalette(cmap)
                mask4_rgb.save(os.path.join(pred_save_path4_rgb, id[i]))
                mask4_seg = Image.fromarray(pred4_seg[i].astype(np.uint8)).convert('P')
                mask4_seg.putpalette(cmap)
                mask4_seg.save(os.path.join(pred_save_path4_semantic, id[i]))

                mask5 = Image.fromarray(out5[i].astype(np.uint8))
                mask5.save(os.path.join(pred_save_path5, id[i]))
                mask5_rgb = mask5.convert('P')
                mask5_rgb.putpalette(cmap)
                mask5_rgb.save(os.path.join(pred_save_path5_rgb, id[i]))
                mask5_seg = Image.fromarray(pred5_seg[i].astype(np.uint8)).convert('P')
                mask5_seg.putpalette(cmap)
                mask5_seg.save(os.path.join(pred_save_path5_semantic, id[i]))
                #
                #
                mask6 = Image.fromarray(out6[i].astype(np.uint8))
                mask6.save(os.path.join(pred_save_path6, id[i]))
                mask6_rgb = mask6.convert('P')
                mask6_rgb.putpalette(cmap)
                mask6_rgb.save(os.path.join(pred_save_path6_rgb, id[i]))
                mask6_seg = Image.fromarray(pred6_seg[i].astype(np.uint8)).convert('P')
                mask6_seg.putpalette(cmap)
                mask6_seg.save(os.path.join(pred_save_path6_semantic, id[i]))

                mask_bn = Image.fromarray(out_bn[i]*255)
                mask_bn.save(os.path.join(pred_save_pathcd, id[i]))
                print(id[i])
            pred[pair[0]][out_bn == 0] = 0
            pred[pair[1]][out_bn == 0] = 0
            label[pair[0]][mask_bn == 0] = 0
            label[pair[1]][mask_bn == 0] = 0
            metric.add_batch(pred[pair[0]], label[pair[0]].numpy())
            metric.add_batch(pred[pair[1]], label[pair[1]].numpy())
            change_ratio, score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = metric.evaluate_SECOND()
            tbar.set_description(
                "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, SC_Precision: %.4f, SC_Recall: %.4f" % (
                    miou, sek, score, Fscd, OA, SC_Precision, SC_Recall))
            # metric.add_batch(out_ls[pair[0]], label_ls[pair[0]].numpy())
            # metric.add_batch(out_ls[pair[-1]], label_ls[pair[-1]].numpy())

            # score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = metric.evaluate_SECOND()
            # change_ratio, OA, mIoU, Sek, Fscd, Score, Precision_scd, Recall_scd = metric.evaluate_inference()
            # metric.reset()
        metric.color_map_DynamicEarth(pred_dir)      #需根据数据集调整函数

        change_ratio, OA, mIoU, Sek, Fscd, Score, Precision_scd, Recall_scd = metric.evaluate_inference()

        print('==>change_ratio', change_ratio)
        print('==>oa', OA)
        print('==>miou', mIoU)
        print('==>sek', Sek)
        print('==>Fscd', Fscd)
        print('==>score', Score)
        print('==>SC_Precision', Precision_scd)
        print('==>SC_Recall', Recall_scd)

        time_use = time.time() - begin_time

    metric_file = os.path.join(pred_dir, 'metric.txt')
    f = open(metric_file, 'w', encoding='utf-8')
    f.write("Data：" + str(args.data_name) + '\n')
    f.write("model：" + str(args.Net_name) + '\n')
    f.write("##################### metric #####################"+'\n')
    f.write("infer time (s) ：" + str(round(time_use, 2)) + '\n')
    # f.write("Params (Mb) ：" + str(round(Params/1e6, 2)) + '\n')
    # f.write("FLOPs (Gbps) ：" + str(round(FLOPs/1e9, 2)) + '\n')
    f.write('\n')
    f.write("change_ratio (%) ：" + str(round(change_ratio * 100, 2)) + '\n')
    f.write("OA (%) ：" + str(round(OA * 100, 2)) + '\n')
    f.write("mIoU (%) ：" + str(round(mIoU * 100, 2)) + '\n')
    f.write("Sek (%) ：" + str(round(Sek * 100, 2)) + '\n')
    f.write("Fscd (%) ：" + str(round(Fscd * 100, 2)) + '\n')
    f.write("Score (%) ：" + str(round(Score * 100, 2)) + '\n')
    f.write("Precision_scd (%) ：" + str(round(Precision_scd * 100, 2)) + '\n')
    f.write("Recall_scd (%) ：" + str(round(Recall_scd * 100, 2)) + '\n')

    f.close()


if __name__ == "__main__":
    args = Options().parse()
    inference(args)