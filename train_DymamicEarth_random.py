import datasets.MultiSiamese_RS_ST_TL_DynamocEarth as RS
# from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth_random as Net
# from models.proposed_MTGrootV3D import GSTMSCD_random as Net
# from models.BiSRNet import BiSRNet_NMT as Net
# from models.EGMSNet import EGMSNet_NMT as Net
# from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth_random_base as Net
# from models.TED import TED_NMT as Net
# from models.HRSCD_str4 import HRSCD_str4_NMT as Net
# from models.SSCDl import SSCDl_NMT as Net
# from models.SCanNet import SCanNet_NMT as Net
from models.HRSCD_str3 import HRSCD_str3_dymamic as Net
from utils.palette import color_map
from utils.metric import IOUandSek
from utils.loss import ChangeSimilarity, DiceLoss
from itertools import combinations
import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_path = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default=r"DynamicEarth")
        parser.add_argument("--Net_name", type=str, default="HRSCD.str3_random_update1")
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default=r"/home/h3c/LXG_data/DynamicEarth")
        parser.add_argument("--log_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--val_batch_size", type=int, default=8)
        parser.add_argument("--test_batch_size", type=int, default=8)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                            help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--pretrain_from", type=str,
                            help='train from a checkpoint')
        parser.add_argument("--load_from", type=str,
                            help='load trained model to generate predictions of validation set')
        parser.add_argument("--pretrained", type=bool, default=True,
                            help='initialize the backbone with pretrained parameters')
        parser.add_argument("--tta", dest="tta", action="store_true",
                            help='test_time augmentation')
        parser.add_argument("--warmup", dest="warmup", default=True, action="store_true", help='warm up')  # 默认使用warmup
        parser.add_argument("--save_mask", dest="save_mask", action="store_true",
                            help='save predictions of validation set during training')
        parser.add_argument("--use_pseudo_label", dest="use_pseudo_label", action="store_true",
                            help='use pseudo labels for re-training (must pseudo label first)')
        parser.add_argument("--M", type=int, default=6)
        parser.add_argument("--Lambda", type=float, default=0.00005)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args


class Trainer:
    def __init__(self, args):
        args.log_dir = os.path.join(working_path, 'logs', args.data_name, args.Net_name, args.backbone)
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        self.writer = SummaryWriter(args.log_dir)
        self.args = args

        trainset = RS.Data(mode="train", random_flip=True)
        valset = RS.Data(mode="val", random_flip=True)
        testset = RS.Data(mode="val", random_flip=True)
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=8, drop_last=True)  # num_works原本为8
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=8, drop_last=False)  # num_works原本为8
        # using proposed models
        # self.model = Net(args.backbone, args.pretrained, len(RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
        # print(len(RS.ST_CLASSES))
        # self.model = Net(args.backbone, args.pretrained, len(RS.ST_CLASSES), args.lightweight, args.M, args.Lambda)
        # using comparative models
        self.model = Net(4, num_classes=len(RS.ST_CLASSES))
        if args.pretrain_from:
            self.model.load_state_dict(torch.load(args.pretrain_from), strict=False)
        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        self.criterion_seg = CrossEntropyLoss()
        self.criterion_bn = BCELoss(reduction='none')
        self.criterion_bn_2 = DiceLoss()
        self.criterion_sc = ChangeSimilarity()
        self.optimizer = AdamW([{"params": [param for name, param in self.model.named_parameters()
                                            if "backbone" in name], "lr": args.lr},
                                {"params": [param for name, param in self.model.named_parameters()
                                            if "backbone" not in name], "lr": args.lr * 1}],
                               lr=args.lr, weight_decay=args.weight_decay)
        self.model = self.model.cuda()

        self.iters = 0
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0
        self.seg_best = 0.0
        self.change_best = 0.0

    def training(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_bn = 0.0
        total_loss_similarity = 0.0

        curr_iter = curr_epoch * len(self.trainloader)

        for i, (img1, img2, img3, img4, img5, img6, mask1, mask2, mask3, mask4, mask5, mask6, mask_bn, id) in enumerate(
                tbar):
            # LXG:添加随机数
            pair = sorted(random.sample(range(6), 2))
            # pair = [1,2]
            # print(pair)  # 输出可能是 [0, 1], [0, 2] 或 [1, 2]
            running_iter = curr_iter + i + 1
            img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
            mask1, mask2, mask3, mask4, mask5, mask6, mask_bn = mask1.cuda().long(), mask2.cuda().long(), mask3.cuda().long(), mask4.cuda().long(), mask5.cuda().long(), mask6.cuda().long(), mask_bn.cuda().float()
            out, out_bn = self.model(img1, img2, img3, img4, img5, img6,
                                     pair)  # [b,6,512,512],[b,6,512,512],[b,512,512]

            mask = [mask1, mask2, mask3, mask4, mask5, mask6]
            mask_bn = mask[pair[0]] - mask[pair[1]]
            mask_bn[mask_bn != 0] = 1

            loss1 = self.criterion_seg(out[0], mask1)
            loss2 = self.criterion_seg(out[1], mask2)
            loss3 = self.criterion_seg(out[2], mask3)
            loss4 = self.criterion_seg(out[3], mask4)
            loss5 = self.criterion_seg(out[4], mask5)
            loss6 = self.criterion_seg(out[5], mask6)
            loss_seg = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6) / 6
            loss_similarity = self.criterion_sc(out[pair[0]][:, 0:], out[pair[0]][:, 0:], mask_bn.float())
            loss_bn_1 = self.criterion_bn(out_bn.float(), mask_bn.float())
            loss_bn_1[mask_bn == 1] *= 2
            loss_bn_1 = loss_bn_1.mean()
            loss_bn_2 = self.criterion_bn_2(out_bn.float(), mask_bn.float())
            loss_bn = loss_bn_1 + loss_bn_2
            loss = loss_bn + loss_seg + loss_similarity

            total_loss_seg += loss_seg.item()
            total_loss_similarity += loss_similarity.item()
            total_loss_bn += loss_bn.item()
            total_loss += loss.item()

            self.iters += 1
            if args.warmup:
                warmup_steps = len(self.trainloader) * (args.epochs / 5)
                if warmup_steps and self.iters < warmup_steps:
                    warmup_percent_done = self.iters / warmup_steps
                    lr = args.lr * warmup_percent_done
                else:
                    lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5
            else:
                lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5
            self.optimizer.param_groups[0]["lr"] = lr
            if args.pretrain_from:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0
            else:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f" %
                                 (total_loss / (i + 1), total_loss_seg / (i + 1), total_loss_bn / (i + 1),
                                  total_loss_similarity / (i + 1)))

            self.writer.add_scalar('train total_loss', total_loss / (i + 1), running_iter)
            self.writer.add_scalar('train seg_loss', total_loss_seg / (i + 1), running_iter)
            self.writer.add_scalar('train bn_loss', total_loss_bn / (i + 1), running_iter)
            self.writer.add_scalar('train sc_loss', total_loss_similarity / (i + 1), running_iter)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], running_iter)

    def validation(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.valloader)
        self.model.eval()
        num_classes = len(RS.ST_CLASSES) + 1

        # 存储所有组合的指标
        all_pair_metrics = []
        all_pair_results = {}

        # 生成所有可能的双时相组合
        from itertools import combinations
        all_pairs = list(combinations(range(6), 2))
        pair_names = [f"T{i + 1}-T{j + 1}" for (i, j) in all_pairs]

        # 为每对组合初始化指标计算器
        pair_metrics = {}
        for pair in all_pairs:
            pair_metrics[pair] = IOUandSek(num_classes=num_classes)
            all_pair_results[pair] = {
                'OA': 0, 'mIoU': 0, 'Sek': 0, 'Fscd': 0, 'Score': 0,
                'SC_Precision': 0, 'SC_Recall': 0
            }

        with torch.no_grad():
            for img1, img2, img3, img4, img5, img6, mask1, mask2, mask3, mask4, mask5, mask6, mask_bn, id in tbar:
                # 将数据转移到GPU
                img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
                mask1, mask2, mask3, mask4, mask5, mask6, mask_bn = mask1.cuda().long(), mask2.cuda().long(), mask3.cuda().long(), mask4.cuda().long(), mask5.cuda().long(), mask6.cuda().long(), mask_bn.cuda().float()

                # 循环处理所有时相对
                for pair in all_pairs:
                    # 获取模型对指定时相对的预测
                    all_out, out_bn = self.model(img1, img2, img3, img4, img5, img6, pair)

                    # 转换为numpy数组
                    mask_numpy = [m.cpu().numpy() for m in [mask1, mask2, mask3, mask4, mask5, mask6]]
                    out_numpy = [torch.argmax(o, dim=1).cpu().numpy() for o in all_out]
                    bn_numpy = (out_bn > 0.5).cpu().numpy().astype(np.uint8)

                    t1, t2 = pair

                    # 使用模型输出的二值变化图
                    pred_change = bn_numpy

                    # 处理预测结果：在未变化区域设置为0
                    pred_t1 = out_numpy[t1].copy()
                    pred_t2 = out_numpy[t2].copy()
                    pred_t1[pred_change == 0] = 0
                    pred_t2[pred_change == 0] = 0

                    # 处理真实标签：在未变化区域设置为0
                    true_t1 = mask_numpy[t1].copy()
                    true_t2 = mask_numpy[t2].copy()
                    true_t1[mask_bn.cpu().numpy() == 0] = 0
                    true_t2[mask_bn.cpu().numpy() == 0] = 0

                    # 添加到该组合的metric
                    pair_metrics[pair].add_batch(pred_t1, true_t1)
                    pair_metrics[pair].add_batch(pred_t2, true_t2)

        # 计算每对组合的指标
        total_OA = 0
        total_mIoU = 0
        total_Sek = 0
        total_Fscd = 0
        total_Score = 0
        total_SC_Precision = 0
        total_SC_Recall = 0

        for pair in all_pairs:
            # 使用 evaluate_inference 方法获取所有指标
            change_ratio, OA, mIoU, Sek, Fscd, Score, SC_Precision, SC_Recall = pair_metrics[pair].evaluate_inference()

            # 存储结果
            all_pair_results[pair] = {
                'OA': OA, 'mIoU': mIoU, 'Sek': Sek, 'Fscd': Fscd, 'Score': Score,
                'SC_Precision': SC_Precision, 'SC_Recall': SC_Recall
            }

            # 累加用于平均值
            total_OA += OA
            total_mIoU += mIoU
            total_Sek += Sek
            total_Fscd += Fscd
            total_Score += Score
            total_SC_Precision += SC_Precision
            total_SC_Recall += SC_Recall

        # 计算平均指标
        num_pairs = len(all_pairs)
        avg_OA = total_OA / num_pairs
        avg_mIoU = total_mIoU / num_pairs
        avg_Sek = total_Sek / num_pairs
        avg_Fscd = total_Fscd / num_pairs
        avg_Score = total_Score / num_pairs
        avg_SC_Precision = total_SC_Precision / num_pairs
        avg_SC_Recall = total_SC_Recall / num_pairs

        # 输出所有组合的指标
        print("\n=== All Pairwise Change Detection Metrics ===")
        for pair, name in zip(all_pairs, pair_names):
            res = all_pair_results[pair]
            print(f"{name}: OA={res['OA']:.4f}, mIoU={res['mIoU']:.4f}, Sek={res['Sek']:.4f}, "
                  f"Fscd={res['Fscd']:.4f}, Score={res['Score']:.4f}, Prec={res['SC_Precision']:.4f}, Recall={res['SC_Recall']:.4f}")

        # 输出平均指标
        print("\n=== Average Metrics ===")
        print(f"Avg OA: {avg_OA:.4f}")
        print(f"Avg mIoU: {avg_mIoU:.4f}")
        print(f"Avg Sek: {avg_Sek:.4f}")
        print(f"Avg Fscd: {avg_Fscd:.4f}")
        print(f"Avg Score: {avg_Score:.4f}")
        print(f"Avg SC_Precision: {avg_SC_Precision:.4f}")
        print(f"Avg SC_Recall: {avg_SC_Recall:.4f}")

        # 保存模型（基于Sek指标）
        if avg_Sek >= self.previous_best:
            model_path = "checkpoints/%s/%s/%s" % \
                         (self.args.data_name, self.args.Net_name, self.args.backbone)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            save_path = "checkpoints/%s/%s/%s/epoch%i_Sek%.2f.pth" % \
                        (self.args.data_name, self.args.Net_name, self.args.backbone,
                         curr_epoch, avg_Sek * 100)

            torch.save(self.model.state_dict(), save_path)
            print(f"\n保存模型: {save_path}")
            self.previous_best = avg_Sek

        # 记录到tensorboard
        self.writer.add_scalar('val_Avg_OA', avg_OA, curr_epoch)
        self.writer.add_scalar('val_Avg_mIoU', avg_mIoU, curr_epoch)
        self.writer.add_scalar('val_Avg_Sek', avg_Sek, curr_epoch)
        self.writer.add_scalar('val_Avg_Fscd', avg_Fscd, curr_epoch)
        self.writer.add_scalar('val_Avg_Score', avg_Score, curr_epoch)
    # def validation(self, epoch):
    #     curr_epoch = epoch
    #     tbar = tqdm(self.valloader)
    #     self.model.eval()
    #     metric = IOUandSek(num_classes=len(RS.ST_CLASSES)+1)
    #     if self.args.save_mask:
    #         cmap = color_map()

    #     with torch.no_grad():
    #         for img1, img2, img3, img4, img5, img6, mask1, mask2, mask3, mask4, mask5, mask6, mask_bn, id in tbar:
    #             img1, img2, img3, img4, img5, img6 = img1.cuda(), img2.cuda(), img3.cuda(), img4.cuda(), img5.cuda(), img6.cuda()
    #             pair = [0, 5]
    #             out, out_bn = self.model(img1, img2, img3, img4, img5, img6, pair)
    #             mask = [mask1, mask2, mask3, mask4, mask5, mask6]
    #             mask_bn = mask[pair[0]] - mask[pair[1]]
    #             mask_bn[mask_bn != 0] = 1

    #             out = [torch.argmax(out[0], dim=1).cpu().numpy(), torch.argmax(out[1], dim=1).cpu().numpy(),
    #                    torch.argmax(out[2], dim=1).cpu().numpy(), torch.argmax(out[3], dim=1).cpu().numpy(),
    #                    torch.argmax(out[4], dim=1).cpu().numpy(), torch.argmax(out[5], dim=1).cpu().numpy()]

    #             out_bn = (out_bn > 0.5).cpu().numpy().astype(np.uint8)
    #             out[pair[0]][out_bn == 0] = 0
    #             out[pair[-1]][out_bn == 0] = 0
    #             mask[pair[0]][mask_bn == 0] = 0
    #             mask[pair[-1]][mask_bn == 0] = 0
    #             metric.add_batch(out[pair[0]], mask[pair[0]].numpy())
    #             metric.add_batch(out[pair[1]], mask[pair[1]].numpy())

    #             score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = metric.evaluate_SECOND()

    #             tbar.set_description(
    #                 "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, SC_Precision: %.4f, SC_Recall: %.4f" % (
    #                 miou, sek, score, Fscd, OA, SC_Precision, SC_Recall))

    #         if score >= self.previous_best:
    #             model_path = "checkpoints/%s/%s/%s" % \
    #                          (self.args.data_name, self.args.Net_name, self.args.backbone)
    #             if not os.path.exists(model_path): os.makedirs(model_path)
    #             torch.save(self.model.state_dict(),
    #                        "checkpoints/%s/%s/%s/epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth" %
    #                        (self.args.data_name, self.args.Net_name, self.args.backbone, curr_epoch, score * 100,
    #                         miou * 100, sek * 100, Fscd * 100, OA * 100))

    #             self.previous_best = score

    #         self.writer.add_scalar('val_Score', score, curr_epoch)
    #         self.writer.add_scalar('val_mIOU', miou, curr_epoch)
    #         self.writer.add_scalar('val_Sek', sek, curr_epoch)
    #         self.writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    #         self.writer.add_scalar('val_OA', OA, curr_epoch)


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    if args.load_from:
        trainer.validation()

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.5f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training(epoch)
        trainer.validation(epoch)


