import datasets.MultiSiamese_RS_ST_TL_DynamocEarth as RS
# from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth as Net
# from models.proposed_MTGrootV3D import MTGrootV3D_SV3_DynamicEarth_random_base as Net
# from models.FEMCD.FEMCD import FEMCD_net_dynamic as Net
# from models.proposed_MTGrootV3D import GSTMSCD_GOST_TSSCS as Net
from models.UNet_3D import UNet3D as Net
# from models.BiSRNet import BiSRNet_NMT as Net
# from models.EGMSNet import EGMSNet_NMT as Net
# from models.TED import TED_NMT as Net
# from models.HRSCD_str4 import HRSCD_str4_NMT as Net
# from models.SSCDl import SSCDl_NMT as Net
# from models.SCanNet import SCanNet_NMT as Net
# from models.HRSCD_str3 import HRSCD_str3_MT as Net
from utils.palette import color_map
from utils.metric import IOUandSek
from utils.loss import ChangeSimilarity, DiceLoss
import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_path = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default=r"DynamicEarth")
        parser.add_argument("--Net_name", type=str, default="UNet3D")
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default=r"/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/DynamicEarthNet_process/train/DynamicEarth512")
        # parser.add_argument("--data_root", type=str, default=r"/media/lenovo/文档/Long-term-SCD/wusu512_process/debug_data")
        parser.add_argument("--log_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--val_batch_size", type=int, default=2)
        parser.add_argument("--test_batch_size", type=int, default=2)
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
        parser.add_argument("--warmup", dest="warmup", default=True, action="store_true", help='warm up') #默认使用warmup
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
                                      pin_memory=False, num_workers=8, drop_last=True)   # num_works原本为8
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=8, drop_last=False)   # num_works原本为8
        # using proposed models
        self.model = Net(in_channel=4, n_classes=len(RS.ST_CLASSES), timesteps=6, dropout=0.2)
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
                                           if "backbone" not in name], "lr": args.lr*1}],
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
        curr_iter = curr_epoch * len(self.trainloader)

        for i, (img1, img2, img3, img4, img5, img6, mask1, mask2, mask3, mask4, mask5, mask6, mask_bn, id) in enumerate(tbar):
            running_iter = curr_iter + i + 1
            img1, img2, img3, img4, img5, img6 = img1.float().cuda(), img2.float().cuda(), img3.float().cuda(), img4.float().cuda(), img5.float().cuda(), img6.float().cuda()
            mask1, mask2, mask3, mask4, mask5, mask6, mask_bn = mask1.cuda().long(), mask2.cuda().long(), mask3.cuda().long(), mask4.cuda().long(), mask5.cuda().long(), mask6.cuda().long(), mask_bn.cuda().float()
            mask = [mask1, mask2, mask3, mask4, mask5, mask6]
            mask = torch.stack(mask, dim=1)
            input = [img1, img2, img3, img4, img5, img6]  # 每个形状 [2, 4, 512, 512]
            input = torch.stack(input, dim=2)
            outs = self.model(input)    #[b,6,512,512],[b,6,512,512],[b,512,512]
            loss = self.criterion_seg(outs, mask)
            total_loss += loss.item()

            self.iters += 1
            if args.warmup:
                warmup_steps = len(self.trainloader) * (args.epochs/5)
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
            # 使用TL LOSS
            tbar.set_description("Loss: %.3f" % (total_loss / (i + 1)))
            self.writer.add_scalar('train total_loss', total_loss / (i + 1), running_iter)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], running_iter)

    def validation(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric = IOUandSek(num_classes=len(RS.ST_CLASSES))
        if self.args.save_mask:
            cmap = color_map()

        with torch.no_grad():
            for img1, img2, img3, img4, img5, img6, mask1, mask2, mask3, mask4, mask5, mask6, mask_bn, id in tbar:
                img1, img2, img3, img4, img5, img6 = img1.float().cuda(), img2.float().cuda(), img3.float().cuda(), img4.float().cuda(), img5.float().cuda(), img6.float().cuda()
                input = [img1, img2, img3, img4, img5, img6]  # 每个形状 [2, 4, 512, 512]
                input = torch.stack(input, dim=2)
                out = self.model(input)
                # 初始化存储列表
                time_series_predictions = []
                # 循环提取每个时相的数据
                for time_idx in range(6):
                    # 提取当前时相的预测结果
                    time_pred = out[:, :, time_idx, :, :]  # 索引操作
                    assert time_pred.shape == (2, 8, 512, 512)
                    # 添加到结果列表
                    time_series_predictions.append(time_pred)
                out1 = torch.argmax(time_series_predictions[0], dim=1).cpu().numpy()
                out2 = torch.argmax(time_series_predictions[1], dim=1).cpu().numpy()
                out3 = torch.argmax(time_series_predictions[2], dim=1).cpu().numpy()
                out4 = torch.argmax(time_series_predictions[3], dim=1).cpu().numpy()
                out5 = torch.argmax(time_series_predictions[4], dim=1).cpu().numpy()
                out6 = torch.argmax(time_series_predictions[5], dim=1).cpu().numpy()
                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out2, mask2.numpy())
                metric.add_batch(out3, mask3.numpy())
                metric.add_batch(out4, mask4.numpy())
                metric.add_batch(out5, mask5.numpy())
                metric.add_batch(out6, mask6.numpy())
                miou, oa, f1, precision, recall = metric.evaluate_classification()
                tbar.set_description(
                    "miou: %.4f, OA: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f" % (miou, oa, f1, precision, recall))
            if miou >= self.previous_best:
                model_path = "checkpoints/%s/%s/%s" % \
                             (self.args.data_name, self.args.Net_name, self.args.backbone)
                if not os.path.exists(model_path): os.makedirs(model_path)
                torch.save(self.model.state_dict(),
                           "checkpoints/%s/%s/%s/epoch%i_mIOU%.2f_Fscd%.2f_OA%.2f.pth" %
                           (self.args.data_name, self.args.Net_name, self.args.backbone, curr_epoch,
                            miou * 100, f1 * 100, oa * 100))
                self.previous_best = miou
            self.writer.add_scalar('val_mIOU', miou, curr_epoch)
            self.writer.add_scalar('val_F1', f1, curr_epoch)
            self.writer.add_scalar('val_OA', oa, curr_epoch)
            self.writer.add_scalar('val_precision', precision, curr_epoch)
            self.writer.add_scalar('val_recall', recall, curr_epoch)


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


