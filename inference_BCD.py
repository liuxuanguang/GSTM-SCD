import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import PIL.Image as Image
from datasets.change_detection import ChangeDetection_LEVIR_CD
from models.proposed_BiGrootV import BiGrootV3D_SV3_BCD as Net
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
from utils.palette import color_map
from utils.metric import IOUandSek
from tqdm import tqdm
from torch.utils.data import DataLoader
from thop import profile
import time
import argparse
import copy

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Building Change Detection')
        parser.add_argument("--data_name", type=str, default="FZ")
        parser.add_argument("--Net_name", type=str, default="Proposed-FZ-Testing")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                           help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default=r"/media/lenovo/课题研究/博士小论文数据/语义变化检测数据集/S1_EXPEND")
        parser.add_argument("--load_from", type=str,
                            default=r"/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/LEVIR-CD+/LEVIR-CD+BiGrootV_SV3_FZ-0305/resnet34/epoch96_Recall99.29_Precision97.64_OA99.20_F198.46_IoU96.96._KC97.91.pth")
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

def inference(args):
    begin_time = time.time()
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results', args.data_name, args.Net_name, args.backbone)
    pred_save_path3 = os.path.join(pred_dir, 'pred_change')

    if not os.path.exists(pred_save_path3): os.makedirs(pred_save_path3)

    testset = ChangeDetection_LEVIR_CD(root=args.data_root, mode="test")
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                pin_memory=True, num_workers=0, drop_last=False)
    model = Net(args.backbone, args.pretrained, len(ChangeDetection_LEVIR_CD.CLASSES)-1, args.lightweight, args.M, args.Lambda)

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)

    model = model.cuda()
    model.eval()

    # calculate Pamrams and FLOPs
    for vi, data in enumerate(testloader):
        if vi == 0:
            img1, img2, _, id = data
            img1, img2 = img1.cuda().float(), img2.cuda().float()
            break
    FLOPs, Params = profile(model, (img1, img2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

    tbar = tqdm(testloader)
    metric = IOUandSek(num_classes=len(ChangeDetection_LEVIR_CD.CLASSES))
    with torch.no_grad():
        for img1, img2, label1, id in tbar:
            img1, img2 = img1.cuda(), img2.cuda()

            out_bn = model(img1, img2)
            out_bn = ((out_bn > 0.5).cpu().numpy()).astype(np.uint8)

            cmap = color_map()

            for i in range(out_bn.shape[0]):

                mask_bn = Image.fromarray(out_bn[i]*255)
                mask_bn.save(os.path.join(pred_save_path3, id[i]))

            metric.add_batch(out_bn, label1.numpy())

        Recall, Precision, OA, F1, IoU, KC = metric.evaluate_BCD()

        print('==>Recall', Recall)
        print('==>Precision', Precision)
        print('==>OA', OA)
        print('==>F1', F1)
        print('==>IoU', IoU)
        print('==>KC', KC)

        time_use = time.time() - begin_time

    metric_file = os.path.join(pred_dir, 'metric.txt')
    f = open(metric_file, 'w', encoding='utf-8')
    f.write("Data：" + str(args.data_name) + '\n')
    f.write("model：" + str(args.Net_name) + '\n')
    f.write("##################### metric #####################"+'\n')
    f.write("infer time (s) ：" + str(round(time_use, 2)) + '\n')
    f.write("Params (Mb) ：" + str(round(Params/1e6, 2)) + '\n')
    f.write("FLOPs (Gbps) ：" + str(round(FLOPs/1e9, 2)) + '\n')
    f.write('\n')
    f.write("Recall (%) ：" + str(round(Recall * 100, 2)) + '\n')
    f.write("Precision (%) ：" + str(round(Precision * 100, 2)) + '\n')
    f.write("OA (%) ：" + str(round(OA * 100, 2)) + '\n')
    f.write("F1 (%) ：" + str(round(F1 * 100, 2)) + '\n')
    f.write("IoU (%) ：" + str(round(IoU * 100, 2)) + '\n')
    f.write("KC (%) ：" + str(round(KC * 100, 2)) + '\n')

    f.close()


if __name__ == "__main__":
    args = Options().parse()
    inference(args)