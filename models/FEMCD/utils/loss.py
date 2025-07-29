import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from functools import partial

import utils.loss_functions as fc
from utils.loss_functions import sigmoid_focal_loss, reduced_focal_loss


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        # ~表示布尔值取反
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss


class js_div(_Loss):
    def __init__(self):
        super().__init__()
        self.KLDivLoss = nn.KLDivLoss(reduction='none')
    def forward(self, p_out, q_out, get_softmax=True):
        if get_softmax:
            p_out = F.softmax(p_out, dim=1)
            q_out = F.softmax(q_out, dim=1)
        log_mean_out = ((p_out + q_out)/2).log()
        js = 0.5*self.KLDivLoss(log_mean_out, p_out) + 0.5*self.KLDivLoss(log_mean_out, q_out)
        return js

class Similarity(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, p_out, q_out, mask_bin):
        # mask_bin = 1 - mask_bin
        p = torch.argmax(p_out, dim=1)
        q = torch.argmax(q_out, dim=1)

        # print('mask_bin', mask_bin.shape)
        # exit(0)

        p[mask_bin == 1] = -1
        q[mask_bin == 1] = -1
        # q_out = (mask_bin * q_out).long()
        loss = 0.5*self.criterion(p_out, q.long()) + 0.5*self.criterion(q_out, p.long())
        return loss



class TverskyLoss(_Loss):
    __name__ = "dice_loss"

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, y_pr, y_gt):
        return 1 - fc.tversky(y_pr, y_gt, beta=1, eps=self.eps, threshold=None)

class BCELoss(_Loss):
    __name__ = "bce_loss"

    def __int__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, target):
        if type(outputs) in [tuple, list]:
            outputs = outputs[0]

        bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return bce_loss(outputs, target)


class WBCELoss(_Loss):
    __name__ = "wbce_loss"

    def __int__(self, reduction="mean", betas=[0.8, 0.2]):
        super(WBCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, target):
        bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return bce_loss(outputs, target)


class NLLLoss(_Loss):
    __name__ = "nll_loss"

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, outputs, target):
        """outputs.shape = [B, 2, H, W], target.shape = [B, H, W]"""
        nll_loss = nn.NLLLoss(weight=self.weight, reduction=self.reduction)
        # print(f"type: {type(outputs)}")
        # print(f"len: {len(outputs)}")
        # if type(outputs) in [tuple, list]:
        #     outputs = outputs[0]

        return nll_loss(outputs, target.long().squeeze(dim=1))


class OhemBCELoss(_Loss):
    __name__ = "OhemBCELoss"

    def __init__(self, weight=None, threshold=0.7, min_kept=1000, reduction="mean"):
        super(OhemBCELoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.ths = threshold
        self.min_kept = min_kept

    def forward(self, predict, target):
        bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        bce_loss_matrix = bce_loss.contiguous().view(-1, )

        # ========================================================== #
        batch_kept = self.min_kept * target.size(0)
        prob_out = torch.sigmoid(predict)

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        # gather: 以tmp_target [n, 1, h, w]的spatial dim[h, w]的值作为prob_out通道维dim=1的索引，取出相应通道的值
        prob = prob_out.gather(dim=1, index=tmp_target.unsqueeze(dim=1))
        mask = target.contiguous().view(-1,) == 1
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, self.thresh)
        # ========================================================== #

        sort_loss_matrix = bce_loss_matrix[mask][sort_indices]
        select_loss_matrix = sort_loss_matrix[sort_prob < threshold]

        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')


class DiceLoss(_Loss):
    __name__ = "dice_loss"

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - fc.f_score(y_pr, y_gt, beta=1, eps=self.eps, threshold=None, activation=self.activation)


class BCEDiceLoss(_Loss):

    __name__ = "bce_dice_loss"

    def __init__(self, dice_weight=0.5, pos_weight=None, eps=1e-7):
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        self.dice_loss = DiceLoss(eps)
        self.dice_weight = dice_weight

    def forward(self, outputs, target):

        return self.bce_loss(outputs, target) + self.dice_weight * self.dice_loss(outputs, target)


class MultiBCEDiceLoss(_Loss):

    __name__ = "multi_bcedice_loss"

    def __init__(self, dice_weight=1., pos_weight=False):
        super().__init__()

        if pos_weight:
            self.pos_weight = torch.ones([1]) * 20
        else:
            self.pos_weight = None

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=self.pos_weight)
        self.dice_loss = DiceLoss(eps=1e-7)
        self.dice_weight = dice_weight

    def forward(self, outputs, target):

        if type(outputs) is list and len(outputs) == 5:
            loss = 0.
            # print('*'*50)
            # print(self.pos_weight)
            # print(outputs[0].shape)
            for index, loss_weight in enumerate([1., 1., 1., 1., 1.]):
                loss += loss_weight * (self.bce_loss(outputs[index], target) + self.dice_weight * self.dice_loss(outputs[index], target))

        elif type(outputs) is list and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target) + self.dice_weight * self.dice_loss(outputs[-1], target)
        else:
            raise ValueError

        return loss


class MultiSEGLoss(_Loss):

    __name__ = "multi_seg_loss"

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, outputs, target):

        if type(outputs) is tuple and len(outputs) == 3:
            loss = 0.
            for index, loss_weight in enumerate([1.0, 0.2, 0.2]):
                loss += loss_weight * self.bce_loss(outputs[index], target)

        elif type(outputs) is tuple and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target)
        else:
            raise ValueError

        return loss


class JaccardLoss(_Loss):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - fc.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class JaccardLogLoss(_Loss):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        iou = fc.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)
        return - torch.log(iou)


class BCEJaccardLoss(_Loss):
    """
    Loss = -\alpha * SoftJaccard + (1 - \alpha) * BCE

    """

    __name__ = "bce_jaccard_loss"

    def __init__(self, jaccard_weight=0.3, use_ohem=False):
        super().__init__()
        if use_ohem:
            print("=> use ohem")
            self.bce_loss = OhemBCELoss(weight=None, threshold=0.7, min_kept=100000, reduction="mean")
        else:
            self.bce_loss = nn.BCELoss(reduction="mean")

        self.jaccard_weight = jaccard_weight
        # self.jaccard_weight = True

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.bce_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()


            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss

    def forward(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.bce_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            # forward output without nn.Sigmoid
            # jaccard_output = torch.sigmoid(outputs)
            jaccard_output = outputs
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class BCEFocalLoss(_Loss):
    __name__ = "bce_focal_loss"

    def __init__(
            self,
            alpha=0.5,
            gamma=2,
            ignore_index=None,
            reduction='mean',
            reduced=False,
            threshold=0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduced:
            self.focal_loss = partial(reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction)
        else:
            self.focal_loss = partial(sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem."""
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = self.focal_loss(label_input, label_target)
        return loss


class FocalJaccardLoss(BCEFocalLoss):
    """Loss = - \alpha * SoftJaccard + (1 - \alpha) Focal"""

    __name__ = "focal_jaccard_loss"

    def __init__(self, alpha=0.5, gamma=2, reduction='mean', reduced=False, jaccard_weight=0.7, threshold=0.5):
        super().__init__()
        self.loss = BCEFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction, reduced=reduced, threshold=threshold)
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = (1 - self.jaccard_weight) * self.loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class MultiBCELoss(torch.nn.BCEWithLogitsLoss):
    """2D Cross Entropy Loss with Multi-Loss"""

    __name__ = "multi_bce_loss"

    def __init__(self, weight=None, reduction="mean"):
        super(MultiBCELoss, self).__init__(weight, reduction)

    def forward(self, preds, target):

        pred1, pred2, pred3 = tuple(preds)

        loss1 = super(MultiBCELoss, self).forward(pred1, target)
        loss2 = super(MultiBCELoss, self).forward(pred2, target)
        loss3 = super(MultiBCELoss, self).forward(pred3, target)
        loss = loss1 + loss2 + loss3
        return loss


class MultiBCEJaccardLoss(BCEJaccardLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""

    __name__ = "multi_bcejaccard_loss"

    def __init__(self, jaccard_weight=0.3, aux_weight=0., pos_weight=0., use_ohem=False):
        super(MultiBCEJaccardLoss, self).__init__(jaccard_weight, use_ohem)
        self.aux_weight = aux_weight
        self.pos_weight = pos_weight

        if pos_weight is not None:
            # self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
            self.bce_loss = nn.BCELoss(reduction="mean")
    def __call__(self, *preds, target):
        """
         __call__(self, *preds, target) for dp: loss = self.loss(predictions, target=y)
         __call__(self, preds, target) for ddp: loss = self.loss(predictions, target=y)
        """

        assert type(preds) in [list, tuple], "net forward preds must return list or tuple."
        # print(f"type_preds = {type(preds[0])}, len_preds = {len(preds[0])}")
        # print(f"type_preds = {type(preds)}, len_preds = {len(preds)}")
        # exit(1)
        # print('loss中的维度', preds)
        # print('*'*50)
        # print('pred[0]的维度', len(preds[0]), len(preds))
        # print('len(preds[0]', len(preds[0]))
        if len(preds[0]) >= 1:
            preds = preds[0]
            # print(preds[0].shape)
        # print('len(preds)', len(preds))
        # exit(0)
        if len(preds) == 3:
            # print(f"preds(type, len) = ({type(preds)}, {len(preds)})")
            # preds(type, len) = (<class 'tuple'>, 3)

            pred1, pred2, pred3 = tuple(preds)

            loss1 = super(MultiBCEJaccardLoss, self).forward(pred1, target)
            loss2 = super(MultiBCEJaccardLoss, self).forward(pred2, target)

            if self.pos_weight is not None:
                loss3 = self.bce_loss(pred3, target)
                loss = loss1 + self.aux_weight*loss2 + self.pos_weight*loss3
            else:
                loss3 = super(MultiBCEJaccardLoss, self).forward(pred3, target)
                loss = loss1 + loss2 + loss3

        elif len(preds) == 2:
            pred1, pred2 = tuple(preds)
            loss1 = super(MultiBCEJaccardLoss, self).forward(pred1, target)

            """
            if self.pos_weight is not None:
                loss2 = self.bce_loss(pred2, target)
                loss = loss1 + self.pos_weight * loss2
            elif self.aux_weight is not None:
                loss2 = super(MultiBCEJaccardLoss, self).forward(pred2, target)
                loss = loss1 + self.aux_weight * loss2
            """

            loss2 = super(MultiBCEJaccardLoss, self).forward(pred2, target)
            loss = loss1 + self.aux_weight * loss2

        else:
            loss = super(MultiBCEJaccardLoss, self).forward(preds[0], target)

        return loss


class AffinityLoss(_Loss):

    __name__ = "affinity_loss"

    def __init__(self, eps=1e-15, ths=0.5):
        super().__init__()
        self.eps = eps
        self.ths = ths

    def __call__(self, pred_affinity_org, gt_affinity):
        pred_affinity_org = torch.sigmoid(pred_affinity_org.detach())
        pred_affinity = pred_affinity_org
        pred_affinity[pred_affinity_org >= self.ths] = 1.0
        pred_affinity[pred_affinity_org < self.ths] = 0.0
        gt_affinity = gt_affinity.float()

        # [N, HW, HW] => [N, HW, 1] 行和sum(dim=-1), 同时去掉最后一维dims = dims - 1
        tp = (pred_affinity * gt_affinity).sum(dim=-1)
        tp_plus_fp = pred_affinity.sum(dim=-1)
        tp_plus_fn = gt_affinity.sum(dim=-1)

        tn = ((torch.ones_like(pred_affinity) - pred_affinity) * (torch.ones_like(gt_affinity) - gt_affinity)).sum(dim=-1)
        tn_plus_fp = (1 - gt_affinity).sum(dim=-1)

        intra_loss_precision = torch.log((tp + self.eps) / (tp_plus_fp + self.eps))
        intra_loss_recall = torch.log((tp + self.eps) / (tp_plus_fn + self.eps))
        inter_loss_specifity = torch.log((tn + self.eps) / (tn_plus_fp + self.eps))

        # mean(dim=1), 对每一行的和取平均，同时dims减一
        gloabal_affinity_loss = -(intra_loss_precision + intra_loss_recall + inter_loss_specifity).mean(dim=1)

        # mean(dim=0), 对批量求平均
        gloabal_affinity_loss = gloabal_affinity_loss.mean(dim=0)

        return gloabal_affinity_loss


class MultiAffinityBCEJaccardLoss(BCEJaccardLoss):
    __name__ = "multi_affinity_bcejaccard_loss"

    def __init__(
        self, jaccard_weight=0.7,
        aux_weight=0.4,
        aff_weight=1.0,
        aff_global_weight=1.0,
        aff_unary_weight=1.0,
        aff_eps=1e-15, aff_ths=0.5
    ):
        super(MultiAffinityBCEJaccardLoss, self).__init__(jaccard_weight)
        self.aux_weight = aux_weight

        self.aff_weight = aff_weight
        self.aff_global_weight = aff_global_weight
        self.aff_unary_weight = aff_unary_weight
        if None not in [aff_unary_weight, aff_global_weight]:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
            self.aff_loss = AffinityLoss(aff_eps, aff_ths)

    def __call__(self, preds, target):

        assert type(preds) in [list, tuple], "net forward preds must return list or tuple."

        if len(preds) == 3:
            # torch.autograd.set_detect_anomaly(True)

            pred1, pred2, pred3 = tuple(preds)

            loss1 = super(MultiAffinityBCEJaccardLoss, self).forward(pred1, target)
            loss2 = super(MultiAffinityBCEJaccardLoss, self).forward(pred2, target)

            b, c, h, w = target.shape
            downsample_target = F.interpolate(target, size=(int(h // 16), int(w // 16)), mode="bilinear", align_corners=True).view(b, c, -1)
            idea_aff_map = torch.matmul(
                downsample_target.permute(0, 2, 1).contiguous(),
                downsample_target
            )
            # print("idea_aff_map : ", idea_aff_map.shape, idea_aff_map.max(), idea_aff_map.min())

            if self.aff_global_weight is not None:
                # loss3 = self.aff_unary_weight * self.bce_loss(pred3, idea_aff_map) + \
                #         self.aff_global_weight * self.aff_loss(pred3, idea_aff_map)

                loss3 = self.aff_unary_weight * self.bce_loss(pred3, idea_aff_map) + \
                        self.aff_global_weight * self.aff_loss(pred3, idea_aff_map)

                loss = loss1 + self.aux_weight * loss2 + self.aff_weight * loss3
            else:
                # loss3 = super(MultiAffinityBCEJaccardLoss, self).forward(pred3, target)
                loss3 = self.aff_unary_weight * self.bce_loss(pred3, target)
                loss = loss1 + self.aux_weight * loss2 + loss3

        elif len(preds) == 2:
            pred1, pred2 = tuple(preds)

            loss1 = super(MultiAffinityBCEJaccardLoss, self).forward(pred1, target)
            loss2 = super(MultiAffinityBCEJaccardLoss, self).forward(pred2, target)
            loss = loss1 + self.aux_weight * loss2

        else:
            loss = super(MultiAffinityBCEJaccardLoss, self).forward(preds[0], target)

        return loss


class MultiJacSEGLoss(_Loss):

    __name__ = "multi_jacseg_loss"

    def __init__(self):
        super().__init__()
        self.bce_loss = BCEJaccardLoss()

    def forward(self, outputs, target):

        if type(outputs) is tuple and len(outputs) == 3:
            loss = 0.
            for index, loss_weight in enumerate([1.0, 0.2, 0.2]):
                loss += loss_weight * self.bce_loss(outputs[index], target)

        elif type(outputs) is tuple and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target)
        else:
            raise ValueError

        return loss


class DSIFN_Loss(_Loss):

    __name__ = "DSIFN_loss"

    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # self.bce_loss = torch.nn.BCELoss()
        self.dice_loss = DiceLoss(eps=1e-7)
        self.sigmoid = torch.nn.Sigmoid()
    def __dice__(self, output, target):

        smooth = 1.
        iflat = self.sigmoid(output).view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        return dice_loss

    def forward(self, outputs, target):
        # print(target)
        
        if type(outputs) is tuple and len(outputs) == 5:
            loss = 0.
            for index, loss_weight in enumerate([1.0] * 5):
                loss += loss_weight * (self.bce_loss(outputs[index], target) + self.__dice__(outputs[index], target))
                # loss += loss_weight * (self.bce_loss(outputs[index], target) + self.dice_loss(outputs[index], target))
        elif type(outputs) is tuple and len(outputs) == 1:
            loss = self.bce_loss(outputs[-1], target) + self.__dice__(outputs[-1], target)
        else:
            raise ValueError

        return loss