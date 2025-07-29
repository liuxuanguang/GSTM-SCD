import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
):
    """
    Compute binary focal loss between target and output logits.
    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    Args:
        outputs: Tensor of arbitrary shape
        targets: Tensor of the same shape as input
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            "none" | "mean" | "sum" | "batchwise_mean".
            "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
            elements in the output,
            "sum": the output will be summed.
    See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py  # noqa: E501
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none"
    )
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def reduced_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    gamma: float = 2.0,
    reduction="mean"
):
    """
    Compute reduced focal loss between target and output logits.
    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    Args:
        outputs: Tensor of arbitrary shape
        targets: Tensor of the same shape as input
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            "none" | "mean" | "sum" | "batchwise_mean".
            "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
            elements in the output,
            "sum": the output will be summed.
            Note: :attr:`size_average` and :attr:`reduce`
            are in the process of being deprecated,
            and in the meantime, specifying either of those two args
            will override :attr:`reduction`.
            "batchwise_mean" computes mean loss per sample in batch.
            Default: "mean"
    See https://arxiv.org/abs/1903.01347
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none"
    )
    pt = torch.exp(logpt)

    # compute the loss
    focal_reduction = ((1. - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1

    loss = -focal_reduction * logpt

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def iou(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    iou = tp / (tp + fp + fn)

    Source:
        https://github.com/catalyst-team/catalyst/
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """
    
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )
    pr = activation_fn(pr)
    if threshold is not None:
        pr = (pr > threshold).float()

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """

    f1_score = 2*tp / (2*tp + fn + fp)

    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F_1 score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)
    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def tversky(pr, gt, beta=2, eps=1e-7, threshold=None):
    """

    f1_score = 2*tp / (2*tp + fn + fp)

    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F_1 score
    """
    # activation and binary threshold
    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = (tp + eps) \
            / (tp +  0.3*fp + 0.7*fn + eps)
    return score

