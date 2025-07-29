import torch
import torch.nn as nn


class FSBM(nn.Module):
    def __init__(self, in_channel, k):
        super(FSBM, self).__init__()
        self.k = k
        self.stripconv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, fm):
        b, c, w, h = fm.shape
        fms = torch.split(fm, w // self.k, dim=2)
        fms_conv = map(self.stripconv, fms)
        fms_pool = list(map(self.avgpool, fms_conv))
        fms_pool = torch.cat(fms_pool, dim=2)
        fms_softmax = torch.softmax(fms_pool, dim=2)  # every parts has one score [B*C*K*1]
        fms_softmax_boost = torch.repeat_interleave(fms_softmax, w // self.k, dim=2)
        alpha = 0.5
        fms_boost = fm + alpha*(fm * fms_softmax_boost)

        beta = 0.5
        fms_max = torch.max(fms_softmax, dim=2, keepdim=True)[0]
        fms_softmax_suppress = torch.clamp((fms_softmax < fms_max).float(), min=beta)
        fms_softmax_suppress = torch.repeat_interleave(fms_softmax_suppress, w // self.k, dim=2)
        fms_suppress = fm * fms_softmax_suppress

        return fms_boost, fms_suppress
    
