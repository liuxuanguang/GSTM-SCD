# from classification.models.tree_scan_utils.tree_scan_core import MinimumSpanningTree
import torch
from einops import rearrange, repeat
from GrootV.classification.models.grootv import GrootVLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T1 = torch.randn(1, 680, 4, 4).to(device)
T2 = torch.randn(1, 680, 4, 4).to(device)
B, C, H, W = T1.size()
T12 = torch.empty(B, C, H, 2 * W).cuda()
T12[:, :, :, 0: W] = T1
T12[:, :, :, W: 2 * W] = T2
model = GrootVLayer(channels=680).to(device)
T12 = T12.permute(0, 2, 3, 1)
out = model(T12)
b, c, h, w = T1.shape
fea4tree_hw = rearrange(T1, 'b d (h w) -> b d h w', h=h, w=w)  # B d L
# fusion_fea = MinimumSpanningTree("Cosine", torch.exp)