"""
Multi-temporal U-TAE Implementation
Inspired by U-TAE Implementation (Vivien Sainte Fare Garnot (github/VSainteuf))
"""
import numpy as np
import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoder


class MultiLTAE(nn.Module):
    def __init__(
            self,
            in_channels=128,
            n_head=16,
            d_k=4,
            dropout=0.2,
            T=730,
            offset=0,
            return_att=False,
            positional_encoding=True
    ):
        super(MultiLTAE, self).__init__()
        self.in_channels = in_channels
        self.mlp = [in_channels, in_channels]
        self.return_att = return_att
        self.n_head = n_head
        self.d_model = in_channels
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head, offset=offset
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape  # 输入形状: [batch, time, channels, height, width]

        # 处理pad_mask (从 [B, T] 扩展为 [B, T, H, W])
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            pad_mask = pad_mask.expand(-1, -1, h, w)  # [B, T, H, W]
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        # 重排维度 ([B, time, C, H, W] -> [B*H*W, time, C])
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        # 处理位置编码
        if self.positional_encoder is not None and batch_positions is not None:
            # 确保batch_positions是二维张量 [batch, time]
            if batch_positions.dim() == 4:
                batch_positions = batch_positions.squeeze(-1).squeeze(-1)

            # 扩展为四维 [B, T, H, W]
            bp = batch_positions.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
            bp = bp.expand(-1, -1, h, w)  # [B, T, H, W]
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

            out = out + self.positional_encoder(bp)

        # 注意力头处理
        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        # 合并多头输出
        out = out.permute(1, 2, 0, 3).contiguous().view(sz_b * h * w, seq_len, -1)

        # MLP处理
        out = self.dropout(self.mlp(out.view(sz_b * h * w * seq_len, -1)))
        if self.out_norm is not None:
            out = self.out_norm(out)

        # 恢复原始形状 [B, time, C, H, W]
        out = out.view(sz_b, h, w, seq_len, -1).permute(0, 3, 4, 1, 2)

        # 处理注意力输出
        attn = attn.view(self.n_head, sz_b, h, w, seq_len, seq_len).permute(
            0, 1, 4, 5, 2, 3
        )

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = self.fc1_q(v).view(sz_b, seq_len, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k).permute(0, 2, 1)  # (n*b) x dk x lk

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        output, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(n_head, sz_b, seq_len, seq_len)
        output = output.view(n_head, sz_b, seq_len, d_in // n_head)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(k, q)
        attn = attn / self.temperature

        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn
